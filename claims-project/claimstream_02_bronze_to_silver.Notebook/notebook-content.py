# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "264f79fb-2fbe-47e8-ad9b-229fafec37ed",
# META       "default_lakehouse_name": "lh_claimstream",
# META       "default_lakehouse_workspace_id": "d0015547-fce2-4e72-852f-fe93fd125cd4",
# META       "known_lakehouses": [
# META         {
# META           "id": "264f79fb-2fbe-47e8-ad9b-229fafec37ed"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # ClaimStream: 02 Bronze to Silver
# 
# Reads the `bronze_*` tables and writes conformed, validated `silver_*` tables:
# 
# | Output | What happens |
# |---|---|
# | `silver_customers` | exact duplicates removed, province codes upper-cased, names trimmed, email normalised + `email_missing` flag |
# | `silver_policies` | **SCD Type 2** built from the bronze change log: `valid_from` (inclusive), `valid_to` (exclusive), `is_current` |
# | `silver_claims` | typed, deduplicated, `report_lag_days` derived; claims with `reported_date < loss_date` are quarantined |
# | `silver_claim_payments` | deduplicated; invalid amounts, and payments whose parent claim was quarantined, are quarantined |
# | `silver_coverage_types` | straight copy |
# | `silver_quarantine` | every rejected row with a reason and the original record as JSON (overwritten each run) |
# | `silver_dq_log` | one row per rule per run: how many rows failed and what was done (appended, so you keep history) |
# 
# Attach the same default Lakehouse as in notebook 01. Silver tables are overwritten on each run, so the notebook is safe to rerun.


# CELL ********************

from datetime import datetime, timezone
from pyspark.sql import functions as F, Window

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
HIGH_DATE = "9999-12-31"          # valid_to for the current version of a policy

dq_log = []                       # (run_id, table, rule, rows_failed, action)
quarantine_parts = []             # DataFrames in a common schema

def log_dq(table, rule, rows_failed, action):
    dq_log.append((RUN_ID, table, rule, int(rows_failed), action))
    print(f"[DQ] {table:<24} {rule:<34} {int(rows_failed):>8,}  -> {action}")

def to_quarantine(df, source_table, key_col, reason):
    return df.select(
        F.lit(source_table).alias("source_table"),
        F.col(key_col).cast("string").alias("record_key"),
        F.lit(reason).alias("reason"),
        F.to_json(F.struct(*df.columns)).alias("payload"),
        F.lit(RUN_ID).alias("run_id"),
        F.current_timestamp().alias("quarantined_at"),
    )

def write_silver(df, name):
    out = df.withColumn("_silver_loaded_at", F.current_timestamp())
    (out.write.mode("overwrite").option("overwriteSchema", "true")
        .format("delta").saveAsTable(f"silver_{name}"))
    print(f"wrote silver_{name}: {out.count():,} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- silver_customers ------------------------------------------------------
b = spark.table("bronze_customers")
n_raw = b.count()

w = Window.partitionBy("customer_id").orderBy(F.col("_ingested_at").desc())
dedup = (b.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn"))
log_dq("customers", "duplicate_customer_id", n_raw - dedup.count(), "kept one row per customer_id")

n_lower = dedup.filter(F.col("province") != F.upper(F.col("province"))).count()
log_dq("customers", "province_not_uppercase", n_lower, "upper-cased")

n_null_email = dedup.filter(F.col("email").isNull()).count()
log_dq("customers", "email_missing", n_null_email, "kept, flagged email_missing")

silver_customers = dedup.select(
    "customer_id",
    F.initcap(F.trim("first_name")).alias("first_name"),
    F.initcap(F.trim("last_name")).alias("last_name"),
    F.lower(F.trim("email")).alias("email"),
    F.col("email").isNull().alias("email_missing"),
    F.upper(F.trim("province")).alias("province"),
    "birth_date",
    "created_date",
)
write_silver(silver_customers, "customers")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- silver_policies (SCD Type 2) -----------------------------------------
# Bronze is a change log: one row per policy version, ordered by record_updated_at.
# Each version is valid from its own record_updated_at (inclusive) until the next
# version's record_updated_at (exclusive); the latest version runs to 9999-12-31.
p = spark.table("bronze_policies")
n_raw = p.count()

p = p.dropDuplicates(["policy_id", "record_updated_at"])
log_dq("policies", "duplicate_policy_version", n_raw - p.count(), "dropped")

w = Window.partitionBy("policy_id").orderBy("record_updated_at")
silver_policies = (
    p.withColumn("version_no", F.row_number().over(w))
     .withColumn("next_from", F.lead("record_updated_at").over(w))
     .withColumn("valid_from", F.col("record_updated_at"))
     .withColumn("valid_to", F.coalesce(F.col("next_from"), F.to_date(F.lit(HIGH_DATE))))
     .withColumn("is_current", F.col("next_from").isNull())
     .withColumn("policy_version_id", F.concat_ws("_v", F.col("policy_id"), F.col("version_no")))
     .select("policy_version_id", "policy_id", "version_no", "customer_id", "coverage_code",
             "effective_date", "end_date", "annual_premium", "status",
             "valid_from", "valid_to", "is_current")
)

n_orphan = silver_policies.join(silver_customers.select("customer_id"), "customer_id", "left_anti").count()
log_dq("policies", "customer_not_in_silver", n_orphan, "flag only (expected 0)")
write_silver(silver_policies, "policies")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- silver_claims ----------------------------------------------------------
c = spark.table("bronze_claims")
n_raw = c.count()
c = c.dropDuplicates(["claim_id"])
log_dq("claims", "duplicate_claim_id", n_raw - c.count(), "dropped")

bad_dates = F.coalesce(F.col("reported_date") < F.col("loss_date"), F.lit(False))
rejected_claims = c.filter(bad_dates)
good_claims = c.filter(~bad_dates)
log_dq("claims", "reported_before_loss", rejected_claims.count(), "quarantined")
quarantine_parts.append(to_quarantine(rejected_claims, "claims", "claim_id", "reported_date earlier than loss_date"))

silver_claims = good_claims.select(
    "claim_id", "policy_id", "loss_date", "reported_date",
    F.datediff("reported_date", "loss_date").alias("report_lag_days"),
    F.upper(F.trim("status")).alias("status"),
    (F.upper(F.trim("status")) == "OPEN").alias("is_open"),
    "cause_of_loss", "reserve_amount",
)

n_orphan = silver_claims.join(silver_policies.select("policy_id").distinct(), "policy_id", "left_anti").count()
log_dq("claims", "policy_not_in_silver", n_orphan, "flag only (expected 0)")
write_silver(silver_claims, "claims")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- silver_claim_payments ------------------------------------------------
pay = spark.table("bronze_claim_payments")
n_raw = pay.count()
pay = pay.dropDuplicates(["payment_id"])
log_dq("claim_payments", "duplicate_payment_id", n_raw - pay.count(), "dropped")

bad_amount = F.coalesce(F.col("amount") <= 0, F.lit(True))
rej_amount = pay.filter(bad_amount)
log_dq("claim_payments", "non_positive_amount", rej_amount.count(), "quarantined")
quarantine_parts.append(to_quarantine(rej_amount, "claim_payments", "payment_id", "amount missing or <= 0"))
pay = pay.filter(~bad_amount)

# a payment is only valid if its parent claim made it into silver_claims
valid_claims = silver_claims.select("claim_id")
rej_parent = pay.join(valid_claims, "claim_id", "left_anti")
log_dq("claim_payments", "parent_claim_missing_or_rejected", rej_parent.count(), "quarantined")
quarantine_parts.append(to_quarantine(rej_parent, "claim_payments", "payment_id", "parent claim missing or quarantined"))
silver_payments = pay.join(valid_claims, "claim_id", "left_semi")

write_silver(silver_payments, "claim_payments")
write_silver(spark.table("bronze_coverage_types"), "coverage_types")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- quarantine + DQ log --------------------------------------------------
quarantine = quarantine_parts[0]
for part in quarantine_parts[1:]:
    quarantine = quarantine.unionByName(part)
(quarantine.write.mode("overwrite").option("overwriteSchema", "true")
    .format("delta").saveAsTable("silver_quarantine"))
print("wrote silver_quarantine:", spark.table("silver_quarantine").count(), "rows")

dq_df = spark.createDataFrame(dq_log, "run_id string, table_name string, rule string, rows_failed long, action string") \
             .withColumn("logged_at", F.current_timestamp())
dq_df.write.mode("append").format("delta").saveAsTable("silver_dq_log")
print("appended", dq_df.count(), "rows to silver_dq_log for run", RUN_ID)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- Reconciliation checks (these fail loudly if something is off) --------
cnt = lambda t: spark.table(t).count()
q = spark.table("silver_quarantine")

# every bronze row must be accounted for: kept, dropped as a duplicate (logged), or quarantined
rej_claims = q.filter(F.col("source_table") == "claims").count()
rej_pay = q.filter(F.col("source_table") == "claim_payments").count()
assert cnt("silver_claims") + rej_claims == cnt("bronze_claims"), "claims do not reconcile"
assert cnt("silver_claim_payments") + rej_pay == cnt("bronze_claim_payments"), "payments do not reconcile"

# SCD2 integrity: exactly one current row per policy, and no gaps/overlaps between versions
sp = spark.table("silver_policies")
n_policies = sp.select("policy_id").distinct().count()
assert sp.filter("is_current").count() == n_policies, "each policy must have exactly one current version"
w = Window.partitionBy("policy_id").orderBy("valid_from")
broken = (sp.withColumn("prev_to", F.lag("valid_to").over(w))
            .filter(F.col("prev_to").isNotNull() & (F.col("prev_to") != F.col("valid_from"))).count())
assert broken == 0, "SCD2 versions have gaps or overlaps"
assert cnt("silver_customers") == spark.table("silver_customers").select("customer_id").distinct().count()

print("All reconciliation checks passed.")
for t in ["customers", "policies", "claims", "claim_payments", "coverage_types"]:
    print(f"{t:<16} bronze {cnt('bronze_' + t):>8,}  silver {cnt('silver_' + t):>8,}")

display(spark.table("silver_dq_log").filter(F.col("run_id") == RUN_ID).orderBy("table_name", "rule"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
