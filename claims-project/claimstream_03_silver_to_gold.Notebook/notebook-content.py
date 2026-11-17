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

# # ClaimStream: 03 Silver to Gold
# 
# Builds the star schema that the semantic model will sit on.
# 
# | Table | Grain | Notes |
# |---|---|---|
# | `gold_dim_date` | one row per day | 2019-01-01 to 2026-12-31 |
# | `gold_dim_customer` | one row per customer | no names/emails: only what analysis needs (province, age band) |
# | `gold_dim_policy` | one row per policy **version** | the SCD2 table from silver |
# | `gold_dim_coverage` | one row per coverage type | |
# | `gold_fact_claims` | one row per claim | joined to the policy version **in force on the loss date** (point-in-time join) |
# | `gold_fact_premium_monthly` | policy version x month | earned premium, pro-rated by days |
# | `gold_mart_loss_ratio_monthly` | month x coverage x province | claim counts, paid, incurred, earned premium, loss ratio |
# 
# **Business rules (change them here if you disagree):**
# - `incurred` for a closed claim = total paid. For an open claim = the larger of its reserve and what has been paid so far.
# - `outstanding` = incurred minus paid (zero for closed and denied claims).
# - Earned premium = annual premium x days in force in the month / 365, using the policy version's premium for the days that version was in force.
# - Loss ratio = incurred / earned premium, by **loss month** (accident-period basis).
# 
# Attach the same default Lakehouse as in notebooks 01 and 02.


# CELL ********************

from pyspark.sql import functions as F

AS_OF_DATE = "2026-10-29"      # must match AS_OF in notebook 01
DATE_START, DATE_END = "2019-01-01", "2026-12-31"

def write_gold(df, name):
    (df.write.mode("overwrite").option("overwriteSchema", "true")
       .format("delta").saveAsTable(f"gold_{name}"))
    print(f"wrote gold_{name}: {spark.table('gold_' + name).count():,} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- Dimensions -------------------------------------------------------------
dim_date = (
    spark.sql(f"SELECT explode(sequence(to_date('{DATE_START}'), to_date('{DATE_END}'), interval 1 day)) AS date")
    .select(
        (F.year("date") * 10000 + F.month("date") * 100 + F.dayofmonth("date")).alias("date_key"),
        "date",
        F.year("date").alias("year"),
        F.quarter("date").alias("quarter"),
        F.month("date").alias("month"),
        F.date_format("date", "MMM").alias("month_name"),
        F.trunc("date", "month").alias("month_start"),
        F.date_format("date", "yyyy-MM").alias("year_month"),
        F.date_format("date", "E").alias("day_name"),
        F.dayofweek("date").isin(1, 7).alias("is_weekend"),
    )
)
write_gold(dim_date, "dim_date")

sc = spark.table("silver_customers")
age = F.floor(F.months_between(F.to_date(F.lit(AS_OF_DATE)), F.col("birth_date")) / 12)
dim_customer = sc.select(
    "customer_id", "province", "created_date",
    F.when(age < 25, "18-24").when(age < 35, "25-34").when(age < 45, "35-44")
     .when(age < 55, "45-54").when(age < 65, "55-64").otherwise("65+").alias("age_band"),
)
write_gold(dim_customer, "dim_customer")

dim_policy = spark.table("silver_policies").drop("_silver_loaded_at")
write_gold(dim_policy, "dim_policy")

dim_coverage = spark.table("silver_coverage_types").drop("_silver_loaded_at")
write_gold(dim_coverage, "dim_coverage")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- gold_fact_claims (point-in-time join to the policy version in force) ----
claims = spark.table("silver_claims")
pay = (spark.table("silver_claim_payments").groupBy("claim_id")
       .agg(F.sum("amount").alias("paid_amount"), F.count("*").alias("payment_count")))

pv = dim_policy.select(
    F.col("policy_id").alias("pv_policy_id"), "policy_version_id", "coverage_code",
    "customer_id", "annual_premium", "valid_from", "valid_to",
)

fact_claims = (
    claims.join(pv, (claims.policy_id == pv.pv_policy_id)
                    & (claims.loss_date >= pv.valid_from) & (claims.loss_date < pv.valid_to), "left")
          .join(pay, "claim_id", "left")
          .withColumn("paid_amount", F.coalesce("paid_amount", F.lit(0.0)))
          .withColumn("payment_count", F.coalesce("payment_count", F.lit(0)))
          .withColumn("incurred_amount",
                      F.when(F.col("is_open"), F.greatest("reserve_amount", "paid_amount")).otherwise(F.col("paid_amount")))
          .withColumn("outstanding_amount", F.col("incurred_amount") - F.col("paid_amount"))
          .withColumn("loss_date_key", F.date_format("loss_date", "yyyyMMdd").cast("int"))
          .select("claim_id", "policy_version_id", "policy_id", "customer_id", "coverage_code",
                  "loss_date_key", "loss_date", "reported_date", "report_lag_days",
                  "status", "is_open", "cause_of_loss",
                  "paid_amount", "payment_count", "reserve_amount", "incurred_amount", "outstanding_amount")
)
write_gold(fact_claims, "fact_claims")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- gold_fact_premium_monthly (earned premium, pro-rated by day) ------------
# Each policy version earns its premium from valid_from until the earlier of valid_to,
# the policy's cancellation date, or the as-of date (all end dates are exclusive).
asof = F.to_date(F.lit(AS_OF_DATE))
v = (dim_policy
     .withColumn("start_d", F.col("valid_from"))
     .withColumn("end_d", F.least(F.col("valid_to"), F.coalesce(F.col("end_date"), asof), asof))
     .filter(F.col("end_d") > F.col("start_d")))

months = v.withColumn("month_start", F.explode(F.expr(
    "sequence(trunc(start_d, 'month'), trunc(date_sub(end_d, 1), 'month'), interval 1 month)")))

fact_premium = (
    months
    .withColumn("month_end", F.add_months("month_start", 1))
    .withColumn("days_in_force", F.datediff(F.least("end_d", "month_end"), F.greatest("start_d", "month_start")))
    .filter(F.col("days_in_force") > 0)
    .withColumn("earned_premium", F.col("annual_premium") * F.col("days_in_force") / 365.0)
    .select("policy_version_id", "policy_id", "customer_id", "coverage_code",
            "month_start", "days_in_force", "annual_premium", "earned_premium")
)
write_gold(fact_premium, "fact_premium_monthly")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- gold_mart_loss_ratio_monthly --------------------------------------------
fp = spark.table("gold_fact_premium_monthly")
fc = spark.table("gold_fact_claims")
dc = spark.table("gold_dim_customer").select("customer_id", "province")
keys = ["month_start", "coverage_code", "province"]

prem_agg = (fp.join(dc, "customer_id").groupBy(*keys)
            .agg(F.sum("earned_premium").alias("earned_premium"),
                 F.countDistinct("policy_id").alias("policies_in_force")))

claim_agg = (fc.withColumn("month_start", F.trunc("loss_date", "month")).join(dc, "customer_id")
             .groupBy(*keys)
             .agg(F.count("*").alias("claim_count"),
                  F.sum("paid_amount").alias("paid_amount"),
                  F.sum("incurred_amount").alias("incurred_amount"),
                  F.sum("outstanding_amount").alias("outstanding_amount")))

mart = (prem_agg.join(claim_agg, keys, "full_outer")
        .na.fill(0, ["earned_premium", "policies_in_force", "claim_count", "paid_amount", "incurred_amount", "outstanding_amount"])
        .withColumn("loss_ratio", F.when(F.col("earned_premium") > 0, F.col("incurred_amount") / F.col("earned_premium")))
        .withColumn("claim_frequency_per_100_policies",
                    F.when(F.col("policies_in_force") > 0, F.col("claim_count") * 100.0 / F.col("policies_in_force"))))
write_gold(mart, "mart_loss_ratio_monthly")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- Checks (these fail loudly) ---------------------------------------------
fc = spark.table("gold_fact_claims")
sc_n = spark.table("silver_claims").count()

# 1. point-in-time join must not drop or duplicate claims, and every claim must find a policy version
assert fc.count() == sc_n, f"fact_claims has {fc.count()} rows but silver_claims has {sc_n} (join fan-out or loss)"
assert fc.filter(F.col("policy_version_id").isNull()).count() == 0, "claims without a matching policy version"

# 2. all silver payments land in the fact
paid_fact = fc.agg(F.sum("paid_amount")).first()[0]
paid_silver = spark.table("silver_claim_payments").agg(F.sum("amount")).first()[0]
assert abs(paid_fact - paid_silver) < 0.01, "paid amounts do not reconcile"

# 3. earned premium equals an independent day-count calculation per policy version
fp = spark.table("gold_fact_premium_monthly")
expected = v.agg(F.sum(F.col("annual_premium") * F.datediff("end_d", "start_d") / 365.0)).first()[0]
actual = fp.agg(F.sum("earned_premium")).first()[0]
assert abs(actual - expected) / expected < 1e-9, f"earned premium mismatch: {actual} vs {expected}"

# 4. incurred never below paid
assert fc.filter(F.col("incurred_amount") < F.col("paid_amount")).count() == 0

print("All gold checks passed.")
print(f"claims: {sc_n:,} | paid: {paid_silver:,.0f} | earned premium: {actual:,.0f}")

display(spark.table("gold_mart_loss_ratio_monthly")
        .groupBy("coverage_code")
        .agg(F.round(F.sum("earned_premium"), 0).alias("earned_premium"),
             F.round(F.sum("incurred_amount"), 0).alias("incurred"),
             F.round(F.sum("incurred_amount") / F.sum("earned_premium"), 3).alias("loss_ratio"),
             F.sum("claim_count").alias("claims"))
        .orderBy("coverage_code"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
