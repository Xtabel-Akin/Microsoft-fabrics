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

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# # ClaimStream: 01 Generate bronze data
# 
# Generates a synthetic P&C insurance dataset and lands it in the attached Lakehouse as **bronze** Delta tables:
# `bronze_customers`, `bronze_policies`, `bronze_claims`, `bronze_claim_payments`, `bronze_coverage_types`.
# 
# **Before running:** attach your Lakehouse as the default Lakehouse for this notebook (left panel > Add Lakehouse).
# 
# The data is seeded, so reruns give identical output. It contains **intentional data quality problems** (listed in the last code cell) for the silver layer to detect and fix. All names and emails are fictional (example.com).

# CELL ********************

import numpy as np
import pandas as pd

# ---- Config -------------------------------------------------------------
SEED = 42
N_CUSTOMERS = 100_000
POLICIES_PER_CUSTOMER = 1.4
AS_OF = np.datetime64("2026-10-29")          # "today" for the dataset
HISTORY_START = np.datetime64("2019-01-01")
BATCH_ID = "batch_001"
SOURCE = "synthetic_generator_v1"

rng = np.random.default_rng(SEED)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- Reference data & helpers -------------------------------------------
COVERAGES = {
    "AUTO":       dict(name="Personal Auto",   lob="Personal",   premium=1600, freq=0.12, sev_median=4500,  share=0.45,
                       causes=["Collision", "Theft", "Glass", "Hail", "Vandalism"]),
    "HOME":       dict(name="Homeowners",      lob="Personal",   premium=1400, freq=0.08, sev_median=7000,  share=0.30,
                       causes=["Water damage", "Fire", "Wind", "Theft", "Liability"]),
    "TENANT":     dict(name="Tenant",          lob="Personal",   premium=350,  freq=0.06, sev_median=2500,  share=0.15,
                       causes=["Theft", "Water damage", "Fire", "Liability"]),
    "COMMERCIAL": dict(name="Small Business",  lob="Commercial", premium=4500, freq=0.14, sev_median=11000, share=0.10,
                       causes=["Property damage", "Liability", "Theft", "Business interruption"]),
}
codes = list(COVERAGES)

FIRST = ["Olivia","Liam","Emma","Noah","Ava","Lucas","Mia","Ethan","Chloe","Mason","Aisha","Wei","Priya","Mateo","Sofia","Amir","Fatima","Jean","Marie","Omar"]
LAST = ["Smith","Tremblay","Singh","Chen","Martin","Roy","Wilson","Patel","Brown","Gagnon","Lee","Campbell","Nguyen","Okafor","Kowalski","Silva","Ahmed","Morin","Taylor","Ross"]
PROVINCES = ["ON","QC","BC","AB","MB","SK","NS","NB"]
PROV_WEIGHTS = [0.40,0.22,0.15,0.12,0.04,0.03,0.025,0.015]

def to_iso(a):
    # numpy datetime64 array -> list of 'YYYY-MM-DD' strings (None for NaT)
    idx = pd.to_datetime(a)
    return [None if pd.isna(x) else x.strftime("%Y-%m-%d") for x in idx]

def days(a):
    return np.asarray(a).astype("timedelta64[D]")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- Customers ----------------------------------------------------------
n = N_CUSTOMERS
span_days = int((AS_OF - HISTORY_START).astype(int))
created = HISTORY_START + days(rng.integers(0, span_days - 30, n))

first = rng.choice(FIRST, n)
last = rng.choice(LAST, n)
province = rng.choice(PROVINCES, n, p=PROV_WEIGHTS)
birth = np.datetime64("2007-01-01") - days(rng.integers(0, 66 * 365, n))
email = np.array([f"{f}.{l}{i}@example.com".lower() for i, (f, l) in enumerate(zip(first, last), start=1)], dtype=object)

# intentional dirt: ~1% missing emails, ~2% lowercase province codes
email[rng.random(n) < 0.01] = None
province = np.where(rng.random(n) < 0.02, np.char.lower(province), province)

customers = pd.DataFrame({
    "customer_id": [f"C{i:07d}" for i in range(1, n + 1)],
    "first_name": first,
    "last_name": last,
    "email": email,
    "province": province,
    "birth_date": to_iso(birth),
    "created_date": to_iso(created),
})

# intentional dirt: ~0.5% exact duplicate rows
dups = customers.sample(frac=0.005, random_state=SEED)
customers = pd.concat([customers, dups]).sample(frac=1, random_state=SEED).reset_index(drop=True)
print("customers:", len(customers))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- Policies (change log: initial row + premium amendments) -------------
n_pol = int(n * POLICIES_PER_CUSTOMER)
cust_idx = np.concatenate([np.arange(n), rng.integers(0, n, n_pol - n)])   # every customer has >= 1 policy
avail = (AS_OF - created[cust_idx]).astype(int)
eff = created[cust_idx] + days((rng.random(n_pol) * avail).astype(int))

coverage = rng.choice(codes, n_pol, p=[COVERAGES[c]["share"] for c in codes])
base_premium = np.array([COVERAGES[c]["premium"] for c in coverage])
premium = (base_premium * rng.lognormal(0, 0.25, n_pol)).round(2)

# ~30% of policies are cancelled at some point; the rest are active (auto-renewing)
span_to_now = (AS_OF - eff).astype(int)
cancelled = (rng.random(n_pol) < 0.30) & (span_to_now > 30)
end_date = np.where(cancelled, eff + days((rng.random(n_pol) * span_to_now).astype(int)), np.datetime64("NaT", "D"))
exposure_end = np.where(cancelled, end_date, AS_OF)
exposure_days = (exposure_end - eff).astype(int)
status = np.where(cancelled, "CANCELLED", "ACTIVE")
policy_ids = np.array([f"P{i:07d}" for i in range(1, n_pol + 1)])

initial = pd.DataFrame({
    "policy_id": policy_ids,
    "customer_id": [f"C{i + 1:07d}" for i in cust_idx],
    "coverage_code": coverage,
    "effective_date": to_iso(eff),
    "end_date": to_iso(end_date),
    "annual_premium": premium,
    "status": status,
    "record_updated_at": to_iso(eff),
})

# ~35% of eligible policies get a later premium amendment -> same policy_id, new row
eligible = np.where(exposure_days > 60)[0]
pick = rng.choice(eligible, size=int(len(eligible) * 0.35), replace=False)
amend_at = eff[pick] + days(rng.integers(30, exposure_days[pick]))
amend = initial.iloc[pick].copy()
amend["annual_premium"] = (premium[pick] * rng.uniform(0.95, 1.15, len(pick))).round(2)
amend["record_updated_at"] = to_iso(amend_at)

policies = pd.concat([initial, amend]).sample(frac=1, random_state=SEED).reset_index(drop=True)
print("policies:", len(policies), "(", len(initial), "policies +", len(amend), "amendments )")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- Claims & payments --------------------------------------------------
freq = np.array([COVERAGES[c]["freq"] for c in coverage])
k = rng.poisson(freq * exposure_days / 365)                  # claims per policy
pol_rep = np.repeat(np.arange(n_pol), k)
n_cl = len(pol_rep)
cov_cl = coverage[pol_rep]

loss = eff[pol_rep] + days((rng.random(n_cl) * exposure_days[pol_rep]).astype(int))
lag = np.ceil(rng.exponential(4, n_cl)).astype(int)
reported = np.minimum(loss + days(lag), AS_OF)

sev_median = np.array([COVERAGES[c]["sev_median"] for c in cov_cl])
severity = rng.lognormal(np.log(sev_median), 0.9).round(2)

# status depends on claim age: older claims are more likely closed
age = (AS_OF - reported).astype(int)
closed = rng.random(n_cl) < np.clip(age / 150, 0.05, 0.95)
denied = closed & (rng.random(n_cl) < 0.12)
paid = closed & ~denied
is_open = ~closed
claim_status = np.select([denied, paid], ["DENIED", "CLOSED"], default="OPEN")
reserve = np.where(is_open, severity, 0.0)

cause = np.empty(n_cl, dtype=object)
for code_ in codes:
    m = cov_cl == code_
    cause[m] = rng.choice(COVERAGES[code_]["causes"], m.sum())

# payments: closed claims paid in 1-3 instalments; 30% of open claims have one partial payment
paid_idx = np.where(paid)[0]
partial_idx = np.where(is_open & (rng.random(n_cl) < 0.30))[0]
base_idx = np.concatenate([paid_idx, partial_idx])
n_inst = np.concatenate([rng.integers(1, 4, len(paid_idx)), np.ones(len(partial_idx), dtype=int)])
share = np.concatenate([np.ones(len(paid_idx)), rng.uniform(0.3, 0.6, len(partial_idx))])
pay_claim = np.repeat(base_idx, n_inst)
pay_n = np.repeat(n_inst, n_inst)
pay_share = np.repeat(share, n_inst)
pay_amount = (severity[pay_claim] * pay_share / pay_n * rng.uniform(0.85, 1.0, len(pay_claim))).round(2)
pay_date = np.minimum(reported[pay_claim] + days(rng.integers(5, 120, len(pay_claim))), AS_OF)

# intentional dirt: ~0.3% of claims get a reported_date BEFORE the loss_date
bad = rng.random(n_cl) < 0.003
reported_out = np.where(bad, loss - days(rng.integers(1, 6, n_cl)), reported)

claim_ids = np.array([f"CL{i:08d}" for i in range(1, n_cl + 1)])
claims = pd.DataFrame({
    "claim_id": claim_ids,
    "policy_id": policy_ids[pol_rep],
    "loss_date": to_iso(loss),
    "reported_date": to_iso(reported_out),
    "status": claim_status,
    "cause_of_loss": cause,
    "reserve_amount": reserve,
})
payments = pd.DataFrame({
    "payment_id": [f"PM{i:08d}" for i in range(1, len(pay_claim) + 1)],
    "claim_id": claim_ids[pay_claim],
    "payment_date": to_iso(pay_date),
    "amount": pay_amount,
    "method": rng.choice(["EFT", "CHEQUE", "INTERAC"], len(pay_claim), p=[0.6, 0.15, 0.25]),
})
coverage_types = pd.DataFrame([
    dict(coverage_code=c, coverage_name=v["name"], line_of_business=v["lob"]) for c, v in COVERAGES.items()
])
print("claims:", len(claims), "| payments:", len(payments))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- Sanity checks (pandas, before writing anything) ---------------------
assert policies["customer_id"].isin(customers["customer_id"]).all(), "orphan policies"
assert claims["policy_id"].isin(policies["policy_id"]).all(), "orphan claims"
assert payments["claim_id"].isin(claims["claim_id"]).all(), "orphan payments"
assert claims["claim_id"].is_unique and payments["payment_id"].is_unique
assert (payments["amount"] > 0).all()

print({
    "customers": len(customers), "policies": len(policies), "claims": len(claims),
    "payments": len(payments), "coverage_types": len(coverage_types),
})
print("total rows:", len(customers) + len(policies) + len(claims) + len(payments) + len(coverage_types))
print("claims by status:", claims["status"].value_counts().to_dict())
print("avg claims/policy:", round(len(claims) / n_pol, 3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- Write bronze Delta tables to the default Lakehouse -------------------
from pyspark.sql import functions as F

def to_spark(pdf, date_cols=()):
    pdf = pdf.astype(object).where(pdf.notna(), None)   # NaN -> None so Spark infers clean string/null columns
    sdf = spark.createDataFrame(pdf)
    for c in date_cols:
        sdf = sdf.withColumn(c, F.to_date(F.col(c)))
    return sdf

TABLES = {
    "customers":      (customers,      ["birth_date", "created_date"]),
    "policies":       (policies,       ["effective_date", "end_date", "record_updated_at"]),
    "claims":         (claims,         ["loss_date", "reported_date"]),
    "claim_payments": (payments,       ["payment_date"]),
    "coverage_types": (coverage_types, []),
}

for name, (pdf, date_cols) in TABLES.items():
    sdf = (to_spark(pdf, date_cols)
           .withColumn("_ingested_at", F.current_timestamp())
           .withColumn("_source", F.lit(SOURCE))
           .withColumn("_batch_id", F.lit(BATCH_ID)))
    (sdf.write.mode("overwrite").option("overwriteSchema", "true")
        .format("delta").saveAsTable(f"bronze_{name}"))
    print(f"wrote bronze_{name}: {sdf.count():,} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ---- Verify what landed, and list the intentional dirt for silver to fix ---
for name in TABLES:
    print(name, spark.table(f"bronze_{name}").count())

display(spark.sql("""
    SELECT status, COUNT(*) AS claims, ROUND(SUM(reserve_amount), 0) AS open_reserve
    FROM bronze_claims GROUP BY status ORDER BY claims DESC
"""))

print("""
Intentional data quality issues (silver should detect and handle each):
  1. bronze_customers: ~0.5% exact duplicate rows
  2. bronze_customers: ~1% null emails
  3. bronze_customers: ~2% lowercase province codes
  4. bronze_claims:    ~0.3% reported_date earlier than loss_date
  5. bronze_policies:  change log (several rows per policy_id) -> build SCD2 in silver
""")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
