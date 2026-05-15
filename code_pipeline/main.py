# BI Assignment 1 - data pipeline
# Builds the star schema from the 4 source CSVs.
# Run from project root: python code_pipeline/main.py

import os
import numpy as np
import pandas as pd

# ----- paths -----
SOURCE_DIR = "source_data"
OUTPUT_DIR = "code_pipeline/output"

# ----- thresholds for the derived dimensions -----
# Perceptions of corruption: 0-1, higher = more corruption perceived
CORRUPTION_HIGH = 0.75
CORRUPTION_MEDIUM = 0.50
# Freedom to make life choices: 0-1, higher = more freedom
FREEDOM_HIGH = 0.75
# Log GDP per capita split into high/low income
GDP_HIGH = 9.7

# ----- country name fixes -----
# different files spell some countries differently, we map everything to the
# happiness report spelling
NAME_FIX = {
    "Congo": "Congo (Brazzaville)",
    "Democratic Republic of Congo": "Congo (Kinshasa)",
    "Czechia": "Czech Republic",
    "Hong Kong": "Hong Kong S.A.R. of China",
    "Cote d'Ivoire": "Ivory Coast",
    "Palestine": "Palestinian Territories",
    "Eswatini": "Swaziland",
    "Taiwan": "Taiwan Province of China",
}


def fix_name(n):
    if pd.isna(n):
        return n
    n = str(n).strip()
    return NAME_FIX.get(n, n)


def corruption_tier(v):
    if pd.isna(v):
        return "Unknown"
    if v >= CORRUPTION_HIGH:
        return "High"
    if v >= CORRUPTION_MEDIUM:
        return "Medium"
    return "Low"


def freedom_tier(v):
    if pd.isna(v):
        return "Unknown"
    return "High" if v >= FREEDOM_HIGH else "Low"


def gdp_bracket(v):
    if pd.isna(v):
        return "Unknown"
    return "High Income" if v >= GDP_HIGH else "Low Income"


# ===== 1. load =====
print("Loading...")
happiness = pd.read_csv(os.path.join(SOURCE_DIR, "world-happiness-report.csv"))
happiness_2021 = pd.read_csv(os.path.join(SOURCE_DIR, "world-happiness-report-2021.csv"))
internet = pd.read_csv(os.path.join(SOURCE_DIR, "number-of-internet-users.csv"))
population = pd.read_csv(os.path.join(SOURCE_DIR, "popolazione-globale-per-paese-1950-2024.csv"))


# ===== 2. clean =====
print("Cleaning...")

# happiness: drop rows without country/year, normalize names
happiness = happiness.dropna(subset=["Country name", "year"]).copy()
happiness["Country name"] = happiness["Country name"].apply(fix_name)
happiness["year"] = happiness["year"].astype(int)

# region info from 2021 file (we only need country -> region mapping)
region = happiness_2021[["Country name", "Regional indicator"]].dropna(subset=["Country name"]).copy()
region["Country name"] = region["Country name"].apply(fix_name)
region = region.drop_duplicates("Country name")

# internet: drop OWID aggregate rows (regions, world, income groups)
internet = internet[~internet["Code"].fillna("").str.startswith("OWID")].copy()
internet = internet.dropna(subset=["Entity", "Year"])
internet = internet.rename(columns={
    "Entity": "country_name",
    "Code": "iso_code",
    "Year": "year",
    "Number of Internet users": "number_of_internet_users",
})
internet["country_name"] = internet["country_name"].apply(fix_name)
internet["year"] = internet["year"].astype(int)

# population
population = population.dropna(subset=["country", "year"]).copy()
population = population.rename(columns={"country": "country_name"})
population["country_name"] = population["country_name"].apply(fix_name)
population["year"] = population["year"].astype(int)


# ===== 3. dim_geography =====
# one row per country from the happiness report (which drives the fact grain)
print("Building dim_geography...")
geo = happiness[["Country name"]].drop_duplicates().rename(columns={"Country name": "country_name"})
geo = geo.sort_values("country_name").reset_index(drop=True)

# iso_code: prefer population, fallback to internet
iso_pop = population[["country_name", "iso_code"]].dropna().drop_duplicates("country_name")
iso_net = internet[["country_name", "iso_code"]].dropna().drop_duplicates("country_name").rename(
    columns={"iso_code": "iso_code_net"})
geo = geo.merge(iso_pop, on="country_name", how="left").merge(iso_net, on="country_name", how="left")
geo["iso_code"] = geo["iso_code"].fillna(geo["iso_code_net"])
geo = geo.drop(columns=["iso_code_net"])

# regional_indicator from 2021 file
geo = geo.merge(region.rename(columns={"Country name": "country_name", "Regional indicator": "regional_indicator"}),
                on="country_name", how="left")
geo["regional_indicator"] = geo["regional_indicator"].fillna("Unknown")

# surrogate key
geo["country_key"] = ["G%04d" % i for i in range(1, len(geo) + 1)]
dim_geography = geo[["country_key", "country_name", "iso_code", "regional_indicator"]]


# ===== 4. dim_date =====
print("Building dim_date...")
years = sorted(happiness["year"].unique())
dim_date = pd.DataFrame({
    "date_key": range(1, len(years) + 1),
    "year": years,
})


# ===== 5. derive tier/bracket per happiness row (used for both fact + dims) =====
h = happiness.copy()
h["corruption_tier"] = h["Perceptions of corruption"].apply(corruption_tier)
h["freedom_tier"] = h["Freedom to make life choices"].apply(freedom_tier)
# if either tier is Unknown, the row goes to the single "Unknown" governance bucket
mask_unknown = (h["corruption_tier"] == "Unknown") | (h["freedom_tier"] == "Unknown")
h.loc[mask_unknown, "corruption_tier"] = "Unknown"
h.loc[mask_unknown, "freedom_tier"] = "Unknown"
h["gdp_bracket"] = h["Log GDP per capita"].apply(gdp_bracket)


# ===== 6. dim_governance =====
print("Building dim_governance...")
combos = h[["corruption_tier", "freedom_tier"]].drop_duplicates()
# keep the Unknown row, then sort the real ones
unknown_row = combos[(combos["corruption_tier"] == "Unknown")]
real = combos[~(combos["corruption_tier"] == "Unknown")].sort_values(["corruption_tier", "freedom_tier"])
dim_governance = pd.concat([unknown_row, real]).reset_index(drop=True)
dim_governance["governance_key"] = ["GOV%02d" % i for i in range(0, len(dim_governance))]
dim_governance = dim_governance[["governance_key", "corruption_tier", "freedom_tier"]]


# ===== 7. dim_economy =====
print("Building dim_economy...")
brackets = h["gdp_bracket"].drop_duplicates().tolist()
# Unknown first (key 00), then the rest
ordered = (["Unknown"] if "Unknown" in brackets else []) + sorted([b for b in brackets if b != "Unknown"])
dim_economy = pd.DataFrame({
    "economy_key": ["ECO%02d" % i for i in range(0, len(ordered))],
    "gdp_bracket": ordered,
})


# ===== 8. fact table =====
print("Building fact table...")
fact = h.rename(columns={"Country name": "country_name"})

# join dims to get keys
fact = fact.merge(dim_geography[["country_key", "country_name"]], on="country_name", how="left")
fact = fact.merge(dim_date, on="year", how="left")
fact = fact.merge(dim_governance, on=["corruption_tier", "freedom_tier"], how="left")
fact = fact.merge(dim_economy, on="gdp_bracket", how="left")

# add population + internet users
fact = fact.merge(population[["country_name", "year", "population"]], on=["country_name", "year"], how="left")
fact = fact.merge(internet[["country_name", "year", "number_of_internet_users"]],
                  on=["country_name", "year"], how="left")

# derived measure: internet penetration rate (in %)
# leave as NaN when either input is missing - don't fill with 0, that would skew Tableau averages
fact["internet_penetration_rate"] = np.where(
    (fact["population"] > 0) & fact["number_of_internet_users"].notna(),
    100 * fact["number_of_internet_users"] / fact["population"],
    np.nan,
)

# rename measures to snake_case to match the schema
fact = fact.rename(columns={
    "Life Ladder": "life_ladder",
    "Log GDP per capita": "log_gdp_per_capita",
    "Social support": "social_support",
    "Healthy life expectancy at birth": "healthy_life_expectancy",
    "Freedom to make life choices": "freedom_raw_score",
    "Perceptions of corruption": "corruption_raw_score",
    "Positive affect": "positive_affect",
    "Negative affect": "negative_affect",
})

fact = fact[[
    "country_key", "date_key", "governance_key", "economy_key",
    "life_ladder", "log_gdp_per_capita", "social_support", "healthy_life_expectancy",
    "freedom_raw_score", "corruption_raw_score", "positive_affect", "negative_affect",
    "population", "number_of_internet_users", "internet_penetration_rate",
]]


# ===== 9. quick sanity checks =====
# basic stuff to make sure nothing is broken before we ship to Tableau
assert dim_geography["country_key"].is_unique
assert dim_date["date_key"].is_unique
assert fact[["country_key", "date_key"]].duplicated().sum() == 0
for fk, dim_df, dim_key in [("country_key", dim_geography, "country_key"),
                            ("date_key", dim_date, "date_key"),
                            ("governance_key", dim_governance, "governance_key"),
                            ("economy_key", dim_economy, "economy_key")]:
    orphans = set(fact[fk].dropna()) - set(dim_df[dim_key])
    assert not orphans, "orphan keys in %s: %s" % (fk, orphans)


# ===== 10. save =====
os.makedirs(OUTPUT_DIR, exist_ok=True)
dim_geography.to_csv(os.path.join(OUTPUT_DIR, "dim_geography.csv"), index=False)
dim_date.to_csv(os.path.join(OUTPUT_DIR, "dim_date.csv"), index=False)
dim_governance.to_csv(os.path.join(OUTPUT_DIR, "dim_governance.csv"), index=False)
dim_economy.to_csv(os.path.join(OUTPUT_DIR, "dim_economy.csv"), index=False)
fact.to_csv(os.path.join(OUTPUT_DIR, "fact_country_year_metrics.csv"), index=False)

print()
print("dim_geography:", len(dim_geography))
print("dim_date:", len(dim_date))
print("dim_governance:", len(dim_governance))
print("dim_economy:", len(dim_economy))
print("fact rows:", len(fact))
print("internet_penetration_rate filled in %.1f%% of rows" %
      (100 * fact["internet_penetration_rate"].notna().mean()))
print("Done.")
