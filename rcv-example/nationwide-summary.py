
import pandas as pd

# Get some stuff from important places!
apportionment = pd.read_csv("./data/demographics/apportionment.csv")
configurations = pd.read_csv("./data/demographics/configurations.csv")
nationwide = pd.read_csv("./data/demographics/nationwide.csv")
elections = pd.read_csv("./data/demographics/elections.csv", thousands=",")
polls = pd.read_csv("./data/demographics/polls-all.csv")

# Select some columns.
nationwide = nationwide[["ABBR", "STATE", "TOTPOP20", "VAP20", "POCVAP20", "POCVAP20%", "POCVAP20SEATS"]]
nationwide["STATE"] = nationwide["STATE"].str.title()

# Pare down data.
apportionment = apportionment[["STATE", "REPRESENTATIVES"]]
configurations = configurations[["STATE", "3", "4", "5"]]
polls = polls[[
    "STATE", "2020_PRES_WHITE", "2020_PRES_WHITE_D", "2020_PRES_WHITE_R",
    "2020_PRES_POC", "2020_PRES_POC_D", "2020_PRES_POC_R"
]]

votes = ["2020_PRES_D","2020_PRES_R","2020_PRES_3P"]
elections[votes] = elections[votes].astype(int)

# Now, do some more stuff!
for data in [elections, polls, apportionment, configurations]:
    data["STATE"] = data["STATE"].str.title()
    nationwide = nationwide.merge(data, on="STATE", how="outer")

# Write summary to file.
nationwide.to_csv("./data/demographics/summary.csv", index=False)
