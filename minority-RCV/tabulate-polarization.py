
import pandas as pd
import us
import jsonlines
import json
import numpy as np
from collections import Counter
from pathlib import Path

from ModelingResult import ModelingResult

# Read in statewide summaries and merge.
statewide = pd.read_csv("./data/demographics/summary.csv")
polarization = pd.read_csv("./data/demographics/polarization.csv")
summary = statewide.merge(polarization, on="STATE")

# Get only the ones with more than two districts.
# summary = summary[summary["REPRESENTATIVES"] > 2]

# Now, for each of the non-focus states, we want to discuss the breakdown of the
# seats, how many seats we'd expect PoC to win under a proportional system, and
# relate the results (in a 4x4 subtable maybe?)
subtable = summary[
    [
        "STATE", "pp", "ww", "ppadj", "wwadj", "2020_PRES_POC_D", "2020_PRES_WHITE_R",
        "2020_PRES_D", "2020_PRES_R", "POCVAP20%"
    ]
]

# Correct some typing.
subtable["pp"] = subtable["pp"].round(2)
subtable["ww"] = subtable["ww"].round(2)
subtable["ppadj"] = subtable["ppadj"].round(2)
subtable["wwadj"] = subtable["wwadj"].round(2)
subtable["2020_PRES_POC_D"] = subtable["2020_PRES_POC_D"].round(2)
subtable["2020_PRES_WHITE_R"] = subtable["2020_PRES_WHITE_R"].round(2)
subtable["2020_PRES_D%"] = ((subtable["2020_PRES_D"]/(subtable["2020_PRES_R"]+subtable["2020_PRES_D"]))*100).round(2).astype(str) + "%"
subtable["POCVAP20%"] = (subtable["POCVAP20%"]*100).round(2).astype(str) + "%"
subtable["2020_PRES_POC_D"] = subtable["2020_PRES_POC_D"].fillna("--")

subtable = subtable[[
    "STATE", "POCVAP20%", "2020_PRES_D%", "2020_PRES_POC_D", "pp", "ppadj",
    "2020_PRES_WHITE_R","wwadj"
]]

# Change some names!
subtable = subtable.rename({
    "STATE": "State",
    "POCVAP20%": "POCVAP%",
    "2020_PRES_D%": "Biden Share",
    "2020_PRES_POC_D": r"Poll",
    "pp": r"Estimate",
    "ppadj": r"$\pi_{cc}$",
    "2020_PRES_WHITE_R": r"Poll",
    "wwadj": r"$\pi_{ww}$",
}, axis=1)

# Write the table to file.
subtable.to_csv("./output/polarization-all.csv", index=False)
