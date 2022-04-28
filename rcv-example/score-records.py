
import jsonlines
import us
import sys
from pathlib import Path

# Do some pathing.
location = us.states.lookup(sys.argv[-2].title(), field="name")
bias = sys.argv[-1]

read = Path(f"./output/chains/{location.name.lower()}/{bias}.jsonl")
write = Path(f"./output/records/{location.name.lower()}")

# Cread in plan data and create an empty list of records.
with jsonlines.open(read) as r: plans = list(r)
records = []

# Get the number of districts in the state.
districts = sum(plans[0]["MAGNITUDE"].values())

# For each of the plans, "invert" the dictionaries so organized by district->properties,
# not properties->district.
for plan in plans:
    dkeys = list(plan["population"])

    # Get all the standard stuff.
    record = {
        district: {
            prop: plan[prop][district]
            for prop in ["population", "POCVAP20", "VAP20", "MAGNITUDE"]
        }
        for district in dkeys
    }

    # Get the POCVAP percentage.
    for district in dkeys:
        record[district]["POCVAP20%"] = record[district]["POCVAP20"]/record[district]["VAP20"]

    # Get the number of "POC seats"
    for district in dkeys:
        threshold = 1/(record[district]["MAGNITUDE"]+1)
        record[district]["POCSEATS"] = record[district]["POCVAP20%"]//threshold

    records.append(record)

# Write back to file.
if not write.exists(): write.mkdir()
with jsonlines.open(write/f"{bias}.jsonl", mode="w") as w: w.write_all(records)
