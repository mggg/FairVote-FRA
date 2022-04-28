
# Deal with the focus states first.
focus=(
    "massachusetts"
    "maryland"
    "florida"
    "illinois"
    "texas"
    "alabama"
    "arizona"
    "colorado"
    "georgia"
    "indiana"
    "kentucky"
    "louisiana"
    "michigan"
    "minnesota"
    "missouri"
    "new jersey"
    "north carolina"
    "oregon"
    "pennsylvania"
    "south carolina"
    "tennessee"
    "virginia"
)

for state in ${nationwide[@]}; do
    python plots-single-pocvap.py "$state"
done
