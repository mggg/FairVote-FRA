
# Deal with the focus states first.
focus=(
    "massachusetts:neutral"
    "massachusetts:tilted"
    "maryland:neutral"
    "maryland:tilted"
    "texas:neutral"
    "texas:tilted"
    "illinois:tilted"
    "florida:tilted"
    "florida:neutral"
    "illinois:neutral"
)

for ((i=0; i<${#focus[@]}; i++)); do
    # Get the name of the state and the type of chain.
    state=${focus[$i]}
    IFS=":"
    read loc bias <<< "$state"

    python score-records.py $loc $bias
done

nationwide=(
    "alabama"
    "arizona"
    "california"
    "colorado"
    "georgia"
    "indiana"
    "kentucky"
    "louisiana"
    "michigan"
    "minnesota"
    "missouri"
    "new jersey"
    "new york"
    "north carolina"
    "ohio"
    "oregon"
    "pennsylvania"
    "south carolina"
    "tennessee"
    "virginia"
    "washington"
    "wisconsin"
)

for ((i=0; i<${#nationwide[@]}; i++)); do
    state=${nationwide[$i]}

    # Create records.
    python score-records.py "$state" neutral
done
