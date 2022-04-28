
# Deal with the focus states first.
focus=(
    "massachusetts"
    "maryland"
    "florida"
    "illinois"
    "texas"
)

for ((i=0; i<${#focus[@]}; i++)); do
    # Get the name of the state and the type of chain.
    state=${focus[$i]}
    IFS=":"
    read loc bias <<< "$state"

    # Do these plots!
    python plots-seatshare-by-ensemble.py "$state"
done

# Then do the other ones!
nationwide=(
    "alabama"
    "alaska"
    "arizona"
    "arkansas"
    "california"
    "colorado"
    "connecticut"
    "delaware"
    "georgia"
    "hawaii"
    "idaho"
    "indiana"
    "iowa"
    "kansas"
    "kentucky"
    "louisiana"
    "maine"
    "mississippi"
    "minnesota"
    "mississippi"
    "missouri"
    "montana"
    "nebraska"
    "nevada"
    "new hampshire"
    "new jersey"
    "new mexico"
    "new york"
    "north carolina"
    "north dakota"
    "ohio"
    "oklahoma"
    "oregon"
    "pennsylvania"
    "rhode island"
    "south carolina"
    "south dakota"
    "tennessee"
    "utah"
    "vermont"
    "virginia"
    "washington"
    "west virginia"
    "wyoming"
)

for state in ${nationwide[@]}; do
    python plots-seatshare-by-ensemble.py "$state"
done
