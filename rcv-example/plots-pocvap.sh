
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
    python plots-split-pocvap.py "$state"
done

# Then do the other ones!
nationwide=(
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
