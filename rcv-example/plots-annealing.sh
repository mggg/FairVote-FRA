
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
    python plots-annealing.py "$state"
done
