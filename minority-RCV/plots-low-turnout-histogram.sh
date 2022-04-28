
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
    # python plots-results-by-threshold-pie-unscaled.py "$state"
    # python plots-results-by-threshold-pie-scaled.py "$state"
    # python plots-results-by-threshold-scatter-split.py "$state"
    python plots-low-turnout-histogram.py "$state"
done
