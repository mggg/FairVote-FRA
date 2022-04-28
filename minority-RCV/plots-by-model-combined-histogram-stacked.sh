
# Deal with the focus states first.
focus=(
    "massachusetts"
    "maryland"
    "florida"
    "illinois"
    "texas"
)

for state in ${focus[@]}; do
    # Do these plots!
    python plots-by-model-combined-histogram-stacked.py "$state"
done
