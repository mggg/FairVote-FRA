
jurisdictions=("florida" "illinois" "maryland" "massachusetts" "texas")

for bias in "neutral" "tilted"; do
    for jurisdiction in "${jurisdictions[@]}"; do
        python models-ensemble.py $jurisdiction $bias
    done
done
