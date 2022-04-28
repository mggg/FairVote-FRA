
# Go through and zip all the results for publishing on git.
for directory in output/results/*; do
    if [[ $directory == *.zip ]]; then 
        continue
    fi
    zip "$directory".zip "$directory" -r
done

zip configurations.zip configurations.json
zip configurations-lowturnout.zip configurations-lowturnout.json
