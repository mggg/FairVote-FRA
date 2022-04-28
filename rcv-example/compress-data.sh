
# Go through and zip all the results for publishing on git.
for directory in output/chains/*; do
    if [[ $directory == *.zip ]]; then 
        continue
    fi
    zip "$directory".zip "$directory"/neutral-assignments.jsonl
done

# for directory in output/records/*; do
#     if [[ $directory == *.zip ]]; then 
#         continue
#     fi
#     zip "$directory".zip "$directory" -r
# done
