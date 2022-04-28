#!/bin/bash

states=(
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

# Check if the file already exists, then re-create it.
if test -f "jobs.txt"; then
    JOBS="$(cat jobs.txt) "
else
    touch jobs.txt
    JOBS=""
fi

# Check if the relationship file already exists, then re-create it.
if test -f "chains.txt"; then
    CHAINS="$(cat chains.txt)\n"
else
    touch chains.txt
    CHAINS=""
fi

for ((i=0; i<${#states[@]}; i++)); do
    state=${states[$i]}
    state=$(tr -s ' ' '-' <<< $state)

    # Capture the output from the sbatch command.
    output=$(sbatch chain.slurm "$state" neutral)

    # Get the individual pieces of output.
    IFS=" "
    read a b c identifier <<< $output

    # Write identifiers and names to file.
    JOBS+="$identifier "
    CHAINS+="$identifier $state neutral\n"
done

# Write to files.
echo -e $CHAINS | column -t > chains.txt
echo -n $JOBS > jobs.txt

