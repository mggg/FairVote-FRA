#!/bin/bash

states=(
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
    "michigan"
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
    "wisconsin"
    "wyoming"
)

# Check if the file already exists, then re-create it.
if test -f "jobs.txt"; then
    JOBS="$(cat jobs.txt) "
else
    touch jobs.txt
    JOBS=""
fi

# Check if the relationship file already exists, then re-create it.
if test -f "simulations.txt"; then
    CHAINS="$(cat simulations.txt)\n"
else
    touch simulations.txt
    CHAINS=""
fi

for ((i=0; i<${#states[@]}; i++)); do
    state=${states[$i]}
    state=$(tr -s ' ' '-' <<< $state)

    for district in {0..50}; do
        # Capture the output from the sbatch command.
        output=$(sbatch rcv-simulation.slurm "$state" neutral $district)

        # Get the individual pieces of output.
        IFS=" "
        read a b c identifier <<< $output

        # Write identifiers and names to file.
        JOBS+="$identifier "
        CHAINS+="$identifier $state neutral $district\n"
    done
done

# Write to files.
echo -e $CHAINS | column -t > simulations.txt
echo -n $JOBS > jobs.txt

