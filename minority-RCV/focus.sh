#!/bin/bash
# Creates jobs for each of the chains.

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

for bias in "tilted" "neutral"; do
    for jurisdiction in "florida" "texas" "illinois" "maryland" "massachusetts"; do
        for district in {0..49}; do
            # Capture the output from the sbatch command.
            output=$(sbatch rcv-simulation.slurm $jurisdiction $bias $district)

            # Get the individual pieces of output.
            IFS=" "
            read a b c identifier <<< $output

            # Write identifiers and names to file.
            JOBS+="$identifier "
            CHAINS+="$identifier $jurisdiction $bias $district\n"
        done
    done
done

# Write to files.
echo -e $CHAINS | column -t > simulations.txt
echo -n $JOBS > jobs.txt
