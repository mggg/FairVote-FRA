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
if test -f "chains.txt"; then
    CHAINS="$(cat chains.txt)\n"
else
    touch chains.txt
    CHAINS=""
fi

for bias in "tilted" "neutral"; do
    for jurisdiction in "florida" "texas" "maryland" "massachusetts" "illinois"; do
        # Capture the output from the sbatch command.
        output=$(sbatch chain.slurm $jurisdiction $bias)

        # Get the individual pieces of output.
        IFS=" "
        read a b c identifier <<< $output

        # Write identifiers and names to file.
        JOBS+="$identifier "
        CHAINS+="$identifier $jurisdiction $bias\n"
    done
done

# Write to files.
echo -e $CHAINS | column -t > chains.txt
echo -n $JOBS > jobs.txt
