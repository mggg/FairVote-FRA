#!/bin/bash

# Run some specific chains.

specific=(
    "new york:neutral"
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

for ((i=0; i<${#specific[@]}; i++)) do
    chain=${specific[$i]}
    chain=$(tr -s ' ' '-' <<< $chain)

    # Get location and the bias from the list of specific things.
    IFS=":" read location bias <<< $chain

    # Start the chain!
    output=$(sbatch chain.slurm "$location" $bias)

    # Get the individual pieces of output.
    IFS=" "
    read a b c identifier <<< $output

    # Write to file.
    CHAINS+="$identifier $location $bias\n"
    JOBS+="$identifier "
done

echo -e $CHAINS | column -t > chains.txt
echo -n $JOBS > jobs.txt
