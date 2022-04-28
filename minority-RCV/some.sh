#!/bin/bash

# Run some specific chains.

specific=(
    "california:neutral"
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

for ((i=0; i<${#specific[@]}; i++)) do
    chain=${specific[$i]}
    chain=$(tr -s ' ' '-' <<< $chain)

    # Get location and the bias from the list of specific things.
    IFS=":" read location bias <<< $chain

    # Start the chain!
    for district in {0..4}; do
        output=$(sbatch rcv-simulation.slurm "$location" $bias $district)

        # Get the individual pieces of output.
        IFS=" "
        read a b c identifier <<< $output

        # Write to file.
        CHAINS+="$identifier $location $bias $district\n"
        JOBS+="$identifier "
    done
done

echo -e $CHAINS | column -t > simulations.txt
echo -n $JOBS > jobs.txt
