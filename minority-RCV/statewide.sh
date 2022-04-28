statewide=(
    "oklahoma"
    "connecticut"
    "utah"
    "nevada"
    "mississippi"
    "kansas"
    "iowa"
    "arkansas"
    "new-mexico"
    "nebraska"
    "west-virginia"
    "rhode-island"
    "new-hampshire"
    "montana"
    "maine"
    "idaho"
    "hawaii"
    "wyoming"
    "vermont"
    "north-dakota"
    "south-dakota"
    "delaware"
    "alaska"
)

for ((i=0; i<${#statewide[@]}; i++)) do
    location=${statewide[$i]}
    output=$(sbatch rcv-simulation.slurm "$location" neutral 0)

    # Get the individual pieces of output.
    IFS=" "
    read a b c identifier <<< $output

    # Write to file.
    CHAINS+="$identifier $location neutral 0\n"
    JOBS+="$identifier "
done

echo -e $CHAINS | column -t > simulations.txt
echo -n $JOBS > jobs.txt