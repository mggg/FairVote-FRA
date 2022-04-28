
import json
import pandas as pd

def optimal(groupings):
    """
    Finds the optimal district configuration, according to the FRA and FairVote.
    The rules specified by the FRA and FairVote say that, given two configurations
    A and B, A should be preferred over B if:
    
        1. A has more representatives from odd-numbered districts;
        2. A has the same number of representatives from odd-numbered districts,
            and:
            a. A has the same number of representatives from four-member districts,
                and:
                i. A has more five-member districts, or;
            b. A has fewer representatives from four-member districts.

    Otherwise, B is preferred.

    Args:
        groupings (list): A list of tuples which correspond to the number of
            3-, 4-, and 5-member districts in a given plan.

    Returns:
        The optimal configuration according to the above rules.
    """

    def compare(A, B):
        magnitudes = (3, 4, 5)
        A3, A4, A5 = (e*m for e, m in zip(A, magnitudes))
        B3, B4, B5 = (e*m for e, m in zip(B, magnitudes))

        # Get the number of representatives from odd-numbered districts in both
        # configurations.
        Aodd = A3 + A5
        Bodd = B3 + B5
        
        if Aodd > Bodd:
            # If A has more odd representatives than B, it is preferred over B.
            return A
        elif Aodd == Bodd:
            # If A has the same number of odd representatives as B, A is preferred
            # if: A has the same number of four-member districts as B, and has more
            # five-member districts than B; or A has fewer four-member districts
            # than B.
            if A4 == B4:
                if A5 > B5: return A
                else: return B
            elif A4 > B4: return B
            elif A4 < B4: return A
        # Finally, if A has fewer odd representatives than B, B is preferred over
        # A.
        return B

    # Hill-climb.
    largest = groupings[0]
    for grouping in groupings[1:]: largest = compare(largest, grouping) 

    return largest

def groupings(n, M):
    """
    Outer call for `_groupings()`.

    Args:
        n (int): The number of districts we'll be combining.
        M (list): Magnitudes.

    Returns:
        A list of tuples of length `len(M)`. For example, if `M = [3, 4, 5]`,
        then the tuple `(2, 0, 1)` represents a districting plan with two
        three-member districts and one five-member district (for a total of
        three districts represented by 11 members).
    """
    complete, universe = [], set()
    _groupings(n, M, complete, universe)
    return complete

def _groupings(n, M, complete, universe, partial=None):
    """
    Given a number of districts and a list of magnitudes, admits all combinations
    of n districts into magnitude-sized groups.

    Args:
        n (int): The number of districts we'll be combining.
        M (list): Magnitudes.
        universe (set): Solutions we've already seen.
        partial (list, optional): A subproblem solution; if `None`, then we create
            a partial subproblem to begin the recursion.
    """
    # First, check our termination conditions: the first is that n=0, which indicates
    # we've found a grouping which adds up properly; the second is that n is negative,
    # in which case we've gone too far.
    if n == 0: complete.append(tuple(partial))
    if n < min(M): return

    for i, k in enumerate(M):
        # If we don't have a partial solution, create one. Otherwise, copy the
        # partial solution as a subsolution and pass that.
        if not partial: subsolution = [0 for _  in range(len(M))]
        else: subsolution = partial.copy()

        # Otherwise, check if we've seen this partial solution before: if we have,
        # then we're re-computing something and we should simply continue, disregarding
        # the solution being actively computed; if not, add it to the list of
        # solutions and be done.
        subsolution[i] += 1
        if tuple(subsolution) in universe: continue
        else: universe.add(tuple(subsolution))

        # Recursive calls!
        _groupings(n-k, M, complete, universe, subsolution)


if __name__ == "__main__":
    # For each of the locations we're redistricting in, create a JSON file with
    # some state district groupings.
    locations = pd.read_csv("./data/demographics/apportionment.csv")
    locations = locations[locations["REPRESENTATIVES"] > 5]
    smaller = locations[locations["REPRESENTATIVES"] < 6]
    apportionment = dict(zip(locations["STATE"], locations["REPRESENTATIVES"]))

    # Create an empty assignment with three groupings per state.
    chunkings = {}

    for location, districts in apportionment.items():
        g = groupings(districts, [3, 4, 5])
        g = list(sorted(g, key=lambda k: k[-1]))
        chunkings[location.lower()] = {
            "districts": districts,
            "groupings": g,
            "optimal": optimal(g)
        }

    # Write the groupings to file.
    with open("./groupings.json", "w") as f: json.dump(chunkings, f, indent=2)

    # Just take the optimal ones and write them to columns on a CSV.
    records = []

    for location in chunkings:
        records.append({
            "STATE": location.title(),
            "3": chunkings[location]["optimal"][0],
            "4": chunkings[location]["optimal"][1],
            "5": chunkings[location]["optimal"][2],
        })

    # Write to file.
    pd.DataFrame.from_records(records).to_csv("./data/demographics/configurations.csv", index=False)
