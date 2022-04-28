
from re import sub
from pydantic import BaseModel
from typing import Optional
from itertools import product
from itertools import combinations_with_replacement as cwr
import numpy as np

class ModelingResult(BaseModel):
    """
    POC support for POC candidates of choice.
    """
    pp: float

    """
    POC support for white candidates of choice.
    """
    pw: float

    """
    White support for white candidates of choice.
    """
    ww: float

    """
    White support for POC candidates of choice.
    """
    wp: float

    """
    Number of simulations.
    """
    simulations: int

    """
    POC share.
    """
    pocshare: float

    """
    Number of ballots.
    """
    ballots: int

    """
    Seats available.
    """
    seats: int

    """
    Candidates.
    """
    candidates: int

    """
    POC candidates.
    """
    poccandidates: Optional[int] = None

    """
    White candidates.
    """
    wcandidates: Optional[int] = None

    """
    Concentration parameters.
    """
    concentration: list

    """
    Name of the concentration.
    """
    concentrationname: Optional[str] = None

    """
    List of POC wins.
    """
    pocwins: list

    """
    Name of the model.
    """
    model: str

    """
    Name of the pool.
    """
    pool: Optional[int] = None


def aggregate(ensemble, models, concentrations, subsample=3) -> dict:
    """
    Aggregates ModelResults based on some combination of properties.

    Args:
        ensemble (list): A nested list which contains raw `ModelingResult`s.
        levels (list):

    Returns:
        A nested dictionary with keys corresponding to each sublist of `levels`,
        where values (at the end) are lists of results.
    """
    # Get all the combinations of the levels.
    combinations = list(product(models, concentrations))
    keys = ["model", "concentrationname"]

    # Create buckets for results!
    totals = { model: [] for model in models }
    aggregated = []

    for plan in ensemble:
        for combination in combinations:
            # Create a statewide dictionary for collecting stuff... statewide.
            statewide = {}

            for identifier, district in enumerate(plan):
                properties = list(zip(keys, combination))
                abides = [
                    c.pocwins[:subsample] for c in district if
                    all(dict(c)[key] == property for key, property in properties)
                ]

                statewide[identifier] = abides

            # Now that we have statewide results, collected independently of the
            # number of levels, we can do some counting!
            indexset = list(cwr(range(len(abides)), r=len(statewide)))

            # For each of the index combinations, create arrays of district-wide
            # results!
            for indices in indexset:
                subtotals = np.array([
                    district[index]
                    for index, district in zip(indices, statewide.values())
                ])

                # Now, get all the combinations of the columns of results. This
                # could take a little while.
                colcombinations = list(cwr(range(subsample), r=len(subtotals)))

                # Sum over all the column combinations!
                for sc in colcombinations:
                    i = np.array(sc)
                    results = np.take_along_axis(subtotals, i[:,None], axis=1)
                    s = sum(results.flat)
                    
                    m = combination[0]
                    totals[m].append(s)
                    aggregated.append(s)

    # Now do an aggregated total!
    totals["all"] = aggregated

    # Return.
    return totals
