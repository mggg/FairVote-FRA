
import pandas as pd
import us
import json
from ModelingConfiguration import ModelingConfiguration
import jsonlines
from pathlib import Path
import numpy as np

# Create a function for tranching these data points.
def tranch(pp, wp):
    # Categorize these values into the appropriate scenarios.
    if pp <= 0.6: pp = 0.6
    elif 0.6 < pp <= 0.7: pp = 0.7
    elif 0.7 < pp <= 0.8: pp = 0.8
    else: pp = 0.9

    if wp >= 0.4: wp = 0.4
    elif 0.25 <= wp < 0.4: wp = 0.25
    else: wp = 0.15

    return pp, 1-pp, 1-wp, wp 
    
# Get polarization parameters, creating a schema for each state.
summary = pd.read_csv("./data/demographics/summary.csv")
configurations = { state.name.lower(): {} for state in us.states.STATES }

# Create configurations for each state.
states = summary.to_dict(orient="records")

# Name the concentrations.
concentrations = {
    "A": [0.5]*4,               # Voters more or less agree on candidate order for each group.
    "B": [2,0.5,0.5,0.5],       # POC voters vary on POC candidate rank-order, but other groups agree.
    "C": [2]*4,                 # Voters don't agree on anything.
    "D": [0.5,0.5,2,2],         # POC voters agree, and white voters don't.
    # "E": [1]*4                  # No agreement or disagreement --- it's pandemonium.
}

# Set a flag for when we're doing a turnout-reduced model.
reducedturnout = True

# Create records for polarization parameters.
polarizationrecords = []

# Name the models.
models = ["bradley-terry", "plackett-luce", "crossover", "cambridge"]

# Name the focus states.
focus = [us.states.IL, us.states.FL, us.states.MA, us.states.MD, us.states.TX]

# How many people are represented by a single RCV ballot? And how many simulated
# elections do we conduct per configuration? Also how many sample plans are we
# getting?
divisor = 45
simulations = 5
samplesize = 50

# For each of the states, we have two options: create a single statewide model
# (for states with fewer than 6 representatives) with a high number of simulations;
# or create multiple statewide models (~5) based on plans sampled according to the
# distribution of seats seen across the chain (i.e. if a plan with 5 seats is
# seen one-fifth of the time, at least one of the five plans selected should have
# five seats).
for state in states:
    # Check whether the state has a chain associated with it. If it does, we create
    # the configuration for a single modeling run and continue; otherwise, we get
    # the chain and pick uniformly.
    location = us.states.lookup(state["STATE"], field="name")
    # if reducedturnout and location not in focus: continue

    # Get the two-way vote total.
    twoway = state["2020_PRES_D"] + state["2020_PRES_R"]

    # Get the share of votes cast by white people; from there, calculate the number
    # of votes cast by white people and POC.
    wshare = state["2020_PRES_WHITE"]
    wvote = twoway*wshare
    pvote = twoway-wvote
    
    # Now, get the number of Democratic votes cast by each group.
    wdshare = state["2020_PRES_WHITE_D"]
    dvote = state["2020_PRES_D"]
    wdvote = wvote*wdshare
    pdvote = dvote-wdvote
    assert wdvote+pdvote == dvote

    # Get POC-for-POC parameter and the POC-for-white parameter. Ensure that
    pp = pdvote/pvote
    pw = 1-pp
    assert pp+pw == 1

    # Do the same for Republicans.
    wrshare = state["2020_PRES_WHITE_R"]
    rvote = state["2020_PRES_R"]
    wrvote = wvote*wrshare
    prvote = rvote-wrvote
    assert wrvote+prvote == rvote

    # Get the white-for-white and white-for-POC parameters.
    ww = wrvote/wvote
    wp = 1-ww
    assert ww+wp == 1

    # Save old ones, create new ones!
    opp, opw, oww, owp = pp, pw, ww, wp
    pp, pw, ww, wp = tranch(pp, wp)

    # Create a record!
    polarizationrecords.append(dict(
        STATE = state["STATE"],
        pp = opp,
        pw = opw,
        ww = oww,
        wp = owp,
        ppadj = pp,
        pwadj = pw,
        wwadj = ww,
        wpadj = wp,
    ))

    # Create a default thing.
    defaults = dict(
        pp=pp, pw=pw, ww=ww, wp=wp, simulations=simulations
    )

    # Pools.
    pools = list(zip([
        (1.5, 1/2),
        (1.5, state["POCVAP20%"]),
        (2, 1/2),
        (2, state["POCVAP20%"])
    ], [1, 2, 3, 4]))

    if state["REPRESENTATIVES"] < 6:
        # Create the modeling configuration with the computed parameters and all
        # the models.
        configs = []
        
        for model in models:
            for scenario, concentration in concentrations.items():
                for (multiplier, share), name in pools:
                    configs.append(dict(ModelingConfiguration(
                        **defaults,
                        multiplier=multiplier,
                        poc=share,
                        seats=state["REPRESENTATIVES"],
                        pocshare=state["POCVAP20%"],
                        ballots=state["VAP20"]//divisor,
                        model=model,
                        concentration=concentration,
                        concentrationname=scenario,
                        pool=name,
                        turnout=3/4 if reducedturnout else 1
                    )))

        # Be done with this state.
        configurations[location.name.lower()]["neutral"] = [[configs]]
        continue

    # Otherwise, read in the state chain records. If the state is one of our
    # focus state, we run double the number of simulations (one for the neutral
    # chain and one for the tilted one); otherwise, we do 5 simulations per plan.
    neutralpath = Path(f"./data/records/{location.name.lower()}/neutral.jsonl")
    if not neutralpath.exists(): continue
    with jsonlines.open(neutralpath) as r: neutral = [p for p in r]

    chains = [neutral]
    chaintypes = ["neutral"]

    if location in focus:
        tiltedpath = Path(f"./data/records/{location.name.lower()}/tilted.jsonl")
        with jsonlines.open(tiltedpath) as r: tilted = [p for p in r]
        chains.append(tilted)
        chaintypes.append("tilted")
    
    # Get a sample of 5 things; for each of these plans, we want to simulate an
    # RCV election on each of the districts.
    for chain, chaintype in zip(chains, chaintypes):
        sample = np.random.choice(chain, size=samplesize, replace=False)
        configurations[location.name.lower()][chaintype] = []

        # For each plan in the sample, and for each district in each plan, we
        # create a simulation configuration for each combination of candidate
        # multiplier and proportion of POC candidates. The configurations dict
        # is then structured like:
        #
        # {
        #   state: {
        #       chain type: [
        #           plan[
        #               district[
        #                   configuration
        #               ]
        #           ]
        #       ]
        #   }
        #   ...
        # }
        for plan in sample:
            planconfigs = []

            for district in plan.values():
                configs = []

                # Pools.
                pools = list(zip([
                    (1.5, 1/2),
                    (1.5, district["POCVAP20%"]),
                    (2, 1/2),
                    (2, district["POCVAP20%"])
                ], [1, 2, 3, 4]))
        
                for model in models:
                    for scenario, concentration in concentrations.items():
                        for (multiplier, share), name in pools:
                            configs.append(dict(ModelingConfiguration(
                                **defaults,
                                multiplier=multiplier,
                                poc=share,
                                seats=district["MAGNITUDE"],
                                pocshare=district["POCVAP20%"],
                                ballots=district["VAP20"]//divisor,
                                model=model,
                                concentration=concentration,
                                concentrationname=scenario,
                                pool=name,
                                turnout=3/4 if reducedturnout else 1
                            )))

                planconfigs.append(configs)
        
            configurations[location.name.lower()][chaintype].append(planconfigs)

# Write configurations to file.
turnoutsuffix = "-lowturnout" if reducedturnout else ""
with open(f"configurations{turnoutsuffix}.json", "w") as w: json.dump(configurations, w)

# Write records to file.
pd.DataFrame.from_records(polarizationrecords).to_csv("./data/demographics/polarization.csv", index=False)
