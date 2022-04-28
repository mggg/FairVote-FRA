
import sys
import jsonlines
import warnings
from pathlib import Path
from model_details import (
    Cambridge_ballot_type, BABABA, luce_dirichlet, bradley_terry_dirichlet
)
from ModelingConfiguration import ModelingConfiguration
from ModelingResult import ModelingResult
import us
import json
import jsonlines

# Suppress all warnings.
warnings.filterwarnings("ignore")

# Get the jurisdiction and read in modeling
loc = sys.argv[-3].replace("-", " ")
JURIS = us.states.lookup(loc.title(), field="name")
BIAS = sys.argv[-2]
INDEX = int(sys.argv[-1])

# Are we doing reduced turnout?
REDUCEDTURNOUT = True
turnoutsuffix = "-lowturnout" if REDUCEDTURNOUT else ""

# Read in modeling information and create ModelingConfigurations from each of the
# raw ones.
with open(f"configurations{turnoutsuffix}.json") as r: plans = json.load(r)[JURIS.name.lower()][BIAS]

# The models we'll be using to conduct experiments.
#   1.  Plackett-Luce RCV model. Voter behavior is modeled by draws from a distribution
#       over the possible candidates weighted by the support probabilities defined above;
#       noise drawn from Dirichlet distributions parameterized by the degreement
#       values above jiggles the ballots.
#   2.  Bradley-Terry Model. Each possible ballot ordering is assigned a probability
#       of selection (by a voter) based on the head-to-head ordering of the candidates.
#       Rather than sampling from this distribution directly (by computing probabilities
#       for each ballot), we sample from the space of ballots using a Markov chain.
#   3.  Alternating crossover model. Each voter is assumed to be a "crossover voter"
#       or a "bloc voter," which vote opposite or according to their group membership's
#       consensus choices, respectively. Each voter is assigned a "ballot type"
#       based on their voter type, and these ballots are permuted according to the
#       degreement assigned earlier.
#   4.  Cambridge Sampler (CS) model. Based on each voter's degreement values, they
#       are probabilistically determined to select a bloc or crossover candidate first;
#       after this first candidate has been ranked, the entire ballot is sampled
#       from the distribution of ballot types in the Cambridge dataset.
models = {
    "plackett-luce": luce_dirichlet,
    "bradley-terry": bradley_terry_dirichlet,
    "crossover": BABABA,
    "cambridge": Cambridge_ballot_type
}

# Get the randomized models and the sampling models.
randomized = {"bradley-terry", "plackett-luce"}
sampling = {"crossover", "cambridge"}

# Are we testing?
TEST = True

# Create a bucket for results.
RESULTS = []

# Get the plan we're going to be evaluating. If we're running statewide, then we
# just get the first one.
try: plan = plans[INDEX]
except: plan = plans[0]

for d, district in enumerate(plan):
    DISTRICTRESULTS = []

    for config in district:
        # Get the modeling configuration.
        configuration = ModelingConfiguration(**config)

        # Create a basic run configuration, which is modified based on whether
        # we're testing or not.
        runconfig = dict(
            num_ballots=(100 if TEST else configuration.ballots),
            num_simulations=(3 if TEST else configuration.simulations)
        )

        # Set up the remaining keyword arguments.
        kwargs = dict(
            poc_share=configuration.pocshare,
            poc_support_for_poc_candidates=configuration.pp,
            poc_support_for_white_candidates=configuration.pw,
            white_support_for_white_candidates=configuration.ww,
            white_support_for_poc_candidates=configuration.wp,
            seats_open=configuration.seats,
            num_poc_candidates=configuration.poccandidates,
            num_white_candidates=configuration.wcandidates,
            max_ballot_length=None,
            **runconfig
        )

        # Get the randomized models and the sampling models.
        model = models[configuration.model]

        if configuration.model in randomized:
            local, atlarge = model(concentrations=configuration.concentration, **kwargs)

            mr = ModelingResult(
                pp=configuration.pp,
                pw=configuration.pw,
                ww=configuration.ww,
                wp=configuration.wp,
                simulations=configuration.simulations,
                pocshare=configuration.pocshare,
                ballots=configuration.ballots,
                seats=configuration.seats,
                candidates=configuration.candidates,
                wcandidates=configuration.wcandidates,
                poccandidates=configuration.poccandidates,
                concentration=configuration.concentration,
                concentrationname=configuration.concentrationname,
                pocwins=local,
                model=configuration.model,
                pool=str(configuration.pool)
            )

            DISTRICTRESULTS.append(dict(mr))

        elif configuration.model in sampling:
            # Simulate the ballots and election again.
            local, atlarge = model(scenarios_to_run=[configuration.concentrationname], **kwargs)
            local = local[configuration.concentrationname]

            mr = ModelingResult(
                pp=configuration.pp,
                pw=configuration.pw,
                ww=configuration.ww,
                wp=configuration.wp,
                simulations=configuration.simulations,
                pocshare=configuration.pocshare,
                ballots=configuration.ballots,
                seats=configuration.seats,
                candidates=configuration.candidates,
                wcandidates=configuration.wcandidates,
                poccandidates=configuration.poccandidates,
                concentration=configuration.concentration,
                concentrationname=configuration.concentrationname,
                pocwins=local,
                model=configuration.model,
                pool=configuration.pool
            )

            DISTRICTRESULTS.append(dict(mr))
        
    RESULTS.append(DISTRICTRESULTS)
        
# Write to file!
write = Path(f"./output/results/{JURIS.name.lower()}/")
if not write.exists(): write.mkdir()

with jsonlines.open(write/f"{BIAS}-{INDEX}{turnoutsuffix}.jsonl", mode="w") as w: w.write_all(RESULTS)
