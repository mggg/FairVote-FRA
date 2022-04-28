
import pandas as pd
import numpy as np
from numpy.random import choice
import random
from vote_transfers import cincinnati_transfer
from ballot_generators import luce, luce_uncertainty, paired_comparison, paired_comparison_mcmc
from additional_funcs import remove_cand, recompute_count
import itertools

def rcv_run(ballot_list, cand_list, num_seats, transfer_method, verbose_bool=False):
    winners = []
    cutoff = int(len(ballot_list)/(num_seats+1) + 1)
    candidates = cand_list.copy()
    ballot_list = [x for x in ballot_list if x != []]
    cand_totals = recompute_count(candidates, ballot_list)

    while len(winners) < num_seats:
        remaining_cands = candidates
        if len(remaining_cands) == num_seats - len(winners):
            winners = winners + remaining_cands
            break
        cand_totals = recompute_count(candidates, ballot_list)

        for cand in list(candidates):
            if len(winners) == num_seats:
                    break
            if cand_totals[cand] >= cutoff:
                winners.append(cand)
                transfer_method(cand, ballot_list, "win", cutoff)
                candidates.remove(cand)
                ballot_list = [x for x in ballot_list if x != []]
                cand_totals = recompute_count(candidates, ballot_list)
                if verbose_bool:
                    print("candidate", cand, "elected")

        if len(winners) == num_seats:
            break

        min_count = min(cand_totals.values())
        min_cand_list = [k for k,v in cand_totals.items() if v == min_count]
        min_cand = random.choice(min_cand_list)
     #   min_cand = min(cand_totals, key=cand_totals.get)
        transfer_method(min_cand, ballot_list, "lose", cutoff)
        candidates.remove(min_cand)
        ballot_list = [x for x in ballot_list if x != []]
        cand_totals= recompute_count(candidates, ballot_list)
        if verbose_bool:
            print("candidate", min_cand, "eliminated")
    return winners

def at_large_run(ballots, cand_list, num_seats):
    votes_for_cand = {c:0 for c in cand_list}
    for c in cand_list:
        votes_for_cand[c] = len([b for b in ballots if c in b[:num_seats]])
    ranking = sorted(votes_for_cand.keys(), key=lambda x:votes_for_cand[x], reverse=True)
    return ranking[:num_seats]
