# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:28:14 2020

@author: darac
"""
import pandas as pd
import numpy as np
from numpy.random import choice
import itertools
import random
import matplotlib.pyplot as plt

def luce(num_ballots,
         mean_support_by_race,
         std_support_by_race,
         cand_list,
         vote_portion_by_race,
         race_list):
    ballots_list = []
    for race in vote_portion_by_race.keys():
        num_ballots_race = int(num_ballots*vote_portion_by_race[race])
        cand_support_vec = list(mean_support_by_race[race].values())
        for j in range(num_ballots_race):
            ballot = list(choice(cand_list, len(cand_list), p = cand_support_vec, replace = False))
            ballots_list.append(ballot)
    return ballots_list

def luce_partial_ballot(num_ballots,
         mean_support_by_race,
         std_support_by_race,
         cand_list,
         vote_portion_by_race,
         race_list,
         ballot_len_filled,
         race_partial_ballot):
    ballots_list = []
    for race in vote_portion_by_race.keys():
        num_ballots_race = int(num_ballots*vote_portion_by_race[race])
        cand_support_vec = list(mean_support_by_race[race].values())
        for j in range(num_ballots_race):
            ballot = list(choice(cand_list, len(cand_list), p = cand_support_vec, replace = False))
            if race == race_partial_ballot:
                append_ballot = ballot[:ballot_len_filled]
                ballots_list.append(append_ballot)
            else:
                 append_ballot = ballot
                 ballots_list.append(append_ballot)
    return ballots_list

def luce_custom_ballot(num_ballots,
         mean_support_by_race,
         std_support_by_race,
         cand_list,
         vote_portion_by_race,
         race_list,
         ballot_len_filled,
         race_partial_ballot):
    ballots_list = []
    for race in vote_portion_by_race.keys():

        num_ballots_race = int(num_ballots*vote_portion_by_race[race])
        cand_support_vec = list(mean_support_by_race[race].values())
        for j in range(num_ballots_race):
            ballot = list(choice(cand_list, len(cand_list), p = cand_support_vec, replace = False))
            if race == race_partial_ballot:
                ballot1 = ['Cand 7', 'Cand 8', 'Cand 9', 'Cand 10', 'Cand 11', 'Cand 12']
                cand_list_trunc = cand_list[:len(ballot1)]
                cand_support_trunc = [1/(len(cand_list)-len(ballot1))]*(len(cand_list)-len(ballot1))
                ballot2 = list(choice(cand_list_trunc, len(cand_list_trunc), p = cand_support_trunc, replace = False))
                ballot = ballot1 + ballot2
                append_ballot = ballot[:ballot_len_filled]
                ballots_list.append(append_ballot)
                if ballot_len_filled <=2:
                    print(race, append_ballot, ballot_len_filled)
            else:
                 append_ballot = ballot
                 ballots_list.append(append_ballot)
    return ballots_list


def luce_uncertainty(num_ballots, mean_support_by_race, std_support_by_race, cand_list, vote_portion_by_race, race_list):
    #luce model with additional layer: first draw from distribution around each race's estimated
    #support of each candidate to get their "true" support of the candidates
    #Normalize the outputs and plug into luce model
    ballots_list = []
    for race in vote_portion_by_race.keys():
        num_ballots_race = int(num_ballots*vote_portion_by_race[race])
        cand_support_vec = list(mean_support_by_race[race].values())
        cand_std_vec = list(std_support_by_race[race].values())
        for j in range(num_ballots_race):
            rand_cand_support = [max(.000001, np.random.normal(cand_support_vec[k], cand_std_vec[k])) for k in range(len(cand_list))]
            norm_support_by_race = [x/sum(rand_cand_support) for x in rand_cand_support]
            ballot = list(choice(cand_list, len(cand_list), p = norm_support_by_race, replace = False))
            ballots_list.append(ballot)
    return ballots_list

def paired_comparison_ballot_type(
    preference_strengths,
    portions,
    candidates_by_race,
    ballot_length,
    num_ballots,
    fill_lengths=None
):
    '''
    Custom function for effective sampling when preferred candidates are not
    distinguishable, two races
    '''
    #generate prob.s for ballot types
    num_candidates = [len(x) for x in candidates_by_race.values()]
    if fill_lengths==None:
        fill_lengths=[sum(num_candidates)]*2
    binary_ballot_seed = [0]*num_candidates[0] + [1]*num_candidates[1]
    ballot_types = [] #types of ballots given as strings of 0s and 1s
    ballot_type_probs = {x:[] for x in [0,1]} #corresponding  prob.s by race
    for a_ranks in itertools.combinations(range(ballot_length), num_candidates[0]):
        ballot_type = [int(x not in a_ranks) for x in range(ballot_length)]
        ballot_types.append(ballot_type)
        p = {0:1, 1:1}
        #run over i < j
        for i in range(ballot_length):
            for j in range(i+1, ballot_length):
                for race in [0,1]:
                    if ballot_type[i] == ballot_type[j]:
                        p[race] *= 0.5
                    elif (ballot_type[j] == race) and (ballot_type[i] != race):
                        p[race] *= (1-preference_strengths[race]) #wrong order
                    elif (ballot_type[i] == race) and (ballot_type[j] != race):
                        p[race] *= (preference_strengths[race]) #right order
        for race in [0,1]:
            ballot_type_probs[race].append(p[race])
    for race in [0,1]:
        total_p = sum(ballot_type_probs[race])
        for i, x in enumerate(ballot_type_probs[race]):
            ballot_type_probs[race][i] /= total_p

    #generate some ballots
    ballots = []
    for race in [0,1]:
        for i in range(int(num_ballots*portions[race])):
            ballot_type = np.random.choice(
                range(len(ballot_types)),
                p=np.array(ballot_type_probs[race])
            )
            ballot_type = ballot_types[ballot_type]
            cand_by_race_queue = candidates_by_race.copy()
            within_race_orders = {
                0: list(np.random.permutation(candidates_by_race[0].copy())),
                1: list(np.random.permutation(candidates_by_race[1].copy())),
            }
            ballot = []
            for j in range(ballot_length):
                ballot.append(within_race_orders[ballot_type[j]].pop())
            ballots.append(ballot[:fill_lengths[race]])
    return ballots

def paired_comparison_predefined(
        num_ballots,
        paired_compare_dicts,
        vote_portion_by_race,
        seeds=None,
        sample_interval=1,
        verbose=False,
        fill_lengths=None
    ):
    if fill_lengths==None:
        fill_lengths=[
            len(set(x[0] for y in paired_compare_dicts.values() for x in y))
        ]*len(paired_compare_dicts)

    #generate ballots based on a paired comparison model for each race
    ballots_list = []
    accept = 0
    for r, race in enumerate(vote_portion_by_race):
        cand_list = list(set([x[0] for x in paired_compare_dicts[race]]))
        #keys are ordered pair of candidates, values are prob i>j in pair of candidates
        paired_compare_dict = paired_compare_dicts[race]
        #function for evaluating single ballot in MCMC
        #don't need normalization term here! Exact probability of a particular ballot would be
        #output of this fnction divided by normalization term that MCMC allows us to avoid
        def ballot_prob(ballot):
            pairs_list_ballot = list(itertools.combinations(ballot,2))
            paired_compare_trunc = {k: paired_compare_dict[k] for k in pairs_list_ballot}
            ballot_prob = np.product(list(paired_compare_trunc.values()))
            return ballot_prob
        #starting ballot for mcmc -- has to have >0 probability
        if seeds == None:
            start_ballot = list(np.random.permutation(cand_list))
            while not ballot_prob(start_ballot):
                start_ballot = list(np.random.permutation(cand_list))
        else:
            start_ballot = seeds[race].copy()

        #start MCMC with 'start_ballot'
        num_ballots_race = int(num_ballots*vote_portion_by_race[race])
        race_ballot_list = []
        step = 0
        while len(race_ballot_list) < num_ballots_race: #range(num_ballots_race):
            #proposed new ballot is a random switch of two elements in ballot before
            proposed_ballot = start_ballot.copy()
            j1,j2 = random.sample(range(len(start_ballot)),2)
            proposed_ballot[j1], proposed_ballot[j2] = proposed_ballot[j2], proposed_ballot[j1]

            #acceptance ratio: (note - symmetric proposal function!)
            accept_ratio = min(ballot_prob(proposed_ballot)/ballot_prob(start_ballot),1)
            #accept or reject proposal
            if random.random() < accept_ratio:
                start_ballot = proposed_ballot
                if step % sample_interval == 0:
                    race_ballot_list.append(start_ballot[:fill_lengths[r]])
                accept += 1
            else:
                if step % sample_interval == 0:
                    race_ballot_list.append(start_ballot[:fill_lengths[r]])
            step += 1
        ballots_list = ballots_list + race_ballot_list
    if verbose:
        print("Acceptance ratio: ", accept/num_ballots)
    return ballots_list

def paired_comparison(num_ballots,
                      mean_support_by_race,
                      std_support_by_race,
                      cand_list,
                      vote_portion_by_race,
                      race_list):
    #define probability distribution over all possible ballots for each race.
    #Draw from each race's prob distribution (number of ballots per race dtmd by cvap share)
    ordered_cand_pairs = list(itertools.permutations(cand_list,2))
    ballot_perms = list(itertools.permutations(cand_list, len(cand_list)))

    ballots_list = []
    for race in race_list:
        #make dictionairy of paired comparisons: i.e. prob i>j for all ordered pairs of candidates
        #keys are ordered pair of candidates, values are prob i>j in pair of candidates
        paired_compare_dict = {k: mean_support_by_race[race][k[0]]/(mean_support_by_race[race][k[0]]+mean_support_by_race[race][k[1]]) for k in ordered_cand_pairs}

        #make dictionary of prob that each ballot occurs
        #key is ballot, value is probability its cast by a member of the race
        ballot_prob_dict = {i: {} for i in ballot_perms}
        for i in ballot_perms:
            pairs_list_ballot = list(itertools.combinations(i,2))
            paired_compare_trunc = {k: paired_compare_dict[k] for k in pairs_list_ballot}
            ballot_prob_dict[i] = np.product(list(paired_compare_trunc.values()))
            #normalizing constant
        norm_constant = sum(ballot_prob_dict.values())
        ballot_prob_dict_norm = {i:ballot_prob_dict[i]/norm_constant for i in ballot_perms}

        #generate ballots for race and append to full ballot list
        num_ballots_race = int(num_ballots*vote_portion_by_race[race])
        ballot_choices = list(ballot_prob_dict_norm.keys())
        ballot_probs = list(ballot_prob_dict_norm.values())
        draw_ballots = random.choices(population = ballot_choices, weights = ballot_probs, k = num_ballots_race)
        ballots_list = ballots_list + draw_ballots

    return ballots_list

def paired_comparison_mcmc(num_ballots,
                           mean_support_by_race,
                           std_support_by_race,
                           cand_list,
                           vote_portion_by_race,
                           race_list,
                           seeds=None,
                           sample_interval=10,
                           verbose = True):
    #Sample from probability distribution for each race using MCMC - don't explicitly
    #compute probability of each ballot in advance
    #Draw from each race's prob distribution (number of ballots per race dtmd by cvap share)
    ordered_cand_pairs = list(itertools.permutations(cand_list,2))
    ballots_list = []

    for race in race_list:
        #make dictionairy of paired comparisons: i.e. prob i>j for all ordered pairs of candidates
        #keys are ordered pair of candidates, values are prob i>j in pair of candidates
        paired_compare_dict = {k: mean_support_by_race[race][k[0]]/(mean_support_by_race[race][k[0]]+mean_support_by_race[race][k[1]]) for k in ordered_cand_pairs}
        #starting ballot for mcmc
        start_ballot = list(np.random.permutation(cand_list))
        #function for evaluating single ballot in MCMC
        #don't need normalization term here! Exact probability of a particular ballot would be
        #output of this fnction divided by normalization term that MCMC allows us to avoid
        track_ballot_prob = []
        def ballot_prob(ballot):
            pairs_list_ballot = list(itertools.combinations(ballot,2))
            paired_compare_trunc = {k: paired_compare_dict[k] for k in pairs_list_ballot}
            ballot_prob = np.product(list(paired_compare_trunc.values()))
            return ballot_prob

        #start MCMC with 'start_ballot'
        num_ballots_race = int(num_ballots*vote_portion_by_race[race])
        race_ballot_list = []
        step = 0
        accept = 0
        while len(race_ballot_list) < num_ballots_race: #range(num_ballots_race):
            #proposed new ballot is a random switch of two elements in ballot before
            proposed_ballot = start_ballot.copy()
            j1,j2 = random.sample(range(len(start_ballot)),2)
            proposed_ballot[j1], proposed_ballot[j2] = proposed_ballot[j2], proposed_ballot[j1]

            #acceptance ratio: (note - symmetric proposal function!)
            accept_ratio = min(ballot_prob(proposed_ballot)/ballot_prob(start_ballot),1)
            #accept or reject proposal
            if random.random() < accept_ratio:
                start_ballot = proposed_ballot
                if step % sample_interval == 0:
                    race_ballot_list.append(start_ballot)
                accept += 1
            else:
                if step % sample_interval == 0:
                    race_ballot_list.append(start_ballot)
            step += 1
        ballots_list = ballots_list + race_ballot_list
        if verbose:
            if step > 0:
                print("Acceptance ratio for {} voters: ".format(race), accept/step)
       # plt.plot(track_ballot_prob)
    return ballots_list
