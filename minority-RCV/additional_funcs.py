# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:28:46 2020

@author: darac
"""

import pandas as pd
import numpy as np
from numpy.random import choice

def remove_cand(cand, ballot_list):
    for n, ballot in enumerate(ballot_list):
        new_ballot = []
        for c in ballot:
            if c!= cand:
                new_ballot.append(c)
        ballot_list[n]= new_ballot

def recompute_count(candidates, ballot_list):
    cand_totals = {}
    for cand in candidates:
        cand_totals[cand] = len([ballot for ballot in ballot_list if ballot[0] == cand])
    return cand_totals  
        