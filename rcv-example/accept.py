
from random import random
import math

def mh(column):
    """
    Preferentially selects plans based on the value of `column`. Implementation
    of the Metropolis-Hastings criterion.
    """
    def _(partition):
        # Calculate the ratio of scores if the passed Partition has a parent.
        ratio = partition[column]/partition.parent[column] if partition.parent and partition.parent[column] != 0 else 1
        return min(1, ratio) >= random()

    return _


def preference(numerator, denominator):
    """
    An updater which scores plans according to the scoring function on pp. 15 of
    http://dx.doi.org/10.2139/ssrn.3778021.
    """
    def score(proportion):
        if proportion >= 1/2: return 1
        elif proportion >= 0.35: return (proportion-.35)/.15
        else: return 0

    def _(partition):
        return sum(
            score(p/q) for p, q in zip(partition[numerator].values(), partition[denominator].values())
        )

    return _


def mmpreference(numerator, denominator, statewide):
    """
    A scoring method which awards points based on the POCVAP shares of each
    district, based on the relevant STV threshold.

    Args:
        numerator (str): The column name in the numerator of the proportion.
        denominator (str): The column name in the denominator of the proportion.
        statewide (float): The statewide POCVAP share.
    """
    def partialseatscapped(seats, proportion):
        """
        Given a seats value and a proportion value, increase the score by 1 for
        every potential seat in each district (i.e. how many election thresholds)
        are crossed.
        """
        # Get the minimum electoral threshold.
        threshold = 1/(seats+1)

        # If the population proportion is below the threshold, the district gets
        # 0 points.
        if proportion < threshold: return 0

        # If the proportion is above the threshold but not at proportionality,
        # we return the seat share (by proportionality).
        elif proportion >= threshold and proportion < statewide: return seats*proportion

        # If the proportion is *greater* than the statewide share, we award the
        # statewide proportionality share. This, in effect, is a *penalty* for
        # having too high a POCVAP share.
        elif proportion >= statewide: return seats*statewide

    def thresholds(seats, proportion):
        # Get the minimum electoral threshold.
        threshold = 1/(seats+1)

        # Return the number of thresholds crossed.
        return math.floor(proportion/threshold)

    def partialseats(seats, proportion):
        # Get the minimum electoral threshold and return the number of *partial*
        # seats.
        threshold = 1/(seats+1)
        return proportion/threshold

    def _(P):
        return sum(
            partialseats(P["MAGNITUDE"][district], P[numerator][district]/P[denominator][district])
            for district in P.parts
        )
    
    return _

def districts(N, districts):
    """
    Calculates the population intervals for magnitude-3, -4, and -5 districts.
    """
    epsilon = 0.05
    single = N/districts

    # Calculate the intervals.
    left_3, right_3 = single*3*(1-epsilon), single*3*(1+epsilon)
    left_4, right_4 = single*4*(1-epsilon), single*4*(1+epsilon)
    left_5, right_5 = single*5*(1-epsilon), single*5*(1+epsilon)

    def _(c):
        if left_3 <= c <= right_3: return 3
        elif left_4 <= c <= right_4: return 4
        elif left_5 <= c <= right_5: return 5

    return _


def seats(percent, members):
    """
    Calculates the number of "seats" won given a percentage.
    """
    droop = 1/(members+1)
    return math.floor(percent/droop)


def seatsupdater(P):
    """
    An updater for calculating the predicted number of seats.
    """
    return {
        district: seats(P["POCVAP20"][district]/P["VAP20"][district], P["MAGNITUDE"][district])
        for district in P.parts
    }

def totalseats(P):
    return sum(
        seats(P["POCVAP20"][district]/P["VAP20"][district], P["MAGNITUDE"][district])
        for district in P.parts
    )


def logistic(L, k, N, midpoint):
    """
    Returns a closure for getting values from a logistic function.

    Args:
        L (float): The maximum(/minimum) value achieved by the function.
        k (float): Growth rate.
        midprop (float): Point at which growth is steepest.
        N (int): Total number of steps.
    """
    midpoint = midpoint*N

    def _(t):
        """
        Closure for getting a value from the logistic function parameterized by
        L, k, and the midpoint.
        """
        return L/(1+math.exp(-k*(t-midpoint)))

    return _


def logicycle(L, k, N, midpoint, cold=0, cycles=None):
    r"""
    Stitches together logistic curves to form a cycle of temperature decreases
    and increases.

    Args:
        L (float): The maximum(/minimum) value achieved by the function.
        k (float): Growth rate.
        N (int): Total number of steps.
        midpoint (float): A number in between 0 and 1 which indicates the midpoint
            of the cycle, and the point at which the temperature is at its lowest.
            If the number of cycles is \(c\) and the midpoint of the cycle is
            \(m\), then the temperature function reaches its minimum at step
            \(\lfloor N/c \rfloor \cdot m \cdot i\), where \(i\) is the index of
            the current cycle.
        cold (int, optional): The number of steps where we hold the temperature
            at its minimum possible value in each cycle.
    """
    # If there are no cycles, we're just doing a logistic curve.
    if not cycles: return logistic(L, k, N, midpoint)

    # Get the cycle length and the midpoint of the cycle. If there is no cycle
    # specified -- i.e. we only want to cool down and never heat up -- then we
    # simply return the logistic value for t.
    cyclelength = math.floor(N/cycles)
    cyclemidpoint = cyclelength*midpoint

    # Select the start and end times for holding at minimum temperature.
    coldstart = cyclemidpoint-(cold/2) if cold/2 >= 1 else cyclemidpoint
    coldend = coldstart+cold if cold > 1 else cyclemidpoint
    
    # Set a relatively small offset so the function is smooth-ish.
    offset = 10e-4

    # Find when we have to start and end cooling in the cycle based on the start
    # and end cooling times.
    leftmidpoint = coldstart + (math.log(L/(offset*L)-1)/-k)
    rightmidpoint = coldend + (math.log(L/(offset*L)-1)/k)

    # If the left or right tails are beyond 0 or the cycle length, warn the user.
    lefttail = (math.log(L/(offset)-1)/-k) + leftmidpoint
    righttail = (math.log(L/(offset)-1)/k) + rightmidpoint
    
    if lefttail < 0 or righttail > cyclelength:
        print(
            "Under this annealing schedule, the maximum temperature of 0 will not be " + \
            "achieved. To prevent undesirable side effects, please visualize the " + \
            "annealing schedule and adjust parameters accordingly."
        )

    # A function for getting the temperature based on the above parameters.
    def temp(t):
        # Get the location of t in the cycle.
        t = t%cyclelength

        # Now, decide whether we're in the first logistic curve, the plateau, or
        # the second logistic curve.
        if t < coldstart: return L/(1+math.exp(-k*(t-leftmidpoint)))
        elif coldstart <= t <= coldend: return L
        else: return L/(1+math.exp(k*(t-rightmidpoint)))
    
    return temp

def annealing(score, beta, step="step", maximize=False):
    """
    Computes the score of a given partition by sampling from the Boltzmann
    distribution, parameterized by a temperature parameter beta.
    """
    def _(P):
        temperature = -beta(P[step])
        delta = 0 if not P.parent else (P.parent[score]-P[score] if maximize else P[score]-P.parent[score])
        return math.exp(temperature*delta)
    
    return _

def step(P):
    return P.parent["STEP"] + 1 if P.parent else 0
    
