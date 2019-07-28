import random
import numpy as np

from itertools import chain
from collections import defaultdict
from operator import attrgetter, itemgetter
from deap import base, tools, algorithms
from deap.tools.emo import assignCrowdingDist

def get_utility(fit, preference):
    wf = np.array(fit.wvalues)
    assert(wf.shape[0]==preference.shape[1])
    return np.dot(wf, preference.transpose())

def line_dominates(fit, other_fit, preference):
    utility = get_utility(fit, preference)
    other_utility = get_utility(other_fit, preference)
    return (utility > other_utility).all()

def sortIndividual(individuals, preferences, k, first_front_only=False):
    if k == 0:
        return []

    map_fit_ind = defaultdict(list)
    for ind in individuals:
        map_fit_ind[ind.fitness].append(ind)
    fits = list(map_fit_ind.keys())

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i + 1:]:
            if line_dominates(fit_i, fit_j, preferences):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif line_dominates(fit_j, fit_i, preferences):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts

def selectW(W, fits):
    metrix = np.dot(fits, np.transpose(W))
    windex = metrix.argmin(axis=1)
    windex = np.unique(windex)
    return windex


def sortPreference(individuals, preferences, k, first_front_only=False):
    if k == 0:
        return []

    fits = []
    for ind in individuals:
        fits.append(ind.fitness.values)

    fronts = []
    dominated_preferences = []

    # Rank first Pareto front
    windex = selectW(preferences, fits)
    for i, preference in enumerate(preferences):
        if i in windex:
            fronts.append(preference)
        else:
            dominated_preferences.append(preference)

    pareto_sorted = len(fronts)

    if not first_front_only:
        N = min(len(preferences), k)
        while pareto_sorted < N:
            preferences = dominated_preferences
            dominated_preferences = []
            windex = selectW(preferences, fits)
            for i, preference in enumerate(preferences):
                if i in windex:
                    fronts.append(preference)
                    pareto_sorted += 1
                else:
                    dominated_preferences.append(preference)

    return fronts

# coSelect method, return subset of individuals and preferencecs
def coSelect(individuals, preferences, k, k_p):
    pareto_fronts = sortIndividual(individuals, preferences, k)

    for front in pareto_fronts:
        assignCrowdingDist(front)

    chosen_m = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen_m)
    if k > 0:
        sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen_m.extend(sorted_front[:k])

    preference_fronts = sortPreference(chosen_m, preferences, k_p)
    chosen_w = preference_fronts[:k_p]

    return chosen_m, chosen_w

def weightGenerator(shape):
    weight = np.random.random(shape)
    weight = weight / np.sum(weight)
    return weight

def weightConcat(preference_a, preference_b, axis=0):
    preference_a = np.array(preference_a)
    preference_b = np.array(preference_b)
    return np.concatenate((preference_a, preference_b), axis=axis)

def normalize(matrix):
    # trival virable avoids nan
    t_v = 0.00001
    matrix_max, matrix_min = matrix.max(axis=0), matrix.min(axis=0)
    matrix = (matrix - matrix_min) / (matrix_max + t_v - matrix_min)
    return matrix

def fitness_normalization(fits):
    pn_fits = [fit for fit in fits]
    matrix_fits = np.asarray(pn_fits)
    matrix_fits = normalize(matrix_fits)
    return list(map(tuple, matrix_fits))

def coEnvolve(pop, toolbox, ngen, npreference, nobj, cxpb, mutpb, seed=None):
    random.seed(seed)
    MU = len(pop)

    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    preferences = weightGenerator((npreference, nobj))

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    fitnesses = fitness_normalization(fitnesses)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop, preferences = toolbox.select(pop, preferences, len(pop), len(preferences))

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        fitnesses = fitness_normalization(fitnesses)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        preferences_c = weightGenerator((npreference, nobj))
        # Select the next generation population from parents and offspring
        pop, preferences = toolbox.select(pop + offspring, weightConcat(preferences, preferences_c), MU, npreference)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    return pop, logbook