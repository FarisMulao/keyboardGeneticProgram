#!/usr/bin/python3
# -*- coding: utf-8 -*-

# %%
import random
from typing import TypedDict
import math
import json
import datetime
import numpy as np

# ./corpus/2_count.py specificies this same structure:
# Positions =  01234   56789   01234
LEFT_DVORAK = "',.PY" "AOEUI" ";QJKX"
LEFT_QWERTY = "QWERT" "ASDFG" "ZXCVB"
LEFT_COLEMK = "QWFPG" "ARSTD" "ZXCVB"
LEFT_WORKMN = "QDRWB" "ASHTG" "ZXMCV"

LEFT_DISTAN = "22222" "11112" "22222"
LEFT_ERGONO = "11112" "11112" "22323"
LEFT_EDGE_B = "12345" "12345" "12345"

# 8 non-thumb fingers are used for touch typing:
LEFT_FINGID = "01233" "01233" "01233"

# Positions     56   7890123   456789   01234
RIGHT_DVORAK = "[]" "FGCRL/=" "DHTNS-" "BMWVZ"
RIGHT_QWERTY = "-=" "YUIOP[]" "HJKL;'" "NM,./"
RIGHT_COLEMK = "-=" "JLUY;[]" "HNEIO'" "KM,./"
RIGHT_WOKRMN = "-=" "JFUP;[]" "YNEOI'" "KL,./"

RIGHT_DISTAN = "34" "2222223" "211112" "22222"
RIGHT_ERGONO = "33" "3111134" "211112" "21222"
RIGHT_EDGE_B = "21" "7654321" "654321" "54321"

# Non-thumb fingers are numbered [0,...,7]
RIGHT_FINGID = "77" "4456777" "445677" "44567"

DVORAK = LEFT_DVORAK + RIGHT_DVORAK
QWERTY = LEFT_QWERTY + RIGHT_QWERTY
COLEMAK = LEFT_COLEMK + RIGHT_COLEMK
WORKMAN = LEFT_WORKMN + RIGHT_WOKRMN
# BEST1 = "-IOUX'REAG/QZYJ[]WCLDBK=FNSTH;MPV,."
# BEST2 = "-LIOU'REAG/QZYJ[]WCXDBK=FNSTH;MPV,."
# BEST3 = "-LIOU'REAY/WQXZ[]JCDGBK=FNSTH;MPV,."
BEST = "-LIOU'REAY/WQXZK=JCDGB[]FNSTH;MPV,."

DISTANCE = LEFT_DISTAN + RIGHT_DISTAN
ERGONOMICS = LEFT_ERGONO + RIGHT_ERGONO
PREFER_EDGES = LEFT_EDGE_B + RIGHT_EDGE_B

FINGER_ID = LEFT_FINGID + RIGHT_FINGID

with open(file="typing_data/manual-typing-data_qwerty.json", mode="r") as f:
    data_qwerty = json.load(fp=f)
with open(file="typing_data/manual-typing-data_dvorak.json", mode="r") as f:
    data_dvorak = json.load(fp=f)
data_values = list(data_qwerty.values()) + list(data_dvorak.values())
mean_value = sum(data_values) / len(data_values)
data_combine = []
for dv, qw in zip(DVORAK, QWERTY):
    if dv in data_dvorak.keys() and qw in data_qwerty.keys():
        data_combine.append((data_dvorak[dv] + data_qwerty[qw]) / 2)
    elif dv in data_dvorak.keys() and qw not in data_qwerty.keys():
        data_combine.append(data_dvorak[dv])
    elif dv not in data_dvorak.keys() and qw in data_qwerty.keys():
        data_combine.append(data_qwerty[qw])
    else:
        # Fill missing data with the mean
        data_combine.append(mean_value)


class Individual(TypedDict):
    genome: str
    fitness: int


Population = list[Individual]


def render_keyboard(individual: Individual) -> str:
    layout = individual["genome"]
    fitness = individual["fitness"]
    """Prints the keyboard in a nice way"""
    return (
        f"______________  ________________\n"
        f" ` 1 2 3 4 5 6  7 8 9 0 " + " ".join(layout[15:17]) + " Back\n"
        f"Tab " + " ".join(layout[0:5]) + "  " + " ".join(layout[17:24]) + " \\\n"
        f"Caps " + " ".join(layout[5:10]) + "  " + " ".join(layout[24:30]) + " Enter\n"
        f"Shift "
        + " ".join(layout[10:15])
        + "  "
        + " ".join(layout[30:35])
        + " Shift\n"
        f"\nAbove keyboard has fitness of: {fitness}"
    )


def initialize_individual(genome: str, fitness: int) -> Individual:
    """
    Purpose:        Create one individual
    Parameters:     genome as string, fitness as integer (higher better)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a dict[str, int]
    Modifies:       Nothing
    """
    return Individual(genome=genome, fitness=fitness)


def initialize_pop(example_genome: str, pop_size: int) -> Population:
    """
    Purpose:        Create population to evolve
    Parameters:     Goal string, population size as int
    User Input:     no
    Prints:         no
    Returns:        a population, as a list of Individuals
    Modifies:       Nothing
    """
    pop = []
    for i in range(pop_size):
        individual = initialize_individual(example_genome, 0)
        pop.append(individual)
    return pop


def recombine_pair(parent1: Individual, parent2: Individual) -> Population:
    """
    Purpose:        Recombine two parents to produce two children
    Parameters:     Two parents as Individuals
    User Input:     no
    Prints:         no
    Returns:        A population of size 2, the children
    Modifies:       Nothing
    """
    length = len(parent1["genome"])  # readability
    split = random.choice(range(length))
    child1_genome = parent1["genome"][:split] + parent2["genome"][split:]
    child2_genome = parent2["genome"][:split] + parent1["genome"][split:]

    # make sure children are valid we don't want to repeat letters
    for i in parent1["genome"][split:]:
        if i not in child2_genome:
            child2_genome += i

    for j in parent1["genome"][:split]:
        if j not in child2_genome:
            child2_genome += j

    for k in parent2["genome"][split:]:
        if k not in child1_genome:
            child1_genome += k

    for l in parent2["genome"][:split]:
        if l not in child1_genome:
            child1_genome += l

    child1 = initialize_individual(child1_genome, 0)
    child2 = initialize_individual(child2_genome, 0)
    return [child1, child2]


def recombine_group(parents: Population, recombine_rate: float) -> Population:
    """
    Purpose:        Recombines a whole group, returns the new population
                    Pair parents 1-2, 2-3, 3-4, etc..
                    Recombine at rate, else clone the parents.
    Parameters:     parents and recombine rate
    User Input:     no
    Prints:         no
    Returns:        New population of children
    """
    children = []
    for i in range(0, len(parents) - 1, 2):
        mom = parents[i]
        dad = parents[i + 1]
        if random.random() < recombine_rate:
            dau, son = recombine_pair(mom, dad)
        else:
            dau, son = mom, dad

        children.extend([dau, son])
    return children


def mutate_individual(parent: Individual, mutate_rate: float) -> Individual:
    """
    Purpose:        Mutate one individual
    Parameters:     One parents as Individual, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    """
    mutation = list(parent["genome"])
    length = len(parent["genome"])
    for i in range(length):
        if random.random() < mutate_rate:
            j = random.choice(list(range(len(parent["genome"])))) 
            mutation[i], mutation[j] = mutation[j], mutation[i]
    mutated_genome = "".join(mutation)
    return initialize_individual(mutated_genome, 0)


def mutate_group(children: Population, mutate_rate: float) -> Population:
    """
    Purpose:        Mutates a whole Population, returns the mutated group
    Parameters:     Population, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    """
    mutants = []
    for child in children:
        mutants.append(mutate_individual(child, mutate_rate))
    return mutants


def evaluate_individual(individual: Individual) -> None:
    """
    Purpose:        Computes and modifies the fitness for one individual
                    Assumes and relies on the logc of ./corpus/2_counts.py
    Parameters:     One Individual
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The individual (mutable object)
    """
    layout = individual["genome"]

    # This checks if the keyboard is valid.
    # It is a permutation of the given keys, without duplicates or deletions.
    if set(dvorak["genome"]) != set(layout) or len(dvorak["genome"]) != len(layout):
        individual["fitness"] = int(1e9)
        return

    fitness = 0

    # Basic return to home row, with no differential cost for repeats.
    for pos, key in enumerate(layout):
        fitness += count_dict[key] * int(DISTANCE[pos])

    # Top-down guess at ideal ergonomics
    for pos, key in enumerate(layout):
        fitness += count_dict[key] * int(ERGONOMICS[pos])

    # Keybr.com querty-dvorak average data as estimate of real hand
    for pos, key in enumerate(layout):
        # These float numbers range from (0-1],
        # transformed so higher floats are better/faster:
        fitness += 2 * (count_dict[key] / data_combine[pos])
        # 2 * was just to increase the emphasis on this real data a bit

    # Symbols should be toward edges.
    for pos, key in enumerate(layout):
        if key in "-[],.';/=":
            fitness += int(PREFER_EDGES[pos])

    # Vowels on the left, Consosants on the right
    for pos, key in enumerate(layout):
        if key in "AEIOUY" and pos > 14:
            fitness += 3

    # [] {} () <> should be adjacent.
    # () are fixed by design choice (number line).
    # [] and {} are on same keys.
    # Perhaps ideally, <> and () should be on same keys too...
    right_edges = [4, 9, 14, 16, 23, 29, 34]
    for pos, key in enumerate(layout):
        # order of (x or y) protects index on far right:
        if key == "[" and (pos in right_edges or "]" != layout[pos + 1]):
            fitness += 1
        if key == "," and (pos in right_edges or "." != layout[pos + 1]):
            fitness += 1

    # High transitional probabilities should be rolls
    # For example, 2-char sequences: in, ch, th, re, er, etc.
    # Rolls are rewarded inwards on the hand.
    # Left is left to right, and right is right to left.
    # left_edges = [0, 5, 10, 15, 17, 24, 30]
    # This is the left half of keyboard:
    for pos in range(len(layout) - 1):
        if pos in right_edges:
            continue
        char1 = layout[pos]
        char2 = layout[pos + 1]
        dict_key = char1 + char2
        # This is the right half of keyboard
        if pos > 14:
            char1, char2 = char2, char1
        fitness -= count_run2_dict[dict_key]

    # Penalize any 2 character run that occurs on the same finger,
    # in proportion to the count of the run.
    # If they don't occur on the same finger, no penalty.
    for keypair, freq in count_run2_dict.items():
        key1pos = layout.find(keypair[0])
        key2pos = layout.find(keypair[1])
        if FINGER_ID[key1pos] == FINGER_ID[key2pos]:
            fitness += freq

    individual["fitness"] = fitness


def evaluate_group(individuals: Population) -> None:
    """
    Purpose:        Computes and modifies the fitness for population
    Parameters:     Objective string, Population
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The Individuals, all mutable objects
    """
    for i in individuals:
        if i["fitness"] == 0:
            evaluate_individual(i)
    return


def rank_group(individuals: Population) -> None:
    """
    Purpose:        Create one individual
    Parameters:     Population of Individuals
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The population's order (a mutable object)
    """
    individuals.sort(key=lambda i: i["fitness"]) 
    return


def parent_select(individuals: Population, number: int) -> Population:
    """
    Purpose:        Choose parents in direct probability to their fitness
    Parameters:     Population, the number of individuals to pick.
    User Input:     no
    Prints:         no
    Returns:        Sub-population
    Modifies:       Nothing
    """
    parents = []
    fit = []
    for i in individuals:
        fit_val = i["fitness"]
        fit.append(fit_val)
    parents = random.choices(individuals, fit, k=number)
    return parents


def survivor_select(individuals: Population, pop_size: int) -> Population:
    """
    Purpose:        Picks who gets to live!
    Parameters:     Population, and population size to return.
    User Input:     no
    Prints:         no
    Returns:        Population, of pop_size
    Modifies:       Nothing
    """
    return individuals[:pop_size]


def evolve(example_genome: str, pop_size: int = 100) -> Population:
    """
    Purpose:        A whole EC run, main driver
    Parameters:     The evolved population of solutions
    User Input:     No
    Prints:         Updates every time fitness switches.
    Returns:        Population
    Modifies:       Various data structures
    """
    # To debug doctest test in pudb
    # Highlight the line of code below below
    # Type 't' to jump 'to' it
    # Type 's' to 'step' deeper
    # Type 'n' to 'next' over
    # Type 'f' or 'r' to finish/return a function call and go back to caller
    mutate_rate = 0.2  # high mutation rate = SUCCESS!!
    recombine_rate = 0.8

    pop = initialize_pop(example_genome, pop_size)
    evaluate_group(pop)
    rank_group(pop)
    for i in range(60000): 
        parents = parent_select(pop, pop_size)
        children = recombine_group(parents, recombine_rate)

        mutants = mutate_group(children, mutate_rate)
        evaluate_group(mutants)

        temp = pop + mutants
        rank_group(temp)
        pop = survivor_select(temp, pop_size)

        if i % 100 == 0:
            print("progress:", i, "/ 20000", "best:", pop[0])
    return pop


seed = False

if __name__ == "__main__":
    divider = "===================================================="
    # Execute doctests to protect main:
    # import doctest

    # doctest.testmod()
    # doctest.testmod(verbose=True)

    if seed:
        random.seed(42)

    with open("corpus/counts.json") as fhand:
        count_dict = json.load(fhand)
    # print({k: v for k, v in sorted(count_dict.items(), key=lambda item: item[1], reverse=True)})
    # print("Above is the order of frequency of letters in English.")

    # print("Counts of characters in big corpus, ordered by freqency:")
    # ordered = sorted(count_dict, key=count_dict.__getitem__, reverse=True)
    # for key in ordered:
    #     print(key, count_dict[key])

    with open("corpus/counts_run2.json") as fhand:
        count_run2_dict = json.load(fhand)
    # print({k: v for k, v in sorted(count_run2_dict.items(), key=lambda item: item[1], reverse=True)})
    # print("Above is the order of frequency of letter-pairs in English.")

    print(divider)
    print(
        f"Number of possible permutations of standard keyboard: {math.factorial(len(DVORAK)):,e}"
    )
    print("That's a huge space to search through")
    print("The messy landscape is a difficult to optimize multi-modal space")
    print("Lower fitness is better.")

    print(divider)
    print("\nThis is the Dvorak keyboard:")
    dvorak = Individual(genome=DVORAK, fitness=0)
    evaluate_individual(dvorak)
    print(render_keyboard(dvorak))

    print(divider)
    print("\nThis is the Workman keyboard:")
    workman = Individual(genome=WORKMAN, fitness=0)
    evaluate_individual(workman)
    print(render_keyboard(workman))

    print(divider)
    print("\nThis is the Colemak keyboard:")
    colemak = Individual(genome=COLEMAK, fitness=0)
    evaluate_individual(colemak)
    print(render_keyboard(colemak))

    print(divider)
    print("\nThis is the QWERTY keyboard:")
    qwerty = Individual(genome=QWERTY, fitness=0)
    evaluate_individual(qwerty)
    print(render_keyboard(qwerty))

    print(divider)
    print("\nThis is a random layout:")
    badarr = list(DVORAK)
    random.shuffle(badarr)
    badstr = "".join(badarr)
    badkey = Individual(genome=badstr, fitness=0)
    evaluate_individual(badkey)
    print(render_keyboard(badkey))

    print(divider)
    print("Below, we print parts of the fitness map (not keyboards themselves)")

    print("\n\nThis is the distance assumption:")
    dist = Individual(genome=DISTANCE, fitness=0)
    print(render_keyboard(dist))

    print("\n\nThis is another human-invented ergonomics assumption:")
    ergo = Individual(genome=ERGONOMICS, fitness=0)
    print(render_keyboard(ergo))

    print("\n\nThis is the edge-avoidance mechanism for special characters:")
    edges = Individual(genome=PREFER_EDGES, fitness=0)
    print(render_keyboard(edges))

    print("\n\nThis is real typing data, transformed so bigger=better:")
    realdata = "".join(
        [str(int(round(reaction_time * 10, 0) - 1)) for reaction_time in data_combine]
    )
    real_rt = Individual(genome=realdata, fitness=0)
    print(render_keyboard(real_rt))

    print("\n\nThis is the finger typing map:")
    edges = Individual(genome=FINGER_ID, fitness=0)
    print(render_keyboard(edges))

    print(divider)
    input("Press any key to start")
    # population = evolve(example_genome=DVORAK)
    population = evolve(example_genome=BEST)

    print("Here is the best layout:")
    print(render_keyboard(population[0]))

    grade = 0
    if qwerty["fitness"] < population[0]["fitness"]:
        grade = 0
    elif colemak["fitness"] < population[0]["fitness"]:
        grade = 50
    elif workman["fitness"] < population[0]["fitness"]:
        grade = 60
    elif dvorak["fitness"] < population[0]["fitness"]:
        grade = 70
    else:
        grade = 80

    with open(file="results.txt", mode="w") as f:
        f.write(str(grade))

    with open(file="best.json", mode="w") as f:
        f.write(json.dumps(population[0]))

    with open(file="best.txt", mode="w") as f:
        f.write(render_keyboard(population[0]))
