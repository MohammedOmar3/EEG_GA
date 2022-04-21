import random
import numpy as np
from joblib.numpy_pickle_utils import xrange
from scipy import rand

chr1 = [random.randint(0, 1) for x in range(7)]
chr2 = [random.randint(0, 1) for x in range(7)]
mylist = [0, 1, 2, 3, 4, 5, 6]

# print(chr1[1])
# print(chr2[1])
pop_size = 20
cro_length = 20
active_el = 2
fitness = list()
pop = list()
new_pop = list()


# pairs = []


def init_pop():
    random.shuffle(mylist)
    for y in range(0, pop_size):
        cr = [random.randint(0, 0) for x in range(cro_length)]
        options = np.random.choice(cro_length, active_el, replace=False)
        # print(options)
        for x in options:
            cr[x] = 1
        # print(cr)
        pop.append(cr)
    # print(pop)


def function(individual):
    str1 = ''.join(str(e) for e in individual)
    dec_number = int(str1, 2)
    return dec_number


def fitness_score(pop):
    for i in range(pop_size):
        str1 = ''.join(str(e) for e in pop[i])
        dec_number = int(str1, 2)
        fitness.insert(i, dec_number)
        # print(str1)
        # print(dec_number)
    fit_value = fitness.copy()
    fit_value.sort(reverse=True)
    best = fit_value[0]
    # print(best)


def aver_fit(fitness):
    sum = 0
    for x in range(len(fitness)):
        sum = sum + fitness[x]
    print(sum / len(fitness))


def fitness_evaluation(pop):
    tmpfit = list()
    for i in range(len(pop)):
        str1 = ''.join(str(e) for e in pop[i])
        dec_number = int(str1, 2)
        tmpfit.insert(i, dec_number)
    print(tmpfit)
    return(tmpfit)
    # print(str1)
    # print(dec_number)


def rws(population, fitnesses, num, size):
    """ Roulette selection, implemented according to:
        <http://stackoverflow.com/questions/177271/roulette
        -selection-in-genetic-algorithms/177278#177278>
    """
    print(fitnesses)
    total_fitness = float(sum(fitnesses))
    print(total_fitness)
    rel_fitness = [f / total_fitness for f in fitnesses]
    print(rel_fitness)
    # Generate probability intervals for each individual
    probs = [sum(rel_fitness[:i + 1]) for i in range(len(rel_fitness))]
    print(probs)
    np.random.choice(np.arange(len(fitnesses)), size=size, replace=True, p=probs)
    # Draw new population
    new_population = []
    for n in xrange(num):
        r = np.random.rand()
        print(r)
        for (i, individual) in enumerate(population):
            if r <= probs[i]:
                new_population.append(individual)
                break
    print(new_population)
    return new_population


def rws2(pop, fitness, n):
    pairs = []
    print(fitness)
    total_fitness = float(sum(fitness))
    print(total_fitness)
    rel_fitness = [f / total_fitness for f in fitness]
    print(rel_fitness)
    for x in range(0, int(pop_size / n)):
        idx = np.random.choice(np.arange(len(pop)), size=n, replace=False, p=rel_fitness)
        print(idx)
        pairs.append(idx)
    print(pairs)
    return (pairs)
    # print(len(pairs))
    # print(pairs[0][0])


# population, fitness

# init_pop()
# print(pop)
# fitness_score(pop)
# rws2(pop, fitness, 2)
# evolve(2, pop)
# fitness_score(pop)
# new_pop = rws(pop, fitness, 4)
# fitness_score(new_pop)


# print(mylist)
# print(chr1)


# print(chr2)


def crossover(chr1, chr2):
    cr1, cr2 = chr1.copy(), chr2.copy()
    allactive = []
    active = []
    done = False
    for x in range(0, cro_length):
        if cr1[x] != cr2[x]:
            # print("valid")
            allactive.append(x)
    while not done:
        active = random.sample(allactive, 2)
        if cr1[active[0]] != cr1[active[1]]:
            for y in active:
                swap(cr1, cr2, y)
                done = True
    # print(allactive)
    # print(active)
    return cr1, cr2


def swap(list1, list2, a):
    list1[a], list2[a] = list2[a], list1[a]


def mutate(chr1):
    active_el2 = list()
    inactive_el = list()
    for x in range(0, cro_length):
        if chr1[x] == 0:
            inactive_el.append(x)
        elif chr1[x] == 1:
            active_el2.append(x)
    i = random.choice(active_el2)
    j = random.choice(inactive_el)
    # print(active_el)
    # print(inactive_el)
    # print(i, j)
    chr1[i], chr1[j] = chr1[j], chr1[i]


def selection(pop, new_pop):
    pop2 = pop.copy()
    for x in range(len(new_pop)):
        pop2.append(new_pop[x])
    print(len(pop2))
    pop2.sort(key=function, reverse=True)
    print(pop2)
    fitness_evaluation(pop2)
    pop2 = pop2[:pop_size]
    print(len(pop2))
    fitness_evaluation(pop2)
    aver_fit(fitness_evaluation(pop2))


def evolve(n, pop):
    idx = rws2(pop, fitness, 2)
    print(idx)
    for x in range(0, len(idx)):
        offspring = crossover(pop[idx[x][0]], pop[idx[x][1]])
        mutate(offspring[0])
        mutate(offspring[1])
        for y in range(len(offspring)):
            new_pop.append(offspring[y])
    print(new_pop)
    print(len(new_pop))

    # offspring = crossover(pop[idx[0]], pop[idx[1]])
    # print(offspring[1])
    # mutate(offspring[1])
    # new_pop.append(offspring[0])
    # new_pop.append(offspring[1])
    # print(new_pop)


# mutate()
# init_pop()
# print(pop[0])
# print(pop[1])
# cr1, cr2 = crossover(pop[0], pop[1])
# print(cr1)
# print(cr2)
# crossover(pop[2], pop[3])
# print(pop[0])
# print(pop[1])
# mutate(pop[0])
# print(pop[0])
# print(chr1)
# print(chr2)

# init_pop()
# print(pop)
# fitness_score(pop)
# evolve(pop_size, pop)

init_pop()
print(pop)
print(len(pop))
fitness_evaluation(pop)
fitness_score(pop)
# rws2(pop, fitness, 2)
evolve(2, pop)
fitness_evaluation(pop)
fitness_evaluation(new_pop)
aver_fit(fitness_evaluation(pop))
aver_fit(fitness_evaluation(new_pop))
selection(pop,new_pop)
