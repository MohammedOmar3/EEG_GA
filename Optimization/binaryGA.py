import numpy as np
import random


def swap(l1, l2, a):
    l1[a], l2[a] = l2[a], l1[a]


def function(individual):
    str1 = ''.join(str(e) for e in individual)
    dec_number = int(str1, 2)
    return dec_number


def fit(pop):
    tmpfit = list()
    for i in range(len(pop)):
        str1 = ''.join(str(e) for e in pop[i])
        dec_number = int(str1, 2)
        tmpfit.insert(i, dec_number)
    # print(tmpfit)
    return tmpfit


def aver(fitness):
    sum = 0

    for x in range(len(fitness)):
        sum = sum + fitness[x]
    return (sum / len(fitness))


class GA():
    """
    0-1 Binary Genetic Algorithm
    """

    def __init__(self, pop_size, n, nA, p_c, p_m, ):
        """
        :param pop_size: population size
        :param n: electrodes in chromosome
        :param nA: active electrodes
        :param p_c: crossover probability
        :param p_m: mutation probability
        """
        self.pop_size = pop_size
        self.n = n
        self.nA = nA
        self.p_c = p_c
        self.p_m = p_m
        self.fitness = []
        self.population = []
        self.new_population = []
        self.parents = []

    def initialization(self):
        for y in range(0, self.pop_size):
            cr = [random.randint(0, 0) for x in range(self.n)]
            options = np.random.choice(self.n, self.nA, replace=False)
            # print(options)
            for x in options:
                cr[x] = 1
            # print(cr)
            self.population.append(cr)
        print(self.population)

    def evaluation(self):
        self.fitness.clear()
        for i in range(pop_size):
            str1 = ''.join(str(e) for e in self.population[i])
            dec_number = int(str1, 2)
            self.fitness.insert(i, dec_number)
        print(self.fitness)
        print("Average Population Fitness: ", aver(self.fitness))

    def crossover(self, chr1, chr2):
        cr1, cr2 = chr1.copy(), chr2.copy()
        allactive = []
        done = False
        for x in range(0, self.n):
            if cr1[x] != cr2[x]:
                # print("valid")
                allactive.append(x)
        if len(allactive) > 2:
            while not done:
                active = random.sample(allactive, 2)
                if cr1[active[0]] != cr1[active[1]]:
                    for y in active:
                        swap(cr1, cr2, y)
                        done = True
        return cr1, cr2

    def mutate(self, p1):
        active_el = list()
        inactive_el = list()
        for x in range(0, self.n):
            if p1[x] == 0:
                inactive_el.append(x)
            elif p1[x] == 1:
                active_el.append(x)
        i = random.choice(active_el)
        j = random.choice(inactive_el)
        p1[i], p1[j] = p1[j], p1[i]

    def rws(self, n):
        pairs = []
        total_fitness = float(sum(self.fitness))
        rel_fitness = [f / total_fitness for f in self.fitness]
        for x in range(0, int(pop_size / n)):
            idx = np.random.choice(np.arange(len(self.population)), size=n, replace=False, p=rel_fitness)
            pairs.append(idx)
        return pairs

    def selection(self):
        pop2 = self.population.copy()
        for x in range(len(self.new_population)):
            pop2.append(self.new_population[x])
        pop2.sort(key=function, reverse=True)
        self.population = pop2[:pop_size]
        self.new_population.clear()
        print(self.population)

    def evolve(self):
        idx = self.rws(2)
        for x in range(0, len(idx)):
            prob = np.random.rand()
            # print(prob)
            if np.random.rand() < self.p_c:
                offspring = self.crossover(self.population[idx[x][0]], self.population[idx[x][1]])
                # print("done crossover")
                if np.random.rand() < self.p_m:
                    self.mutate(offspring[0])
                    self.mutate(offspring[1])
                    # print("done mutate")
                for y in range(0, len(offspring)):
                    self.new_population.append(offspring[y])


if __name__ == "__main__":
    pop_size = 100
    n = 63
    nA = 4
    p_c = 0.8
    p_m = 0.02
    g = 5

    pop = GA(pop_size=pop_size, n=n, nA=nA, p_c=p_c, p_m=p_m)
    pop.initialization()
    for x in range(g):
        print("Generation: ", x)
        pop.evaluation()
        pop.evolve()
        pop.selection()
        print()
    print("Final Evaluation")
    pop.evaluation()
