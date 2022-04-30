import numpy as np
import random
import re
import sys
from pathlib import Path
import csv
import pandas as pd
from collections import OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
import pickle
from sklearn.model_selection import train_test_split

# sys.stdout = open('output.txt','wt')  # Prints out output from running code to a txt file

data_bc = pd.read_csv('train_test.csv')
label_bc = data_bc["Label"]

classifiers = ['LinearSVM', 'RadialSVM',
               'Logistic', 'RandomForest',
               'AdaBoost', 'DecisionTree',
               'KNeighbors', 'GradientBoosting']

models = [svm.SVC(kernel='linear'),
          svm.SVC(kernel='rbf'),
          LogisticRegression(max_iter=1000),
          RandomForestClassifier(n_estimators=200, random_state=0),
          AdaBoostClassifier(random_state=0),
          DecisionTreeClassifier(random_state=0),
          KNeighborsClassifier(),
          GradientBoostingClassifier(random_state=0)]


def split(df, label):
    # Splits dataset into Training and Testing
    X_tr, X_te, Y_tr, Y_te = train_test_split(df, label, test_size=0.25, random_state=42)
    return X_tr, X_te, Y_tr, Y_te


def plot(score, x, y, c="b"):
    gen = [1, 2, 3, 4, 5]
    plt.figure(figsize=(6, 4))
    ax = sns.pointplot(x=gen, y=score, color=c)
    ax.set(xlabel="Generation", ylabel="Accuracy")
    ax.set(ylim=(x, y))


def swap(l1, l2, a):
    # used for crossover, swaps bits in a binary
    l1[a], l2[a] = l2[a], l1[a]


def fill_electrodes():
    # Returns electrode index for feature in dataset, [0-61, 0-61]
    headers = data_bc.columns.values
    electrodes = list()
    for x in range(len(headers) - 1):
        electrodes.append(int(re.sub('.*?([0-9]*)$', r'\1', headers[x])))
    electrodes.append(headers[len(headers) - 1])
    return electrodes


class GA:
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
        self.fitness = []  # Population fitness
        self.new_fitness = []  # Population fitness
        self.population = []  # Population
        self.new_population = []  # New Population consisting of Offsprings

        # Variables for performance testing
        self.best_fitness = float
        self.active_nA = list()
        self.all_fitness = list()

    def initialization(self):
        """
        Method generates a population of chromosomes with N: electrodes and nA: active electrodes.
        :return: Populates self.population
        """
        for y in range(0, self.pop_size):
            cr = [random.randint(0, 0) for x in range(self.n)]
            options = np.random.choice(self.n, self.nA, replace=False)
            # print(options)
            for x in options:
                cr[x] = 1
            # print(cr)
            self.population.append(cr)
        print(self.population)

    def active_electrodes(self, population):
        """
        This method checks what electrodes are active and returns every feature column that is active for that chromosome.
        active_el = active electrodes in chromosome (index)
        active_cols = columns in dataset that have features for active electrodes(active_el)
        :return: Returns Columns in dataset that are active in regards to active electrode in a chromosome.
        """
        active_el = list()
        active_cols = list()
        for x in range(self.pop_size):
            tmp = list()
            for i in range(self.n):
                # Checks for active electrode in chromosome
                if population[x][i] == 1:
                    # Adds electrode index to temperory list which is appended to active_el
                    tmp.append(i)
            active_el.append(tmp)
        electrodes = fill_electrodes()
        for k in range(len(active_el)):
            custom = list()
            for y in active_el[k]:
                # Checks what electrodes match up with dataset feature column names.
                custom.extend([i for (i, z) in enumerate(electrodes) if z == y])
            active_cols.append(custom)
        self.active_nA = self.active_nA + active_el
        print(active_el)
        return active_cols

    def fitness_score(self, bool):
        '''
        Method to work out fitness(accuracy_score) of each chromosome in population
        :return: scores = list of fitness
        '''
        scores = []
        if bool:
            chromosome = self.active_electrodes(self.population)
        elif not bool:
            chromosome = self.active_electrodes(self.new_population)
        for x in range(len(self.population)):
            logmodel.fit(X_train.iloc[:, chromosome[x]], Y_train)
            predictions = logmodel.predict(X_test.iloc[:, chromosome[x]])
            scores.append(accuracy_score(Y_test, predictions))
        if bool:
            self.fitness = scores
        elif not bool:
            return scores

    def crossover(self, chr1, chr2):
        cr1, cr2 = chr1.copy(), chr2.copy()
        allactive = []
        done = False
        for x in range(0, self.n):
            if cr1[x] != cr2[x]:
                # print("valid")
                allactive.append(x)
        # If chromosome doesn't have more than 2 active electrodes it can swap, don't do crossover and leave it how
        # it is
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
        """
        Roulette Wheel Selection
        :param n:
        :return:
        """
        pairs = []
        total_fitness = float(sum(self.fitness))
        rel_fitness = [f / total_fitness for f in self.fitness]
        for x in range(0, int(self.pop_size / n)):
            idx = np.random.choice(np.arange(len(self.population)), size=n, replace=False, p=rel_fitness)
            pairs.append(idx)
        return pairs

    def evolve(self):
        idx = self.rws(2)
        offspring = list()
        for x in range(0, len(idx)):
            prob = np.random.rand()
            if prob < self.p_c:
                # print("Crossover has happened")
                tmp = self.crossover(self.population[idx[x][0]], self.population[idx[x][1]])
                if np.random.rand() < self.p_m:
                    self.mutate(tmp[0])
                    self.mutate(tmp[1])
                for y in range(0, len(tmp)):
                    offspring.append(tmp[y])
            elif prob > self.p_c:
                # print("crossover has not happened")
                offspring.append(self.population[idx[x][0]])
                offspring.append(self.population[idx[x][1]])
        self.new_population = offspring

    def selection(self):
        # combines both previous population and new population into a single list
        total_pop = self.population.copy()
        for x in range(len(self.new_population)):
            total_pop.append(self.new_population[x])

        # combines both previous fitness and new fitness into a single list
        previous_fitness = self.fitness.copy()
        new_fitness = self.fitness_score(bool=False)
        print(new_fitness)
        total_fitness = previous_fitness + new_fitness
        self.all_fitness = total_fitness

        zipped = zip(total_fitness, total_pop)  # Links both total_fitness and total_pop together in regards to index
        sorted2 = sorted(zipped)
        sorted_list1 = [element for _, element in sorted2]  # Sorts zipped based on fitness values of each chromosome
        # in the population
        
        # Removes Duplicates from total population
        final_list = list(k for k,_ in itertools.groupby(sorted_list1))
        sorted_list1 = final_list
        
        sorted_list1.reverse() # Arranges new population from the chromosome with highest fitness to lowest

        # Takes the top half of the combined population with the highest fitness values.
        self.population = sorted_list1[:self.pop_size]
        # Clears new_population for the next generation iteration.
        self.new_population.clear()

    def performance(self):
        """

        :return:
        """
        per_fit = self.all_fitness.copy()
        per_fit.sort(reverse=True)
        zipped = zip(self.all_fitness, self.active_nA)
        sorted2 = sorted(zipped)
        sorted_list1 = [element for _, element in sorted2]
        sorted_list1.reverse()
        average_score = sum(self.all_fitness)/len(self.all_fitness)
        print("Best Electrode Positioning:", sorted_list1[0], ", with fitness score:", per_fit[0])
        print("Average accuracy score:", average_score)
        self.active_nA.clear()


if __name__ == "__main__":
    pop_size = 100
    n = 62
    nA = 4
    p_c = 0.8
    p_m = 0.02
    g = 30

    pop = GA(pop_size=pop_size, n=n, nA=nA, p_c=p_c, p_m=p_m)
    X_train, X_test, Y_train, Y_test = split(data_bc, label_bc)
    logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
    pop.initialization()

    for x in range(g):
        print("Generation: ", x)
        pop.fitness_score(bool=True)
        print(pop.fitness)
        pop.evolve()
        pop.selection()
        pop.performance()
        print()

    # Save Model.
    filename = 'finalized_model.sav'
    pickle.dump(logmodel, open(filename, 'wb'))

    # Load Model
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    # print(result)
