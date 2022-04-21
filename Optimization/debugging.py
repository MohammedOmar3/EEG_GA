import numpy as np
import random
import re
from pathlib import Path
import csv
import pandas as pd
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

from sklearn.model_selection import train_test_split

data_bc = pd.read_csv('datasets.csv')
label_bc = data_bc["Label"]


def split(df, label):
    X_tr, X_te, Y_tr, Y_te = train_test_split(df, label, test_size=0.25, random_state=42)
    return X_tr, X_te, Y_tr, Y_te


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


def acc_score(df, label):
    Score = pd.DataFrame({"Classifier": classifiers})
    j = 0
    acc = []
    X_train, X_test, Y_train, Y_test = split(df, label)
    for i in models:
        model = i
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        acc.append(accuracy_score(Y_test, predictions))
        j = j + 1
    Score["Accuracy"] = acc
    Score.sort_values(by="Accuracy", ascending=False, inplace=True)
    Score.reset_index(drop=True, inplace=True)
    return Score


def plot(score, x, y, c="b"):
    gen = [1, 2, 3, 4, 5]
    plt.figure(figsize=(6, 4))
    ax = sns.pointplot(x=gen, y=score, color=c)
    ax.set(xlabel="Generation", ylabel="Accuracy")
    ax.set(ylim=(x, y))


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
    return sum / len(fitness)


def fill_electrodes():
    headers = data_bc.columns.values
    electrodes = list()
    for x in range(len(headers) - 1):
        electrodes.append(int(re.sub('.*?([0-9]*)$', r'\1', headers[x])))
    electrodes.append(headers[len(headers) - 1])
    return electrodes


def col_checker(active_electrodes):
    i = 0
    electrodes = fill_electrodes()
    print(len(electrodes))
    for x in range(len(electrodes)):
        if electrodes[x] in active_electrodes:
            i = i + 1
    print(i)


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
        self.fitness = []
        self.population = []
        self.new_population = []
        self.parents = []

    def evaluation_null(self, pop):
        """
        Computes fitness of each chromosome in population
        :param pop:
        :return:
            Fitness array of each chromosome
        """
        self.fitness.clear()
        for i in range(self.pop_size):
            str1 = ''.join(str(e) for e in pop[i])
            dec_number = int(str1, 2)
            self.fitness.insert(i, dec_number)
            print(str1)
            print(dec_number)
        fit_value = self.fitness.copy()
        fit_value.sort(reverse=True)

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

    def active_electrodes(self):
        """
        This method checks what electrodes are active and returns every feature column that is active for that chromosome
        :return: Returns Columns that are active in regards to active electrode in a chromosome.
        """
        active_el = list()
        active_cols = list()
        for x in range(self.pop_size):
            tmp = list()
            for i in range(self.n):
                if self.population[x][i] == 1:
                    tmp.append(i)
            active_el.append(tmp)
        electrodes = fill_electrodes()
        for k in range(len(active_el)):
            custom = list()
            for y in active_el[k]:
                custom.extend([i for (i, z) in enumerate(electrodes) if z == y])
            active_cols.append(custom)
        print(active_el)
        return active_cols

    def fitness_score(self):
        scores = []
        self.fitness.clear()
        logmodel = RandomForestClassifier(n_estimators=200, random_state=0)
        X_train,X_test,Y_train,Y_test = split(data_bc,label_bc)
        chromosome = self.active_electrodes()
        for x in range(len(self.population)):
            logmodel.fit(X_train.iloc[:, chromosome[x]], Y_train)
            predictions = logmodel.predict(X_test.iloc[:, chromosome[x]])
            scores.append(accuracy_score(Y_test, predictions))
        self.fitness = scores
        print(self.fitness)
        # scores, population = np.array(scores), np.array(self.population)
        # inds = np.argsort(scores)
        # return list(scores[inds][::-1]), list(population[inds, :][::-1])

    def evaluation(self):
        self.fitness.clear()
        for i in range(self.pop_size):
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
        for x in range(0, int(self.pop_size / n)):
            idx = np.random.choice(np.arange(len(self.population)), size=n, replace=False, p=rel_fitness)
            pairs.append(idx)
        return pairs

    def selection(self):
        pop2 = self.population.copy()
        for x in range(len(self.new_population)):
            pop2.append(self.new_population[x])
        pop2.sort(key=function, reverse=True)
        self.population = pop2[:self.pop_size]
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
    pop_size = 4
    n = 62
    nA = 4
    p_c = 0.8
    p_m = 0.02
    g = 5

    pop = GA(pop_size=pop_size, n=n, nA=nA, p_c=p_c, p_m=p_m)
    pop.initialization()
    #active_cols = pop.active_electrodes()
    #df = data_bc.iloc[:, active_cols[0]]
    #print(df)
    # df.to_csv('example2')

    for x in range(g):
        print("Generation: ", x)
        # pop.evaluation()
        pop.fitness_score()
        pop.active_electrodes()
        pop.evolve()
        pop.selection()
        print()
    print("Final Evaluation")
    # pop.evaluation()
