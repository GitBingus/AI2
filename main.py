import random
import copy
import statistics
import matplotlib.pyplot as plt # type: ignore -- matplotlib is installed and works, but mypy doesn't know that

import math

"""
main.py
This script implements a genetic algorithm to optimize a function using a population of individuals
"""

offspring = []

N = 20
P = 50
MUTRATE = 0.1
GENERATIONS = 500

MIN = -5.12
MAX = 5.12
MUTSTEP = 0.1

""" 
### TEST MINIMUM AND MAXIMUM OF POPULATION SWEEP

    MUTSTEP / MUTRATE       Low (0.025)     Medium (0.05)       High (0.1)
    LOW (0.025)             70.17           63.55               66.50
    MED (0.05)              70.25           62.13               59.82
    HIGH (0.1)              60.94           57.36               56.55 -- WINNER --

###
"""

def maxGenes(x):
    N = len(x)
    return 10 * N + math.fsum(
        gene**2 - 10 * math.cos(2 * math.pi * gene)
        for gene in x
    )

class individual:
    def __init__(self):
        self.gene = [0] * N
        self.fitness = 0

population = []

for x in range(0, P):
    tempgene = []
    for y in range(0, N):
        tempgene.append(random.uniform(MIN, MAX))

    newind = individual()
    newind.gene = tempgene.copy()
    population.append(newind)

def test_function(ind):
    return maxGenes(ind.gene)


# Store fitness results for the graph
average_fitness = []
best_fitness = []

for generation in range(GENERATIONS):
    offspring = []

    for ind in population:
        ind.fitness = test_function(ind)

    average_fitness.append(
        sum(ind.fitness for ind in population) / P
    )
    best_fitness.append(
        min(ind.fitness for ind in population)
    )

    # Selection
    for i in range(0, P):
        parent1 = random.randint(0, P-1)
        off1 = copy.deepcopy(population[parent1])
        parent2 = random.randint(0, P-1)
        off2 = copy.deepcopy(population[parent2])

        if off1.fitness < off2.fitness:
            offspring.append(off1)
        else:
            offspring.append(off2)

    # Crossover
    toff1 = individual()
    toff2 = individual()
    temp = individual()

    for i in range(0, P, 2):
        toff1 = copy.deepcopy(offspring[i])
        toff2 = copy.deepcopy(offspring[i+1])
        temp = copy.deepcopy(offspring[i])

        crosspoint = random.randint(1, N)

        for j in range(crosspoint, N):
            toff1.gene[j] = toff2.gene[j]
            toff2.gene[j] = temp.gene[j]

        offspring[i] = copy.deepcopy(toff1)
        offspring[i+1] = copy.deepcopy(toff2)

    # Mutation
    for i in range( 0, P ):
        newind = individual();
        newind.gene = []

        for j in range( 0, N ):
            gene = offspring[i].gene[j]
            mutprob = random.random()

            if mutprob < MUTRATE:
                alter = random.uniform(-MUTSTEP,MUTSTEP)
                gene = gene + alter
                if gene > MAX:
                    gene = MAX
                if gene < MIN:
                    gene = MIN

            newind.gene.append(gene)
        offspring[i] = copy.deepcopy(newind)

    # Update population for the next generation
    population = copy.deepcopy(offspring)

    # print(f"Generation {generation + 1}: Average Fitness = {average_fitness[-1]}, Best Fitness = {best_fitness[-1]}")

runMean = (f'{statistics.mean(best_fitness):.2f}')

if __name__ == "__main__":
    # Plot the fitness results
    plt.plot(average_fitness, label="Average fitness")
    plt.plot(best_fitness, label="Best fitness")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over Generations")
    plt.legend()
    plt.grid(True)
    plt.show()