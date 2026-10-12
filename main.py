import random
import copy
import matplotlib.pyplot as plt

offspring = []

N = 10
P = 50
MUTRATE = 1 / N
GENERATIONS = 50

class individual:
    def __init__(self):
        self.gene = [0] * N
        self.fitness = 0

population = []

for x in range(0, P):
    tempgene = []
    for y in range(0, N):
        tempgene.append(random.randint(0, 1))

    newind = individual()
    newind.gene = tempgene.copy()
    population.append(newind)

def test_function(ind):
    utility = 0
    for i in range(N):
        utility = utility + ind.gene[i]
    return utility


# Store fitness results for the graph
average_fitness = []
best_fitness = []

for generation in range(GENERATIONS):
    offspring = []

    for ind in population:
        ind.fitness = test_function(ind)

    # Record fitness
    average_fitness.append(
        sum(ind.fitness for ind in population) / P
    )
    best_fitness.append(
        max(ind.fitness for ind in population)
    )

    # Selection
    for i in range(0, P):
        parent1 = random.randint(0, P-1)
        off1 = copy.deepcopy(population[parent1])
        parent2 = random.randint(0, P-1)
        off2 = copy.deepcopy(population[parent2])

        if off1.fitness > off2.fitness:
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
    for i in range(0, P):
        newind = individual()
        newind.gene = []

        for j in range(0, N):
            gene = offspring[i].gene[j]
            mutprob = random.random()

            if mutprob < MUTRATE:
                if gene == 1:
                    gene = 0
                else:
                    gene = 1

            newind.gene.append(gene)

        offspring[i] = copy.deepcopy(newind)

    # Update population for the next generation
    population = copy.deepcopy(offspring)


# Plot the fitness results
plt.plot(average_fitness, label="Average fitness")
plt.plot(best_fitness, label="Best fitness")

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness over Generations")
plt.legend()
plt.grid(True)
plt.show()