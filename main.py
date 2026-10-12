import random
import copy

offspring = []

N = 10
P = 50

MUTRATE = 1 / N

class individual:
    def __init__(self):
        self.gene = [0]*N
        self.fitness = 0

population = []

for x in range (0, P):
    tempgene=[]
    for y in range (0, N):
        tempgene.append(random.randint(0,1))
        
    newind = individual()
    newind.gene = tempgene.copy()
    population.append(newind)

def test_function( ind ):
    utility=0
    for i in range(N):
        utility = utility + ind.gene[i]
    return utility


for ind in population:
    ind.fitness = test_function(ind)


for i in range (0, P):
    parent1 = random.randint( 0, P-1 )
    off1 = copy.deepcopy(population[parent1])
    parent2 = random.randint( 0, P-1 )
    off2 = copy.deepcopy(population[parent2])
    if off1.fitness > off2.fitness:
        offspring.append( off1 )
    else:
        offspring.append( off2 )


toff1 = individual()
toff2 = individual()
temp = individual()
for i in range( 0, P, 2 ):
    toff1 = copy.deepcopy(offspring[i])
    toff2 = copy.deepcopy(offspring[i+1])
    temp = copy.deepcopy(offspring[i])
    crosspoint = random.randint(1,N)

    for j in range (crosspoint, N):
        toff1.gene[j] = toff2.gene[j]
        toff2.gene[j] = temp.gene[j]

    offspring[i] = copy.deepcopy(toff1)
    offspring[i+1] = copy.deepcopy(toff2)


for i in range( 0, P ):
    newind = individual()
    newind.gene = []
    for j in range( 0, N ):
        gene = offspring[i].gene[j]
        mutprob = random.random()
        if mutprob < MUTRATE:
            if (gene == 1):
                gene = 0
            else:
                gene = 1

        newind.gene.append(gene)
    offspring[i] = copy.deepcopy(newind)


