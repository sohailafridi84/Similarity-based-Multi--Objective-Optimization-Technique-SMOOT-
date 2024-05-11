import numpy
import random
import time
import sys

from solution import solution
import xlsxwriter



def crossoverPopulaton(population, scores, popSize, crossoverProbability, keep):
    global bestIndividual1
    # initialize a new population
    newPopulation = numpy.empty_like(population)
    newPopulation[0:keep] = population[0:keep]
    
    # Create pairs of parents. The number of pairs equals the number of individuals divided by 2
    for i in range(keep, popSize, 2):
        # pair of parents selection
        parent1, parent2 = pairSelection(population, scores, popSize)
        crossoverLength = min(len(parent1), len(parent2))
        parentsCrossoverProbability = random.uniform(0.0, 1.0)
        if parentsCrossoverProbability < crossoverProbability:
            offspring1, offspring2 = crossover(crossoverLength, parent1, parent2)
        else:
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()

        # Add offsprings to population
        newPopulation[i] = numpy.copy(offspring1)
        newPopulation[i + 1] = numpy.copy(offspring2)
    ga = newPopulation[i+1]

    bestIndividual1  = ga.copy()
    
    

    return newPopulation


def mutatePopulaton(population, popSize, mutationProbability, keep, lb, ub):
    for i in range(keep, popSize):
        # Mutation
        offspringMutationProbability = random.uniform(0.0, 1.0)
        if offspringMutationProbability < mutationProbability:
            mutation(population[i], len(population[i]), lb, ub)


def elitism(population, scores, bestIndividual, bestScore):
    # get the worst individual
    worstFitnessId = selectWorstIndividual(scores)

    # replace worst cromosome with best one from previous generation if its fitness is less than the other
    if scores[worstFitnessId] > bestScore:
        population[worstFitnessId] = numpy.copy(bestIndividual)
        scores[worstFitnessId] = numpy.copy(bestScore)


def selectWorstIndividual(scores):
    maxFitnessId = numpy.where(scores == numpy.max(scores))
    maxFitnessId = maxFitnessId[0][0]
    return maxFitnessId


def pairSelection(population, scores, popSize):
    parent1Id = rouletteWheelSelectionId(scores, popSize)
    parent1 = population[parent1Id].copy()

    parent2Id = rouletteWheelSelectionId(scores, popSize)
    parent2 = population[parent2Id].copy()

    return parent1, parent2


def rouletteWheelSelectionId(scores, popSize):
    ##reverse score because minimum value should have more chance of selection
    reverse = max(scores) + min(scores)
    reverseScores = reverse - scores.copy()
    sumScores = sum(reverseScores)
    pick = random.uniform(0, sumScores)
    current = 0
    for individualId in range(popSize):
        current += reverseScores[individualId]
        if current > pick:
            return individualId


def crossover(individualLength, parent1, parent2):
    # The point at which crossover takes place between two parents.
    crossover_point = random.randint(0, individualLength - 1)
    # The new offspring will have its first half of its genes taken from the first parent and second half of its genes taken from the second parent.
    offspring1 = numpy.concatenate(
        [parent1[0:crossover_point], parent2[crossover_point:]]
    )
    # The new offspring will have its first half of its genes taken from the second parent and second half of its genes taken from the first parent.
    offspring2 = numpy.concatenate(
        [parent2[0:crossover_point], parent1[crossover_point:]]
    )

    return offspring1, offspring2


def mutation(offspring, individualLength, lb, ub):
    mutationIndex = random.randint(0, individualLength - 1)
    mutationValue = random.uniform(lb[mutationIndex], ub[mutationIndex])
    offspring[mutationIndex] = mutationValue


def clearDups(Population, lb, ub):
    newPopulation = numpy.unique(Population, axis=0)
    oldLen = len(Population)
    newLen = len(newPopulation)
    if newLen < oldLen:
        nDuplicates = oldLen - newLen
        newPopulation = numpy.append(
            newPopulation,
            numpy.random.uniform(0, 1, (nDuplicates, len(Population[0])))
            * (numpy.array(ub) - numpy.array(lb))
            + numpy.array(lb),
            axis=0,
        )

    return newPopulation


def calculateCost(objf, population, popSize, lb, ub):
    scores = numpy.full(popSize, numpy.inf)

    # Loop through individuals in population
    for i in range(0, popSize):
        # Return back the search agents that go beyond the boundaries of the search space
        population[i] = numpy.clip(population[i], lb, ub)

        # Calculate objective function for each search agent
        scores[i] = objf(population[i, :])

    return scores


def sortPopulation(population, scores):
    sortedIndices = scores.argsort()
    population = population[sortedIndices]
    scores = scores[sortedIndices]

    return population, scores


def GA(objf, lb, ub, dim, popSize, iters):

    workbook = xlsxwriter.Workbook(objf.__name__+'_'+'GA.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    cp = 1  # crossover Probability
    mp = 0.01  # Mutation Probability
    keep = 2
    # elitism parameter: how many of the best individuals to keep from one generation to the next

    s = solution()

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    bestIndividual = numpy.zeros(dim)
    scores = numpy.random.uniform(0.0, 1.0, popSize)
    bestScore = float("inf")

    ga = numpy.zeros((popSize, dim))
    for i in range(dim):
        ga[:, i] = numpy.random.uniform(0, 1, popSize) * (ub[i] - lb[i]) + lb[i]
    convergence_curve = numpy.zeros(iters)

    print('GA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(iters):

        # crossover
        ga = crossoverPopulaton(ga, scores, popSize, cp, keep)

        # mutation
        mutatePopulaton(ga, popSize, mp, keep, lb, ub)

        ga = clearDups(ga, lb, ub)

        scores = calculateCost(objf, ga, popSize, lb, ub)

        bestScore = min(scores)

        # Sort from best to worst
        ga, scores = sortPopulation(ga, scores)

        convergence_curve[l] = bestScore

        if l % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(l + 1)
                    + " the best fitness is "
                    + str(bestScore)
                ]
            )
            bestIndividual = bestIndividual1
            for i in range(len(bestIndividual)):
                if(bestIndividual[i] > 0.5):

                    bestIndividual[i]=1
                    worksheet.write(row, col, bestIndividual[i])
                    row += 1
                else:
                    bestIndividual[i]=0
                    worksheet.write(row, col,     bestIndividual[i])
                    row += 1
            row = 0
            col +=1
    workbook.close()
            # print ("Best Individual Solution" ,bestIndividual1)


    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "GA"
    s.objfname = objf.__name__

   


    return s
