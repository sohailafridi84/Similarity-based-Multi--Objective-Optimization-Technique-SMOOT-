# -*- coding: utf-8 -*-

import math
import numpy
import random
import time
from solution import solution
import xlsxwriter



def BAT(objf, lb, ub, dim, N, Max_iteration):


    workbook = xlsxwriter.Workbook(objf.__name__+'_'+'BAT.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    n = N
    # Population size

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    N_gen = Max_iteration  # Number of generations

    A = 0.5
    # Loudness  (constant or decreasing)
    r = 0.5
    # Pulse rate (constant or decreasing)

    Qmin = 0  # Frequency minimum
    Qmax = 2  # Frequency maximum

    d = dim  # Number of dimensions

    # Initializing arrays
    Q = numpy.zeros(n)  # Frequency
    v = numpy.zeros((n, d))  # Velocities
    Convergence_curve = []

    # Initialize the population/solutions
    Sol = numpy.zeros((n, d))
    for i in range(dim):
        Sol[:, i] = numpy.random.rand(n) * (ub[i] - lb[i]) + lb[i]

    S = numpy.zeros((n, d))
    S = numpy.copy(Sol)
    Fitness = numpy.zeros(n)

    # initialize solution for the final results
    s = solution()
    print('BAT is optimizing  "' + objf.__name__ + '"')

    # Initialize timer for the experiment
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Evaluate initial random solutions
    for i in range(0, n):
        Fitness[i] = objf(Sol[i, :])

    # Find the initial best solution and minimum fitness
    I = numpy.argmin(Fitness)
    best = Sol[I, :]
    fmin = min(Fitness)

    # Main loop
    for t in range(0, N_gen):

        # Loop over all bats(solutions)
        for i in range(0, n):
            Q[i] = Qmin + (Qmin - Qmax) * random.random()
            v[i, :] = v[i, :] + (Sol[i, :] - best) * Q[i]
            S[i, :] = Sol[i, :] + v[i, :]

            # Check boundaries
            for j in range(d):
                Sol[i, j] = numpy.clip(Sol[i, j], lb[j], ub[j])

            # Pulse rate
            if random.random() > r:
                S[i, :] = best + 0.001 * numpy.random.randn(d)

            # Evaluate new solutions
            Fnew = objf(S[i, :])

            # Update if the solution improves
            if (Fnew <= Fitness[i]) and (random.random() < A):
                Sol[i, :] = numpy.copy(S[i, :])
                Fitness[i] = Fnew

            # Update the current best solution
            if Fnew <= fmin:
                best = numpy.copy(S[i, :])
                fmin = Fnew

        # update convergence curve
        Convergence_curve.append(fmin)

        if t % 1 == 0:
            print(["At iteration " + str(t) + " the best fitness is " + str(fmin)])

        bestIndividual  = best.copy()
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
        # for i in range(len(bestIndividual)):
        #     if(bestIndividual[i] > 0.5):
        #         bestIndividual[i]=1
        #     else:
        #         bestIndividual[i]=0
            
        # s.bestIndividual = bestIndividual
        # print ("Best Individual Solution" ,bestIndividual)

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "BAT"
    s.objfname = objf.__name__
   

    return s
