import random
import numpy as np
from solution import solution
import time
import xlsxwriter


def RS(objf, lower_boundary, upper_boundary, dimensions, PopSize, max_iter, maximize=False):

    workbook = xlsxwriter.Workbook(objf.__name__+'_'+'RS.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    if not isinstance(lower_boundary, list):
        lower_boundary = [lower_boundary] * dimensions
    if not isinstance(upper_boundary, list):
        upper_boundary = [upper_boundary] * dimensions


    #  = np.array([float()] * dimensions)
    best_solution = np.zeros((PopSize, dimensions))

    for i in range(dimensions):
        best_solution[:,i] = (np.random.uniform(0, 1, PopSize) * (upper_boundary[i] - lower_boundary[i]) + lower_boundary[i])
        # random.uniform(lower_boundary[i], upper_boundary[i])

    Convergence_curve = np.zeros(max_iter)
    s = solution()
    print('RS is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    for i in range(0,max_iter):
        for j in range(0,PopSize):
            solution1 = objf(best_solution[:,i])
            for d in range(dimensions):
                 best_solution[j,d] = np.clip(best_solution[j,d], lower_boundary[d], upper_boundary[d])
        
            
            new_solution = [lower_boundary[d] + random.random() * (upper_boundary[d] - lower_boundary[d]) for d in
                            range(dimensions)]
            # print(new_solution)
            if np.greater_equal(best_solution[j,d], lower_boundary).all() and np.less_equal(best_solution[j,d], upper_boundary).all():
                solution2 = objf(new_solution)
            elif maximize:
                solution2 = -100000.0
            else:
                solution2 = 100000.0

            if solution2 > solution1 and maximize:
                best_solution = best_solution[j,d]
            elif solution2 < solution1 and not maximize:
                best_solution = best_solution[j,d]
            
            # best_fitness = objf(best_solution[:,i])

        Convergence_curve[i] = solution2

        if i % 1 == 0:
            print(
                ["At iteration " + str(i) + " the best fitness is " + str(solution2)]
            )
            bestIndividual  = new_solution.copy()
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
                
            # s.bestIndividual = bestIndividual
    
            # print ("Best Individual Solution" ,bestIndividual)
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "RS"
    s.objfname = objf.__name__

    
    print ("Best Individual Solution" ,solution2)

    return s