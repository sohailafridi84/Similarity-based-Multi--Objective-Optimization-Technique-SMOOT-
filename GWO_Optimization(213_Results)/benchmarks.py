import numpy
import math
from math import*
from decimal import Decimal
import os.path
from csv_parser import CSVParser


# define the function blocks
# def get_test_case(i):
data_set_name = 'dataset/faultmatrix.txt'
pwd = os.path.abspath(os.path.dirname(__file__))
data_set_path = os.path.join(pwd, data_set_name)
parser = CSVParser(data_set_path)
test_case = parser.parse_data(True)

    


def Fun1(x):
    score = 0
    x_temp  = x.copy()
    for i in range(len(x_temp)):
        if(x_temp[i] > 0.5):
            x_temp[i]=1
        else:
            x_temp[i]=0
    
    for i in range(len(x)):
        for j in range(i):
            score += SimilarityFunction(i,j) * x_temp[i]   
    return score

def SimilarityFunction(x1,x2):
    testcase1 = test_case[x1]
    testcase2 = test_case[x2]
    return euclidean_distance(testcase1[1],testcase2[1])

def Fun2(x):
    score = 0
    x_temp  = x.copy()
    for i in range(len(x_temp)):
        if(x_temp[i] > 0.5):
            x_temp[i]=1
        else:
            x_temp[i]=0
    for i in range(len(x)):
        for j in range(i):
            score += SimilarityFunction2(i,j) * x_temp[i]
    return score

def SimilarityFunction2(x1,x2):
    testcase1 = test_case[x1]
    testcase2 = test_case[x2]
    return manhattan_distance(testcase1[1],testcase2[1])

def Fun3(x):
    score = 0
    x_temp  = x.copy()
    for i in range(len(x_temp)):
        if(x_temp[i] > 0.5):
            x_temp[i]=1
        else:
            x_temp[i]=0
    for i in range(len(x)):
        for j in range(i):
            score += SimilarityFunction3(i,j) * x_temp[i]
    return score

def SimilarityFunction3(x1,x2):
    testcase1 = test_case[x1]
    testcase2 = test_case[x2]
    p = 3
    return minkowski_distance(testcase1[1],testcase2[1],p)

def Fun4(x):
    score = 0
    x_temp  = x.copy()
    for i in range(len(x_temp)):
        if(x_temp[i] > 0.5):
            x_temp[i]=1
        else:
            x_temp[i]=0
    for i in range(len(x)):
        for j in range(i):
            score += SimilarityFunction4(i,j) * x_temp[i]
    return score

def SimilarityFunction4(x1,x2):
    testcase1 = test_case[x1]
    testcase2 = test_case[x2]
    return cosine_similarity(testcase1[1],testcase2[1])

def Fun5(x):
    score = 0
    x_temp  = x.copy()
    for i in range(len(x_temp)):
        if(x_temp[i] > 0.5):
            x_temp[i]=1
        else:
            x_temp[i]=0
    for i in range(len(x)):
        for j in range(i):
            score += SimilarityFunction5(i,j) * x_temp[i]
    return score

def SimilarityFunction5(x1,x2):
    testcase1 = test_case[x1]
    testcase2 = test_case[x2]
    return jaccard_similarity(testcase1[1],testcase2[1])



def euclidean_distance(x1,x2):
    
    return sqrt(sum(pow(a-b,2) for a, b in zip(x1, x2)))

def manhattan_distance(x1,x2):

    return sum(abs(a-b) for a,b in zip(x1,x2))

def minkowski_distance(x1,x2,p_value):

    return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x1, x2)),p_value)

def nth_root(value, n_root):

    root_value = 1/float(n_root)
    return round ((value) ** (root_value),3)

def cosine_similarity(x1,x2):

    numerator = sum(a*b for a,b in zip(x1,x2))
    denominator = square_rooted(x1)*square_rooted(x2)
    return round(numerator/float(denominator),3)

def square_rooted(x1):

    return round(sqrt(sum([a*a for a in x1])),3)

def jaccard_similarity(x1,x2):


    intersection_cardinality = len(set.intersection(*[set(x1), set(x2)]))
    union_cardinality = len(set.union(*[set(x1), set(x2)]))
    return intersection_cardinality/float(union_cardinality)



def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "Fun1": ["Fun1", 0, 1, 4209],
        "Fun2": ["Fun2", 0, 1, 4209],
        "Fun3": ["Fun3", 0, 1, 4209],
        "Fun4": ["Fun4", 0, 1, 4209],
        "Fun5": ["Fun5", 0, 1, 4209],
        
           
    }
    return param.get(a, "nothing")

