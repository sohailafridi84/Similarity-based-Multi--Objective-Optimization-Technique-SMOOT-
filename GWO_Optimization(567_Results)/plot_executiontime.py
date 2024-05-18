import pandas as pd
from functools import partial
import timeit
import numpy as np
import matplotlib.pyplot as plt


def run( optimizer, objectivefunc):
    plt.ioff()
    # fileResultsData = pd.read_csv(results_directory + "/experiment.csv")

    for j in range(0, len(objectivefunc)):
        objective_name = objectivefunc[j]
        for i in range(len(optimizer)):
            optimizer_name = optimizer[i]
            execution_time = timeit.timeit(optimizer_name, )
            plt.plot(execution_time, label=optimizer_name)
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.02))
        plt.grid()
        fig_name =  "/convergence-" + objective_name + ".png"
        plt.savefig(fig_name, bbox_inches="tight")
        plt.clf()
        # plt.show()