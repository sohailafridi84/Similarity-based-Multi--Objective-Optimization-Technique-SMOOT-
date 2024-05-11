from optimizer import run

# "PSO","GA","BAT","FFA","GWO","CS","DE" ,"RS"
optimizer = ["PSO","GA","BAT","FFA","GWO","CS","DE" ,"RS"]

# "Fun1","Fun2","Fun3","Fun4","Fun5"
objectivefunc = ["Fun1"]

# Select number of repetitions for each experiment.
NumOfRuns = 1

# Select general parameters for all optimizers (population size, number of iterations) ....
params = {"PopulationSize": 20, "Iterations": 30}

export_flags = {
    "Export_avg": True,
    "Export_details": True,
    "Export_convergence": True,
    "Export_boxplot": True,
    "Export_executiontime": False,
}


run(optimizer, objectivefunc, NumOfRuns, params, export_flags)
