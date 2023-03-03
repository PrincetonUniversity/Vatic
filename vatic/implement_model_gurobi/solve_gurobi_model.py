import time

# Set Gurobi Solving Model Status
mipgap = 0.01
threads = 9
logfiel = '/Users/jf3375/Desktop/Vatic_Run/solve_logfile'
model.Params.MIPGap = mipgap
model.Params.Threads = threads
model.Params.LogFiel = logfile

solvemodel_start_time = time.time()
model.optimize()
print('solvemodel_time', time.time()- solvemodel_start_time)


if model.status == 4:
    model.Params.DualReductions = 0
    model.optimize()
    print(model.status)
    if model.status == 3:
        # drop binary constraints
        model_relax = model.relax()
        model_relax.optimize()
        print(model_relax.status)
