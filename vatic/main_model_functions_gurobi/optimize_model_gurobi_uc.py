import time

# Set Gurobi Solving Model Status
import gurobi

mipgap = 0.01
threads = 9

model.Params.MIPGap = mipgap
model.Params.Threads = threads

# logfile = '/Users/jf3375/Desktop/Gurobi/output'
# model.Params.LogFile = logfile
# model.Params.OutputFlag = 1

# Tune the Model
# Based on tuning, the original set of parameters is fastest
# model.Params.TuneCriterion = 2
# Increase the number of tuning trials to control the randomness of MIP models
# model.Params.TuneTrials = 100
# model.Params.TuneResults = -1
# model.tune()
# for i in range(model.tuneResultCount):
#     print(i)
#     model.write('/Users/jf3375/Desktop/Gurobi/output/Tune_UnitCommitment.prm')

# Output and Read the Current Model Parameter File to set Model Parameters
# model.read('/Users/jf3375/Desktop/Gurobi/output/Tune_UnitCommitment.prm')

#Reset all parameters
# gurobi.resetParams()

# solvemodel_start_time = time.time()
# model.optimize()
# print('solvemodel_time', time.time()- solvemodel_start_time)

# Check inconsistent constraints and debug
# If this returns error, that means all constraints form the feasible set
# model.computeIIS()
# model.write('UnitCommitment.ilp')

# Save Gurobi Model Results
for v in model.getVars():
    print(v.x)