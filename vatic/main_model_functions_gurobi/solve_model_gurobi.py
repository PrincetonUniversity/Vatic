import time

from ..model_data import VaticModelData

from vatic.models_gurobi import _save_uc_results


def solve_model(model, relaxed, mipgap, threads, outputflag) -> VaticModelData:
    model.Params.OutputFlag = outputflag
    model.Params.MIPGap = mipgap
    model.Params.Threads = threads

    starttime = time.time()
    model.optimize()
    md = _save_uc_results(model, relaxed)
    model_solve_time = time.time()-starttime
    md._data['system']['solver_runtime'] = model_solve_time
    return VaticModelData(md._data)



