
import time
from ..model_data import VaticModelData
from ..models_gurobi._utils_gurobi import _save_uc_results


def solve_model(model, relaxed, mipgap, threads, outputflag) -> VaticModelData:
    model.Params.OutputFlag = outputflag
    model.Params.MIPGap = mipgap
    model.Params.Threads = threads

    solvemodel_start_time = time.time()
    model.optimize()
    solve_time = time.time() - solvemodel_start_time

    md = _save_uc_results(model, relaxed)
    md.data['system']['solver_runtime'] = solve_time

    return VaticModelData(md.data)
