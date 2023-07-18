"""
Solve unit commitment or economic dispatch models
"""

import time
from vatic.model_data import VaticModelData
from vatic.models_gurobi import _save_uc_results

import gurobipy as gp


def solve_model(
        model: gp.Model,
        relaxed: bool,
        mipgap: float,
        threads: int,
        outputflag: int,
) -> VaticModelData:
    """
    Solve the gurobi model of unit commitment or economic dispatch
    and save results in VaticModelData

    Parameters
    ----------
    model: gurobipy.Model
        Gurobi model of unitCommitment or economic dispatch
    relaxed: bool
        True to relax all binary variables in the model
    mipgap: float
        The relative gap used to terminate the solver
    threads: int
        The number of threads used for parallelization
    outputflag: int
        1 to print the solver output in the console

    Returns
    -------
    VaticModelData
    """

    # Set parameters of gurobi model
    model.Params.OutputFlag = outputflag
    model.Params.MIPGap = mipgap
    model.Params.Threads = threads

    starttime = time.time()
    # Optimize gurobi models
    model.optimize()
    # Save the solving results at model data
    md = _save_uc_results(model, relaxed)
    model_solve_time = time.time() - starttime
    md._data["system"]["solver_runtime"] = model_solve_time
    return VaticModelData(md._data)
