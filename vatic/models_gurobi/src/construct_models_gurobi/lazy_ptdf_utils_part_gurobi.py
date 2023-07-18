"""
Helpers functions for flow verification across dcopf and models
"""

import vatic.models_gurobi.src.transmission_gurobi.branch as libbranch

from egret.common.log import logger

import copy as cp


def add_monitored_flow_tracker(mb):
    mb._idx_monitored = list()
    mb._interfaces_monitored = list()
    mb._contingencies_monitored = list()

    # add these if there are no slacks
    # so we don't have to check later
    # for these attributes
    if not hasattr(mb, "_pf_slack_pos"):
        mb._pf_slack_pos = []
    if not hasattr(mb, "_pfi_slack_pos"):
        mb._pfi_slack_pos = []
    if not hasattr(mb, "_pfc_slack_pos"):
        mb._pfc_slack_pos = []


def _iter_over_initial_set(branches, branches_in_service, PTDF):
    for bn in branches_in_service:
        branch = branches[bn]
        if "lazy" in branch and not branch["lazy"]:
            if bn in PTDF.branchname_to_index_masked_map:
                i = PTDF.branchname_to_index_masked_map[bn]
                yield i, bn
            else:
                logger.warning(
                    "Branch {0} has flag 'lazy' set to False but is excluded \
                    from monitored set based on kV limits".format(bn)
                )


def _generate_branch_thermal_bounds(mb, bn, thermal_limit):
    if bn in mb._pf_slack_pos.keys():
        if bn not in mb._pf_slack_pos:
            neg_slack = mb._pf_slack_neg[bn]
            pos_slack = mb._pf_slack_pos[bn]
            assert len(mb._pf_slack_pos) == len(mb._pf_slack_neg)
            new_var = True
        else:  # the constraint could have been added and removed
            neg_slack = mb._pf_slack_neg[bn]
            pos_slack = mb._pf_slack_pos[bn]
            new_var = False
        # initialize to 0.
        neg_slack.value = 0.0
        pos_slack.value = 0.0
    else:
        neg_slack = None
        pos_slack = None
        new_var = False

    return (
        libbranch.generate_thermal_bounds(
            mb._pf[bn], -thermal_limit, thermal_limit, neg_slack, pos_slack
        ),
        new_var,
    )


def add_initial_monitored_branches(
        mb, parent_model, branches, branches_in_service, ptdf_options, PTDF,
        period
):

    rel_ptdf_tol = ptdf_options["rel_ptdf_tol"]
    abs_ptdf_tol = ptdf_options["abs_ptdf_tol"]

    viol_in_mb = mb._idx_monitored
    for i, bn in _iter_over_initial_set(branches, branches_in_service, PTDF):
        thermal_limit = PTDF.branch_limits_array_masked[i]
        mb._pf[bn] = libbranch.get_power_flow_expr_ptdf_approx(
            mb, bn, PTDF, abs_ptdf_tol=abs_ptdf_tol, rel_ptdf_tol=rel_ptdf_tol
        )
        constr, _ = _generate_branch_thermal_bounds(mb, bn, thermal_limit)
        lb, expr, ub = constr[0], constr[1], constr[2]
        viol_in_mb.append(i)

        parent_model._ineq_pf_branch_thermal_ub[
            (bn, period)
        ] = parent_model.addConstr(
            (expr <= ub),
            name="ineq_pf_branch_thermal_upper_bounds_branch[{}]_period[{}]".format(
                bn, period
            ),
        )
        parent_model._ineq_pf_branch_thermal_lb[
            (bn, period)
        ] = parent_model.addConstr(
            (expr >= lb),
            name="ineq_pf_branch_thermal_lower_bounds_branch[{}]_period[{}]".format(
                bn, period
            ),
        )


def populate_default_ptdf_options(ptdf_options):
    if ptdf_options is None:
        ptdf_options = dict()
    else:
        # get a copy
        ptdf_options = cp.deepcopy(ptdf_options)
    if "rel_ptdf_tol" not in ptdf_options:
        ptdf_options["rel_ptdf_tol"] = 1.0e-6
    if "abs_ptdf_tol" not in ptdf_options:
        ptdf_options["abs_ptdf_tol"] = 1.0e-10
    if "abs_flow_tol" not in ptdf_options:
        ptdf_options["abs_flow_tol"] = 1.0e-3
    if "rel_flow_tol" not in ptdf_options:
        ptdf_options["rel_flow_tol"] = 1.0e-5
    if "lazy_rel_flow_tol" not in ptdf_options:
        ptdf_options["lazy_rel_flow_tol"] = -0.01
    if "iteration_limit" not in ptdf_options:
        ptdf_options["iteration_limit"] = 100000
    if "lp_iteration_limit" not in ptdf_options:
        ptdf_options["lp_iteration_limit"] = 100
    if "max_violations_per_iteration" not in ptdf_options:
        ptdf_options["max_violations_per_iteration"] = 5
    if "lazy" not in ptdf_options:
        ptdf_options["lazy"] = True
    if "branch_kv_threshold" not in ptdf_options:
        ptdf_options["branch_kv_threshold"] = None
    if "kv_threshold_type" not in ptdf_options:
        ptdf_options["kv_threshold_type"] = "one"
    if "pre_lp_iteration_limit" not in ptdf_options:
        ptdf_options["pre_lp_iteration_limit"] = 100
    if "active_flow_tol" not in ptdf_options:
        ptdf_options["active_flow_tol"] = 50.0
    if "lp_cleanup_phase" not in ptdf_options:
        ptdf_options["lp_cleanup_phase"] = True
    return ptdf_options


def check_and_scale_ptdf_options(ptdf_options, baseMVA):
    # scale to base MVA
    ptdf_options["abs_ptdf_tol"] /= baseMVA
    ptdf_options["abs_flow_tol"] /= baseMVA
    ptdf_options["active_flow_tol"] /= baseMVA

    # lowercase keyword options
    ptdf_options["kv_threshold_type"] = ptdf_options[
        "kv_threshold_type"
    ].lower()

    rel_flow_tol = ptdf_options["rel_flow_tol"]
    abs_flow_tol = ptdf_options["abs_flow_tol"]

    rel_ptdf_tol = ptdf_options["rel_ptdf_tol"]
    abs_ptdf_tol = ptdf_options["abs_ptdf_tol"]

    lazy_rel_flow_tol = ptdf_options["lazy_rel_flow_tol"]

    max_violations_per_iteration = ptdf_options["max_violations_per_iteration"]

    if max_violations_per_iteration < 1 or (
            not isinstance(max_violations_per_iteration, int)
    ):
        raise Exception(
            "max_violations_per_iteration must be an integer least 1, max_violations_per_iteration={}".format(
                max_violations_per_iteration
            )
        )

    if abs_flow_tol < lazy_rel_flow_tol:
        raise Exception(
            "abs_flow_tol (when scaled by baseMVA) cannot be less than lazy_flow_tol"
            " abs_flow_tol={0}, lazy_rel_flow_tol={1}, baseMVA={2}".format(
                abs_flow_tol * baseMVA, lazy_rel_flow_tol, baseMVA
            )
        )

    if ptdf_options["kv_threshold_type"] not in ["one", "both"]:
        raise Exception(
            "kv_threshold_type must be either 'one' (for at least one end of the line"
            " above branch_kv_threshold) or 'both' (for both end of the line above"
            " branch_kv_threshold), kv_threshold_type={}".format(
                ptdf_options["kv_threshold_type"]
            )
        )

    if abs_flow_tol < 1e-6:
        logger.warning(
            "WARNING: abs_flow_tol={0}, which is below the numeric threshold of most solvers.".format(
                abs_flow_tol * baseMVA
            )
        )
    if abs_flow_tol < rel_ptdf_tol * 10:
        logger.warning(
            "WARNING: abs_flow_tol={0}, rel_ptdf_tol={1}, which will likely result in violations. Consider raising abs_flow_tol or lowering rel_ptdf_tol.".format(
                abs_flow_tol * baseMVA, rel_ptdf_tol
            )
        )
    if rel_ptdf_tol < 1e-6:
        logger.warning(
            "WARNING: rel_ptdf_tol={0}, which is low enough it may cause numerical issues in the solver. Consider rasing rel_ptdf_tol.".format(
                rel_ptdf_tol
            )
        )
    if abs_ptdf_tol < 1e-12:
        logger.warning(
            "WARNING: abs_ptdf_tol={0}, which is low enough it may cause numerical issues in the solver. Consider rasing abs_ptdf_tol.".format(
                abs_ptdf_tol * baseMVA
            )
        )
