"""Representations of grid states used as optimization self.model input/output."""

from __future__ import annotations

import time
import math
import pandas as pd
from abc import ABC, abstractmethod
from itertools import product
from copy import copy, deepcopy
from typing import Optional, Any

import gurobipy as gp
from gurobipy import GRB, tupledict, LinExpr, quicksum
from ordered_set import OrderedSet
from .ptdf_utils import (VirtualPTDFMatrix, BasePointType,
                         _MaximalViolationsStore, _LazyViolations,
                         add_violations)


def _much_less_than(v1, v2):
    return v1 < v2 and not math.isclose(v1, v2)


def constr_genr(constr):
    return (constr[k] for k in constr)


def _step_coeff(upper, lower, susd):
    if lower < susd < upper:
        return upper - susd

    elif susd <= lower:
        return upper - lower
    elif upper <= susd:
        return 0

    print("Something went wrong, step_coeff is returning None...")
    print("lower: ", lower)
    print("susd: ", susd)
    print("upper: ", upper)

    return None


class VaticModelError(Exception):
    pass


class BaseModel(ABC):

    model_name = "VaticModel"

    DEFAULT_PTDF_OPTIONS = {'rel_ptdf_tol': 1.e-6,
                            'abs_ptdf_tol': 1.e-10,
                            'abs_flow_tol': 1.e-3,
                            'rel_flow_tol': 1.e-5,
                            'lazy_rel_flow_tol': -0.01,
                            'iteration_limit': 100000,
                            'lp_iteration_limit': 100,
                            'max_violations_per_iteration': 5,
                            'lazy': True,
                            'branch_kv_threshold': None,
                            'kv_threshold_type': 'one',
                            'pre_lp_iteration_limit': 100,
                            'active_flow_tol': 50.,
                            'lp_cleanup_phase': True}

    def __init__(self,
                 grid_template: dict, renews_data: pd.DataFrame,
                 load_data: pd.DataFrame, reserve_factor: float,
                 sim_state: Optional[Any] = None,
                 future_status: Optional[dict[str, int]] = None) -> None:

        if not (renews_data.index == load_data.index).all():
            raise ValueError("Renewable generator outputs and load demands "
                             "must come from the same times!")

        self.RenewOutput = renews_data.transpose()
        self.Demand = load_data.transpose()
        self.time_steps = renews_data.index.to_list()
        self.future_status = future_status
        self.verbose = 0

        self.gen_costs = {
            gen: (grid_template['CostPiecewiseValues'][gen][-1]
                  / grid_template['CostPiecewisePoints'][gen][-1])
            for gen in grid_template['CostPiecewiseValues']
            }

        self.PTDFoptions = dict()
        self.PTDF = None
        self.BaseMVA = 1.
        self.BaseKV = 1e3
        self.ReferenceBusAngle = 0.

        self.InitialTime = 1
        self.mins_per_step = 60
        self.NumTimePeriods = self.RenewOutput.shape[1]
        self.TimePeriods = list(range(self.InitialTime,
                                      self.NumTimePeriods + 1))

        self.ThermalGenerators = OrderedSet(
            grid_template['ThermalGenerators'])
        self.RenewableGenerators = OrderedSet(
            grid_template['NondispatchableGenerators'])

        # easier to do this here than in rajan_takriti due to sim_state
        self.FixedCommitment = tupledict({
            (g, t): (None if sim_state is None
                     else sim_state.get_commitments().loc[g, t])
            for g, t in product(*self.thermal_periods)
            })

        self.Demand.columns = self.TimePeriods
        self.RenewOutput.columns = self.TimePeriods

        # set aside a proportion of the total demand as the self.model's reserve
        # requirement (if any) at each time point
        self.ReserveReqs = reserve_factor * self.Demand.sum(axis=0)

        self.Buses = OrderedSet(grid_template['Buses'])
        self.TransmissionLines = grid_template['TransmissionLines']
        self.LineOutOfService = {
            line: False for line in self.TransmissionLines}

        self.branches = {
            line: {'from_bus': grid_template['BusFrom'][line],
                   'to_bus': grid_template['BusTo'][line],

                   'reactance': grid_template['Impedence'][line],
                   'rating_long_term': grid_template['ThermalLimit'][line],
                   'rating_short_term': grid_template['ThermalLimit'][line],
                   'rating_emergency': grid_template['ThermalLimit'][line],

                   'in_service': True, 'branch_type': 'line',
                   'angle_diff_min': -90, 'angle_diff_max': 90,
                   }

            for line in grid_template['TransmissionLines']
            }

        self.buses = {bus: {'base_kv': 1e3} for bus in grid_template['Buses']}
        self.ReferenceBus = self.Buses[0]
        self.ThermalGeneratorsAtBus = grid_template['ThermalGeneratorsAtBus']
        self.NondispatchableGeneratorsAtBus = grid_template[
            'NondispatchableGeneratorsAtBus']

        self.MinPowerOutput = {(g, t): grid_template['MinimumPowerOutput'][g]
                               for g, t in product(*self.thermal_periods)}
        self.MaxPowerOutput = {(g, t): grid_template['MaximumPowerOutput'][g]
                               for g, t in product(*self.thermal_periods)}

        for g, t in product(*self.renew_periods):
            if g in self.RenewOutput.index:
                self.MaxPowerOutput[g, t] = self.RenewOutput.loc[g, t]
            else:
                self.MaxPowerOutput[g, t] = 0.

            if g in grid_template['NondispatchRenewables']:
                self.MinPowerOutput[g, t] = copy(self.MaxPowerOutput[g, t])
            else:
                self.MinPowerOutput[g, t] = 0.

        self.ScaledMinUpTime = tupledict({
            g: min(grid_template['MinimumUpTime'][g], self.NumTimePeriods)
            for g in self.ThermalGenerators
            })
        self.ScaledMinDownTime = tupledict({
            g: min(grid_template['MinimumDownTime'][g], self.NumTimePeriods)
            for g in self.ThermalGenerators
            })

        self.NominalRampUpLimit = grid_template['NominalRampUpLimit']
        self.NominalRampDownLimit = grid_template['NominalRampDownLimit']
        self.StartupRampLimit = grid_template['StartupRampLimit']
        self.ShutdownRampLimit = grid_template['ShutdownRampLimit']

        if sim_state:
            self.UnitOnT0State, self.PowerGeneratedT0 = deepcopy(
                sim_state.init_states)

        else:
            self.UnitOnT0State = grid_template['UnitOnT0State']
            self.PowerGeneratedT0 = grid_template['PowerGeneratedT0']

        self.UnitOnT0 = tupledict({
            g: init_st > 0. for g, init_st in self.UnitOnT0State.items()})

        (self.PiecewiseGenerationPoints,
            self.PiecewiseGenerationCosts,
            self.MinProductionCost) = self._piecewise_adjustment_helper(
                    grid_template['CostPiecewisePoints'],
                    grid_template['CostPiecewiseValues']
                    )

        self.prod_indices = [
            (g, t, i) for g in self.ThermalGenerators for t in self.TimePeriods
            for i in range(len(self.PiecewiseGenerationPoints[g]) - 1)
            ]

        self.InitTimePeriodsOnline = tupledict({
            g: int(min(self.NumTimePeriods,
                       round(max(0,
                                 grid_template['MinimumUpTime'][g]
                                 - self.UnitOnT0State[g]))))
            if unit_on else 0 for g, unit_on in self.UnitOnT0.items()
            })

        self.InitTimePeriodsOffline = tupledict({
            g: int(min(self.NumTimePeriods,
                       round(max(0,
                                 grid_template['MinimumDownTime'][g]
                                 + self.UnitOnT0State[g]))))
            if not unit_on else 0 for g, unit_on in self.UnitOnT0.items()
            })

        self.StartupLags = grid_template['StartupLags']
        self.StartupCosts = grid_template['StartupCosts']
        self.StartupCostIndices = {g: range(len(lags))
                                   for g, lags in self.StartupLags.items()}

        # empty until we find good storage data
        self.Storage = list()
        self.StorageAtBus = {b: list() for b in self.Buses}

        self.StageSet = ['Stage_1', 'Stage_2']
        self.solve_time = None
        self.results = None

        self.model = None

    def get_commitments(self, gen: str) -> list[bool]:
        return [self.FixedCommitment[gen, t] for t in self.TimePeriods]

    def _initialize_model(self, ptdf, ptdf_options) -> None:
        self.model = gp.Model(self.model_name)
        self.model._ptdf_options = self.DEFAULT_PTDF_OPTIONS

        self.model._CommitmentTimeInStage = {
            'Stage_1': self.TimePeriods, 'Stage_2': list()}
        self.model._GenerationTimeInStage = {
            'Stage_1': list(), 'Stage_2': self.TimePeriods}

        self.model._ptdf_options = ptdf_options
        if ptdf:
            self.PTDF = ptdf

        self.model._StartupCurve = {g: tuple() for g in self.ThermalGenerators}
        self.model._ShutdownCurve = {g: tuple() for g in self.ThermalGenerators}
        self.model._LoadMismatchPenalty = 1e4
        self.model._ReserveShortfallPenalty = 1e3

        self.model._ThermalLimits = {br: bdict['rating_long_term']
                                     for br, bdict in self.branches.items()}

    @property
    def binary_variables(self):
        return [self.model._UnitOn,
                self.model._UnitStart, self.model._UnitStop]

    @property
    def thermal_periods(self):
        return self.ThermalGenerators, self.TimePeriods

    @property
    def renew_periods(self):
        return self.RenewableGenerators, self.TimePeriods

    @property
    def gen_periods(self):
        return (self.ThermalGenerators | self.RenewableGenerators,
                self.TimePeriods)

    @property
    def startup_costs_indices(self):
        return [(g, s, t) for g in self.ThermalGenerators
                for s in self.StartupCostIndices[g] for t in self.TimePeriods]

    def relax_binaries(self) -> None:
        for varray in self.binary_variables:
            for var in varray.values():
                var.setAttr('vtype', 'C')

        self.model.update()

    def enforce_binaries(self) -> None:
        for varray in self.binary_variables:
            for var in varray.values():
                var.setAttr('vtype', 'B')

        self.model.update()

    @abstractmethod
    def generate(self,
                 relax_binaries: bool, ptdf_options,
                 ptdf, objective_hours: int) -> None:
        pass

    def _lazy_ptdf_uc_solve_loop(self, iters=100) -> None:
        duals = False

        for i in range(iters):
            flows = {}
            viol_num = {}
            mon_viol_num = {}
            viol_lazy = {}

            # check_violations
            for t, mb in self.model._TransmissionBlock.items():
                pfv, pfv_i, va = self.PTDF.calculate_PFV(mb, masked=True)

                viols_store = _MaximalViolationsStore(
                    max_viol_add=5, baseMVA=self.BaseMVA, time=t,
                    prepend_str='[MIP phase] '
                    )

                if self.PTDF.branches_keys_masked:
                    viols_store.check_and_add_violations(
                        'branch', pfv, self.PTDF.lazy_branch_limits,
                        self.PTDF.enforced_branch_limits,
                        -self.PTDF.lazy_branch_limits,
                        -self.PTDF.enforced_branch_limits, mb['idx_monitored'],
                        self.PTDF.branches_keys_masked
                        )

                viol_num[t] = viols_store.total_violations
                mon_viol_num[t] = viols_store.monitored_violations
                flows[t] = {'PFV': pfv, 'PFV_I': pfv_i}

                viol_lazy[t] = _LazyViolations(
                    branch_lazy_violations=set(
                        viols_store.get_violations_named('branch')),
                    interface_lazy_violations=set(
                        viols_store.get_violations_named('interface')),
                    contingency_lazy_violations=set(
                        viols_store.get_violations_named('contingency'))
                    )

            total_viol_num = sum(viol_num.values())
            total_mon_viol_num = sum(mon_viol_num.values())

            # this flag is for if we found violations **and** every
            # violation is in the model
            all_viol_in_model = total_viol_num > 0
            all_viol_in_model &= total_viol_num == total_mon_viol_num

            # this flag is for if we're going to terminate this iteration,
            # either because there are no violations in this solution
            # **or** because every violation is already in the model
            terminate_this_iter = (total_viol_num == 0) or all_viol_in_model

            iter_status_str = f"[MIP phase] iteration {i},"\
                              f" found {total_viol_num} violation(s)"

            if total_mon_viol_num:
                iter_status_str += f", {total_mon_viol_num} of "\
                                   "which are already monitored"

            if self.verbose:
                print(iter_status_str)

            if terminate_this_iter:
                if duals:
                    solver.add_duals()

                break

            # lazy_ptdf_violation_adder
            total_flow_constr_added = 0
            for t in self.TimePeriods:
                self.model = add_violations(self.model, t, viol_lazy[t],
                                            self.ptdf_options, self.PTDF)

                total_flow_constr_added += len(viol_lazy[t])

            if self.verbose:
                print(f"[MIP phase] iteration {i}, added "
                      f"{total_flow_constr_added} flow constraint(s)")

            self.model.update()
            self.model.optimize()

    def solve(self, relaxed, mipgap, threads, outputflag):
        self.model.Params.OutputFlag = outputflag
        self.model.Params.MIPGap = mipgap
        self.model.Params.Threads = threads

        start_time = time.time()

        self.relax_binaries()
        self.model.optimize()
        self._lazy_ptdf_uc_solve_loop()

        self.enforce_binaries()
        self.model.optimize()
        self._lazy_ptdf_uc_solve_loop()

        self.solve_time = time.time() - start_time
        self.results = self._parse_model_results()

    def _parse_model_results(self):
        if not self.model:
            raise ValueError("Model must be solved before its results "
                             "can be parsed!")

        rampup_avail_constrs = {'_EnforceMaxAvailableT0RampUpRates',
                                '_EnforceMaxAvailableTkRampUpRates',
                                '_AncillaryServiceRampUpLimit',
                                '_power_limit_from_start',
                                '_power_limit_from_stop',
                                '_power_limit_from_startstop',
                                '_power_limit_from_startstops',
                                '_max_power_limit_from_starts',
                                '_EnforceScaledT0NominalRampDownLimits',
                                '_EnforceScaledTkNominalRampDownLimits',
                                '_EnforceMaxCapacity',
                                '_OAVUpperBound',
                                '_EnforceGenerationLimits'}

        cur_rampup_constrs = [getattr(self.model, constr)
                              for constr in rampup_avail_constrs
                              if hasattr(self.model, constr)]

        results = {
            'power_generated': {
                (g, t): self.model._PowerGenerated[g, t].getValue()
                for g, t in product(*self.thermal_periods)
                },

            'commitment': {
                (g, t): self.model._UnitOn[g, t].x
                for g, t in product(*self.thermal_periods)
                },

            'commitment_cost': {
                (g, t): (self.model._NoLoadCost[g, t].getValue()
                         + self.model._StartupCost[g, t].x)
                for g, t in product(*self.thermal_periods)
                },

            'production_cost': {
                (g, t): self.model._ProductionCost[g, t].x
                for g, t in product(*self.thermal_periods)
                },

            'headroom': {(g, t): 0.
                         for g, t in product(*self.thermal_periods)},

            'no_load_cost': {(g, t): self.model._NoLoadCost[g, t].getValue()
                             for g, t in product(*self.thermal_periods)},
            'startup_cost': {(g, t): self.model._StartupCost[g, t].x
                             for g, t in product(*self.thermal_periods)},

            'load_shedding': {(b, t): (self.model._LoadShedding[b, t].x
                                       if (b, t) in self.model._LoadShedding
                                       else 0.)
                              for b in self.Buses for t in self.TimePeriods},

            'over_generation': {
                (b, t): (self.model._OverGeneration[b, t].x
                         if (b, t) in self.model._OverGeneration else 0.)
                for b in self.Buses for t in self.TimePeriods
                },

            'reserve_shortfall': {t: self.model._ReserveShortfall[t].x
                                  for t in self.TimePeriods},

            }

        for g, t in product(*self.thermal_periods):
            slack_list = [constr[g, t].slack for constr in cur_rampup_constrs
                          if (g, t) in constr]

            if slack_list:
                results['headroom'][g, t] = min(slack_list)

        for g, t in product(*self.renew_periods):
            results['power_generated'][g, t] = self.model.\
                _RenewablePowerUsed[g, t].x

        flows = dict()
        voltage_angles = dict()
        p_balances = dict()
        pl_dict = dict()

        for t in self.TimePeriods:
            cur_block = self.model._TransmissionBlock[t]
            cur_ptdf = cur_block['PTDF']
            pfv, pfv_i, va = cur_ptdf.calculate_PFV(cur_block)

            for i, br in enumerate(cur_ptdf.branches_keys):
                flows[t, br] = pfv[i]

            for i, bs in enumerate(cur_ptdf.buses_keys):
                voltage_angles[t, bs] = va[i]

                if (bs, t) in self.model._LoadGenerateMismatch:
                    p_balances[bs, t] \
                        = self.model._LoadGenerateMismatch[bs, t]
                else:
                    p_balances[bs, t] = 0.

                pl_dict[bs, t] = self.model._TransmissionBlock[t]['pl'][bs]

        results['flows'] = flows
        results['voltage_angles'] = voltage_angles
        results['p_balances'] = p_balances
        results['pl'] = pl_dict

        results['reserve_shortfall'] = {t: self.model._ReserveShortfall[t].x
                                        for t in self.TimePeriods}
        results['total_cost'] = self.model.ObjVal

        return results

    @property
    def flows(self):
        if not self.results:
            raise VaticModelError(
                "Cannot retrieve transmission line flows "
                "until self.model has been generated and solved!"
                )

        return pd.Series(self.results['flows']).unstack()

    def is_generator_on(self, gen: str) -> bool:
        return self.FixedCommitment[gen, self.InitialTime]

    def was_generator_on(self, gen: str) -> bool:
        return self.UnitOnT0[gen]

    @property
    def system_price(self):
        tot_demand = self.Demand[self.InitialTime].sum()

        if tot_demand > 0:
            tot_costs = sum(
                v for (g, t), v in self.results['commitment_cost'].items()
                if t == self.InitialTime
                )

            tot_costs += sum(
                v for (g, t), v in self.results['production_cost'].items()
                if t == self.InitialTime
                )

            price = tot_costs / tot_demand
        else:
            price = 0.

        return price

    @property
    def headrooms(self) -> dict[str, float]:
        return {gen: self.results['reserves_provided'][gen, self.InitialTime]
                for gen in self.ThermalGenerators}

    def _piecewise_adjustment_helper(self, points, costs):
        new_points, new_vals, p_mins = dict(), dict(), dict()

        for g in self.ThermalGenerators:
            if g not in points:
                raise ValueError(f"Missing cost curve points for "
                                 f"thermal generator `{g}`!")

            if g not in costs:
                raise ValueError(f"Missing cost curve values for "
                                 f"thermal generator `{g}`!")

            p_min, p_max = (self.MinPowerOutput[g, self.InitialTime],
                            self.MaxPowerOutput[g, self.InitialTime])

            gen_min = 0.
            gen_points = []
            gen_vals = []

            # input_func = _eliminate_piecewise_duplicates(input_func)
            set_p_min = False

            # NOTE: this implicitly inserts a (0.,0.)
            #       into every cost array
            prior_output, prior_cost = 0., 0.

            for output, cost in zip(points[g], costs[g]):
                ## catch this case
                if math.isclose(output, p_min) and math.isclose(output, p_max):
                    gen_points.append(0.)
                    gen_vals.append(0.)
                    gen_min = cost

                    break

                ## output < p_min
                elif _much_less_than(output, p_min):
                    pass

                ## p_min == output
                elif math.isclose(output, p_min):
                    assert set_p_min is False
                    gen_points.append(0.)
                    gen_vals.append(0.)
                    gen_min = cost
                    set_p_min = True

                ## p_min < output
                elif (_much_less_than(p_min, output)
                      and _much_less_than(output, p_max)):
                    if not set_p_min:
                        gen_points.append(0.)
                        gen_vals.append(0.)

                        price = (cost - prior_cost) / (output - prior_output)
                        gen_min = (p_min - prior_output) * price + prior_cost

                        gen_points.append(output - p_min)
                        gen_vals.append((output - p_min) * price)
                        set_p_min = True

                    else:
                        gen_points.append(output - p_min)
                        gen_vals.append(cost - gen_min)

                elif math.isclose(output, p_max) or _much_less_than(p_max, output):
                    if not set_p_min:
                        gen_points.append(0.)
                        gen_vals.append(0.)

                        price = (cost - prior_cost) / (output - prior_output)
                        gen_min = (p_min - prior_output) * price + prior_cost
                        gen_points.append(p_max - p_min)

                        if math.isclose(output, p_max):
                            gen_vals.append(cost - gen_min)
                        else:
                            gen_vals.append((p_max - p_min) * price)

                        set_p_min = True

                    else:
                        gen_points.append(p_max - p_min)

                        if math.isclose(output, p_max):
                            gen_vals.append(cost - gen_min)
                        else:
                            price = (cost - prior_cost) / (output - prior_output)
                            gen_vals.append((p_max - prior_output)
                                            * price + prior_cost - gen_min)

                    break

                else:
                    raise ValueError(
                        "Unexpected case in _piecewise_adjustment_helper, "
                        "p_min={}, p_max={}, output={}".format(
                            p_min, p_max, output)
                        )

                prior_output, prior_cost = output, cost

            new_points[g] = gen_points
            new_vals[g] = gen_vals
            p_mins[g] = gen_min

        return new_points, new_vals, p_mins

    def garver_3bin_vars(self, relax_binaries):
        vtype = GRB.CONTINUOUS if relax_binaries else GRB.BINARY

        self.model._UnitOn = self.model.addVars(
            *self.thermal_periods, lb=0, ub=1, vtype=vtype, name='UnitOn')
        self.model._UnitStart = self.model.addVars(
            *self.thermal_periods, lb=0, ub=1, vtype=vtype, name='UnitStart')
        self.model._UnitStop = self.model.addVars(
            *self.thermal_periods, lb=0, ub=1, vtype=vtype, name='UnitStop')

    def garver_power_vars(self):
        self.model._PowerGeneratedAboveMinimum = self.model.addVars(
            *self.thermal_periods,
            lb=0, ub={(g, t): (self.MaxPowerOutput[g, t]
                               - self.MinPowerOutput[g, t])
                      for g, t in product(*self.thermal_periods)},
            name='PowerGeneratedAboveMinimum'
            )

        self.model._PowerGenerated = tupledict({
            (g, t): (self.model._PowerGeneratedAboveMinimum[g, t]
                     + self.MinPowerOutput[g, t] * self.model._UnitOn[g, t])
            for g, t in product(*self.thermal_periods)
            })

        self.model._PowerGeneratedStartupShutdown = tupledict({
            (g, t): self._add_power_generated_startup_shutdown(
                g, t)
            for g, t in product(*self.thermal_periods)
            })

    def damcikurt_ramping(self):
        # enforce_max_available_ramp_up_rates_rule #
        t0_upower_constrs = dict()
        tk_upower_constrs = dict()

        t0_uramp_periods = {
            (g, self.InitialTime) for g in self.ThermalGenerators
            if (self.UnitOnT0[g]
                and self.NominalRampUpLimit[g] < (
                        self.MaxPowerOutput[g, self.InitialTime]
                        - self.PowerGeneratedT0[g]
                        ))
            }

        for g, t in t0_uramp_periods:
            pvars, pcoefs = self._get_initial_max_power_available_lists(g, t)

            pvars += [self.model._UnitOn[g, t], self.model._UnitStart[g, t]]
            pcoefs += [-self.NominalRampUpLimit[g] + self.MinPowerOutput[g, t],
                       -self.StartupRampLimit[g] + self.NominalRampUpLimit[g]]

            t0_upower_constrs[g, t] = LinExpr(
                pcoefs, pvars) <= self.PowerGeneratedT0[g]

        tk_uramp_periods = {
            (g, t) for g, t in product(*self.thermal_periods)
            if (t > self.InitialTime
                and self.NominalRampUpLimit[g] < (
                        self.MaxPowerOutput[g, t]
                        - self.MinPowerOutput[g, t - 1]
                ))
            }

        for g, t in tk_uramp_periods:
            pvars, pcoefs = self._get_initial_max_power_available_lists(g, t)

            pvars += [self.model._UnitOn[g, t], self.model._UnitStart[g, t]]
            pcoefs += [
                -self.NominalRampUpLimit[g]
                - self.MinPowerOutput[g, t - 1] + self.MinPowerOutput[g, t],
                - self.StartupRampLimit[g]
                + self.MinPowerOutput[g, t - 1] + self.NominalRampUpLimit[g]
                ]

            neg_vars, neg_coefs = self._get_generation_above_minimum_lists(
                g, t - 1, negative=True)

            tk_upower_constrs[g, t] = LinExpr(
                pcoefs + neg_coefs, pvars + neg_vars) <= 0.

        self.model._EnforceMaxAvailableT0RampUpRates = self.model.addConstrs(
            constr_genr(t0_upower_constrs),
            name='enforce_max_available_t0_ramp_up_rates'
            )
        self.model._EnforceMaxAvailableTkRampUpRates = self.model.addConstrs(
            constr_genr(tk_upower_constrs),
            name='enforce_max_available_tk_ramp_up_rates'
            )

        # enforce_ramp_down_limits_rule #
        t0_dpower_constrs = dict()
        tk_dpower_constrs = dict()

        t0_dramp_periods = {
            (g, self.InitialTime) for g in self.ThermalGenerators
            if (self.UnitOnT0[g]
                and (self.NominalRampDownLimit[g]
                     < (self.PowerGeneratedT0[g]
                        - self.MinPowerOutput[g, self.InitialTime])
                     or self.ShutdownRampLimit[g] < self.PowerGeneratedT0[g]))
            }

        tk_dramp_periods = {
            (g, t) for g, t in product(*self.thermal_periods)
            if (t > self.InitialTime
                and (self.NominalRampDownLimit[g] < (
                            self.MaxPowerOutput[g, t - 1]
                            - self.MinPowerOutput[g, t]
                    )))

            }

        for g, t in t0_dramp_periods:
            power_vars, power_coefs = self._get_generation_above_minimum_lists(
                g, t)

            power_vars += [self.model._UnitStop[g, t]]
            power_coefs += [self.ShutdownRampLimit[g]
                            - self.MinPowerOutput[g, t]
                            - self.NominalRampDownLimit[g]]

            power_lhs = (self.PowerGeneratedT0[g]
                         - (self.NominalRampDownLimit[g]
                            + self.MinPowerOutput[g, t]) * self.UnitOnT0[g])

            t0_dpower_constrs[g, t] = LinExpr(
                power_coefs, power_vars) >= power_lhs

        for g, t in tk_dramp_periods:
            power_vars, power_coefs = self._get_generation_above_minimum_lists(
                g, t)

            power_vars += [self.model._UnitOn[g, t - 1],
                           self.model._UnitStop[g, t]]
            power_coefs += [self.NominalRampDownLimit[g]
                            + self.MinPowerOutput[g, t]
                            - self.MinPowerOutput[g, t - 1],
                            self.ShutdownRampLimit[g]
                            - self.MinPowerOutput[g, t]
                            - self.NominalRampDownLimit[g]]

            neg_vars, neg_coefs = self._get_generation_above_minimum_lists(
                g, t - 1, negative=True)

            tk_dpower_constrs[g, t] = LinExpr(
                power_coefs + neg_coefs, power_vars + neg_vars) >= 0

        self.model._EnforceScaledT0RampDownLimits = self.model.addConstrs(
            constr_genr(t0_dpower_constrs),
            name='enforce_max_available_t0_ramp_down_rates'
            )
        self.model._EnforceScaledTkRampDownLimits = self.model.addConstrs(
            constr_genr(tk_dpower_constrs),
            name='enforce_max_available_tk_ramp_down_rates'
            )

    def piecewise_production_sum_rule(self, g, t):
        linear_vars = [
            self.model._PiecewiseProduction[g, t, i]
            for i in range(len(self.PiecewiseGenerationPoints[g]) - 1)
            ]

        linear_coefs = [1.] * len(linear_vars)
        linear_vars.append(self.model._PowerGeneratedAboveMinimum[g, t])
        linear_coefs.append(-1.)

        return LinExpr(linear_coefs, linear_vars) == 0

    def piecewise_production_limits_rule(self, g, t, i, tightened=True):
        # these can always be tightened based on SU/SD, regardless of the
        # ramping/aggregation
        # since PowerGenerationPiecewisePoints are scaled to
        # MinimumPowerOutput, we need to scale Startup/Shutdown ramps
        # to it as well

        upper = self.PiecewiseGenerationPoints[g][i + 1]
        lower = self.PiecewiseGenerationPoints[g][i]
        SU = self.StartupRampLimit[g]
        minP = self.MinPowerOutput[g, t]
        su_step = _step_coeff(upper, lower, SU - minP)

        if t < self.NumTimePeriods:
            SD = self.ShutdownRampLimit[g]
            UT = self.ScaledMinUpTime[g]
            sd_step = _step_coeff(upper, lower, SD - minP)

            if UT > 1:
                linear_vars = [self.model._PiecewiseProduction[g, t, i],
                               self.model._UnitOn[g, t],
                               self.model._UnitStart[g, t],
                               self.model._UnitStop[g, t + 1]]
                linear_coefs = [-1., (upper - lower), -su_step, -sd_step]

                return LinExpr(linear_coefs, linear_vars) >= 0

            # MinimumUpTime[g] <= 1
            else:
                linear_vars = [self.model._PiecewiseProduction[g, t, i],
                               self.model._UnitOn[g, t],
                               self.model._UnitStart[g, t], ]
                linear_coefs = [-1., (upper - lower), -su_step, ]

                if tightened:
                    coef = -max(sd_step - su_step, 0)

                    if coef != 0:
                        linear_vars.append(self.model._UnitStop[g, t + 1])
                        linear_coefs.append(coef)

                return LinExpr(linear_coefs, linear_vars) >= 0

        # t >= value(m.NumTimePeriods)
        else:
            linear_vars = [self.model._PiecewiseProduction[g, t, i],
                           self.model._UnitOn[g, t],
                           self.model._UnitStart[g, t], ]
            linear_coefs = [-1., (upper - lower), -su_step, ]

            return LinExpr(linear_coefs, linear_vars) >= 0

    def piecewise_production_limits_rule2(self, g, t, i, tightened=True):
        ### these can always be tightened based on SU/SD, regardless of the
        # ramping/aggregation
        ### since PowerGenerationPiecewisePoints are scaled to
        # MinimumPowerOutput, we need to scale Startup/Shutdown ramps
        # to it as well
        UT = self.ScaledMinUpTime[g]

        if UT <= 1 and t < self.NumTimePeriods:
            upper = self.PiecewiseGenerationPoints[g][i + 1]
            lower = self.PiecewiseGenerationPoints[g][i]
            SD = self.ShutdownRampLimit[g]
            minP = self.MinPowerOutput[g, t]

            sd_step = _step_coeff(upper, lower, SD - minP)
            linear_vars = [self.model._PiecewiseProduction[g, t, i],
                           self.model._UnitOn[g, t],
                           self.model._UnitStop[g, t + 1], ]
            linear_coefs = [-1., (upper - lower), -sd_step, ]

            if tightened:
                SU = self.StartupRampLimit[g]
                su_step = _step_coeff(upper, lower, SU - minP)
                coef = -max(su_step - sd_step, 0)

                if coef != 0:
                    linear_vars.append(self.model._UnitStart[g, t])
                    linear_coefs.append(coef)

            return LinExpr(linear_coefs, linear_vars) >= 0

        ## MinimumUpTime[g] > 1 or we added it in the
        # t == value(m.NumTimePeriods) clause above
        else:
            return None

    def uptime_rule(self, g, t):
        linear_vars = [self.model._UnitStart[g, tk]
                       for tk in range(t - self.ScaledMinUpTime[g] + 1, t + 1)]
        linear_coefs = [1.] * len(linear_vars)

        linear_vars += [self.model._UnitOn[g, t]]
        linear_coefs += [-1.]

        return LinExpr(linear_coefs, linear_vars) <= 0

    def downtime_rule(self, g, t):
        linear_vars = [self.model._UnitStop[g, tk]
                       for tk in range(t - self.ScaledMinDownTime[g] + 1,
                                       t + 1)]
        linear_coefs = [1.] * len(linear_vars)

        linear_vars += [self.model._UnitOn[g, t]]
        linear_coefs += [1.]

        return LinExpr(linear_coefs, linear_vars) <= 1

    def logical_rule(self, g, t):
        if t == self.InitialTime:
            linear_vars = [self.model._UnitOn[g, t],
                           self.model._UnitStart[g, t],
                           self.model._UnitStop[g, t]]
            linear_coefs = [1., -1., 1.]

            return LinExpr(linear_coefs, linear_vars) == self.UnitOnT0[g]

        else:
            linear_vars = [self.model._UnitOn[g, t],
                           self.model._UnitOn[g, t - 1],
                           self.model._UnitStart[g, t],
                           self.model._UnitStop[g, t]]
            linear_coefs = [1., -1, -1., 1.]

            return LinExpr(linear_coefs, linear_vars) == 0

    def get_hot_startup_pairs(self, g):
        ## for speed, if we don't have different startups

        if len(self.StartupLags[g]) <= 1:
            return []

        first_lag = self.StartupLags[g][0]
        last_lag = self.StartupLags[g][-1]
        init_time = self.TimePeriods[0]
        after_last_time = self.TimePeriods[-1] + 1

        for t_prime in self.model._ValidShutdownTimePeriods[g]:
            t_first = first_lag + t_prime
            t_last = last_lag + t_prime

            if t_first < init_time:
                t_first = init_time

            if t_last > after_last_time:
                t_last = after_last_time

            for t in range(t_first, t_last):
                yield t_prime, t

    def shutdown_match_rule(self, begin_times, g, t):
        init_time = self.TimePeriods[0]

        linear_vars = [self.model._StartupIndicator[g, t, t_p]
                       for t_p in begin_times[g, t]]
        linear_coefs = [1.] * len(linear_vars)

        if t < init_time:
            return LinExpr(linear_coefs, linear_vars) <= 1

        else:
            linear_vars.append(self.model._UnitStop[g, t])
            linear_coefs.append(-1.)

            return LinExpr(linear_coefs, linear_vars) <= 0

    def compute_startup_cost_rule(self, g, t):
        startup_lags = self.StartupLags[g]
        startup_costs = self.StartupCosts[g]
        last_startup_cost = startup_costs[-1]

        linear_vars = [self.model._StartupCost[g, t], self.model._UnitStart[g, t]]
        linear_coefs = [-1., last_startup_cost]

        for tp in self.model._ShutdownsByStartups[g, t]:
            for s in self.StartupCostIndices[g]:
                this_lag = startup_lags[s]
                next_lag = startup_lags[s + 1]

                if this_lag <= t - tp < next_lag:
                    linear_vars.append(self.model._StartupIndicator[g, tp, t])
                    linear_coefs.append(startup_costs[s] - last_startup_cost)

                    break

        return LinExpr(linear_coefs, linear_vars) == 0

    def _get_power_generated_lists(self, g, t):
        return ([self.model._PowerGeneratedAboveMinimum[g, t],
                 self.model._UnitOn[g, t]],
                [1., self.MinPowerOutput[g, t]])

    def _get_max_power_available_lists(self, g, t):
        return ([self.model._MaximumPowerAvailableAboveMinimum[g, t],
                 self.model._UnitOn[g, t]],
                [1., self.MinPowerOutput[g, t]])

    @abstractmethod
    def _get_generation_above_minimum_lists(self, g, t, negative=False):
        pass

    def _get_initial_max_power_available_lists(self, g, t):
        linear_vars, linear_coefs = self._get_max_power_available_lists(g, t)
        linear_vars.append(self.model._UnitOn[g, t])
        linear_coefs.append(-self.MaxPowerOutput[g, t])

        return linear_vars, linear_coefs

    def _get_look_back_periods(self, g, t, ut_end):
        p_max_gt = self.MaxPowerOutput[g, t]
        ramping_tot = 0

        end = (t - self.InitialTime if ut_end is None
               else min(ut_end, t - self.InitialTime))

        if end <= 0:
            return end

        for i in range(1, end + 1):
            startup_gi = self.StartupRampLimit[g]
            ramping_tot += self.NominalRampUpLimit[g]

            if startup_gi + ramping_tot >= p_max_gt:
                ## the prior index what the correct one
                return i - 1

        # then we can go to the end
        return i

    def _get_look_forward_periods(self, g, t, ut_end):
        p_max_gt = self.MaxPowerOutput[g, t]
        ramping_tot = 0

        end = (self.NumTimePeriods - t - 1 if ut_end is None
               else min(ut_end, self.NumTimePeriods - t - 1))

        if end <= 0:
            return end

        for i in range(1, end + 1):
            shutdown_gi = self.ShutdownRampLimit[g]
            ramping_tot += self.NominalRampDownLimit[g]

            if shutdown_gi + ramping_tot >= p_max_gt:
                ## the prior index what the correct one
                return i - 1

        ## then we can go to the end
        return i

    def _add_power_generated_startup_shutdown(self, g, t):
        linear_vars, linear_coefs = self._get_power_generated_lists(g, t)

        # first, discover if we have startup/shutdown
        # curves in the self.model
        self.model_has_startup_shutdown_curves = False
        for s in self.model._StartupCurve.values():
            if len(s) > 0:
                self.model_has_startup_shutdown_curves = True
                break

        if not self.model_has_startup_shutdown_curves:
            for s in self.model._ShutdownCurve.values():
                if len(s) > 0:
                    self.model_has_startup_shutdown_curves = True
                    break

        if self.model_has_startup_shutdown_curves:
            # check the status vars to see if we're compatible
            # with startup/shutdown curves
            if self.model._status_vars not in ['garver_2bin_vars', 'garver_3bin_vars', 'garver_3bin_relaxed_stop_vars',
                                          'ALS_state_transition_vars']:
                raise RuntimeError(
                    f"Status variable formulation {self.model._status_vars} is not compatible with startup or shutdown curves")

            startup_curve = self.model._StartupCurve[g]
            shutdown_curve = self.model._ShutdownCurve[g]
            time_periods_before_startup = self.model._TimePeriodsBeforeStartup[g]
            time_periods_since_shutdown = self.model._TimePeriodsSinceShutdown[g]

            future_startup_past_shutdown_production = 0.
            future_startup_power_index = time_periods_before_startup + self.model._NumTimePeriods - t
            if future_startup_power_index <= len(startup_curve) - 1:
                future_startup_past_shutdown_production += startup_curve[future_startup_power_index]

            past_shutdown_power_index = time_periods_since_shutdown + t
            if past_shutdown_power_index <= len(shutdown_curve) - 1:
                future_startup_past_shutdown_production += shutdown_curve[past_shutdown_power_index]

            linear_vars, linear_coefs = self.model._get_power_generated_lists(self.model, g, t)
            for startup_idx in range(1, min(len(startup_curve), self.model._NumTimePeriods + 1 - t)):
                linear_vars.append(self.model._UnitStart[g, t + startup_idx])
                linear_coefs.append(startup_curve[startup_idx])
            for shutdown_idx in range(1, min(len(shutdown_curve), t + 1)):
                linear_vars.append(self.model._UnitStop[g, t - shutdown_idx + 1])
                linear_coefs.append(shutdown_curve[shutdown_idx])
            return LinExpr(linear_coefs, linear_vars) + future_startup_past_shutdown_production

            ## if we're here, then we can use 1-bin self.models
            ## and no need to do the additional work
        return LinExpr(linear_coefs, linear_vars)

    def _add_reserve_shortfall(self):
        # add_reserve_shortfall (fixed=False) #

        self.model._ReserveShortfall = self.model.addVars(
            self.TimePeriods, lb=0, ub=[self.ReserveReqs[t]
                                        for t in self.TimePeriods],
            name='ReserveShortfall'
            )

    @property
    def ptdf_options(self) -> dict:
        """See egret.common.lazy_ptdf_utils"""
        new_options = dict()

        for ptdf_k, default_val in self.DEFAULT_PTDF_OPTIONS.items():
            if ptdf_k in self.PTDFoptions:
                new_options[ptdf_k] = self.PTDFoptions[ptdf_k]
            else:
                new_options[ptdf_k] = default_val

        new_options['abs_ptdf_tol'] /= self.BaseMVA
        new_options['abs_flow_tol'] /= self.BaseMVA
        new_options['active_flow_tol'] /= self.BaseMVA

        return new_options

    def file_non_dispatchable_vars(self):
        self.model._RenewablePowerUsed = self.model.addVars(
            *self.renew_periods, lb=self.MinPowerOutput,
            ub=self.MaxPowerOutput, name='NondispatchablePowerUsed'
            )

    def piecewise_production_costs_rule(self, g, t):
        if (g, t, 0) in self.prod_indices:
            points = self.PiecewiseGenerationPoints[g]
            costs = self.PiecewiseGenerationCosts[g]

            linear_coefs = [(costs[i + 1] - costs[i])
                            / (points[i + 1] - points[i])
                            for i in range(len(points) - 1)]
            linear_vars = [self.model._PiecewiseProduction[g, t, i]
                           for i in range(len(points) - 1)]

            linear_coefs.append(-1.)
            linear_vars.append(self.model._ProductionCost[g, t])

            return LinExpr(linear_coefs, linear_vars) == 0

        else:
            return self.model._ProductionCost[g, t] == 0

    def rajan_takriti_ut_dt(self):
        for g, t in product(*self.thermal_periods):
            if self.FixedCommitment[g, t] is not None:
                self.model._UnitOn[g, t].lb = self.FixedCommitment[g, t]
                self.model._UnitOn[g, t].ub = self.FixedCommitment[g, t]

        for g in self.ThermalGenerators:
            if self.InitTimePeriodsOnline[g] != 0:
                for t in range(self.TimePeriods[0],
                               self.InitTimePeriodsOnline[g]
                               + self.TimePeriods[0]):
                    self.model._UnitOn[g, t].ub = 1
                    self.model._UnitOn[g, t].lb = 1

            if self.InitTimePeriodsOffline[g] != 0:
                for t in range(self.TimePeriods[0],
                               self.InitTimePeriodsOffline[g]
                               + self.TimePeriods[0]):
                    self.model._UnitOn[g, t].ub = 0
                    self.model._UnitOn[g, t].lb = 0

        uptime_periods = [(g, t) for g, t in product(*self.thermal_periods)
                          if t >= self.ScaledMinUpTime[g]]
        downtime_periods = [(g, t) for g, t in product(*self.thermal_periods)
                            if t >= self.ScaledMinDownTime[g]]

        self.model._UpTime = self.model.addConstrs(
            (self.uptime_rule(g, t) for g, t in uptime_periods), name='UpTime')

        self.model._DownTime = self.model.addConstrs(
            (self.downtime_rule(g, t)
             for g, t in downtime_periods), name='DownTime'
            )

    def ptdf_power_flow(self):
        over_gen_maxes = {}
        over_gen_times_per_bus = {b: list() for b in self.Buses}
        load_shed_maxes = {}
        load_shed_times_per_bus = {b: list() for b in self.Buses}

        for b in self.Buses:

            # storage, for now, does not have time-varying parameters
            storage_max_injections = 0.
            storage_max_withdraws = 0.

            for s in self.StorageAtBus[b]:
                storage_max_injections += self.model._MaximumPowerOutputStorage[s]
                storage_max_withdraws += self.model._MaximumPowerInputStorage[s]

            for t in self.TimePeriods:
                max_injections = storage_max_injections
                max_withdrawls = storage_max_withdraws

                for g in self.ThermalGeneratorsAtBus[b]:
                    p_max = self.MaxPowerOutput[g, t]
                    p_min = self.MinPowerOutput[g, t]

                    if p_max > 0:
                        max_injections += p_max
                    if p_min < 0:
                        max_withdrawls += -p_min

                for n in self.NondispatchableGeneratorsAtBus[b]:
                    p_max = self.MaxPowerOutput[n, t]
                    p_min = self.MinPowerOutput[n, t]

                    if p_max > 0:
                        max_injections += p_max
                    if p_min < 0:
                        max_withdrawls += -p_min

                load = self.Demand.loc[b, t]

                if load > 0:
                    max_withdrawls += load
                elif load < 0:
                    max_injections += -load

                if max_injections > 0:
                    over_gen_maxes[b, t] = max_injections
                    over_gen_times_per_bus[b].append(t)

                if max_withdrawls > 0:
                    load_shed_maxes[b, t] = max_withdrawls
                    load_shed_times_per_bus[b].append(t)

        self.model._OverGenerationBusTimes = list(over_gen_maxes.keys())
        self.model._LoadSheddingBusTimes = list(load_shed_maxes.keys())

        self.model._OverGeneration = self.model.addVars(
            self.model._OverGenerationBusTimes,
            lb=0, ub=[over_gen_maxes[k]
                      for k in self.model._OverGenerationBusTimes],
            name='OverGeneration'
            )

        self.model._LoadShedding = self.model.addVars(
            self.model._LoadSheddingBusTimes,
            lb=0, ub=[load_shed_maxes[key]
                      for key in self.model._LoadSheddingBusTimes],
            name='LoadShedding'
            )

        #####################################################
        # load "shedding" can be both positive and negative #
        #####################################################
        self.model._LoadGenerateMismatch = {
            (b, t): 0. for b in self.Buses for t in self.TimePeriods}

        for b, t in self.model._LoadSheddingBusTimes:
            self.model._LoadGenerateMismatch[b, t] += self.model._LoadShedding[b, t]
        for b, t in self.model._OverGenerationBusTimes:
            self.model._LoadGenerateMismatch[b, t] -= self.model._OverGeneration[b, t]

        self.model._LoadMismatchCost = {}
        for t in self.TimePeriods:
            self.model._LoadMismatchCost[t] = 0.

        for b, t in self.model._LoadSheddingBusTimes:
            self.model._LoadMismatchCost[t] += (
                    self.model._LoadMismatchPenalty
                    * self.model._LoadShedding[b, t]
                    )

        for b, t in self.model._OverGenerationBusTimes:
            self.model._LoadMismatchCost[t] += (
                    self.model._LoadMismatchPenalty
                    * self.model._OverGeneration[b, t]
                    )

        # for interface violation costs at a time step
        self.model._BranchViolationCost = {t: 0 for t in self.TimePeriods}

        # for interface violation costs at a time step
        self.model._InterfaceViolationCost = {t: 0 for t in self.TimePeriods}

        # for contingency violation costs at a time step
        self.model._ContingencyViolationCost = {t: 0 for t in self.TimePeriods}

        self.model._p_nw = self.model.addVars(
            self.Buses, self.TimePeriods,
            lb=-GRB.INFINITY, ub=GRB.INFINITY, name='p_nw'
            )

        # set up the empty self.model block for each time period to add constraints
        self.model._TransmissionBlock = dict()
        for tm in self.TimePeriods:
            block = {'tm': tm,
                     'gens_by_bus': {bus: [bus] for bus in self.Buses},
                     'pg': dict(), 'idx_monitored': list()}

            for b in self.Buses:
                start_shut = quicksum(
                    self.model._PowerGeneratedStartupShutdown[g, tm]
                    for g in self.ThermalGeneratorsAtBus[b]
                    )

                out_store = quicksum(self.model._PowerOutputStorage[s, tm]
                                     for s in self.StorageAtBus[b])
                in_store = quicksum(self.model._PowerInputStorage[s, tm]
                                    for s in self.StorageAtBus[b])

                non_dispatch = quicksum(
                    self.model._RenewablePowerUsed[g, tm]
                    for g in self.NondispatchableGeneratorsAtBus[b]
                    )

                block['pg'][b] = (start_shut + non_dispatch
                                  + out_store - in_store
                                  + self.model._LoadGenerateMismatch[b, tm])

            # _PTDF_DCOPF_NETWORK_MODEL
            bus_loads = {bus: self.Demand.loc[bus, tm] for bus in self.Buses}
            block['pl'] = bus_loads

            block['branches_in_service'] = tuple(
                line for line in self.TransmissionLines
                if not self.LineOutOfService[line]
                )

            # declare_eq_p_net_withdraw_at_bus  (dc...branches = None) #
            for bus in self.Buses:
                block[f'eq_withdraw_{bus}'] = self.model.addConstr(
                    (self.model._p_nw[bus, tm] == (
                            (block['pl'][bus] if bus_loads[bus] != 0.0
                             else 0.0)
                            - quicksum(block['pg'][g]
                                       for g in block['gens_by_bus'][bus]))),

                    name=f"_eq_p_net_withdraw_at_bus[{bus}]_at_period[{tm}]"
                    )

            p_expr = quicksum(block['pg'][bus_name] for bus_name in self.Buses
                              if len(block['gens_by_bus'][bus_name]) != 0)
            p_expr -= quicksum(block['pl'][bus] for bus in self.Buses
                               if bus_loads[bus] is not None)

            block['eq_balance'] = self.model.addConstr(
                (p_expr == 0.0), name=f"eq_p_balance_at_period{tm}")

            if not self.PTDF:
                self.PTDF = VirtualPTDFMatrix(
                    self.branches, self.buses, self.ReferenceBus,
                    BasePointType.FLATSTART, self.ptdf_options,
                    branches_keys=block['branches_in_service'],
                    buses_keys=self.Buses
                    )

            block['PTDF'] = self.PTDF
            block['p_nw'] = {b: self.model._p_nw[b, tm] for b in self.Buses}
            block['pf'] = {branch: None for branch in self.branches}

            self.model._TransmissionBlock[tm] = block

    def add_objective(self):
        self.model._NoLoadCost = tupledict({
            (g, t): self.MinProductionCost[g] * self.model._UnitOn[g, t]
            for g, t in product(*self.thermal_periods)
            })

        self.model._TotalProductionCost = {t: sum(self.model._ProductionCost[g, t]
                                             for g in self.ThermalGenerators)
                                      for t in self.TimePeriods}

        self.model._CommitmentStageCost = {
            st: sum(sum(self.model._NoLoadCost[g, t] + self.model._StartupCost[g, t]
                        for g in self.ThermalGenerators)
                    for t in self.model._CommitmentTimeInStage[st])
            for st in self.StageSet
            }

        self.model._ReserveShortfallCost = {
            t: self.model._ReserveShortfallPenalty * self.model._ReserveShortfall[t]
            for t in self.TimePeriods
            }

        self.model._GenerationStageCost = {
            st: (sum(sum(self.model._ProductionCost[g, t]
                        for g in self.ThermalGenerators)
                    for t in self.model._GenerationTimeInStage[st])
                 + sum(self.model._LoadMismatchCost[t]
                      for t in self.model._GenerationTimeInStage[st])
                 + sum(self.model._ReserveShortfallCost[t]
                      for t in self.model._GenerationTimeInStage[st])
                 + sum(self.model._StorageCost[s, t] for s in self.Storage
                       for t in self.model._GenerationTimeInStage[st]))
            for st in self.StageSet
            }

        self.model._StageCost = {st: (self.model._GenerationStageCost[st]
                                      + self.model._CommitmentStageCost[st])
                                 for st in self.StageSet}

        self.model.setObjective(quicksum(self.model._StageCost[st]
                                         for st in self.StageSet),
                                GRB.MINIMIZE)

    def _add_zero_cost_hours(self, objective_hours):
        if objective_hours:
            zero_cost_hours = self.TimePeriods.copy()

            for i, t in enumerate(self.TimePeriods):
                if i < objective_hours:
                    zero_cost_hours.remove(t)
                else:
                    break

            cost_gens = {g for g, _ in self.model._ProductionCost}
            for t in zero_cost_hours:
                for g in cost_gens:
                    self.model.remove(self.model._ProductionCostConstr[g, t])
                    self.model._ProductionCost[g, t].lb = 0.
                    self.model._ProductionCost[g, t].ub = 0.

    @property
    def forecastables(self) -> pd.DataFrame:
        use_gen = self.RenewOutput.copy().transpose()
        use_load = self.Demand.copy().transpose()

        use_gen.columns = pd.MultiIndex.from_tuples(
            [('RenewGen', gen) for gen in use_gen.columns],
            names=('AssetType', 'Asset')
            )
        use_load.columns = pd.MultiIndex.from_tuples(
            [('LoadBus', bus) for bus in use_load.columns],
            names=('AssetType', 'Asset')
            )

        fcsts = pd.concat([use_gen, use_load], axis=1)
        fcsts.index = self.time_steps

        return fcsts


class RucModel(BaseModel):

    model_name = "UnitCommitment"

    def _parse_model_results(self):
        results = super()._parse_model_results()

        results['reserves_provided'] = {
            (g, t): self.model._ReserveProvided[g, t].getValue()
            for g, t in product(*self.thermal_periods)
            }

        results['power_generated_above_min'] = {
            (g, t): self.model._PowerGeneratedAboveMinimum[g, t].x
            for g, t in product(*self.thermal_periods)
            }

        results['power_avail_above_min'] = {
            (g, t): self.model._MaximumPowerAvailableAboveMinimum[g, t].x
            for g, t in product(*self.thermal_periods)
            }

        return results

    def _get_generation_above_minimum_lists(self, g, t, negative=False):
        linear_vars = [self.model._PowerGeneratedAboveMinimum[g, t]]
        linear_coefs = [-1.] if negative else [1.]

        return linear_vars, linear_coefs

    def garver_power_avail_vars(self):
        self.model._MaximumPowerAvailableAboveMinimum = self.model.addVars(
            *self.thermal_periods,
            lb=0, ub={(g, t): (self.MaxPowerOutput[g, t]
                               - self.MinPowerOutput[g, t])
                      for g, t in product(*self.thermal_periods)},
            name='MaximumPowerAvailableAboveMinimum'
            )

        self.model._MaximumPowerAvailable = tupledict({
            (g, t): (self.model._MaximumPowerAvailableAboveMinimum[g, t]
                     + self.MinPowerOutput[g, t] * self.model._UnitOn[g, t])
            for g, t in product(*self.thermal_periods)
            })

        self.model._ReserveProvided = tupledict({
            (g, t): (self.model._MaximumPowerAvailableAboveMinimum[g, t]
                     - self.model._PowerGeneratedAboveMinimum[g, t])
            for g, t in product(*self.thermal_periods)
            })

        self.model._EnforceGeneratorOutputLimitsPartB = self.model.addConstrs(
            ((self.model._PowerGeneratedAboveMinimum[g, t]
              - self.model._MaximumPowerAvailableAboveMinimum[g, t] <= 0)
             for g, t in product(*self.thermal_periods)),
            name='EnforceGeneratorOutputLimitsPartB'
            )

    def pgg_KOW_gen_limits(self):
        power_startlimit_constrs = dict()
        power_stoplimit_constrs = dict()
        power_startstoplimit_constrs = dict()

        for g, t in product(*self.thermal_periods):
            linear_vars, linear_coefs = self._get_initial_max_power_available_lists(
                g, t)

            # _MLR_GENERATION_LIMITS_UPTIME_1 (tightened) #
            if self.ScaledMinUpTime[g] == 1:
                start_vars = linear_vars + [self.model._UnitStart[g, t]]
                start_coefs = linear_coefs + [self.MaxPowerOutput[g, t]
                                              - self.StartupRampLimit[g]]

                if t < self.NumTimePeriods:
                    startramp_coef = (self.StartupRampLimit[g]
                                      - self.ShutdownRampLimit[g])

                    if startramp_coef > 0:
                        start_vars += [self.model._UnitStop[g, t + 1]]
                        start_coefs += [startramp_coef]

                    stop_vars = linear_vars + [self.model._UnitStop[g, t + 1]]
                    stop_coefs = linear_coefs + [
                        self.MaxPowerOutput[g, t] - self.ShutdownRampLimit[g]]

                    stopramp_coef = (self.ShutdownRampLimit[g]
                                     - self.StartupRampLimit[g])

                    if stopramp_coef > 0:
                        stop_vars += [self.model._UnitStart[g, t]]
                        stop_coefs += [stopramp_coef]

                    power_stoplimit_constrs[g, t] = LinExpr(
                        stop_coefs, stop_vars) <= 0

                power_startlimit_constrs[g, t] = LinExpr(
                    start_coefs, start_vars) <= 0

            # _PAN_GUAN_GENERATION_LIMITS (w/o uptime-1 generators) #
            else:
                end_ut = (self.ScaledMinUpTime[g] - 2
                          if t < self.NumTimePeriods
                          else self.ScaledMinUpTime[g] - 1)

                for i in range(self._get_look_back_periods(g, t, end_ut) + 1):
                    linear_vars += [self.model._UnitStart[g, t - i]]

                    linear_coefs += [self.MaxPowerOutput[g, t]
                                     - self.StartupRampLimit[g]
                                     - sum(self.NominalRampUpLimit[g]
                                           for j in range(1, i + 1))]

                if t < self.NumTimePeriods:
                    linear_vars += [self.model._UnitStop[g, t + 1]]
                    linear_coefs += [self.MaxPowerOutput[g, t]
                                     - self.ShutdownRampLimit[g]]

                power_startstoplimit_constrs[g, t] = LinExpr(
                    linear_coefs, linear_vars) <= 0

        self.model._power_limit_from_start = self.model.addConstrs(
            constr_genr(power_startlimit_constrs),
            name='_power_limit_from_start'
            )

        self.model._power_limit_from_stop = self.model.addConstrs(
            constr_genr(power_stoplimit_constrs),
            name='_power_limit_from_stop'
            )

        self.model._power_limit_from_startstop = self.model.addConstrs(
            constr_genr(power_startstoplimit_constrs),
            name='_power_limit_from_start_stop_pan_guan_gentile'
            )

    def kow_gen_limits(self):
        gener_starts_constrs = dict()
        gener_startstops_constrs = dict()

        # max_power_limit_from_starts_rule #
        for g, t in product(*self.thermal_periods):
            time_ru = self._get_look_back_periods(g, t, None)

            if (t < self.NumTimePeriods
                    and time_ru > max(0, self.ScaledMinUpTime[g] - 2)):
                start_vars, start_coefs = self._get_initial_max_power_available_lists(
                    g, t)

                start_vars += [self.model._UnitOn[g, t]]
                start_coefs += [-self.MaxPowerOutput[g, t]]

                for i in range(min(time_ru, self.ScaledMinUpTime[g] - 1,
                                   t - self.InitialTime) + 1):
                    start_vars += [self.model._UnitStart[g, t - i]]
                    start_coefs += [
                        self.MaxPowerOutput[g, t]
                        - self.StartupRampLimit[g]
                        - sum(self.NominalRampUpLimit[g]
                              for j in range(1, i + 1))
                        ]

                gener_starts_constrs[g, t] = LinExpr(
                    start_coefs, start_vars) <= 0

        # power_limit_from_start_stops_rule #
        for g, t in product(*self.thermal_periods):
            sd_time_limit = self._get_look_forward_periods(
                g, t, self.ScaledMinUpTime[g] - 1)

            if sd_time_limit > 0:
                su_time_limit = self._get_look_back_periods(
                    g, t, self.ScaledMinUpTime[g] - 2 - sd_time_limit)

                start_vars, start_coefs = self._get_initial_max_power_available_lists(
                    g, t)

                for i in range(sd_time_limit + 1):
                    start_vars += [self.model._UnitStop[g, t + i + 1]]
                    start_coefs += [
                        self.MaxPowerOutput[g, t]
                        - self.ShutdownRampLimit[g]
                        - sum(self.NominalRampDownLimit[g]
                              for j in range(1, i + 1))
                        ]

                for i in range(su_time_limit + 1):
                    start_vars += [self.model._UnitStart[g, t - i]]
                    start_coefs += [
                        self.MaxPowerOutput[g, t]
                        - self.StartupRampLimit[g]
                        - sum(self.NominalRampUpLimit[g]
                              for j in range(1, i + 1))
                        ]

                if self.ScaledMinUpTime[g] < (max(0, su_time_limit)
                                              + max(0, sd_time_limit) + 2):
                    if (t - su_time_limit - 1) >= self.InitialTime:
                        coef = max(
                            0, self.MaxPowerOutput[g, t]
                               - self.ShutdownRampLimit[g]
                               - sum(self.NominalRampDownLimit[g]
                                     for j in range(1, sd_time_limit + 1))

                               - (self.MaxPowerOutput[g, t]
                                  - self.StartupRampLimit[g]
                                  - sum(self.NominalRampUpLimit[g]
                                        for j in range(1, su_time_limit + 2)))
                            )

                        if coef != 0:
                            start_vars += [self.model._UnitStart[
                                               g, t - su_time_limit - 1]]
                            start_coefs += [coef]

                gener_startstops_constrs[g, t] = LinExpr(
                    start_coefs, start_vars) <= 0

        self.model._max_power_limit_from_starts = self.model.addConstrs(
            constr_genr(gener_starts_constrs),
            name='_max_power_limit_from_starts'
            )

        self.model._max_power_limit_from_start_stop = self.model.addConstrs(
            constr_genr(gener_startstops_constrs),
            name='_power_limit_from_start_stop_KOW'
            )

    def compute_production_costs_rule(self, g, t, avg_power):
        ## piecewise points for power buckets
        piecewise_points = self.PiecewiseGenerationPoints[g]
        piecewise_eval = [0] * (len(piecewise_points) - 1)

        ## fill the buckets (skip the first since it's min power)
        for l in range(len(piecewise_eval)):
            ## fill this bucket all the way
            if avg_power >= piecewise_points[l + 1]:
                piecewise_eval[l] = (piecewise_points[l + 1]
                                     - piecewise_points[l])

            ## fill the bucket part way and stop
            elif avg_power < piecewise_points[l + 1]:
                piecewise_eval[l] = avg_power - piecewise_points[l]
                break

        # slope * production
        return sum((self.model._PiecewiseProductionCosts[g, t, l + 1]
                    - self.model._PiecewiseProductionCosts[g, t, l])
                   / (piecewise_points[l + 1] - piecewise_points[l])
                   * piecewise_eval[l] for l in range(len(piecewise_eval)))

    def kow_production_costs_tightened(self):
        self.model._PiecewiseProduction = self.model.addVars(
            self.prod_indices,
            lb=0., ub=[self.PiecewiseGenerationPoints[g][i + 1]
                       - self.PiecewiseGenerationPoints[g][i]
                       for g, t, i in self.prod_indices],
            name='PiecewiseProduction'
            )

        self.model._PiecewiseProductionSum = self.model.addConstrs(
            (self.piecewise_production_sum_rule(g, t)
             for g, t, _ in self.prod_indices), name='PiecewiseProductionSum'
            )

        self.model._PiecewiseProductionLimits = self.model.addConstrs(
            (self.piecewise_production_limits_rule(g, t, i)
             for g, t, i in self.prod_indices),
            name='PiecewiseProductionLimits'
            )

        limits_periods = [(g, t, i) for g, t, i in self.prod_indices
                          if self.ScaledMinUpTime[g] <= 1
                          and t < self.NumTimePeriods]

        self.model._PiecewiseProductionLimits2 = self.model.addConstrs(
            (self.piecewise_production_limits_rule2(g, t, i)
             for g, t, i in limits_periods),
            name='PiecewiseProductionLimits2'
            )

        self.model._ProductionCost = self.model.addVars(
            *self.thermal_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY,
            name='ProductionCost'
            )

        self.model._ProductionCostConstr = self.model.addConstrs(
            (self.piecewise_production_costs_rule(g, t)
             for g, t in product(*self.thermal_periods)),
            name='ProductionCostConstr'
            )

        self.model._ComputeProductionCosts = self.compute_production_costs_rule

    def kow_startup_costs(self, relax_binaries):
        vtype = GRB.CONTINUOUS if relax_binaries else GRB.BINARY

        self.model._ValidShutdownTimePeriods = {
            g: (list() if len(self.StartupLags[g]) <= 1
                else self.TimePeriods if self.UnitOnT0State[g] >= 0
                else self.TimePeriods + [self.InitialTime
                                         + int(self.UnitOnT0State[g])])
            for g in self.ThermalGenerators
            }

        self.model._ShutdownHotStartupPairs = {
            g: ([] if len(self.StartupLags) <= 1
                else list(self.get_hot_startup_pairs(g)))
            for g in self.ThermalGenerators
            }

        self.model._StartupIndicator_domain = [
            (g, t_prime, t) for g in self.ThermalGenerators
            for t_prime, t in self.model._ShutdownHotStartupPairs[g]
            ]

        self.model._StartupIndicator = self.model.addVars(
            self.model._StartupIndicator_domain,
            vtype=vtype, name='StartupIndicator'
            )

        self.model._GeneratorShutdownPeriods = [
            (g, t) for g in self.ThermalGenerators
            for t in self.model._ValidShutdownTimePeriods[g]
            ]

        self.model._ShutdownsByStartups = {
            (g, t): [] for g, t in product(*self.thermal_periods)}
        self.model._StartupsByShutdowns = {
            (g, t): [] for g, t in self.model._GeneratorShutdownPeriods}

        for g, t_p, t in self.model._StartupIndicator_domain:
            self.model._ShutdownsByStartups[g, t] += [t_p]
            self.model._ShutdownsByStartups[g, t] += [t_p]
            self.model._StartupsByShutdowns[g, t_p] += [t]

        self.model._StartupMatch = self.model.addConstrs(
            (LinExpr([1.] * len(self.model._ShutdownsByStartups[g, t]) + [-1.],
                     [self.model._StartupIndicator[g, t_prime, t]
                      for t_prime in self.model._ShutdownsByStartups[g, t]]
                     + [self.model._UnitStart[g, t]]) <= 0
             for g, t in product(*self.thermal_periods)),
            name='StartupMatch'
            )

        begin_times = {(g, t): self.model._StartupsByShutdowns[g, t]
                       for g, t in self.model._GeneratorShutdownPeriods
                       if self.model._StartupsByShutdowns[g, t]}

        self.model._ShutdownMatch = self.model.addConstrs(
            (self.shutdown_match_rule(begin_times, g, t)
             for g, t in begin_times),
            name='ShutdownMatch'
            )

        self.model._StartupCost = self.model.addVars(
            *self.thermal_periods, lb=0, ub=GRB.INFINITY, name='StartupCost')

        self.model._ComputeStartupCosts = self.model.addConstrs(
            (self.compute_startup_cost_rule(g, t)
             for g, t in product(*self.thermal_periods)),
            name='ComputeStartupCosts'
            )

    def enforce_reserve_requirements_rule(self, t):
        self.model._LoadGenerateMismatch = tupledict(self.model._LoadGenerateMismatch)

        linear_expr = (
                quicksum(self.model._MaximumPowerAvailable.select('*', t))
                + quicksum(self.model._RenewablePowerUsed.select('*', t))
                + quicksum(self.model._LoadGenerateMismatch.select('*', t))
                + quicksum(self.model._ReserveShortfall.select(t))
                )

        if hasattr(self.model, '_PowerOutputStorage'):
            linear_expr += quicksum(self.model._PowerOutputStorage.select('*', t))

        if hasattr(self.model, '_PowerInputStorage'):
            linear_expr -= quicksum(self.model._PowerInputStorage.select('*', t))

        return linear_expr >= (
            sum(self.Demand.loc[b, t] for b in sorted(self.Buses))
            + self.ReserveReqs[t]
            )

    def ca_reserve_constraints(self):
        self._add_reserve_shortfall()

        # ensure there is sufficient maximal power output available to meet
        # both the demand and the spinning reserve requirements in each time
        # period. encodes Constraint 3 in Carrion and Arroyo.

        # IMPT: In contrast to power balance, reserves are (1) not per-bus
        # and (2) expressed in terms of maximum power available, and not
        # actual power generated.
        self.model._EnforceReserveRequirements = self.model.addConstrs(
            (self.enforce_reserve_requirements_rule(t)
             for t in self.TimePeriods),
            name='EnforceReserveRequirements'
            )

    @property
    def binary_variables(self):
        return super().binary_variables + [self.model._StartupIndicator]

    def generate(self,
                 relax_binaries: bool, ptdf_options: dict,
                 ptdf, objective_hours: int) -> None:
        self._initialize_model(ptdf, ptdf_options)

        self.garver_3bin_vars(relax_binaries)
        self.garver_power_vars()
        self.garver_power_avail_vars()
        self.file_non_dispatchable_vars()

        self.pgg_KOW_gen_limits()
        self.damcikurt_ramping()
        self.kow_production_costs_tightened()
        self.rajan_takriti_ut_dt()
        self.kow_startup_costs(relax_binaries)

        # _3bin_logic can just be written out here
        self.model._Logical = self.model.addConstrs(
            (self.logical_rule(g, t)
             for g, t in product(*self.thermal_periods))
            )

        self.ptdf_power_flow()
        self.ca_reserve_constraints()

        # set up objective
        self.add_objective()
        self._add_zero_cost_hours(objective_hours)

        self.model.update()

        self.model.write("gurobi.lp")


class ScedModel(BaseModel):

    model_name = 'EconomicDispatch'

    def _parse_model_results(self):
        results = super()._parse_model_results()

        results['reserves_provided'] = {
            (g, t): self.model._ReserveProvided[g, t].x
            for g, t in product(*self.thermal_periods)
            }

        results['power_generated_above_min'] = {
            (g, t): self.model._PowerGeneratedAboveMinimum[g, t].x
            for g, t in product(*self.thermal_periods)
            }

        results['power_avail_above_min'] = {
            (g, t): self.model._MaximumPowerAvailableAboveMinimum[
                g, t].getValue()
            for g, t in product(*self.thermal_periods)
            }

        return results

    def _get_max_power_available_lists(self, g, t):
        linear_vars, linear_coefs = self._get_power_generated_lists(g, t)
        linear_vars.append(self.model._ReserveProvided[g, t])
        linear_coefs.append(1.)

        return linear_vars, linear_coefs

    def _get_generation_above_minimum_lists(self, g, t, negative=False):
        linear_vars = [self.model._PowerGeneratedAboveMinimum[g, t]]
        linear_coefs = [-1.] if negative else [1.]

        linear_vars.append(self.model._ReserveProvided[g, t])
        linear_coefs.append(1.)

        return linear_vars, linear_coefs

    def mlr_reserve_vars(self):
        # amount of power produced by each generator above minimum,
        # at each time period. variable for reserves offered
        self.model._ReserveProvided = self.model.addVars(
            *self.thermal_periods,
            lb=0, ub=[self.MaxPowerOutput[g, t] - self.MinPowerOutput[g, t]
                      for g, t in product(*self.thermal_periods)],
            name='ReserveProvided'
            )

        self.model._MaximumPowerAvailableAboveMinimum = {
            (g, t): LinExpr(
                *self._get_generation_above_minimum_lists(g, t)[::-1])
            for g, t in product(*self.thermal_periods)
            }

    def mlr_generation_limits(self):
        power_startlimit_constrs = dict()
        power_stoplimit_constrs = dict()
        power_startstoplimit_constrs = dict()

        for g, t in product(*self.thermal_periods):
            linear_vars, linear_coefs = self._get_initial_max_power_available_lists(g, t)
            start_vars = linear_vars + [self.model._UnitStart[g, t]]
            start_coefs = linear_coefs + [self.MaxPowerOutput[g, t]
                                          - self.StartupRampLimit[g]]

            # _MLR_GENERATION_LIMITS_UPTIME_1 (tightened=False) #
            if self.ScaledMinUpTime[g] == 1:
                if t < self.NumTimePeriods:
                    stop_vars = linear_vars + [self.model._UnitStop[g, t + 1]]
                    stop_coefs = linear_coefs + [
                        self.MaxPowerOutput[g, t] - self.ShutdownRampLimit[g]]

                    power_stoplimit_constrs[g, t] = LinExpr(
                        stop_coefs, stop_vars) <= 0

                power_startlimit_constrs[g, t] = LinExpr(
                    start_coefs, start_vars) <= 0

            # _MLR_generation_limits (w/o uptime-1 generators) #
            else:
                if t < self.NumTimePeriods:
                    linear_vars += [self.model._UnitStop[g, t + 1]]
                    linear_coefs += [self.MaxPowerOutput[g, t]
                                     - self.ShutdownRampLimit[g]]

                power_startstoplimit_constrs[g, t] = LinExpr(
                    linear_coefs, linear_vars) <= 0

        self.model._power_limit_from_start = self.model.addConstrs(
            constr_genr(power_startlimit_constrs),
            name='_power_limit_from_start_mlr'
            )

        self.model._power_limit_from_stop = self.model.addConstrs(
            constr_genr(power_stoplimit_constrs),
            name='_power_limit_from_stop_mlr'
            )

        self.model._power_limit_from_startstop = self.model.addConstrs(
            constr_genr(power_startstoplimit_constrs),
            name='_power_limit_from_startstop_mlr'
            )

    def ca_production_costs(self):
        self.model._PiecewiseProduction = self.model.addVars(
            self.prod_indices,
            lb=0., ub=[self.PiecewiseGenerationPoints[g][i + 1]
                       - self.PiecewiseGenerationPoints[g][i]
                       for g, t, i in self.prod_indices],
            name='PiecewiseProduction'
            )

        self.model._PiecewiseProductionSum = self.model.addConstrs(
            (self.piecewise_production_sum_rule(g, t)
             for g, t, _ in self.prod_indices), name='PiecewiseProductionSum'
            )

        self.model._ProductionCost = self.model.addVars(
            *self.thermal_periods, lb=0, ub=GRB.INFINITY,
            name='ProductionCost'
            )

        self.model._ProductionCostConstr = self.model.addConstrs(
            (self.piecewise_production_costs_rule(g, t)
             for g, t in product(*self.thermal_periods)),
            name='ProductionCostConstr'
            )

    def mlr_startup_costs(self, relax_binaries):
        vtype = GRB.CONTINUOUS if relax_binaries else GRB.BINARY

        self.model._delta = self.model.addVars(
            self.startup_costs_indices, vtype=vtype, name='delta')

        self.model._delta_eq = self.model.addConstrs(
            (LinExpr([1.] * len(self.StartupCostIndices[g]) + [-1],
                     [self.model._delta[g, s, t]
                      for s in self.StartupCostIndices[g]]
                     + [self.model._UnitStart[g, t]]) == 0

             for g, t in product(*self.thermal_periods)),
            name='delta_eq'
            )

        lag_constr_indx = list()
        lag_constrs = dict()

        for g, s, t in self.startup_costs_indices:
            if s < len(self.StartupCostIndices[g]) - 1:
                this_lag = self.StartupCostIndices[g][s]
                next_lag = self.StartupCostIndices[g][s + 1]

                if next_lag + self.UnitOnT0State[g] < t < next_lag:
                    self.model._delta[g, s, t].lb = 0
                    self.model._delta[g, s, t].ub = 0

                elif t >= next_lag:
                    lags = list(range(this_lag, next_lag))
                    lag_constr_indx.append((g, s, t))

                    lag_constrs[g, s, t] = LinExpr(
                        [-1.] * len(lags) + [1.],
                        [self.model._UnitStop[g, t - l] for l in lags]
                        + [self.model._delta[g, s, t]]
                        ) <= 0

        self.model._delta_ineq = self.model.addConstrs(
            (lag_constrs[g, s, t] for g, s, t in lag_constr_indx),
            name='delta_ineq'
            )

        self.model._StartupCost = self.model.addVars(
            *self.thermal_periods, name='StartupCost')

        self.model._ComputeStartupCosts = self.model.addConstrs(
            (LinExpr([self.StartupCosts[g][s]
                      for s in self.StartupCostIndices[g]] + [-1],
                     [self.model._delta[g, s, t]
                      for s in self.StartupCostIndices[g]]
                     + [self.model._StartupCost[g, t]]) == 0

             for g, t in product(*self.thermal_periods)),
            name='ComputeStartupCosts'
            )

    def mlr_reserve_constraints(self):
        self._add_reserve_shortfall()

        self.model._EnforceReserveRequirements = self.model.addConstrs(
            (LinExpr([1.] * (len(self.ThermalGenerators) + 1),
                     [self.model._ReserveProvided[g, t]
                      for g in self.ThermalGenerators]
                     + [self.model._ReserveShortfall[t]])
             >= self.ReserveReqs[t]

             for t in self.TimePeriods),
            name='EnforceReserveRequirements'
            )

    @property
    def binary_variables(self):
        return super().binary_variables + [self.model._delta]

    def generate(self,
                 relax_binaries: bool, ptdf_options: dict,
                 ptdf, objective_hours: int) -> None:

        for gen in self.ThermalGenerators:
            self.ShutdownRampLimit[gen] = 1. + self.MinPowerOutput[
                gen, self.InitialTime]

        self._initialize_model(ptdf, ptdf_options)

        self.garver_3bin_vars(relax_binaries)
        self.garver_power_vars()
        self.mlr_reserve_vars()
        self.file_non_dispatchable_vars()

        self.mlr_generation_limits()
        self.damcikurt_ramping()
        self.ca_production_costs()

        self.rajan_takriti_ut_dt()
        self.mlr_startup_costs(relax_binaries)

        self.ptdf_power_flow()
        self.mlr_reserve_constraints()

        self.add_objective()
        self._add_zero_cost_hours(objective_hours)

        self.model.update()
