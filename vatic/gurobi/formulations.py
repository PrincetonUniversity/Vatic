"""Representations of grid states used as optimization model input/output."""

from __future__ import annotations

import time
import math
import pandas as pd
from abc import ABC, abstractmethod
from itertools import product
from typing import Optional, Any

import gurobipy as gp
from gurobipy import GRB, tupledict, LinExpr, quicksum
from ordered_set import OrderedSet
from .models_gurobi import ptdf_utils_gurobi as ptdf_utils


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

    def garver_3bin_vars(self, model, relax_binaries):
        vtype = GRB.CONTINUOUS if relax_binaries else GRB.BINARY

        model._UnitOn = model.addVars(
            *self.thermal_periods, lb=0, ub=1, vtype=vtype, name='UnitOn')
        model._UnitStart = model.addVars(
            *self.thermal_periods, lb=0, ub=1, vtype=vtype, name='UnitStart')
        model._UnitStop = model.addVars(
            *self.thermal_periods, lb=0, ub=1, vtype=vtype, name='UnitStop')

        return model

    def garver_power_vars(self, model):
        model._PowerGeneratedAboveMinimum = model.addVars(
            *self.thermal_periods,
            lb=0, ub={(g, t): (self.MaxPowerOutput[g, t]
                               - self.MinPowerOutput[g, t])
                      for g, t in product(*self.thermal_periods)},
            name='PowerGeneratedAboveMinimum'
            )

        model._PowerGenerated = tupledict({
            (g, t): (model._PowerGeneratedAboveMinimum[g, t]
                     + self.MinPowerOutput[g, t] * model._UnitOn[g, t])
            for g, t in product(*self.thermal_periods)
            })

        model._PowerGeneratedStartupShutdown = tupledict({
            (g, t): self._add_power_generated_startup_shutdown(model, g, t)
            for g, t in product(*self.thermal_periods)
            })

        return model

    def damcikurt_ramping(self, model):
        # enforce_max_available_ramp_up_rates_rule #
        t0_upower_constrs = dict()
        tk_upower_constrs = dict()

        t0_uramp_periods = {
            (g, t) for g, t in product(*self.thermal_periods)
            if (t == self.InitialTime and self.UnitOnT0[g]
                and self.NominalRampUpLimit[g] < (self.MaxPowerOutput[g, t]
                                                  - self.PowerGeneratedT0[g]))
            }

        for g, t in t0_uramp_periods:
            power_vars, power_coefs = self._get_generation_above_minimum_lists(
                model, g, t)

            power_vars += [model._UnitOn[g, t], model._UnitStart[g, t]]
            power_coefs += [-self.NominalRampUpLimit[g]
                            + self.MinPowerOutput[g, t],
                            -self.StartupRampLimit[g]
                            + self.NominalRampUpLimit[g]]

            t0_upower_constrs[g, t] = LinExpr(
                power_coefs, power_vars) <= self.PowerGeneratedT0[g]

        tk_uramp_periods = {
            (g, t) for g, t in product(*self.thermal_periods)
            if (t > self.InitialTime
                and self.NominalRampUpLimit[g] < (
                        self.MaxPowerOutput[g, t]
                        - self.MinPowerOutput[g, t - 1]
                ))
            }

        for g, t in tk_uramp_periods:
            power_vars, power_coefs = self._get_generation_above_minimum_lists(
                model, g, t)

            power_vars += [model._UnitOn[g, t], model._UnitStart[g, t]]
            power_coefs += [-self.NominalRampUpLimit[g]
                            - self.MinPowerOutput[g, t - 1]
                            + self.MinPowerOutput[g, t],
                            -self.StartupRampLimit[g]
                            + self.MinPowerOutput[g, t - 1]
                            + self.NominalRampUpLimit[g]]

            neg_vars, neg_coefs = self._get_generation_above_minimum_lists(
                model, g, t - 1, negative=True)

            tk_upower_constrs[g, t] = LinExpr(
                power_coefs + neg_coefs, power_vars + neg_vars) <= 0.

        model._EnforceMaxAvailableT0RampUpRates = model.addConstrs(
            constr_genr(t0_upower_constrs),
            name='enforce_max_available_t0_ramp_up_rates'
            )
        model._EnforceMaxAvailableTkRampUpRates = model.addConstrs(
            constr_genr(tk_upower_constrs),
            name='enforce_max_available_tk_ramp_up_rates'
            )

        # enforce_ramp_down_limits_rule #
        t0_dpower_constrs = dict()
        tk_dpower_constrs = dict()

        t0_dramp_periods = {
            (g, t) for g, t in product(*self.thermal_periods)
            if (t == self.InitialTime and self.UnitOnT0[g]
                and (self.NominalRampDownLimit[g] < (
                            self.PowerGeneratedT0[g]
                            - self.MinPowerOutput[g, t]
                    )
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
                model, g, t)

            power_vars += [model._UnitStop[g, t]]
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
                model, g, t)

            power_vars += [model._UnitOn[g, t - 1], model._UnitStop[g, t]]
            power_coefs += [self.NominalRampDownLimit[g]
                            + self.MinPowerOutput[g, t]
                            - self.MinPowerOutput[g, t - 1],
                            self.ShutdownRampLimit[g]
                            - self.MinPowerOutput[g, t]
                            - self.NominalRampDownLimit[g]]

            neg_vars, neg_coefs = self._get_generation_above_minimum_lists(
                model, g, t - 1, negative=True)

            tk_dpower_constrs[g, t] = LinExpr(
                power_coefs + neg_coefs, power_vars + neg_vars) >= 0

        model._EnforceScaledT0NominalRampDownLimits = model.addConstrs(
            constr_genr(t0_dpower_constrs),
            name='enforce_max_available_t0_ramp_down_rates'
            )
        model._EnforceScaledTkNominalRampDownLimits = model.addConstrs(
            constr_genr(tk_dpower_constrs),
            name='enforce_max_available_tk_ramp_down_rates'
            )

        return model

    def piecewise_production_sum_rule(self, model, g, t):
        linear_vars = list(model._PiecewiseProduction[g, t, i] for i in range(
            len(self.PiecewiseGenerationPoints[g]) - 1))

        linear_coefs = [1.] * len(linear_vars)
        linear_vars.append(model._PowerGeneratedAboveMinimum[g, t])
        linear_coefs.append(-1.)

        return LinExpr(linear_coefs, linear_vars) == 0

    def piecewise_production_limits_rule(self, model, g, t, i, tightened=True):
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
                linear_vars = [model._PiecewiseProduction[g, t, i],
                               model._UnitOn[g, t],
                               model._UnitStart[g, t],
                               model._UnitStop[g, t + 1]]
                linear_coefs = [-1., (upper - lower), -su_step, -sd_step]

                return LinExpr(linear_coefs, linear_vars) >= 0

            # MinimumUpTime[g] <= 1
            else:
                linear_vars = [model._PiecewiseProduction[g, t, i],
                               model._UnitOn[g, t],
                               model._UnitStart[g, t], ]
                linear_coefs = [-1., (upper - lower), -su_step, ]

                if tightened:
                    coef = -max(sd_step - su_step, 0)

                    if coef != 0:
                        linear_vars.append(model._UnitStop[g, t + 1])
                        linear_coefs.append(coef)

                return LinExpr(linear_coefs, linear_vars) >= 0

        # t >= value(m.NumTimePeriods)
        else:
            linear_vars = [model._PiecewiseProduction[g, t, i],
                           model._UnitOn[g, t],
                           model._UnitStart[g, t], ]
            linear_coefs = [-1., (upper - lower), -su_step, ]

            return LinExpr(linear_coefs, linear_vars) >= 0

    def piecewise_production_limits_rule2(self,
                                          model, g, t, i, tightened=True):
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
            linear_vars = [model._PiecewiseProduction[g, t, i],
                           model._UnitOn[g, t],
                           model._UnitStop[g, t + 1], ]
            linear_coefs = [-1., (upper - lower), -sd_step, ]

            if tightened:
                SU = self.StartupRampLimit[g]
                su_step = _step_coeff(upper, lower, SU - minP)
                coef = -max(su_step - sd_step, 0)

                if coef != 0:
                    linear_vars.append(model._UnitStart[g, t])
                    linear_coefs.append(coef)

            return LinExpr(linear_coefs, linear_vars) >= 0

        ## MinimumUpTime[g] > 1 or we added it in the
        # t == value(m.NumTimePeriods) clause above
        else:
            return None

    def uptime_rule(self, model, g, t):
        linear_vars = [model._UnitStart[g, tk]
                       for tk in range(t - self.ScaledMinUpTime[g] + 1, t + 1)]
        linear_coefs = [1.] * len(linear_vars)

        linear_vars += [model._UnitOn[g, t]]
        linear_coefs += [-1]

        return LinExpr(linear_coefs, linear_vars) <= 0

    def downtime_rule(self, model, g, t):
        linear_vars = [model._UnitStop[g, tk]
                       for tk in range(t - self.ScaledMinDownTime[g] + 1,
                                       t + 1)]
        linear_coefs = [1.] * len(linear_vars)

        linear_vars += [model._UnitOn[g, t]]
        linear_coefs += [-1]

        return LinExpr(linear_coefs, linear_vars) <= 1

    def logical_rule(self, model, g, t):
        if t == self.InitialTime:
            linear_vars = [model._UnitOn[g, t], model._UnitStart[g, t],
                           model._UnitStop[g, t]]
            linear_coefs = [1., -1., 1.]

            return LinExpr(linear_coefs, linear_vars) <= self.UnitOnT0[g]

        else:
            linear_vars = [model._UnitOn[g, t], model._UnitOn[g, t - 1],
                           model._UnitStart[g, t],
                           model._UnitStop[g, t]]
            linear_coefs = [1., -1, -1., 1.]

            return LinExpr(linear_coefs, linear_vars) == 0

    def get_hot_startup_pairs(self, model, g):
        ## for speed, if we don't have different startups

        if len(self.StartupLags[g]) <= 1:
            return []

        first_lag = self.StartupLags[g][0]
        last_lag = self.StartupLags[g][-1]
        init_time = self.TimePeriods[0]
        after_last_time = self.TimePeriods[-1] + 1

        for t_prime in model._ValidShutdownTimePeriods[g]:
            t_first = first_lag + t_prime
            t_last = last_lag + t_prime

            if t_first < init_time:
                t_first = init_time

            if t_last > after_last_time:
                t_last = after_last_time

            for t in range(t_first, t_last):
                yield t_prime, t

    def shutdown_match_rule(self, model, begin_times, g, t):
        init_time = self.TimePeriods[0]

        linear_vars = [model._StartupIndicator[g, t, t_p]
                       for t_p in begin_times[g, t]]
        linear_coefs = [1.] * len(linear_vars)

        if t < init_time:
            return LinExpr(linear_coefs, linear_vars) <= 1

        else:
            linear_vars.append(model._UnitStop[g, t])
            linear_coefs.append(-1.)

            return LinExpr(linear_coefs, linear_vars) <= 0

    def compute_startup_cost_rule(self, model, g, t):
        startup_lags = self.StartupLags[g]
        startup_costs = self.StartupCosts[g]
        last_startup_cost = startup_costs[-1]

        linear_vars = [model._StartupCost[g, t], model._UnitStart[g, t]]
        linear_coefs = [-1., last_startup_cost]

        for tp in model._ShutdownsByStartups[g, t]:
            for s in self.StartupCostIndices[g]:
                this_lag = startup_lags[s]
                next_lag = startup_lags[s + 1]

                if this_lag <= t - tp < next_lag:
                    linear_vars.append(model._StartupIndicator[g, tp, t])
                    linear_coefs.append(startup_costs[s] - last_startup_cost)

                    break

        return LinExpr(linear_coefs, linear_vars) == 0

    def _get_power_generated_lists(self, model, g, t):
        return ([model._PowerGeneratedAboveMinimum[g, t], model._UnitOn[g, t]],
                [1., self.MinPowerOutput[g, t]])

    def _get_max_power_available_lists(self, model, g, t):
        return ([model._MaximumPowerAvailableAboveMinimum[g, t],
                 model._UnitOn[g, t]],
                [1., self.MinPowerOutput[g, t]])

    @abstractmethod
    def _get_generation_above_minimum_lists(self, model, g, t, negative=False):
        pass

    def _get_initial_max_power_lists(self, model, g, t):
        linear_vars, linear_coefs = self._get_power_generated_lists(
            model, g, t)

        linear_vars.append(model._UnitOn[g, t])
        linear_coefs.append(-self.MaxPowerOutput[g, t])

        return linear_vars, linear_coefs

    def _get_initial_max_power_available_lists(self, model, g, t):
        linear_vars, linear_coefs = self._get_max_power_available_lists(
            model, g, t)

        linear_vars.append(model._UnitOn[g, t])
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

    def _add_power_generated_startup_shutdown(self, model, g, t):
        linear_vars = [model._PowerGeneratedAboveMinimum[g, t],
                       model._UnitOn[g, t]]
        linear_coefs = [1., self.MinPowerOutput[g, t]]

        # first, discover if we have startup/shutdown
        # curves in the model
        model_has_startup_shutdown_curves = False
        for s in model._StartupCurve.values():
            if len(s) > 0:
                model_has_startup_shutdown_curves = True
                break

        if not model_has_startup_shutdown_curves:
            for s in model._ShutdownCurve.values():
                if len(s) > 0:
                    model_has_startup_shutdown_curves = True
                    break

        if model_has_startup_shutdown_curves:
            # check the status vars to see if we're compatible
            # with startup/shutdown curves
            if model._status_vars not in ['garver_2bin_vars', 'garver_3bin_vars', 'garver_3bin_relaxed_stop_vars',
                                          'ALS_state_transition_vars']:
                raise RuntimeError(
                    f"Status variable formulation {model._status_vars} is not compatible with startup or shutdown curves")

            startup_curve = model._StartupCurve[g]
            shutdown_curve = model._ShutdownCurve[g]
            time_periods_before_startup = model._TimePeriodsBeforeStartup[g]
            time_periods_since_shutdown = model._TimePeriodsSinceShutdown[g]

            future_startup_past_shutdown_production = 0.
            future_startup_power_index = time_periods_before_startup + model._NumTimePeriods - t
            if future_startup_power_index <= len(startup_curve) - 1:
                future_startup_past_shutdown_production += startup_curve[future_startup_power_index]

            past_shutdown_power_index = time_periods_since_shutdown + t
            if past_shutdown_power_index <= len(shutdown_curve) - 1:
                future_startup_past_shutdown_production += shutdown_curve[past_shutdown_power_index]

            linear_vars, linear_coefs = model._get_power_generated_lists(model, g, t)
            for startup_idx in range(1, min(len(startup_curve), model._NumTimePeriods + 1 - t)):
                linear_vars.append(model._UnitStart[g, t + startup_idx])
                linear_coefs.append(startup_curve[startup_idx])
            for shutdown_idx in range(1, min(len(shutdown_curve), t + 1)):
                linear_vars.append(model._UnitStop[g, t - shutdown_idx + 1])
                linear_coefs.append(shutdown_curve[shutdown_idx])
            return LinExpr(linear_coefs, linear_vars) + future_startup_past_shutdown_production

            ## if we're here, then we can use 1-bin models
            ## and no need to do the additional work
        return LinExpr(linear_coefs, linear_vars)

    def _add_reserve_shortfall(self, model):
        # add_reserve_shortfall (fixed=False) #

        model._ReserveShortfall = model.addVars(
            self.TimePeriods, lb=0, ub=[self.ReserveReqs[t]
                                        for t in self.TimePeriods],
            name='ReserveShortfall'
            )

        return model

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

        # set aside a proportion of the total demand as the model's reserve
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
            self.MinPowerOutput[g, t] = 0.

            if g in self.RenewOutput.index:
                self.MaxPowerOutput[g, t] = self.RenewOutput.loc[g, t]
            else:
                self.MaxPowerOutput[g, t] = 0.

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

        self.UnitOnT0State = grid_template['UnitOnT0State']
        self.UnitOnT0 = tupledict({
            g: init_st > 0. for g, init_st in self.UnitOnT0State.items()})
        self.PowerGeneratedT0 = grid_template['PowerGeneratedT0']

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
        self.model = None
        self.solve_time = None
        self.results = None

    def get_commitments(self, gen: str) -> list[bool]:
        return [self.FixedCommitment[gen, t] for t in self.TimePeriods]

    def _initialize_model(self, ptdf, ptdf_options) -> gp.Model:
        model = gp.Model(self.model_name)
        model._ptdf_options = self.DEFAULT_PTDF_OPTIONS

        model._CommitmentTimeInStage = {
            'Stage_1': self.TimePeriods, 'Stage_2': list()}
        model._GenerationTimeInStage = {
            'Stage_1': list(), 'Stage_2': self.TimePeriods}

        model._ptdf_options = ptdf_options
        if ptdf:
            self.PTDF = ptdf

        model._StartupCurve = {g: tuple() for g in self.ThermalGenerators}
        model._ShutdownCurve = {g: tuple() for g in self.ThermalGenerators}
        model._LoadMismatchPenalty = 1e4
        model._ReserveShortfallPenalty = 1e3

        return model

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

    @abstractmethod
    def generate(self,
                 relax_binaries: bool, ptdf_options,
                 ptdf, objective_hours: int) -> None:
        pass

    def solve(self, relaxed, mipgap, threads, outputflag):
        self.model.Params.OutputFlag = outputflag
        self.model.Params.MIPGap = mipgap
        self.model.Params.Threads = threads

        start_time = time.time()
        self.model.optimize()
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
                (g, t): self.model._PowerGeneratedStartupShutdown[g, t].getValue()
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

            'load_shedding': {(b, t): self.model._LoadShedding[b, t].x
                              for b in self.Buses for t in self.TimePeriods},
            'over_generation': {(b, t): self.model._OverGeneration[b, t].x
                                for b in self.Buses for t in self.TimePeriods},
            'reserve_shortfall': {t: self.model._ReserveShortfall[t].x
                                  for t in self.TimePeriods},

            }

        pg = pd.Series(results['power_generated']).unstack()

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
            cur_ptdf = cur_block._PTDF
            pfv, pfv_i, va = cur_ptdf.calculate_PFV(cur_block)

            for i, br in enumerate(cur_ptdf.branches_keys):
                flows[t, br] = pfv[i]

            for i, bs in enumerate(cur_ptdf.buses_keys):
                voltage_angles[t, bs] = va[i]

                if (bs, t) in self.model._LoadGenerateMismatch:
                    p_balances[bs, t] \
                        = self.model._LoadGenerateMismatch[bs, t].getValue()
                else:
                    p_balances[bs, t] = 0.

                pl_dict[bs, t] = self.model._TransmissionBlock[t]._pl[bs]

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
            raise VaticModelError("Cannot retrieve transmission line flows "
                                 "until model has been generated and solved!")

        return pd.Series(self.results['flows']).unstack()

    def file_non_dispatchable_vars(self, model: gp.Model) -> gp.Model:
        model._RenewablePowerUsed = model.addVars(
            *self.renew_periods, lb=self.MinPowerOutput,
            ub=self.MaxPowerOutput, name='NondispatchablePowerUsed'
            )

        return model

    def piecewise_production_costs_rule(self, model, g, t):
        if (g, t, 0) in self.prod_indices:
            points = self.PiecewiseGenerationPoints[g]
            costs = self.PiecewiseGenerationCosts[g]

            linear_coefs = [(costs[i + 1] - costs[i])
                            / (points[i + 1] - points[i])
                            for i in range(len(points) - 1)]
            linear_vars = [model._PiecewiseProduction[g, t, i]
                           for i in range(len(points) - 1)]

            linear_coefs.append(-1.)
            linear_vars.append(model._ProductionCost[g, t])

            return LinExpr(linear_coefs, linear_vars) == 0

        else:
            return model._ProductionCost[g, t] == 0

    def rajan_takriti_ut_dt(self, model):
        for g, t in product(*self.thermal_periods):
            if self.FixedCommitment[g, t] is not None:
                model._UnitOn[g, t].lb = self.FixedCommitment[g, t]
                model._UnitOn[g, t].ub = self.FixedCommitment[g, t]

        for g in self.ThermalGenerators:
            if self.InitTimePeriodsOnline[g] != 0:
                for t in range(self.TimePeriods[0],
                               self.InitTimePeriodsOnline[g]
                               + self.TimePeriods[0]):
                    model._UnitOn[g, t].ub = 1
                    model._UnitOn[g, t].lb = 1

            if self.InitTimePeriodsOffline[g] != 0:
                for t in range(self.TimePeriods[0],
                               self.InitTimePeriodsOffline[g]
                               + self.TimePeriods[0]):
                    model._UnitOn[g, t].ub = 0
                    model._UnitOn[g, t].lb = 0

        uptime_periods = [(g, t) for g, t in product(*self.thermal_periods)
                          if t >= self.ScaledMinUpTime[g]]
        downtime_periods = [(g, t) for g, t in product(*self.thermal_periods)
                            if t >= self.ScaledMinDownTime[g]]

        model._UpTime = model.addConstrs((self.uptime_rule(model, g, t)
                                          for g, t in uptime_periods),
                                         name='UpTime')
        model._DownTime = model.addConstrs((self.downtime_rule(model, g, t)
                                            for g, t in downtime_periods),
                                           name='DownTime')

        return model

    def ptdf_power_flow(self, model):
        over_gen_maxes = {}
        over_gen_times_per_bus = {b: list() for b in self.Buses}
        load_shed_maxes = {}
        load_shed_times_per_bus = {b: list() for b in self.Buses}

        for b in self.Buses:

            # storage, for now, does not have time-varying parameters
            storage_max_injections = 0.
            storage_max_withdraws = 0.

            for s in self.StorageAtBus[b]:
                storage_max_injections += model._MaximumPowerOutputStorage[s]
                storage_max_withdraws += model._MaximumPowerInputStorage[s]

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
                else:
                    over_gen_maxes[b, t] = GRB.INFINITY

                if max_withdrawls > 0:
                    load_shed_maxes[b, t] = max_withdrawls
                    load_shed_times_per_bus[b].append(t)
                else:
                    load_shed_maxes[b, t] = GRB.INFINITY

        model._OverGenerationBusTimes = list(over_gen_maxes.keys())
        model._LoadSheddingBusTimes = list(load_shed_maxes.keys())

        model._OverGeneration = model.addVars(
            model._OverGenerationBusTimes,
            lb=0, ub=[over_gen_maxes[k]
                      for k in model._OverGenerationBusTimes],
            name='OverGeneration'
            )

        model._LoadShedding = model.addVars(
            model._LoadSheddingBusTimes,
            lb=0, ub=[load_shed_maxes[key]
                      for key in model._LoadSheddingBusTimes],
            name='LoadShedding'
            )

        # the following constraints are necessarily, at least in the case of
        # CPLEX 12.4, to prevent the appearance of load generation mismatch
        # component values in the range of *negative* e-5. what these small
        # negative values do is to cause the optimal objective to be a very
        # large negative, due to obviously large penalty values for under or
        # over-generation. JPW would call this a heuristic at this point, but
        # it does seem to work broadly. we tried a single global constraint,
        # across all buses, but that failed to correct the problem, and caused
        # the solve times to explode.

        for b in self.Buses:
            if load_shed_times_per_bus[b]:
                linear_vars = list(model._LoadShedding[b, t]
                                   for t in load_shed_times_per_bus[b])
                linear_coefs = [1.] * len(linear_vars)

                model.addConstr(LinExpr(linear_coefs, linear_vars) >= 0,
                                name=f"PosLoadGenerateMismatchTolerance[{b}]")

            if over_gen_times_per_bus[b]:
                linear_vars = list(model._OverGeneration[b, t]
                                   for t in over_gen_times_per_bus[b])
                linear_coefs = [1.] * len(linear_vars)

                model.addConstr(LinExpr(linear_coefs, linear_vars) >= 0,
                                name=f"NegLoadGenerateMismatchTolerance[{b}]")

        #####################################################
        # load "shedding" can be both positive and negative #
        #####################################################
        model._LoadGenerateMismatch = {(b, t): 0. for b in self.Buses
                                       for t in self.TimePeriods}

        for b, t in model._LoadSheddingBusTimes:
            model._LoadGenerateMismatch[b, t] += model._LoadShedding[b, t]
        for b, t in model._OverGenerationBusTimes:
            model._LoadGenerateMismatch[b, t] -= model._OverGeneration[b, t]

        model._LoadMismatchCost = {}
        for t in self.TimePeriods:
            model._LoadMismatchCost[t] = 0.

        for b, t in model._LoadSheddingBusTimes:
            model._LoadMismatchCost[t] += (model._LoadMismatchPenalty
                                           * model._LoadShedding[b, t])

        for b, t in model._OverGenerationBusTimes:
            model._LoadMismatchCost[t] += (model._LoadMismatchPenalty
                                           * model._OverGeneration[b, t])

        # for interface violation costs at a time step
        model._BranchViolationCost = {t: 0 for t in self.TimePeriods}

        # for interface violation costs at a time step
        model._InterfaceViolationCost = {t: 0 for t in self.TimePeriods}

        # for contingency violation costs at a time step
        model._ContingencyViolationCost = {t: 0 for t in self.TimePeriods}

        # set up the empty model block for each time period to add constraints
        model._TransmissionBlock = {}
        block = gp.Model()
        block._gens_by_bus = {bus: [bus] for bus in self.Buses}
        parent_model = model

        for tm in self.TimePeriods:
            block._tm = tm
            block._pg = dict()

            for b in self.Buses:
                start_shut = quicksum(
                    model._PowerGeneratedStartupShutdown[g, tm]
                    for g in self.ThermalGeneratorsAtBus[b]
                    )

                out_store = quicksum(model._PowerOutputStorage[s, tm]
                                     for s in self.StorageAtBus[b])
                in_store = quicksum(model._PowerInputStorage[s, tm]
                                    for s in self.StorageAtBus[b])

                non_dispatch = quicksum(
                    model._RenewablePowerUsed[g, tm]
                    for g in self.NondispatchableGeneratorsAtBus[b]
                    )

                block._pg[b] = (start_shut
                                + out_store - in_store + non_dispatch
                                + model._LoadGenerateMismatch[b, tm])

            self._ptdf_dcopf_network_model(block, tm, parent_model)
            model._TransmissionBlock[tm] = block

        return model

    def _ptdf_dcopf_network_model(self, block, tm, parent_model):
        model = parent_model

        bus_loads = {b: self.Demand.loc[b, tm] for b in self.Buses}
        block._pl = bus_loads

        block._branches_inservice = tuple(
            line for line in self.TransmissionLines
            if not self.LineOutOfService[line]
            )

        block._p_nw_tm = model.addVars(self.Buses,
                                       lb=-GRB.INFINITY, ub=GRB.INFINITY,
                                       name=f'p_nw_{tm}')

        # declare_eq_p_net_withdraw_at_bus  (dc...branches = None) #
        for b in self.Buses:
            parent_model.addConstr(
                (block._p_nw_tm[b] == (
                        (block._pl[b] if bus_loads[b] != 0.0 else 0.0)
                        - quicksum(block._pg[g]
                                   for g in block._gens_by_bus[b]))),
                name=f"_eq_p_net_withdraw_at_bus[{b}]_at_period[{block._tm}]"
                )

        p_expr = quicksum(block._pg[g] for b in self.Buses
                          for g in block._gens_by_bus[b])
        p_expr -= quicksum(block._pl[b] for b in self.Buses
                           if bus_loads[b] is not None)

        parent_model.addConstr(
            (p_expr == 0.0), name=f"eq_p_balance_at_period{block._tm}")

        if not self.PTDF:
            self.PTDF = ptdf_utils.VirtualPTDFMatrix(
                self.branches, self.buses, self.ReferenceBus,
                ptdf_utils.BasePointType.FLATSTART, self.ptdf_options,
                branches_keys=block._branches_inservice,
                buses_keys=self.Buses
                )

        block._PTDF = self.PTDF

        return parent_model

    def add_objective(self, model):
        model._NoLoadCost = tupledict({
            (g, t): self.MinProductionCost[g] * model._UnitOn[g, t]
            for g, t in product(*self.thermal_periods)
            })

        model._TotalProductionCost = {t: sum(model._ProductionCost[g, t]
                                             for g in self.ThermalGenerators)
                                      for t in self.TimePeriods}

        model._CommitmentStageCost = {
            st: sum(sum(model._NoLoadCost[g, t] + model._StartupCost[g, t]
                        for g in self.ThermalGenerators)
                    for t in model._CommitmentTimeInStage[st])
            for st in self.StageSet
            }

        model._ReserveShortfallCost = {
            t: model._ReserveShortfallPenalty * model._ReserveShortfall[t]
            for t in self.TimePeriods
            }

        model._GenerationStageCost = {
            st: sum(sum(model._ProductionCost[g, t]
                        for g in self.ThermalGenerators)
                    for t in model._GenerationTimeInStage[st])
                + sum(model._LoadMismatchCost[t]
                      for t in model._GenerationTimeInStage[st])
                + sum(model._ReserveShortfallCost[t]
                      for t in model._GenerationTimeInStage[st])
                + sum(model._StorageCost[s, t] for s in self.Storage
                      for t in model._GenerationTimeInStage[st])
            for st in self.StageSet
            }

        model._StageCost = {
            st: model._GenerationStageCost[st] + model._CommitmentStageCost[st]
            for st in self.StageSet
            }

        model.setObjective(quicksum(model._StageCost[st]
                                    for st in self.StageSet),
                           GRB.MINIMIZE)

        return model

    def _add_zero_cost_hours(self, model, objective_hours):
        if objective_hours:
            zero_cost_hours = self.TimePeriods.copy()

            for i, t in enumerate(self.TimePeriods):
                if i < objective_hours:
                    zero_cost_hours.remove(t)
                else:
                    break

            cost_gens = {g for g, _ in model._ProductionCost}
            for t in zero_cost_hours:
                for g in cost_gens:
                    model.remove(model._ProductionCostConstr[g, t])
                    model._ProductionCost[g, t].lb = 0.
                    model._ProductionCost[g, t].ub = 0.

        return model

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

    def _get_generation_above_minimum_lists(self, model, g, t, negative=False):
        linear_vars = [model._PowerGeneratedAboveMinimum[g, t]]
        linear_coefs = [-1.] if negative else [1.]

        return linear_vars, linear_coefs

    def garver_power_avail_vars(self, model):
        model._MaximumPowerAvailableAboveMinimum = model.addVars(
            *self.thermal_periods,
            lb=0, ub={(g, t): (self.MaxPowerOutput[g, t]
                               - self.MinPowerOutput[g, t])
                      for g, t in product(*self.thermal_periods)},
            name='MaximumPowerAvailableAboveMinimum'
            )

        model._MaximumPowerAvailable = tupledict({
            (g, t): (model._MaximumPowerAvailableAboveMinimum[g, t]
                     + self.MinPowerOutput[g, t] * model._UnitOn[g, t])
            for g, t in product(*self.thermal_periods)
            })

        model._ReserveProvided = tupledict({
            (g, t): (model._MaximumPowerAvailableAboveMinimum[g, t]
                     - model._PowerGeneratedAboveMinimum[g, t])
            for g, t in product(*self.thermal_periods)
            })

        model._EnforceGeneratorOutputLimitsPartB = model.addConstrs(
            ((model._PowerGeneratedAboveMinimum[g, t]
              - model._MaximumPowerAvailableAboveMinimum[g, t] <= 0)
             for g, t in product(*self.thermal_periods)),
            name='EnforceGeneratorOutputLimitsPartB'
            )

        return model

    def pgg_KOW_gen_limits(self, model):
        power_startlimit_constrs = dict()
        power_stoplimit_constrs = dict()
        power_startstoplimit_constrs = dict()

        for g, t in product(*self.thermal_periods):
            linear_vars, linear_coefs = self._get_initial_max_power_available_lists(
                model, g, t)

            # _MLR_GENERATION_LIMITS_UPTIME_1 (tightened) #
            if self.ScaledMinUpTime[g] == 1:
                start_vars = linear_vars + [model._UnitStart[g, t]]
                start_coefs = linear_coefs + [self.MaxPowerOutput[g, t]
                                              - self.StartupRampLimit[g]]

                if t < self.NumTimePeriods:
                    startramp_coef = (self.StartupRampLimit[g]
                                      - self.ShutdownRampLimit[g])

                    if startramp_coef > 0:
                        start_vars += [model._UnitStop[g, t + 1]]
                        start_coefs += [startramp_coef]

                    stop_vars = linear_vars + [model._UnitStop[g, t + 1]]
                    stop_coefs = linear_coefs + [
                        self.MaxPowerOutput[g, t] - self.ShutdownRampLimit[g]]

                    stopramp_coef = (self.ShutdownRampLimit[g]
                                     - self.StartupRampLimit[g])

                    if stopramp_coef > 0:
                        stop_vars += [model._UnitStart[g, t]]
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
                    linear_vars += [model._UnitStart[g, t - i]]

                    linear_coefs += [self.MaxPowerOutput[g, t]
                                     - self.StartupRampLimit[g]
                                     - sum(self.NominalRampUpLimit[g]
                                           for j in range(1, i + 1))]

                if t < self.NumTimePeriods:
                    linear_vars += [model._UnitStop[g, t + 1]]
                    linear_coefs += [self.MaxPowerOutput[g, t]
                                     - self.ShutdownRampLimit[g]]

                power_startstoplimit_constrs[g, t] = LinExpr(
                    linear_coefs, linear_vars) <= 0

        model._power_limit_from_start = model.addConstrs(
            constr_genr(power_startlimit_constrs),
            name='_power_limit_from_start'
            )

        model._power_limit_from_stop = model.addConstrs(
            constr_genr(power_stoplimit_constrs),
            name='_power_limit_from_stop'
            )

        model._power_limit_from_startstop = model.addConstrs(
            constr_genr(power_startstoplimit_constrs),
            name='_power_limit_from_start_stop_pan_guan_gentile'
            )

        return model

    def kow_gen_limits(self, model):
        gener_starts_constrs = dict()
        gener_startstops_constrs = dict()

        # max_power_limit_from_starts_rule #
        for g, t in product(*self.thermal_periods):
            time_ru = self._get_look_back_periods(g, t, None)

            if (t < self.NumTimePeriods
                    and time_ru > max(0, self.ScaledMinUpTime[g] - 2)):
                start_vars, start_coefs = self._get_initial_max_power_available_lists(
                    model, g, t)

                start_vars += [model._UnitOn[g, t]]
                start_coefs += [-self.MaxPowerOutput[g, t]]

                for i in range(min(time_ru, self.ScaledMinUpTime[g] - 1,
                                   t - self.InitialTime) + 1):
                    start_vars += [model._UnitStart[g, t - i]]
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
                    model, g, t)

                for i in range(sd_time_limit + 1):
                    start_vars += [model._UnitStop[g, t + i + 1]]
                    start_coefs += [
                        self.MaxPowerOutput[g, t]
                        - self.ShutdownRampLimit[g]
                        - sum(self.NominalRampDownLimit[g]
                              for j in range(1, i + 1))
                        ]

                for i in range(su_time_limit + 1):
                    start_vars += [model._UnitStart[g, t - i]]
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
                            start_vars += [
                                model._UnitStart[g, t - su_time_limit - 1]]
                            start_coefs += [coef]

                gener_startstops_constrs[g, t] = LinExpr(
                    start_coefs, start_vars) <= 0

        model._max_power_limit_from_starts = model.addConstrs(
            constr_genr(gener_starts_constrs),
            name='_max_power_limit_from_starts'
            )

        model._max_power_limit_from_start_stop = model.addConstrs(
            constr_genr(gener_startstops_constrs),
            name='_power_limit_from_start_stop_KOW'
            )

        return model

    def compute_production_costs_rule(self, model, g, t, avg_power):
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
        return sum((model._PiecewiseProductionCosts[g, t, l + 1]
                    - model._PiecewiseProductionCosts[g, t, l])
                   / (piecewise_points[l + 1] - piecewise_points[l])
                   * piecewise_eval[l] for l in range(len(piecewise_eval)))

    def kow_production_costs_tightened(self, model):
        model._PiecewiseProduction = model.addVars(
            self.prod_indices,
            lb=0., ub=[self.PiecewiseGenerationPoints[g][i + 1]
                       - self.PiecewiseGenerationPoints[g][i]
                       for g, t, i in self.prod_indices],
            name='PiecewiseProduction'
            )

        model._PiecewiseProductionSum = model.addConstrs(
            (self.piecewise_production_sum_rule(model, g, t)
             for g, t, _ in self.prod_indices), name='PiecewiseProductionSum'
            )

        model._PiecewiseProductionLimits = model.addConstrs(
            (self.piecewise_production_limits_rule(model, g, t, i)
             for g, t, i in self.prod_indices),
            name='PiecewiseProductionLimits'
            )

        limits_periods = [(g, t, i) for g, t, i in self.prod_indices
                          if self.ScaledMinUpTime[g] <= 1
                          and t < self.NumTimePeriods]

        model._PiecewiseProductionLimits2 = model.addConstrs(
            (self.piecewise_production_limits_rule2(model, g, t, i)
             for g, t, i in limits_periods),
            name='PiecewiseProductionLimits2'
            )

        model._ProductionCost = model.addVars(
            *self.thermal_periods, lb=-GRB.INFINITY, ub=GRB.INFINITY,
            name='ProductionCost'
            )

        model._ProductionCostConstr = model.addConstrs(
            (self.piecewise_production_costs_rule(model, g, t)
             for g, t in product(*self.thermal_periods)),
            name='ProductionCostConstr'
            )

        model._ComputeProductionCosts = self.compute_production_costs_rule

        return model

    def kow_startup_costs(self, model, relax_binaries):
        vtype = GRB.CONTINUOUS if relax_binaries else GRB.BINARY

        model._ValidShutdownTimePeriods = {
            g: ([] if len(self.StartupLags[g]) <= 1
                else self.TimePeriods if self.UnitOnT0State[g] >= 0
            else self.TimePeriods + [self.TimePeriods[0]
                                     + int(self.UnitOnT0State[g])])
            for g in self.ThermalGenerators
            }

        model._ShutdownHotStartupPairs = {
            g: ([] if len(self.StartupLags) <= 1
                else list(self.get_hot_startup_pairs(model, g)))
            for g in self.ThermalGenerators
            }

        model._StartupIndicator_domain = [
            (g, t_prime, t) for g in self.ThermalGenerators
            for t_prime, t in model._ShutdownHotStartupPairs[g]
            ]

        model._StartupIndicator = model.addVars(
            model._StartupIndicator_domain,
            vtype=vtype, name='StartupIndicator'
            )

        model._GeneratorShutdownPeriods = [
            (g, t) for g in self.ThermalGenerators
            for t in model._ValidShutdownTimePeriods[g]
            ]

        model._ShutdownsByStartups = {
            (g, t): [] for g, t in product(*self.thermal_periods)}
        model._StartupsByShutdowns = {
            (g, t): [] for g, t in model._GeneratorShutdownPeriods}

        for g, t_p, t in model._StartupIndicator_domain:
            model._ShutdownsByStartups[g, t] += [t_p]
            model._ShutdownsByStartups[g, t] += [t_p]
            model._StartupsByShutdowns[g, t_p] += [t]

        model._StartupMatch = model.addConstrs(
            (LinExpr([1.] * len(model._ShutdownsByStartups[g, t]) + [-1.],
                     [model._StartupIndicator[g, t_prime, t]
                      for t_prime in model._ShutdownsByStartups[g, t]]
                     + [model._UnitStart[g, t]]) <= 0
             for g, t in product(*self.thermal_periods)),
            name='StartupMatch'
            )

        begin_times = {(g, t): model._StartupsByShutdowns[g, t]
                       for g, t in model._GeneratorShutdownPeriods
                       if model._StartupsByShutdowns[g, t]}

        model._ShutdownMatch = model.addConstrs(
            (self.shutdown_match_rule(model, begin_times, g, t)
             for g, t in begin_times),
            name='ShutdownMatch'
            )

        model._StartupCost = model.addVars(
            *self.thermal_periods, lb=0, ub=GRB.INFINITY, name='StartupCost')

        model._StartupIndicator = model.addVars(
            model._StartupIndicator_domain, vtype=GRB.BINARY,
            name='StartupIndicator'
            )

        model._ComputeStartupCosts = model.addConstrs(
            (self.compute_startup_cost_rule(model, g, t)
             for g, t in product(*self.thermal_periods)),
            name='ComputeStartupCosts'
            )

        return model

    def enforce_reserve_requirements_rule(self, model, t):
        model._LoadGenerateMismatch = tupledict(model._LoadGenerateMismatch)

        linear_expr = (
                quicksum(model._MaximumPowerAvailable.select('*', t))
                + quicksum(model._RenewablePowerUsed.select('*', t))
                + quicksum(model._LoadGenerateMismatch.select('*', t))
                + quicksum(model._ReserveShortfall.select(t))
                )

        if hasattr(model, '_PowerOutputStorage'):
            linear_expr += quicksum(model._PowerOutputStorage.select('*', t))

        if hasattr(model, '_PowerInputStorage'):
            linear_expr -= quicksum(model._PowerInputStorage.select('*', t))

        return linear_expr >= (
            sum(self.Demand.loc[b, t] for b in sorted(self.Buses))
            + self.ReserveReqs[t]
            )

    def CA_reserve_constraints(self, model):
        model = self._add_reserve_shortfall(model)

        # ensure there is sufficient maximal power output available to meet
        # both the demand and the spinning reserve requirements in each time
        # period. encodes Constraint 3 in Carrion and Arroyo.

        # IMPT: In contrast to power balance, reserves are (1) not per-bus
        # and (2) expressed in terms of maximum power available, and not
        # actual power generated.
        model._EnforceReserveRequirements = model.addConstrs(
            (self.enforce_reserve_requirements_rule(model, t)
             for t in self.TimePeriods),
            name='EnforceReserveRequirements'
            )

        return model

    def generate(self,
                 relax_binaries: bool, ptdf_options: dict,
                 ptdf, objective_hours: int) -> None:
        model = self._initialize_model(ptdf, ptdf_options)

        model = self.garver_3bin_vars(model, relax_binaries)
        model = self.garver_power_vars(model)
        model = self.garver_power_avail_vars(model)
        model = self.file_non_dispatchable_vars(model)

        model = self.pgg_KOW_gen_limits(model)
        model = self.damcikurt_ramping(model)
        model = self.kow_production_costs_tightened(model)
        model = self.rajan_takriti_ut_dt(model)
        model = self.kow_startup_costs(model, relax_binaries)

        # _3bin_logic can just be written out here
        model._Logical = model.addConstrs(
            (self.logical_rule(model, g, t)
             for g, t in product(*self.thermal_periods))
            )

        model = self.ptdf_power_flow(model)
        model = self.CA_reserve_constraints(model)

        # set up objective
        model = self.add_objective(model)
        model = self._add_zero_cost_hours(model, objective_hours)

        model.update()
        self.model = model


class ScedModel(BaseModel):

    model_name = 'EconomicDispatch'

    def _get_max_power_available_lists(self, model, g, t):
        linear_vars, linear_coefs = self._get_power_generated_lists(
            model, g, t)

        linear_vars.append(model._ReserveProvided[g, t])
        linear_coefs.append(1.)

        return linear_vars, linear_coefs

    def _get_generation_above_minimum_lists(self, model, g, t, negative=False):
        linear_vars = [model._PowerGeneratedAboveMinimum[g, t]]
        linear_coefs = [-1.] if negative else [1.]

        linear_vars.append(model._ReserveProvided[g, t])
        linear_coefs.append(1.)

        return linear_vars, linear_coefs

    def mlr_reserve_vars(self, model):

        # amount of power produced by each generator above minimum,
        # at each time period. variable for reserves offered
        model._ReserveProvided = model.addVars(
            *self.thermal_periods,
            lb=0, ub=[self.MaxPowerOutput[g, t] - self.MinPowerOutput[g, t]
                      for g, t in product(*self.thermal_periods)],
            name='ReserveProvided'
            )

        model._MaximumPowerAvailableAboveMinimum = {
            (g, t): self._get_generation_above_minimum_lists(model, g, t)
            for g, t in product(*self.thermal_periods)
            }

        return model

    def mlr_generation_limits(self, model):
        power_startlimit_constrs = dict()
        power_stoplimit_constrs = dict()
        power_startstoplimit_constrs = dict()

        for g, t in product(*self.thermal_periods):
            linear_vars, linear_coefs = self._get_initial_max_power_lists(
                model, g, t)

            start_vars = linear_vars + [model._UnitStart[g, t]]
            start_coefs = linear_coefs + [self.MaxPowerOutput[g, t]
                                          - self.StartupRampLimit[g]]

            # _MLR_GENERATION_LIMITS_UPTIME_1 (tightened=False) #
            if self.ScaledMinUpTime[g] == 1:
                if t < self.NumTimePeriods:
                    stop_vars = linear_vars + [model._UnitStop[g, t + 1]]
                    stop_coefs = linear_coefs + [
                        self.MaxPowerOutput[g, t] - self.ShutdownRampLimit[g]]

                    power_stoplimit_constrs[g, t] = LinExpr(
                        stop_coefs, stop_vars) <= 0

                power_startlimit_constrs[g, t] = LinExpr(
                    start_coefs, start_vars) <= 0

            # _MLR_generation_limits (w/o uptime-1 generators) #
            else:
                if t < self.NumTimePeriods:
                    linear_vars += [model._UnitStop[g, t + 1]]
                    linear_coefs += [self.MaxPowerOutput[g, t]
                                     - self.ShutdownRampLimit[g]]

                power_startstoplimit_constrs[g, t] = LinExpr(
                    linear_coefs, linear_vars) <= 0

        model._power_limit_from_start = model.addConstrs(
            constr_genr(power_startlimit_constrs),
            name='_power_limit_from_start_mlr'
            )

        model._power_limit_from_stop = model.addConstrs(
            constr_genr(power_stoplimit_constrs),
            name='_power_limit_from_stop_mlr'
            )

        model._power_limit_from_startstop = model.addConstrs(
            constr_genr(power_startstoplimit_constrs),
            name='_power_limit_from_startstop_mlr'
            )

        return model

    def ca_production_costs(self, model):
        model._PiecewiseProduction = model.addVars(
            self.prod_indices,
            lb=0., ub=[self.PiecewiseGenerationPoints[g][i + 1]
                       - self.PiecewiseGenerationPoints[g][i]
                       for g, t, i in self.prod_indices],
            name='PiecewiseProduction'
            )

        model._PiecewiseProductionSum = model.addConstrs(
            (self.piecewise_production_sum_rule(model, g, t)
             for g, t, _ in self.prod_indices), name='PiecewiseProductionSum'
            )

        model._ProductionCost = model.addVars(*self.thermal_periods,
                                              lb=0, ub=GRB.INFINITY,
                                              name='ProductionCost')

        model._ProductionCostConstr = model.addConstrs(
            (self.piecewise_production_costs_rule(model, g, t)
             for g, t in product(*self.thermal_periods)),
            name='ProductionCostConstr'
            )

        return model

    def mlr_startup_costs(self, model, relax_binaries):
        vtype = GRB.CONTINUOUS if relax_binaries else GRB.BINARY

        model._delta = model.addVars(self.startup_costs_indices,
                                     vtype=vtype, name='delta')

        model._delta_eq = model.addConstrs(
            (LinExpr([1.] * len(self.StartupCostIndices[g]) + [-1],
                     [model._delta[g, s, t]
                      for s in self.StartupCostIndices[g]]
                     + [model._UnitStart[g, t]]) == 0

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
                    model._delta[g, s, t].lb = 0
                    model._delta[g, s, t].ub = 0

                elif t >= next_lag:
                    lags = list(range(this_lag, next_lag))
                    lag_constr_indx.append((g, s, t))

                    lag_constrs[g, s, t] = LinExpr(
                        [-1.] * len(lags) + [1.],
                        [model._UnitStop[g, t - l] for l in lags]
                        + [model._delta[g, s, t]]
                        ) <= 0

        model._delta_ineq = model.addConstrs(
            (lag_constrs[g, s, t] for g, s, t in lag_constr_indx),
            name='delta_ineq'
            )

        model._StartupCost = model.addVars(*self.thermal_periods,
                                           name='StartupCost')

        model._ComputeStartupCosts = model.addConstrs(
            (LinExpr([self.StartupCosts[g][s]
                      for s in self.StartupCostIndices[g]] + [-1],
                     [model._delta[g, s, t]
                      for s in self.StartupCostIndices[g]]
                     + [model._StartupCost[g, t]]) == 0

             for g, t in product(*self.thermal_periods)),
            name='ComputeStartupCosts'
            )

        return model

    def mlr_reserve_constraints(self, model):
        model = self._add_reserve_shortfall(model)

        model._EnforceReserveRequirements = model.addConstrs(
            (LinExpr([1.] * (len(self.ThermalGenerators) + 1),
                     [model._ReserveProvided[g, t]
                      for g in self.ThermalGenerators]
                     + [model._ReserveShortfall[t]]) >= self.ReserveReqs[t]

             for t in self.TimePeriods),
            name='EnforceReserveRequirements'
            )

        return model

    def generate(self,
                 relax_binaries: bool, ptdf_options: dict,
                 ptdf, objective_hours: int) -> None:

        for gen in self.ThermalGenerators:
            self.ShutdownRampLimit[gen] = 1. + self.MinPowerOutput[
                gen, self.InitialTime]

        model = self._initialize_model(ptdf, ptdf_options)

        model = self.garver_3bin_vars(model, relax_binaries)
        model = self.garver_power_vars(model)
        model = self.mlr_reserve_vars(model)
        model = self.file_non_dispatchable_vars(model)

        model = self.mlr_generation_limits(model)
        model = self.damcikurt_ramping(model)
        model = self.ca_production_costs(model)

        model = self.rajan_takriti_ut_dt(model)
        model = self.mlr_startup_costs(model, relax_binaries)

        model = self.ptdf_power_flow(model)
        model = self.mlr_reserve_constraints(model)

        model = self.add_objective(model)
        model = self._add_zero_cost_hours(model, objective_hours)

        model.update()
        self.model = model
