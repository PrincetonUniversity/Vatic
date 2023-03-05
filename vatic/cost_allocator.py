from __future__ import annotations

from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np
import math
import os
import time
from datetime import datetime, timedelta
import dill as pickle
import shutil
from copy import deepcopy
import multiprocessing as mp

import gurobipy as gp
from gurobipy import GRB

from .model_data import VaticModelData
from .engines import Simulator

def compute_adjusted_cost(p, piecewise_points: list[float], piecewise_values : list[float]):
    """compute adjusted production cost at given output level p
    where p can be between 0 and pmin (extrapolation using the per MWh price of the first bin)
    """
    assert len(piecewise_points) == len(piecewise_values), 'piecewise points and values mismatch!'
    assert p > 0., "generator dispatch level p must be positive!"

    ## adjusted commitment cost
    unit_price = (piecewise_values[1] - piecewise_values[0]) / (piecewise_points[1] - piecewise_points[0])
    adjusted_commitment_cost = piecewise_values[0] - unit_price * piecewise_points[0]
    
    ## adjusted variable cost
    if 0. < p < piecewise_points[0]:
        adjusted_production_cost = unit_price * p
    
    for i in range(len(piecewise_points) - 1):
        if piecewise_points[i] <= p <= piecewise_points[i + 1]:
            unit_price =  (piecewise_values[i + 1] - piecewise_values[i]) / \
                (piecewise_points[i + 1] - piecewise_points[i])
            adjusted_production_cost = piecewise_values[i] + unit_price * (p - piecewise_points[i]) - adjusted_commitment_cost

    return adjusted_commitment_cost, adjusted_production_cost


def _get_perturb_initial_p_output(simulation_outputs: dict[str, float], 
                                perturb_outputs: dict[str, float], 
                                perturb_list: list[str], 
                                tol: float = 1e-5) -> dict:
    """
    Find difference between simulation and perturbation outputs to be used as initial states for next timestep
    """
    init_p = dict()
    for asset in perturb_list:
        if not math.isclose(simulation_outputs[asset], perturb_outputs[asset], abs_tol=tol):
            init_p[asset] = {'simulation':simulation_outputs[asset], 'perturbation': perturb_outputs[asset]}
    return init_p


def _get_thermal_outputs(model, 
                        thermal_gens: list[str], 
                        thermal_minimum_output: dict[str, float]) -> dict:
    """get thermal generator outputs for the most current timestep from a GUROBI model
    """
    thermal_outputs = dict()

    for gen in thermal_gens:

        var_name = 'PowerGeneratedAboveMinimum(' + gen.replace('-', '_').replace('/', '_') + '_1)'
        var = model.getVarByName(var_name)
        try:
            thermal_outputs[gen] = var.ScenNX + thermal_minimum_output[gen]
        except:
            print('_get_thermal_outputs: cannot find {}'.format(gen))
    
    return thermal_outputs


class ShadowPriceSimulator(Simulator):

    BigPenalty = 1e4
    ModeratelyBigPenalty = 0.

    def __init__(self,
                workdir: Union[Path, str], **siml_args) -> None:

        super().__init__(**siml_args)

        ## grid parameters for perturbation analysis
        self.thermal_generators = self._data_provider.template['ThermalGenerators']
        self.thermal_pmin = self._data_provider.template['MinimumPowerOutput']
        self.thermal_pmax = self._data_provider.template['MaximumPowerOutput']

        self.ScaledNominalRampUpLimit = {
            gen : min(self._data_provider.template['NominalRampUpLimit'][gen], 
                self._data_provider.template['MaximumPowerOutput'][gen]) for gen 
                in self._data_provider.template['ThermalGenerators']
        }

        self.ScaledNominalRampDownLimit = self._data_provider.template['NominalRampDownLimit']
        self.ShutdownRampLimit = self._data_provider.template['ShutdownRampLimit']

        self.NondispatchableGeneratorsAtBus = dict()
        for bus in self._data_provider.template['NondispatchableGeneratorsAtBus']:
            if self._data_provider.template['NondispatchableGeneratorsAtBus'][bus]:
                for gen in self._data_provider.template['NondispatchableGeneratorsAtBus'][bus]:
                    self.NondispatchableGeneratorsAtBus[gen] = bus
        
        ## working directory
        self.workdir = Path(workdir)
        os.makedirs(self.workdir, exist_ok=True)

        self.report_dfs = None
        self.init_thermal_output = dict()
        self.perturb_init_thermal_output = dict()
        self.perturb_init_thermal_output2 = dict()
        self.commits = dict()
        self.shadow_price = dict()
        self.perturb_costs = dict()
        self.EnforceMaxAvailableRampUpRates = dict()
        self.EnforceScaledNominalRampDownLimits = dict()

    def clean_wkdir(self):
        shutil.rmtree(self.workdir)

    def solve_save_sced(self,
                   hours_in_objective: int,
                   sced_horizon: int) -> VaticModelData:

        """This method is adapted from engine.solve_sced. Save sced model to MPS file. 
        """

        sced_model_data = self._data_provider.create_sced_instance(
            self._simulation_state, sced_horizon=sced_horizon)

        ## get thermal generators initial output level
        gen_data = sced_model_data._data['elements']['generator']
        self.init_thermal_output[self._current_timestep.when] = {
            gen: gen_data[gen]['initial_p_output']
            for gen in gen_data if gen_data[gen]['generator_type'] != 'renewable'
        }

        ## set sced shortfall cost to 0
        if not self.sced_shortfall_costs:
            sced_model_data._data['system']['reserve_shortfall_cost'] = 0.

        self._ptdf_manager.mark_active(sced_model_data)

        self._hours_in_objective = hours_in_objective
        if self._hours_in_objective > 10:
            ptdf_options = self._ptdf_manager.look_ahead_sced_ptdf_options
        else:
            ptdf_options = self._ptdf_manager.sced_ptdf_options

        self.sced_model.generate_model(
            sced_model_data, relax_binaries=False, ptdf_options=ptdf_options,
            ptdf_matrix_dict=self._ptdf_manager.PTDF_matrix_dict,
            objective_hours=hours_in_objective
            )
        
        # update in case lines were taken out
        self._ptdf_manager.PTDF_matrix_dict = self.sced_model.pyo_instance._PTDFs

        sced_results = self.sced_model.solve_model(self._sced_solver,
                                                   self.solver_options)
        
        ## save sced model instance to files
        model = self.sced_model.pyo_instance
        filename = 'sced_' + self._current_timestep.when.strftime('%H%M') + '.mps'
        model.write(str(self.workdir / filename), io_options={'symbolic_solver_labels': True})

        self._ptdf_manager.update_active(sced_results)

        ## save model parameters for ``EnforceMaxAvailableRampUpRates`` and 
        ## ``EnforceScaledNominalRampDownLimits`` constraints 
        ## to be used for perturbation analysis
        rampup = dict()
        for gen, timestep in model.EnforceMaxAvailableRampUpRates._data:
            gurobi_constr_name = 'c_u_EnforceMaxAvailableRampUpRates(' + \
                gen.replace('-', '_').replace('/', '_') + '_' + str(timestep) + ')_'

            rampup[gurobi_constr_name] = {
                'UnitOn' : model.UnitOn._data[gen, timestep].value,
                'ScaledNominalRampUpLimit' : model.ScaledNominalRampUpLimit._data[gen, timestep].value,
                'ScaledStartupRampLimit' : model.ScaledStartupRampLimit._data[gen, timestep].value,
                'PowerGeneratedT0' : model.PowerGeneratedT0._data[gen].value
            }

        rampdown = dict()
        for gen, timestep in model.EnforceScaledNominalRampDownLimits._data:
            gurobi_constr_name = 'c_l_EnforceScaledNominalRampDownLimits(' + \
                gen.replace('-', '_').replace('/', '_') + '_' + str(timestep) + ')_'

            rampdown[gurobi_constr_name] = {
                'UnitOn' : model.UnitOn._data[gen, timestep].value,
                'ScaledShutdownRampLimit' : model.ScaledShutdownRampLimit._data[gen, timestep].value,
                'ScaledNominalRampDownLimit' : model.ScaledNominalRampDownLimit._data[gen, timestep].value,
                'ScaledShutdownRampLimitT0' : model.ScaledShutdownRampLimitT0._data[gen].value,
                'PowerGeneratedT0' : model.PowerGeneratedT0._data[gen].value
            }

        self.EnforceMaxAvailableRampUpRates[self._current_timestep.when] = rampup
        self.EnforceScaledNominalRampDownLimits[self._current_timestep.when] = rampdown

        return sced_results

    def call_save_oracle(self) -> None:
        """Solves the real-time economic dispatch and save sced optimization to file.
        This method is adapted from engine.call_oracle.
        """

        self.commits[self._current_timestep] = deepcopy(
            self._simulation_state._commits)

        if self.verbosity > 0:
            print("\nSolving SCED instance")

        current_sced_instance = self.solve_save_sced(hours_in_objective=1,
                                                sced_horizon=self.sced_horizon)

        if self.verbosity > 0:
            print("Solving for LMPs")

        if self.run_lmps:
            lmp_sced = self.solve_lmp(current_sced_instance)
        else:
            lmp_sced = None

        self._simulation_state.apply_sced(current_sced_instance)
        self._prior_sced_instance = current_sced_instance

        self._stats_manager.collect_sced_solution(
            self._current_timestep, current_sced_instance, lmp_sced,
            pre_quickstart_cache=None
            )

    def simulate(self) -> dict[str, pd.DataFrame]:
        """Top-level runner of a simulation. sced model will be saved to MPS files
        """
        simulation_start_time = time.time()

        # create commitments for the first day using an RUC
        self.initialize_oracle()
        self.simulation_times['Init'] += time.time() - simulation_start_time

        for time_step in self._time_manager.time_steps():
            self._current_timestep = time_step

            # run the day-ahead RUC at some point in the day before
            if time_step.is_planning_time:
                plan_start_time = time.time()

                self.call_planning_oracle()
                self.simulation_times['Plan'] += time.time() - plan_start_time

            # run the SCED to simulate this time step
            oracle_start_time = time.time()
            self.call_save_oracle()
            self.simulation_times['Sim'] += time.time() - oracle_start_time

        sim_time = time.time() - simulation_start_time

        if self.verbosity > 0:
            print("Simulation Complete")
            print("Total simulation time: {:.1f} seconds".format(sim_time))

            if self.verbosity > 1:
                print("Initialization time: {:.2f} seconds".format(
                    self.simulation_times['Init']))
                print("Planning time: {:.2f} seconds".format(
                    self.simulation_times['Plan']))
                print("Real-time sim time: {:.2f} seconds".format(
                    self.simulation_times['Sim']))

        self.report_dfs = self._stats_manager.save_output(sim_time)

        ## get thermal outputs at each timestep
        df = self.report_dfs['thermal_detail'].copy(deep=True)
        df['Datetime'] = (pd.to_datetime(df.index.get_level_values(0)) + \
                        pd.to_timedelta(df.index.get_level_values(1), unit='H')).to_pydatetime()
        gb = df.reset_index().groupby('Datetime')
        thermal_outputs = {datetime: dict(zip(df.Generator, df.Dispatch)) for datetime, df in gb}
        self.thermal_outputs = thermal_outputs

        self.cost_df = pd.DataFrame.from_dict(self._stats_manager._sced_stats, orient='index',
                    columns=['fixed_costs', 'variable_costs', 'reserve_shortfall', 'load_shedding', 'over_generation'])
        self.cost_df.rename(columns={'reserve_shortfall': 'reserve_shortfall_costs',
            'load_shedding': 'load_shedding_costs', 
            'over_generation': 'over_generation_costs'},
            inplace=True
            )
        self.cost_df['reserve_shortfall_costs'] *= self.ModeratelyBigPenalty
        self.cost_df['load_shedding_costs'] *= self.BigPenalty
        self.cost_df['over_generation_costs'] *= self.BigPenalty
        self.cost_df.index = self.cost_df.index.map(lambda x: x.when)

        fixed_cost_adjustments, variable_cost_adjustments = [], []
        for timestep in self._stats_manager._sced_stats:
            sced_stats = self._stats_manager._sced_stats[timestep]

            fix_costs = 0.
            variable_costs = 0.
            for th_gen in sced_stats['observed_thermal_states']:
                if (not sced_stats['observed_thermal_states'][th_gen]) and \
                    (sced_stats['observed_thermal_dispatch_levels'][th_gen] > 0.):
                    f, v = compute_adjusted_cost(
                        sced_stats['observed_thermal_dispatch_levels'][th_gen],
                        self._data_provider.template['CostPiecewisePoints'][th_gen],
                        self._data_provider.template['CostPiecewiseValues'][th_gen]
                    )

                    fix_costs += f
                    variable_costs += v

            fixed_cost_adjustments.append(fix_costs)
            variable_cost_adjustments.append(variable_costs)

        self.cost_df['fixed_costs_adjustments'] = fixed_cost_adjustments
        self.cost_df['variable_costs_adjustments'] = variable_cost_adjustments

    def simulate_perturbations(self, perturb_dict: dict[str, dict[datetime, float]]):

        ## sced horizons
        self.gurobi_sced_horizons = [h for h in range(self.sced_horizon)]
        self.perturb_dict = perturb_dict

        ## reset perturb_costs
        self.perturb_costs = dict()

        for current_timestep in self._time_manager.time_steps():

            self._current_timestep = current_timestep
            perturb_outputs = self.perturb_current_timestep()

            effect_timesteps = (
                effect_timestep for effect_timestep in self._time_manager.time_steps()
                if effect_timestep.when > self._current_timestep.when
            )
                
            for effect_timestep in effect_timesteps:

                simulation_outputs = self.init_thermal_output[effect_timestep.when]

                for commit_timestep in self.commits:
                    if commit_timestep.when + self._time_manager.sced_interval == effect_timestep.when:
                        simulation_commits = self.commits[commit_timestep]
                        break

                perturb_init_th_outputs = dict()

                for perturb_gen, output in perturb_outputs.items():
                    
                    ## we do not care about ``OFF`` units
                    for gen in simulation_outputs:
                        if not simulation_commits[gen][0]:
                            output[gen] = simulation_outputs[gen]
                    
                    init_p_output = _get_perturb_initial_p_output(
                        simulation_outputs, output, self.thermal_generators)

                    if init_p_output:
                        # import pdb; pdb.set_trace()
                        perturb_init_th_outputs[perturb_gen] = init_p_output
                
                if not perturb_init_th_outputs:
                    break
                else:
                    perturb_outputs = self.perturb_future_timestep(effect_timestep, perturb_init_th_outputs)

                    if not current_timestep.when in self.perturb_init_thermal_output2:
                        self.perturb_init_thermal_output2[current_timestep.when] = dict()
                    self.perturb_init_thermal_output2[current_timestep.when][effect_timestep.when] = perturb_init_th_outputs
                


    def perturb_current_timestep(self) -> None:
        """perturb simulations at the current timestep
        """

        ## count number of scenarios
        NumScenarios = 0
        for gen in self.perturb_dict:
            for h in range(self.sced_horizon):
                sced_horizon_time = self._current_timestep.when + \
                        h * timedelta(minutes = self._sced_frequency_minutes)
                if self.perturb_dict[gen][sced_horizon_time] != 0.:
                    NumScenarios += 1
        if NumScenarios == 0:
            return {}

        ## solve original model and store optimal values for binary and ``PowerGeneratedAboveMinimum`` variables
        filename = 'sced_' + self._current_timestep.when.strftime('%H%M') + '.mps'
        gurobi_md = gp.read(str(self.workdir / filename))
        gurobi_md.Params.LogToConsole = 0
        gurobi_md.setParam('MIPGap', 1e-12)
        gurobi_md.optimize()

        thermal_optimal_vals = dict()
        for gen in self.thermal_generators:
            gurobi_gen = gen.replace('-', '_').replace('/', '_')
            thermal_optimal_vals[gen] = [gurobi_md.getVarByName(
                'PowerGeneratedAboveMinimum(' + gurobi_gen + '_' + str(h + 1) + ')'
                ).X for h in range(self.sced_horizon)]

        binary_optimal_vals = {
            var.VarName: var.X for var in gurobi_md.getVars() \
                if (var.LB == 0. and var.UB == 1.) and var.VType in (GRB.BINARY, GRB.INTEGER)
            }        

        ## setup multiple scenario model
        gurobi_md = gp.read(str(self.workdir / filename))
        gurobi_md.Params.LogToConsole = 0
        gurobi_md.setParam('MIPGap', 1e-12)

        ## fix binary variables 
        for var in gurobi_md.getVars():
            if (var.LB == 0. and var.UB == 1.) and var.VType in (GRB.BINARY, GRB.INTEGER):
                var.LB = binary_optimal_vals[var.VarName]
                var.UB = binary_optimal_vals[var.VarName]

        ## number of scenarios
        gurobi_md.NumScenarios = NumScenarios

        print('Total number of scenarios is {}'.format(NumScenarios))

        ## get commits 
        ## setup starting point for ON units
        ## fix output level for OFF units

        for timestep in self.commits:
            if timestep.when == self._current_timestep.when:
                commits = self.commits[timestep]
                break

        for gen in self.thermal_generators:
            gurobi_gen = gen.replace('-', '_').replace('/', '_')
            for h in range(self.sced_horizon):
                vname = 'PowerGeneratedAboveMinimum(' + gurobi_gen + '_' + str(h + 1) + ')'
                var = gurobi_md.getVarByName(vname)

                if commits[gen][h]:
                    var.VarHintVal = thermal_optimal_vals[gen][h]
                else:
                    var.LB = thermal_optimal_vals[gen][h]
                    var.UB = thermal_optimal_vals[gen][h]

        perturb = 0.
        scenario = 0

        for gen in self.perturb_dict:

            gurobi_gen = gen.replace('-', '_').replace('/', '_')
            gurobi_bus = self.NondispatchableGeneratorsAtBus[gen].replace('-', '_').replace('/', '_')

            for h in range(self.sced_horizon):

                sced_horizon_time = self._current_timestep.when + \
                    h * timedelta(minutes = self._sced_frequency_minutes)

                if self.perturb_dict[gen][sced_horizon_time] != 0.:

                    gurobi_md.Params.ScenarioNumber = scenario
                    gurobi_md.ScenNName = gen + '_' + str(h)
                    
                    vname = 'NondispatchablePowerUsed(' + gurobi_gen + '_' + str(h + 1) + ')'
                    var = gurobi_md.getVarByName(vname)

                    if var:
                        var.ScenNLB = (
                            max(0., var.LB + self.perturb_dict[gen][sced_horizon_time]) if var.LB > 0. else var.LB
                        )
                        var.ScenNUB = max(0., var.UB + self.perturb_dict[gen][sced_horizon_time])

                    vname = 'OverGeneration(' + gurobi_bus + '_' + str(h + 1) + ')'
                    var = gurobi_md.getVarByName(vname)

                    if var:
                        var.ScenNUB = max(0., var.UB + self.perturb_dict[gen][sced_horizon_time])
                    
                    scenario += 1

                    ## keep track of the largest perturbation
                    perturb = max(perturb, abs(self.perturb_dict[gen][sced_horizon_time]))

        ## look for solution in a small region by
        ## setting LB and UB for ``PowerGeneratedAboveMinimum``
        print('new bounds is {}'.format(perturb))
        print('number of scenarios is {}'.format(NumScenarios))

        for th_gen in self.thermal_generators:

            gurobi_th_gen = th_gen.replace('-', '_').replace('/', '_')
            
            for h in range(self.sced_horizon):

                if commits[th_gen][h]:
                    
                    th_gen_vname = 'PowerGeneratedAboveMinimum(' + gurobi_th_gen + '_' + str(h + 1) + ')'
                    th_gen_var = gurobi_md.getVarByName(th_gen_vname)

                    th_gen_var.LB = max(th_gen_var.LB, thermal_optimal_vals[th_gen][h] - perturb)
                    th_gen_var.UB = min(th_gen_var.UB, thermal_optimal_vals[th_gen][h] + perturb)

        gurobi_md.optimize()

        ## collecting results
        perturb_sced_costs = dict()
        perturb_outputs = dict()

        ## get vars for later reference later
        production_costs_vars = [
            var for var in gurobi_md.getVars() if var.VarName.startswith('ProductionCost') 
            and var.VarName.endswith('_1)')
        ]
        reserve_shortfall_costs_vars = [
            gurobi_md.getVarByName('ReserveShortfall(1)')
        ]
        load_shedding_costs_vars = [
            var for var in gurobi_md.getVars() if var.VarName.startswith('LoadShedding') 
            and var.VarName.endswith('_1')
        ]
        over_generation_costs_vars = [
            var for var in gurobi_md.getVars() if var.VarName.startswith('OverGeneration') 
            and var.VarName.endswith('_1')
        ]
        startup_shuntdown_costs_vars = [
            var for var in gurobi_md.getVars() if (var.VarName.startswith('StartupCost') 
            and var.VarName.endswith('_1)')) or (var.VarName.startswith('ShutdownCost') 
            and var.VarName.endswith('_1)'))
        ]

        ## compute fixed costs
        fixed_costs = 0.
        for gen in commits:
            if commits[gen][0]:
                fixed_costs += self._data_provider.template['CostPiecewiseValues'][gen][0] 

        for scenario in range(gurobi_md.NumScenarios):

            gurobi_md.Params.ScenarioNumber = scenario

            if gurobi_md.ScenNObjVal >= gurobi_md.ModelSense * GRB.INFINITY:
                raise RuntimeError('GUROBI solver is not able to solve scenario {}'.format(scenario))
            else:
                perturb_sced_costs[gurobi_md.ScenNName] = {
                    'fixed_costs' : fixed_costs + sum(var.ScenNX for var in startup_shuntdown_costs_vars),
                    'variable_costs': sum(var.ScenNX for var in production_costs_vars),
                    'reserve_shortfall_costs': self.ModeratelyBigPenalty * sum(var.ScenNX for var in reserve_shortfall_costs_vars),
                    'load_shedding_costs': self.BigPenalty * sum(var.ScenNX for var in load_shedding_costs_vars),
                    'over_generation_costs': self.BigPenalty * sum(var.ScenNX for var in over_generation_costs_vars),
                }
                perturb_outputs[gurobi_md.ScenNName] = _get_thermal_outputs(
                        gurobi_md, self.thermal_generators, self.thermal_pmin)

        self.perturb_costs[self._current_timestep.when] = {self._current_timestep.when : perturb_sced_costs}

        self.perturb_init_thermal_output[self._current_timestep.when] = dict()
        self.perturb_init_thermal_output[self._current_timestep.when][self._current_timestep.when] = perturb_outputs

        return perturb_outputs

    def perturb_future_timestep(self, future_timestep, init_p_outputs) -> None:
        """perturb simulations at the current timestep
        """
        ## solve original model for optimal values for binary and ``PowerGeneratedAboveMinimum`` variables
        filename = 'sced_' + future_timestep.when.strftime('%H%M') + '.mps'
        gurobi_md = gp.read(str(self.workdir / filename))
        gurobi_md.Params.LogToConsole = 0
        gurobi_md.setParam('MIPGap', 1e-12)
        gurobi_md.optimize()

        thermal_optimal_vals = dict()
        for gen in self.thermal_generators:
            gurobi_gen = gen.replace('-', '_').replace('/', '_')
            thermal_optimal_vals[gen] = [gurobi_md.getVarByName(
                'PowerGeneratedAboveMinimum(' + gurobi_gen + '_' + str(h + 1) + ')'
                ).X for h in range(self.sced_horizon)]

        binary_optimal_vals = {
            var.VarName: var.X for var in gurobi_md.getVars() 
            if (var.LB == 0. and var.UB == 1.) and var.VType in (GRB.BINARY, GRB.INTEGER)
            }

        ## reload model and setup multiple scenarios
        gurobi_md = gp.read(str(self.workdir / filename))
        gurobi_md.Params.LogToConsole = 0
        gurobi_md.setParam('MIPGap', 1e-12)

        ## get constraint parameters
        rampup = self.EnforceMaxAvailableRampUpRates[future_timestep.when]
        rampdown = self.EnforceScaledNominalRampDownLimits[future_timestep.when]

        ## fix binary variable value
        for var in gurobi_md.getVars():
            if (var.LB == 0. and var.UB == 1.) and var.VType in (GRB.BINARY, GRB.INTEGER):
                var.LB = binary_optimal_vals[var.VarName]
                var.UB = binary_optimal_vals[var.VarName]

        ## setup multiple scenarios
        gurobi_md.NumScenarios = len(init_p_outputs)

        ## setup starting point for ON units
        ## fix dispatch level for OFF units
        for timestep in self.commits:
            if timestep.when == future_timestep.when:
                commits = self.commits[timestep]
                break

        for gen in self.thermal_generators:
            gurobi_gen = gen.replace('-', '_').replace('/', '_')

            for h in range(self.sced_horizon):
                vname = 'PowerGeneratedAboveMinimum(' + gurobi_gen + '_' + str(h + 1) + ')'    
                var = gurobi_md.getVarByName(vname)

                if commits[gen][h]:
                    var.VarHintVal = thermal_optimal_vals[gen][h]
                else:
                    var.LB = thermal_optimal_vals[gen][h]
                    var.UB = thermal_optimal_vals[gen][h]

        scenario = 0
        perturb = 0.

        for perturb_gen in init_p_outputs:

            gurobi_md.Params.ScenarioNumber = scenario
            gurobi_md.ScenNName = perturb_gen

            ## keep track of perturbation
            eps = 0.

            for gen in init_p_outputs[perturb_gen]:

                gurobi_gen = gen.replace('-', '_').replace('/', '_')

                ## ramp up rate constraint
                constr_name = 'c_u_EnforceMaxAvailableRampUpRates(' + gurobi_gen + '_1)_'
                constr = gurobi_md.getConstrByName(constr_name)
                
                if constr:

                    RHS = rampup[constr_name]['UnitOn'] * ( 
                        rampup[constr_name]['ScaledNominalRampUpLimit'] - self.thermal_pmin[gen]
                        ) + init_p_outputs[perturb_gen][gen]['perturbation']
                    constr.ScenNRHS = RHS
                    eps += abs(constr.RHS - RHS)

                ## ramp down constraint
                constr_name = 'c_l_EnforceScaledNominalRampDownLimits(' + gurobi_gen + '_1)_'
                constr = gurobi_md.getConstrByName(constr_name)

                if constr:

                    ## disable original constraint
                    constr.ScenNRHS = - GRB.INFINITY

                    COEF = init_p_outputs[perturb_gen][gen]['perturbation'] - \
                        self.thermal_pmin[gen] - rampdown[constr_name]['ScaledNominalRampDownLimit']
                    RHS = init_p_outputs[perturb_gen][gen]['perturbation'] - \
                        rampdown[constr_name]['ScaledNominalRampDownLimit'] - self.thermal_pmin[gen]
                    
                    UnitStop = gurobi_md.getVarByName('UnitStop(' + gurobi_gen + '_1)')
                    PowerGeneratedAboveMinimum = gurobi_md.getVarByName('PowerGeneratedAboveMinimum(' + gurobi_gen + '_1)')
                    
                    gurobi_md.addLConstr(PowerGeneratedAboveMinimum + COEF * UnitStop >= - GRB.INFINITY, 
                        perturb_gen + '_' + gen + '_perturbed_constraint')
                    gurobi_md.update()

                    pert_constr = gurobi_md.getConstrByName(perturb_gen + '_' + gen + '_perturbed_constraint')
                    pert_constr.ScenNRHS = RHS
                    gurobi_md.update()

                    eps += abs(constr.RHS - RHS)
            
            perturb = max(perturb, eps)
            scenario += 1

        print('perturb step size is {}'.format(perturb))
        print('number of scenarios is {}'.format(len(init_p_outputs)))

        ## find solution in a small region by
        ## setting LB and UB for ``PowerGeneratedAboveMinimum``
        for th_gen in self.thermal_generators:
            gurobi_th_gen = th_gen.replace('-', '_').replace('/', '_')
            for h in range(self.sced_horizon):
                if commits[th_gen][h]:
                    th_gen_vname = 'PowerGeneratedAboveMinimum(' + gurobi_th_gen + '_' + str(h + 1) + ')'    
                    th_gen_var = gurobi_md.getVarByName(th_gen_vname)

                    th_gen_var.LB = max(th_gen_var.LB, thermal_optimal_vals[th_gen][h] - perturb)
                    th_gen_var.UB = min(th_gen_var.UB, thermal_optimal_vals[th_gen][h] + perturb)

            # gurobi_th_gen = th_gen.replace('-', '_').replace('/', '_')
            # th_gen_vname = 'PowerGeneratedAboveMinimum(' + gurobi_th_gen + '_1)'    
            # th_gen_var = gurobi_md.getVarByName(th_gen_vname)

            # th_gen_var.LB = max(th_gen_var.LB, thermal_optimal_vals[th_gen] - perturb)
            # th_gen_var.UB = min(th_gen_var.UB, thermal_optimal_vals[th_gen] + perturb)

        gurobi_md.optimize()

        ## collect results
        perturb_sced_costs = dict()
        perturb_outputs = dict()

        ## get vars corresponding to current timestep
        production_costs_vars = [var for var in gurobi_md.getVars() 
            if var.VarName.startswith('ProductionCost') and var.VarName.endswith('_1)')
        ]
        reserve_shortfall_costs_vars = [
            gurobi_md.getVarByName('ReserveShortfall(1)')
        ]
        load_shedding_costs_vars = [
            var for var in gurobi_md.getVars() if var.VarName.startswith('LoadShedding') 
            and var.VarName.endswith('_1')
        ]
        over_generation_costs_vars = [
            var for var in gurobi_md.getVars() if var.VarName.startswith('OverGeneration') 
            and var.VarName.endswith('_1')
        ]
        startup_shuntdown_costs_vars = [
            var for var in gurobi_md.getVars() if (var.VarName.startswith('StartupCost') 
            and var.VarName.endswith('_1)')) or (var.VarName.startswith('ShutdownCost') 
            and var.VarName.endswith('_1)'))
        ]

        ## compute fixed costs for the current step
        fixed_costs = 0.
        for gen in commits:
            if commits[gen][0]:
                fixed_costs += self._data_provider.template['CostPiecewiseValues'][gen][0] 

        for scenario in range(gurobi_md.NumScenarios):

            gurobi_md.Params.ScenarioNumber = scenario

            if gurobi_md.ScenNObjVal >= gurobi_md.ModelSense * GRB.INFINITY:
                raise RuntimeError('GUROBI solver is not able to solve scenario {}'.format(scenario))
            else:
                perturb_sced_costs[gurobi_md.ScenNName] = {
                    'fixed_costs' : fixed_costs + sum(var.ScenNX for var in startup_shuntdown_costs_vars),
                    'variable_costs': sum(var.ScenNX for var in production_costs_vars),
                    'reserve_shortfall_costs': self.ModeratelyBigPenalty * sum(var.ScenNX for var in reserve_shortfall_costs_vars),
                    'load_shedding_costs': self.BigPenalty * sum(var.ScenNX for var in load_shedding_costs_vars),
                    'over_generation_costs': self.BigPenalty * sum(var.ScenNX for var in over_generation_costs_vars),
                }

                perturb_outputs[gurobi_md.ScenNName] = _get_thermal_outputs(
                            gurobi_md, self.thermal_generators, self.thermal_pmin)

        self.perturb_costs[self._current_timestep.when][future_timestep.when] = perturb_sced_costs

        self.perturb_init_thermal_output[self._current_timestep.when][future_timestep.when] = perturb_outputs

        return perturb_outputs

    def compute_shadow_price(self, eps : float = 1e-3) -> None:
        """compute shadow price from the perturbation analysis
        """        
        for cost_type in ['fixed_costs', 'variable_costs', 'reserve_shortfall_costs', 'load_shedding_costs', 'over_generation_costs']:
            
            shadow_price = dict()

            for perturb_timestep in self.perturb_costs:

                shadow_price[perturb_timestep] = dict()
                
                for effect_timestep in self.perturb_costs[perturb_timestep]:

                    shadow_price[perturb_timestep][effect_timestep] = dict()

                    costs = self.perturb_costs[perturb_timestep][effect_timestep]

                    for gen_horizon in costs:

                        gen_horizon_splitted = gen_horizon.rsplit('_', 1)
                        gen = gen_horizon_splitted[0]
                        h = int(gen_horizon_splitted[1])

                        stepsize = self.perturb_dict[gen][perturb_timestep + h * timedelta(minutes = self._sced_frequency_minutes)]

                        if stepsize != 0.:
                            gen_h_shadow_price = (
                                costs[gen_horizon][cost_type] - self.cost_df.loc[pd.to_datetime(effect_timestep), cost_type]
                                ) / stepsize
                        else:
                            raise ValueError('perturbation step size is zero, cannot compute shadow price!')

                        if abs(gen_h_shadow_price) > eps:
                            shadow_price[perturb_timestep][effect_timestep][gen, h] = gen_h_shadow_price

                    if not shadow_price[perturb_timestep][effect_timestep]:
                        shadow_price[perturb_timestep].pop(effect_timestep)

            self.shadow_price[cost_type] = shadow_price


class CostAllocator:
    """This abstract class allocates cost of running economic disptch 
    between the baseline and the target scenarios using perturbation analysis.
    """

    def __init__(self, 
                template_data: dict,
                start_date: datetime.date, 
                mipgap: float,
                reserve_factor: float, 
                sced_horizon: int,
                lmp_shortfall_costs: bool,
                init_ruc_file: str | Path | None, 
                verbosity: int,
                target_assets: list[str], 
                baseline_gen_data: pd.DataFrame, 
                baseline_load_data: pd.DataFrame, 
                target_gen_data: pd.DataFrame, 
                target_load_data: pd.DataFrame,
                workdir: Path | str, 
                sensitivity: int = 0.01, 
                scale :int = 8, 
                ref_output: float | None = None) -> None:

        self.template_data = template_data
        self.start_date = start_date
        self.mipgap = mipgap
        self.reserve_factor = reserve_factor
        self.sced_horizon = sced_horizon
        self.lmp_shortfall_costs = lmp_shortfall_costs
        self.init_ruc_file = Path(init_ruc_file)
        self.verbosity = verbosity

        self.target_assets = target_assets
        self.baseline_gen_data = baseline_gen_data
        self.baseline_load_data = baseline_load_data
        self.target_gen_data = target_gen_data
        self.target_load_data = target_load_data
        self.workdir = Path(workdir)
        self.sensitivity = sensitivity
        self.scale = scale
        self.ref_output = ref_output

        ## hard-coded parameters
        self.ruc_every_hours = 24
        
        ## 
        self.nnods = 2 ** self.scale
        self.stepsize = 1. / self.nnods
        self.simulations = {
            nod : {'alpha' : nod * self.stepsize, 
                    'simulated': False, 
                    'shadow_price': None
            } 
            for nod in range(self.nnods + 1)
        }

        ## time step when renewable production to be perturbed 
        self.perturb_timesteps = [
            (self.start_date.to_pydatetime().replace(tzinfo=None) + timedelta(hours=hour)) 
            for hour in range(self.ruc_every_hours + self.sced_horizon - 1)]

        ## time step when perturbations affect costs
        self.effect_timesteps = [
            self.start_date.to_pydatetime().replace(tzinfo=None) + timedelta(hours=hour) 
            for hour in range(self.ruc_every_hours)]

        self.gen_diff = self.target_gen_data - self.baseline_gen_data
        self.load_diff = self.target_load_data - self.baseline_load_data
        
        testing_nodes_df = self.gen_diff.loc[pd.to_datetime(self.perturb_timesteps, utc=True), 
                [('actl', gen) for gen in self.target_assets]
                ].copy(deep=True)
        testing_nodes_df.index = testing_nodes_df.index.map(
                lambda x: x.to_pydatetime().replace(tzinfo=None))
                
        testing_nodes_df.columns = testing_nodes_df.columns.droplevel(0)
        testing_nodes_df.fillna(method='ffill', inplace=True)
        if not self.ref_output:
            self.ref_output = np.abs(testing_nodes_df.to_numpy()).max()
        testing_nodes_df = testing_nodes_df.abs() / (self.ref_output / self.nnods)
        testing_nodes_df = testing_nodes_df.applymap(
            lambda x: 2 if x < 1 else min(2 ** math.ceil(math.log2(x)), 2 ** self.scale)
        )
     
        self.testing_nodes_df = testing_nodes_df

    def simulate_shadow_price(self, 
                              nod: int, 
                              clean_wkdir: bool = True) -> dict:
        """Run simulation at given node and compute shadow price"""

        simulator = ShadowPriceSimulator(
                Path(self.workdir, str(nod)),
                template_data = self.template_data, 
                gen_data = self.baseline_gen_data + self.simulations[nod]['alpha'] * self.gen_diff, 
                load_data = self.baseline_load_data + self.simulations[nod]['alpha'] * self.load_diff,
                out_dir = None, 
                start_date = self.start_date.date(), 
                num_days = 1, 
                solver = 'gurobi',
                solver_options = None, 
                run_lmps = False, 
                mipgap = 1e-12, 
                reserve_factor = self.reserve_factor,
                output_detail = 2,
                prescient_sced_forecasts = True, 
                ruc_prescience_hour = 0,
                ruc_execution_hour = 16, 
                ruc_every_hours = 24, 
                ruc_horizon = 48,
                sced_horizon = self.sced_horizon, 
                lmp_shortfall_costs = False,
                enforce_sced_shutdown_ramprate = False,
                no_startup_shutdown_curves = False, 
                init_ruc_file = self.init_ruc_file, 
                verbosity = 0,
                output_max_decimals = 12, 
                create_plots = False, 
                renew_costs = None,
                save_to_csv = False,
                last_conditions_file = None,
                )

        simulator.simulate()

        # get perturb_dict 
        perturb_dict = {gen : {} for gen in self.target_assets}

        for gen, col in self.testing_nodes_df.iteritems():
            for timestep, num_of_nodes in col.iteritems():
                if num_of_nodes != 0 and nod % (self.nnods // num_of_nodes) == 0:
                    perturb_dict[gen][timestep] = self.sensitivity
                else:
                    perturb_dict[gen][timestep] = 0.

        simulator.simulate_perturbations(perturb_dict)
        simulator.compute_shadow_price()

        cost_df = deepcopy(simulator.cost_df)
        shadow_price = deepcopy(simulator.shadow_price)
        init_th_output = deepcopy(simulator.perturb_init_thermal_output)

        # clean up
        if clean_wkdir:
            simulator.clean_wkdir()
        del simulator

        return {'nod' : nod, 'costs' : cost_df, 'shadow_price' : shadow_price, 'perturb_init_output': init_th_output}

    def run_perturb_analysis(self, 
                             processes: int | None = None, 
                             clean_wkdir : bool = True) -> None:

        if not processes:
            for nod in range(self.nnods + 1):
                simulation_results = self.simulate_shadow_price(nod, clean_wkdir)

                self.simulations[nod]['costs'] = simulation_results['costs']
                self.simulations[nod]['shadow_price'] = simulation_results['shadow_price']
                self.simulations[nod]['perturb_init_output'] = simulation_results['perturb_init_output']
        else:
            if processes > mp.cpu_count():
                processes = mp.cpu_count()

            with mp.Pool(processes=processes) as pool:
                for result in pool.starmap(
                        self.simulate_shadow_price, 
                        [(nod, clean_wkdir) for nod in range(self.nnods + 1)]):

                    nod = result['nod']
                    self.simulations[nod]['costs'] = result['costs']
                    self.simulations[nod]['shadow_price'] = result['shadow_price']
                    self.simulations[nod]['perturb_init_output'] = result['perturb_init_output']


    def compute_allocation(self, 
            allocation_cost_types : list[str] = ['variable_costs', 
            'reserve_shortfall_costs', 'load_shedding_costs', 'over_generation_costs']):

        average_shadow_price = {cost_type : {} for cost_type in allocation_cost_types}
        cost_allocation = {cost_type : {} for cost_type in allocation_cost_types}
        allocated_costs = {cost_type : {} for cost_type in allocation_cost_types}
        allocation_summary_dfs = {}

        diff_df = self.gen_diff.loc[pd.to_datetime(self.perturb_timesteps, utc=True), 
                                    [('actl', asset) for asset in self.target_assets]].copy(deep=True)
        diff_df.fillna(method='ffill', inplace=True)
        diff_df.columns = diff_df.columns.droplevel(0)
        diff_df.index = diff_df.index.map(lambda x: x.to_pydatetime().replace(tzinfo=None))

        for cost_type in allocation_cost_types:
            
            for h in range(self.sced_horizon):

                average_shadow_price[cost_type][h] = dict()
                cost_allocation[cost_type][h] = dict()
                
                for gen in self.target_assets:
                    
                    arr = np.zeros((self.ruc_every_hours, self.ruc_every_hours))
                    
                    for perturb_timestep_index in range(self.ruc_every_hours):
                        
                        perturb_timestep = self.perturb_timesteps[perturb_timestep_index]
                        
                        subdiv = self.testing_nodes_df.loc[perturb_timestep, gen]
                                        
                        if subdiv > 0:
                            spacing = self.nnods // subdiv
                            nodes = [i * spacing for i in range(subdiv + 1)]
                            
                            for effect_timestep_index, effect_timestep in enumerate(self.effect_timesteps):
                                p = 0.
                                for nod in nodes:
                                    try:
                                        shadow_price = self.simulations[nod]['shadow_price'][cost_type][perturb_timestep][effect_timestep][gen, h]
                                    except:
                                        shadow_price = 0.

                                    if nod == 0 or nod == self.nnods:
                                        p += 0.5 * shadow_price
                                    else:
                                        p += shadow_price         
                                arr[effect_timestep_index, perturb_timestep_index] = p / (len(nodes) - 1)
                            
                    average_shadow_price[cost_type][h][gen] = pd.DataFrame(
                        data=arr, index=self.effect_timesteps, 
                        columns=self.perturb_timesteps[h: self.ruc_every_hours + h])
                    
                ## compute cost allocation
                allocated_costs[cost_type][h] = pd.DataFrame(
                    data=np.zeros((self.ruc_every_hours, self.ruc_every_hours)), 
                    index=self.effect_timesteps, columns=self.perturb_timesteps[h : self.ruc_every_hours + h])
                
                for gen in self.target_assets:
                    cost_allocation[cost_type][h][gen] = average_shadow_price[cost_type][h][gen].copy(deep=True)
                    for timestep in cost_allocation[cost_type][h][gen].columns:
                        cost_allocation[cost_type][h][gen][timestep] *= diff_df.loc[timestep, gen]
                        
                    allocated_costs[cost_type][h] += cost_allocation[cost_type][h][gen]            
                
            ## get cost allocation summary
            if cost_type == 'variable_costs':
                allocation_summary_dfs[cost_type] = self.simulations[self.nnods]['costs'][['variable_costs', 'variable_costs_adjustments']] - \
                    self.simulations[0]['costs'][['variable_costs', 'variable_costs_adjustments']]
            else:
                allocation_summary_dfs[cost_type] = self.simulations[self.nnods]['costs'][[cost_type]] - \
                    self.simulations[0]['costs'][[cost_type]]

            allocation_summary_dfs[cost_type][cost_type + '_allocated'] = allocated_costs[cost_type][0].sum(axis=1).values
            for h in range(1, self.sced_horizon):
                allocation_summary_dfs[cost_type][cost_type + '_allocated'] += allocated_costs[cost_type][h].sum(axis=1).values

            allocation_summary_dfs[cost_type] = allocation_summary_dfs[cost_type].round(2)

        self.average_shadow_price = average_shadow_price   
        self.cost_allocation = cost_allocation
        self.allocated_costs = allocated_costs
        self.allocation_summary_dfs = allocation_summary_dfs