"""Running a simulation of alternating UC and ED grid optimization steps."""

from __future__ import annotations

import os
import dill as pickle
from pathlib import Path
import time

import datetime
import math
import pandas as pd
from copy import deepcopy

from .data_providers import PickleProvider
from .model_data import VaticModelData
from .simulation_state import VaticSimulationState, VaticStateWithScedOffset
from .models import UCModel
from .stats_manager import StatsManager
from .time_manager import VaticTimeManager, VaticTime
from .ptdf_manager import VaticPTDFManager

from pyomo.environ import SolverFactory
from pyomo.environ import Suffix as PyomoSuffix
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from egret.common.lazy_ptdf_utils import uc_instance_binary_relaxer


class Simulator:
    """An engine for simulating the operation of a power grid.

    Parameters
    ----------
    template_data       Static grid properties such as line capacities, thermal
                        generator cost curves, network topology, etc.
    gen_data            Forecasted and actual renewable generator outputs.
    load_data           Forecasted and actual load bus demands.

    out_dir         Where to save the results of the simulation.
    start_date      The first day to simulate.
    num_days        How many days, including the first day, to simulate.

    solver_options      A dictionary of parameters regulating the behaviour of
                        the MILP solver used.
    run_lmps            Whether to solve for locational marginal prices after
                        each real-time economic dispatch.
    mipgap      The minimum quality of each optimization model solution, given
                as the ratio of the optimality gap to its objective value.
    reserve_factor      The proportion of forecasted load demand that must be
                        available as thermal generator headroom at each time
                        step in case of unexpected shortfalls.

    output_detail       The amount of information included in the results file.
    prescient_sced_forecasts        If used, forecast errors are removed from
                                    real-time dispatches.
    ruc_prescience_hour         Before this hour, forecasts used by unit
                                commitments will be made more accurate.
    ruc_execution_hour          At which time during the preceding day the
                                day-ahead unit commitment is run (except for
                                the first day).

    ruc_every_hours         How often unit commitments are executed.
    ruc_horizon             For how many hours each unit commitment is run.
    sced_horizon            For how many hours each real-time economic dispatch
                            is run (including the current hour).

    lmp_shortfall_costs     Whether or not calculation of locational marginal
                            prices includes reserve shortfall costs.
    enforce_sced_shutdown_curves        Whether to use generator shutdown
                                        constraints in dispatches.

    no_startup_shutdown_curves          Refrain from inferring startup and
                                        shutdown costs for thermal generators?

    init_ruc_file       Use these cached unit commitments for the first day
                        instead of solving for them from the given forecasts.
    verbosity           How many info messages to print during simulation.
    output_max_decimals     Precision to use for values saved to file.

    create_plots        Save visualizations of grid behaviour to file?
    renew_costs         Use cost curves for renewable generators instead of
                        assuming no costs.
    save_to_csv         Use .csv format for results saved to file instead of
                        Python-pickled archives (.p.gz).
    last_condition_file     Save the final generator states to file. Especially
                            useful for use as `init_ruc_file` above.

    """

    # model formulations used by Egret, with separate types of models defined
    # for unit commitments and economic dispatches
    ruc_formulations = dict(
        params_forml='default_params',
        status_forml='garver_3bin_vars',
        power_forml='garver_power_vars',
        reserve_forml='garver_power_avail_vars',
        generation_forml='pan_guan_gentile_KOW_generation_limits',
        ramping_forml='damcikurt_ramping',
        production_forml='KOW_production_costs_tightened',
        updown_forml='rajan_takriti_UT_DT',
        startup_forml='KOW_startup_costs',
        network_forml='ptdf_power_flow'
        )

    sced_formulations = dict(
        params_forml='default_params',
        status_forml='garver_3bin_vars',
        power_forml='garver_power_vars',
        reserve_forml='MLR_reserve_vars',
        generation_forml='MLR_generation_limits',
        ramping_forml='damcikurt_ramping',
        production_forml='CA_production_costs',
        updown_forml='rajan_takriti_UT_DT',
        startup_forml='MLR_startup_costs',
        network_forml='ptdf_power_flow'
        )

    supported_solvers = ['xpress', 'xpress_direct', 'xpress_persistent',
                         'gurobi', 'gurobi_direct', 'gurobi_persistent',
                         'cplex', 'cplex_direct', 'cplex_persistent',
                         'cbc', 'glpk']
    supported_persistent_solvers = ('xpress', 'gurobi', 'cplex')

    @classmethod
    def _verify_solver(cls, solver_type: str, solver_lbl: str) -> str:
        """Checks that the given MILP solver is available for use."""

        if solver_type not in cls.supported_solvers:
            raise RuntimeError("Unknown {} solver `{}` specified!".format(
                solver_lbl, solver_type))

        if solver_type in cls.supported_persistent_solvers:
            available = SolverFactory(solver_type + '_persistent').available()

            if not available:
                print(f"WARNING: {solver_lbl} Solver {solver_type} supports "
                      f"persistence, which improves the performance of "
                      f"Prescient. Consider installing the Python bindings "
                      f"for {solver_type}.")

            else:
                solver_type += '_persistent'

        if not SolverFactory(solver_type).available():
            raise RuntimeError(
                f"Solver {solver_type} is not available to Pyomo!")

        return solver_type

    def __init__(self,
                 template_data: dict, gen_data: pd.DataFrame,
                 load_data: pd.DataFrame, out_dir: Path | str | None,
                 start_date: datetime.date, num_days: int, solver: str,
                 solver_options: dict, run_lmps: bool, mipgap: float,
                 load_shed_penalty: float, reserve_shortfall_penalty: float,
                 reserve_factor: float, output_detail: int,
                 prescient_sced_forecasts: bool, ruc_prescience_hour: int,
                 ruc_execution_hour: int, ruc_every_hours: int,
                 ruc_horizon: int, sced_horizon: int,
                 lmp_shortfall_costs: bool,
                 enforce_sced_shutdown_ramprate: bool,
                 no_startup_shutdown_curves: bool,
                 init_ruc_file: str | Path | None, verbosity: int,
                 output_max_decimals: int, create_plots: bool,
                 renew_costs, save_to_csv, last_conditions_file) -> None:

        self._ruc_solver = self._verify_solver(solver, 'RUC')
        self._sced_solver = self._verify_solver(solver, 'SCED')

        self.run_lmps = run_lmps
        self.solver_options = solver_options
        self.sced_horizon = sced_horizon
        self.lmp_shortfall_costs = lmp_shortfall_costs

        self._hours_in_objective = None
        self._current_timestep = None
        #time dictionary for profiling
        self.simulation_times = {'Init': 0., 'Plan': 0., 'Sim': 0.}

        # if cost curves for renewable generators are given, use alternate
        # model formulations that do not assume no costs for renewables
        # only change it at unit commitment phase
        if renew_costs is not None:
            self.ruc_formulations['params_forml'] = 'renewable_cost_params'
            self.ruc_formulations[
                'production_forml'] = 'KOW_Vatic_production_costs_tightened'

        self._data_provider = PickleProvider(
            template_data, gen_data, load_data, load_shed_penalty,
            reserve_shortfall_penalty, reserve_factor,
            prescient_sced_forecasts, ruc_prescience_hour, ruc_execution_hour,
            ruc_every_hours, ruc_horizon, enforce_sced_shutdown_ramprate,
            no_startup_shutdown_curves, verbosity, start_date, num_days,
            renew_costs
            )

        self._sced_frequency_minutes = self._data_provider.data_freq
        self._actuals_step_frequency = 60

        self.init_ruc_file = init_ruc_file
        self.verbosity = verbosity

        self._simulation_state = VaticSimulationState(
            ruc_execution_hour, ruc_every_hours, self._sced_frequency_minutes)
        self._prior_sced_instance = None
        self._ptdf_manager = VaticPTDFManager()

        self._time_manager = VaticTimeManager(
            self._data_provider.first_day, self._data_provider.final_day,
            ruc_execution_hour, ruc_every_hours, ruc_horizon,
            self._sced_frequency_minutes
            )

        self._stats_manager = StatsManager(out_dir, output_detail, verbosity,
                                           self._data_provider.init_model,
                                           output_max_decimals, create_plots,
                                           save_to_csv, last_conditions_file)

        self.ruc_model = UCModel(mipgap, output_solver_logs=verbosity > 1,
                                 symbolic_solver_labels=True,
                                 **self.ruc_formulations)
        self.sced_model = UCModel(mipgap, output_solver_logs=verbosity > 1,
                                  symbolic_solver_labels=True,
                                  **self.sced_formulations)

    @profile
    def simulate(self) -> dict[str, pd.DataFrame]:
        """Top-level runner of a simulation's alternating RUCs and SCEDs.

        See prescient.simulator.Simulator.simulate() for the original
        implementation of this logic.

        """
        simulation_start_time = time.time()

        # create commitments for the first day using an initial unit commitment
        self.initialize_oracle()
        self.simulation_times['Init'] += time.time() - simulation_start_time

        # simulate each time period
        for time_step in self._time_manager.time_steps():
            self._current_timestep = time_step

            # run the day-ahead RUC at some point in the day before
            if time_step.is_planning_time:
                plan_start_time = time.time()

                self.call_planning_oracle()
                self.simulation_times['Plan'] += time.time() - plan_start_time

            # run the SCED to simulate this time step
            oracle_start_time = time.time()
            self.call_oracle()
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

        return self._stats_manager.save_output(sim_time)

    @profile
    def initialize_oracle(self) -> None:
        """Creates a day-ahead unit commitment for the simulation's first day.

        This method is a merge of
        prescient.simulator.OracleManager.call_initialization_oracle
        and OracleManager._generate_ruc.

        """
        first_step = self._time_manager.get_first_timestep()

        # if an initial RUC file has been given and it exists then load the
        # pre-solved RUC from it...
        if self.init_ruc_file and self.init_ruc_file.exists():
            sim_actuals = self.create_simulation_actuals(first_step)

            with open(self.init_ruc_file, 'rb') as f:
                ruc = pickle.load(f)

        # ...otherwise, solve the initial RUC
        else:
            sim_actuals, ruc = self.solve_ruc(first_step)

        # if an initial RUC file has been given and it doesn't already exist
        # then save the solved RUC for future use
        if self.init_ruc_file and not self.init_ruc_file.exists():
            with open(self.init_ruc_file, 'wb') as f:
                pickle.dump(ruc, f, protocol=-1)

        self._simulation_state.apply_initial_ruc(ruc, sim_actuals)
        self._stats_manager.collect_ruc_solution(first_step, ruc)

    def call_planning_oracle(self) -> None:
        """Creates a day-ahead unit commitment for the simulation's next day.

        This method is adapted from OracleManager.call_planning_oracle.

        """
        projected_state = self._get_projected_state()

        # find out when this unit commitment will come into effect and solve it
        uc_datetime = self._time_manager.get_uc_activation_time(
            self._current_timestep)
        sim_actuals, ruc = self.solve_ruc(VaticTime(uc_datetime, False, False),
                                          projected_state)

        self._simulation_state.apply_planning_ruc(ruc, sim_actuals)
        self._stats_manager.collect_ruc_solution(self._current_timestep, ruc)

    @profile
    def call_oracle(self) -> None:
        """Solves the real-time economic dispatch for the current time step.

        This method is adapted from OracleManager.call_operation_oracle.

        """
        if self.verbosity > 0:
            print("\nSolving SCED instance")

        current_sced_instance = self.solve_sced(hours_in_objective=1,
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

    def perturb_oracle(
            self,
            perturb_dict: dict[str, float], run_lmps: bool = False,
            ) -> dict[str, float | dict]:
        """Simulates a perturbed economic dispatch for current time step."""

        sced_model_data = self._data_provider.create_sced_instance(
            self._simulation_state, sced_horizon=self.sced_horizon)

        # for each asset with forecastable output/demand values, check to see
        # if we want to perturb it
        for k, sced_data in sced_model_data.get_forecastables():
            if k[1] in perturb_dict:
                if k[0] in {'p_max', 'p_load'}:
                    sced_data[0] = max(sced_data[0] + perturb_dict[k[1]], 0.)

                # handles case of renewables like RTPVs which have
                # p_min equal to p_max
                if k[0] == 'p_min' and sced_data[0] > 0.:
                    sced_data[0] = max(sced_data[0] + perturb_dict[k[1]], 0.)

        # proceed as we would otherwise but don't update the state of the sim,
        # just run the SCED we would have run with the perturbations added
        self._ptdf_manager.mark_active(sced_model_data)

        self.sced_model.generate_model(
            sced_model_data, relax_binaries=False,
            ptdf_options=self._ptdf_manager.sced_ptdf_options,
            ptdf_matrix_dict=self._ptdf_manager.PTDF_matrix_dict,
            objective_hours=1
            )

        # update in case lines were taken out
        self._ptdf_manager.PTDF_matrix_dict = self.sced_model.pyo_instance._PTDFs

        sced_results = self.sced_model.solve_model(self._sced_solver,
                                                   self.solver_options)

        # solve for locational marginal prices if necessary
        if run_lmps:
            lmp_sced = self.solve_lmp(sced_results)
        else:
            lmp_sced = None

        self._stats_manager.collect_sced_solution(self._current_timestep,
                                                  sced_results, lmp_sced,
                                                  pre_quickstart_cache=None)

        return self._stats_manager._sced_stats[self._current_timestep]

    @profile
    def solve_ruc(
            self,
            time_step: VaticTime,
            sim_state_for_ruc: VaticSimulationState | None = None
            ) -> tuple[VaticModelData, VaticModelData]:

        ruc_model_data = self._data_provider.create_deterministic_ruc(
            time_step, sim_state_for_ruc)
        self._ptdf_manager.mark_active(ruc_model_data)

        self.ruc_model.generate_model(
            ruc_model_data, relax_binaries=False,
            ptdf_options=self._ptdf_manager.ruc_ptdf_options,
            ptdf_matrix_dict=self._ptdf_manager.PTDF_matrix_dict
            )

        # update in case lines were taken out
        # TODO: why is this necessary?
        self._ptdf_manager.PTDF_matrix_dict = self.ruc_model.pyo_instance._PTDFs

        # TODO: better error handling
        try:
            ruc_plan = self.ruc_model.solve_model(self._ruc_solver,
                                                  self.solver_options)

        except:
            print("Failed to solve deterministic RUC instance - likely "
                  "because no feasible solution exists!")

            output_filename = "bad_ruc.json"
            ruc_model_data.write(output_filename)
            print("Wrote failed RUC model to file=" + output_filename)
            raise

        self._ptdf_manager.update_active(ruc_plan)
        # TODO: add the reporting stuff in
        #  egret_plugin._solve_deterministic_ruc

        # the RUC instance to simulate only exists to store the actual demand
        # and renewables outputs to be realized during the course of a day. it
        # also serves to provide a concrete instance, from which static data
        # and topological features of the system can be extracted.
        # IMPORTANT: This instance should *not* be passed to any method
        #            involved in the creation of economic dispatch instances,
        #            as that would enable those instances to be prescient.

        if self.verbosity > 0:
            print("\nExtracting scenario to simulate")

        return self.create_simulation_actuals(time_step), ruc_plan

    @profile
    def solve_sced(self,
                   hours_in_objective: int,
                   sced_horizon: int) -> VaticModelData:

        sced_model_data = self._data_provider.create_sced_instance(
            self._simulation_state, sced_horizon=sced_horizon)
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
        self._ptdf_manager.update_active(sced_results)

        return sced_results

    def solve_lmp(self, sced_instance: VaticModelData) -> VaticModelData:
        lmp_sced_instance = deepcopy(sced_instance)

        # in the case of a shortfall in meeting demand or the reserve
        # requirement, the price skyrockets, so we set max price values
        for cost_lbl, max_price in zip(
                ['load_mismatch_cost', 'reserve_shortfall_cost'],
                [10000., 1000.]
                ):
            shortfall_cost = lmp_sced_instance.get_system_attr(
                cost_lbl, max_price)

            if shortfall_cost >= max_price:
                lmp_sced_instance.set_system_attr(cost_lbl, max_price)

        # often we want to avoid having the reserve requirement shortfall make
        # any impact on the prices whatsoever
        if not self.lmp_shortfall_costs:
            lmp_sced_instance.set_system_attr('reserve_shortfall_cost', 0)

        if self.sced_model.pyo_instance is None:
            self._ptdf_manager.mark_active(lmp_sced_instance)

            self.sced_model.generate_model(
                lmp_sced_instance, relax_binaries=True,
                ptdf_options=self._ptdf_manager.lmpsced_ptdf_options,
                ptdf_matrix_dict=self._ptdf_manager.PTDF_matrix_dict,
                objective_hours=self._hours_in_objective
                )

        else:
            uc_instance_binary_relaxer(self.sced_model.pyo_instance,
                                       self.sced_model.solver)

            ## reset the penalties
            update_obj = False
            base_MVA = lmp_sced_instance.get_system_attr('baseMVA')
            new_load_penalty = base_MVA * lmp_sced_instance.get_system_attr(
                'load_mismatch_cost')

            if not math.isclose(
                    new_load_penalty,
                    self.sced_model.pyo_instance.LoadMismatchPenalty.value
                    ):
                self.sced_model.pyo_instance.LoadMismatchPenalty.value = new_load_penalty
                update_obj = True

            new_reserve_penalty = base_MVA * lmp_sced_instance.get_system_attr(
                'reserve_shortfall_cost')

            if not math.isclose(
                    new_reserve_penalty,
                    self.sced_model.pyo_instance.ReserveShortfallPenalty.value
                    ):
                self.sced_model.pyo_instance.ReserveShortfallPenalty.value = new_reserve_penalty
                update_obj = True

            self.sced_model.pyo_instance.model_data = lmp_sced_instance

            if update_obj and isinstance(self.sced_model.solver,
                                         PersistentSolver):
                self.sced_model.solver.set_objective(
                    self.sced_model.pyo_instance.TotalCostObjective)

        self.sced_model.pyo_instance.dual = PyomoSuffix(
            direction=PyomoSuffix.IMPORT)

        try:
            lmp_sced_results = self.sced_model.solve_model(
                solver_options=self.solver_options, relaxed=True,
                set_instance=(self.sced_model.pyo_instance is None),
                )

        except:
            print("Some issue with LMP SCED, writing instance")
            quickstart_uc_filename = (self.options.output_directory
                                      + os.sep + "FAILED_LMP_SCED.json")

            lmp_sced_instance.write(quickstart_uc_filename)
            print(f"Problematic LMP SCED written to {quickstart_uc_filename}")
            raise

        return lmp_sced_results

    def _get_projected_state(self) -> VaticSimulationState:
        """Gets the projection of the simulation state after the plan delay."""

        # if there is no RUC delay, use the current state as is
        if self._simulation_state.ruc_delay == 0:
            if self.verbosity > 0:
                print("\nDrawing UC initial conditions for {} from prior SCED "
                      "instance.".format(self._current_timestep.when))

            return self._simulation_state

        # determine the SCED execution mode, in terms of how discrepancies
        # between forecast and actuals are handled. prescient processing is
        # identical in the case of deterministic and stochastic RUC.
        # persistent processing differs, as there is no point forecast
        # for stochastic RUC.

        if self.verbosity > 0:
            uc_datetime = self._time_manager.get_uc_activation_time(
                self._current_timestep)

            print("Creating and solving SCED to determine UC initial "
                  "conditions for date:", str(uc_datetime.date()),
                  "hour:", uc_datetime.hour)

            if self._data_provider.prescient_sced_forecasts:
                print("Using prescient forecast error model when projecting "
                      "demand and renewables in SCED\n")
            else:
                print("Using persistent forecast error model when projecting "
                      "demand and renewables in SCED\n")

        #TODO: the projected SCED probably doesn't have to be run for a full
        #       24 hours - just enough to get you to midnight and a few hours
        #       beyond (to avoid end-of-horizon effects).
        #       But for now we run for 24 hours.
        proj_hours = min(24, self._simulation_state.timestep_count)
        proj_sced_instance = self.solve_sced(hours_in_objective=proj_hours,
                                             sced_horizon=proj_hours)

        return VaticStateWithScedOffset(self._simulation_state,
                                        proj_sced_instance,
                                        self._simulation_state.ruc_delay)

    def create_simulation_actuals(self,
                                  time_step: VaticTime) -> VaticModelData:
        """
        merge of EgretEngine.create_simulation_actuals
             and egret_plugin.create_simulation_actuals

        Get an Egret model consisting of data to be treated as actuals,
        starting at a given time.
        """

        # Convert time string to time
        start_time = datetime.datetime.combine(
            time_step.date(), datetime.time(time_step.hour()))

        # Get a new model
        total_step_count = self._data_provider.ruc_horizon * 60
        total_step_count //= self._actuals_step_frequency

        if time_step.hour() == 0:
            sim_model = self._data_provider.get_populated_model(
                use_actuals=True, start_time=start_time,
                num_time_periods=total_step_count
                )

        else:
            # only get up to 24 hours of data, then copy it
            timesteps_per_day = 24 * 60 // self._actuals_step_frequency
            steps_to_request = min(timesteps_per_day, total_step_count)

            sim_model = self._data_provider.get_populated_model(
                use_actuals=True, start_time=start_time,
                num_time_periods=steps_to_request
                )

            for _, vals in sim_model.get_forecastables():
                for t in range(timesteps_per_day, total_step_count):
                    vals[t] = vals[t - timesteps_per_day]

        return sim_model
