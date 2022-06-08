"""Running a simulation of alternating UC and ED grid optimization steps."""

import os
import datetime
import time
import math
import dill as pickle
from copy import deepcopy
from typing import Union, Tuple, Dict, Any, Callable

from .data_providers import (
    PickleProvider, AllocationPickleProvider, AutoAllocationPickleProvider)
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

    data_provider_class = PickleProvider

    supported_solvers = ['xpress', 'xpress_direct', 'xpress_persistent',
                         'gurobi', 'gurobi_direct', 'gurobi_persistent',
                         'cplex', 'cplex_direct', 'cplex_persistent',
                         'cbc', 'glpk']
    supported_persistent_solvers = ('xpress', 'gurobi', 'cplex')

    @classmethod
    def _verify_solver(cls, solver_type: str, solver_lbl: str) -> str:
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
                 template_data, gen_data, load_data, out_dir,
                 start_date, num_days, solver, solver_options, mipgap,
                 reserve_factor, light_output, prescient_sced_forecasts,
                 ruc_prescience_hour, ruc_execution_hour, ruc_every_hours,
                 ruc_horizon, sced_horizon, enforce_sced_shutdown_ramprate,
                 no_startup_shutdown_curves, init_ruc_file, verbosity,
                 output_max_decimals, create_plots):
        self._ruc_solver = self._verify_solver(solver, 'RUC')
        self._sced_solver = self._verify_solver(solver, 'SCED')

        self.solver_options = solver_options
        self.sced_horizon = sced_horizon
        self._hours_in_objective = None

        self.simulation_start_time = time.time()
        self.simulation_end_time = None
        self._current_timestep = None

        self._data_provider = self.data_provider_class(
            template_data, gen_data, load_data, reserve_factor,
            prescient_sced_forecasts, ruc_prescience_hour, ruc_execution_hour,
            ruc_every_hours, ruc_horizon, enforce_sced_shutdown_ramprate,
            no_startup_shutdown_curves, verbosity, start_date, num_days
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

        self._stats_manager = StatsManager(out_dir, light_output, verbosity,
                                           self._data_provider.init_model,
                                           output_max_decimals, create_plots)

        self.ruc_model = UCModel(mipgap, output_solver_logs=verbosity > 1,
                                 symbolic_solver_labels=True,
                                 **self.ruc_formulations)
        self.sced_model = UCModel(mipgap, output_solver_logs=verbosity > 1,
                                  symbolic_solver_labels=True,
                                  **self.sced_formulations)

    def simulate(self) -> None:
        """Top-level runner of a simulation's alternating RUCs and SCEDs.

        See prescient.simulator.Simulator.simulate() for the original
        implementation of this logic.

        """
        self.initialize_oracle()

        for time_step in self._time_manager.time_steps():
            self._current_timestep = time_step

            # run the day-ahead RUC at some point in the day before
            if time_step.is_planning_time:
                self.call_planning_oracle()

            # run the SCED to simulate this time step
            self.call_oracle()

        sim_runtime = time.time() - self.simulation_start_time
        if self.verbosity > 0:
            print("Simulation Complete")
            print("Total simulation time: {:.2f} seconds".format(sim_runtime))

        self._stats_manager.save_output(sim_runtime)

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
            with open(self.init_ruc_file, 'rb') as f:
                sim_actuals, ruc = pickle.load(f)

        # ...otherwise, solve the initial RUC
        else:
            sim_actuals, ruc = self.solve_ruc(first_step)

        # if an initial RUC file has been given and it doesn't already exist
        # then save the solved RUC for future use
        if self.init_ruc_file and not self.init_ruc_file.exists():
            with open(self.init_ruc_file, 'wb') as f:
                pickle.dump((sim_actuals, ruc), f, protocol=-1)

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

        lmp_sced = self.solve_lmp(current_sced_instance)
        self._simulation_state.apply_sced(current_sced_instance)
        self._prior_sced_instance = current_sced_instance

        self._stats_manager.collect_sced_solution(
            self._current_timestep, current_sced_instance, lmp_sced,
            pre_quickstart_cache=None
            )

    def solve_ruc(self, time_step, sim_state_for_ruc=None):

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

        try:
            sced_results = self.sced_model.solve_model(self._sced_solver,
                                                       self.solver_options)

        except:
            print("Some isssue with SCED, writing instance")
            print("Problematic SCED from to file")

            # for diagnostic purposes, save the failed SCED instance.
            if lp_filename is not None:
                if lp_filename.endswith(".json"):
                    infeasible_sced_filename = lp_filename[
                                               :-5] + ".FAILED.json"
                else:
                    infeasible_sced_filename = lp_filename + ".FAILED.json"

            else:
                infeasible_sced_filename = (self.options.output_directory
                                            + os.sep + "FAILED_SCED.json")

            sced_model_data.write(infeasible_sced_filename)
            print("Problematic SCED instance written to file="
                  + infeasible_sced_filename)

            raise

        self._ptdf_manager.update_active(sced_results)

        return sced_results

    #TODO: figure out how to produce bus-level LMPs
    def solve_lmp(self, sced_instance):
        lmp_sced_instance = deepcopy(sced_instance)

        # in the case of a shortfall in meeting demand or the reserve
        # requirement, the price skyrockets, so we set max price values
        for cost_lbl, max_price in zip(
                ['load_mismatch_cost', 'reserve_shortfall_cost'],
                [10000., 1000.]
                ):
            shortfall_cost = sced_instance.get_system_attr(cost_lbl, max_price)

            if shortfall_cost >= max_price:
                sced_instance.set_system_attr(cost_lbl, max_price)

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

    def _get_projected_state(self) -> VaticStateWithScedOffset:
        """
        Get the simulation state as we project it will appear
        after the RUC delay.
        """

        # If there is no RUC delay, use the current state as is
        if self._simulation_state.ruc_delay == 0:
            print("")
            print("Drawing UC initial conditions for date:",
                  self._current_timestep.date(),
                  "hour:", self._current_timestep.hour(), "from prior SCED instance.")

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

    def create_simulation_actuals(self, time_step):
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
            timesteps_per_day = 24 * 60 / self._actuals_step_frequency
            steps_to_request = min(timesteps_per_day, total_step_count)

            sim_model = self._data_provider.get_populated_model(
                use_actuals=True, start_time=start_time,
                num_time_periods=steps_to_request
                )

            for _, vals in sim_model.get_forecastables():
                for t in range(timesteps_per_day, total_step_count):
                    vals[t] = vals[t - timesteps_per_day]

        return sim_model


class AllocationSimulator(Simulator):

    ruc_formulations = dict(
        params_forml='renewable_cost_params',
        status_forml='garver_3bin_vars',
        power_forml='garver_power_vars',
        reserve_forml='garver_power_avail_vars',
        generation_forml='pan_guan_gentile_KOW_generation_limits',
        ramping_forml='damcikurt_ramping',
        production_forml='KOW_Vatic_production_costs_tightened',
        updown_forml='rajan_takriti_UT_DT',
        startup_forml='KOW_startup_costs',
        network_forml='ptdf_power_flow'
        )

    data_provider_class = AllocationPickleProvider


class AutoAllocationSimulator(AllocationSimulator):

    def __new__(cls, cost_vals, **sim_args):
        cls.data_provider_class = cls.auto_provider_factory(cost_vals)

        return super().__new__(cls)

    def __init__(self, cost_vals, **sim_args):
        super().__init__(**sim_args)

    @staticmethod
    def auto_provider_factory(cost_vals):
        NewCls = AutoAllocationPickleProvider

        ncosts = len(cost_vals) - 1
        if ncosts == 0:
            NewCls.cost_vals = [(1., float(cost_vals[0]))]

        else:
            NewCls.cost_vals = [(i / ncosts, float(c))
                                for i, c in enumerate(cost_vals)]

        return NewCls
