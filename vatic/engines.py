
from prescient.engine.egret.engine import EgretEngine
from egret.models.unit_commitment import (
    _solve_unit_commitment, _save_uc_results)
from egret.data.model_data import ModelData
from egret.models.unit_commitment import (
    create_tight_unit_commitment_model, _get_uc_model)
from egret.common.lazy_ptdf_utils import uc_instance_binary_relaxer
from egret.common.log import logger as egret_logger

from prescient.simulator.time_manager import PrescientTime
from prescient.data.providers.dat_data_provider import DatDataProvider
from prescient.data.simulation_state.state_with_offset import StateWithOffset
from prescient.simulator import TimeManager, DataManager, StatsManager
from prescient.engine.egret.data_extractors import ScedDataExtractor
from prescient.engine.modeling_engine import ForecastErrorMethod
from prescient.engine.egret.egret_plugin import (
    _copy_initial_state_into_model, _ensure_reserve_factor_honored,
    create_sced_instance, _zero_out_costs,
    )

from prescient.engine.forecast_helper import get_forecastables
from prescient.engine.egret.reporting import (
    report_initial_conditions_for_deterministic_ruc,
    report_demand_for_deterministic_ruc
    )

from prescient.engine.egret.ptdf_manager import PTDFManager
from prescient.data.simulation_state import SimulationState
from prescient.simulator.data_manager import RucPlan
import pyomo.environ as pe

import os
import datetime
from datetime import timedelta
import dateutil
from typing import Union
import logging
import time
from ast import literal_eval


class Simulator(EgretEngine):

    _ptdf_manager: PTDFManager
    _current_state: Union[None, SimulationState]

    def __init__(self, options=None):
        self.simulation_start_time = time.time()
        self.simulation_end_time = None

        self.options = options
        self.data = None
        self.model = None # Pyomo.ConcreteModel
        self._setup_solvers(options)

        self._data_provider = DatDataProvider()
        self._data_provider.initialize(options)

        self._data_manager = DataManager()
        self._data_manager.initialize(self, options)

        self._time_manager = TimeManager()
        self._time_manager.initialize(options)

        self._current_state = None
        self._sced_extractor = ScedDataExtractor()
        self._ptdf_manager = PTDFManager()

        self._stats_manager = StatsManager()
        self._stats_manager.initialize(options)

        self._actuals_step_frequency = 60
        if options.simulate_out_of_sample:
            self._actuals_step_frequency = self._data_provider\
                .negotiate_data_frequency(options.sced_frequency_minutes)

        self.network_constraints = 'ptdf_power_flow'
        self.formulation_list = [
            'garver_3bin_vars',
            'garver_power_vars',
            'MLR_reserve_vars',
            'MLR_generation_limits',
            'damcikurt_ramping',
            'CA_production_costs',
            'rajan_takriti_UT_DT',
            'MLR_startup_costs',
            self.network_constraints,
            ]

        self._last_sced_pyo_model = None
        self._last_sced_pyo_solver = None

    def simulate(self):
        self.initialize_oracle()

        for time_step in self._time_manager.time_steps():
            self._stats_manager.begin_timestep(time_step)
            self._data_manager.update_time(time_step)

            if time_step.is_planning_time:
                self.call_planning_oracle(time_step)

            if time_step.is_ruc_activation_time:
                self._data_manager.activate_pending_ruc(self.options)

            self.call_oracle(time_step)

            self._stats_manager.end_timestep(time_step)

        self._stats_manager.end_simulation()

        print("Simulation Complete")
        self.simulation_end_time = time.time()
        print("Total simulation time: {:.2f} seconds".format(
            self.simulation_end_time - self.simulation_start_time))

    def generate_ruc(self, time_step, sim_state_for_ruc=None):
        ruc_plan = self.solve_deterministic_ruc(self.create_deterministic_ruc(
            time_step, current_state=sim_state_for_ruc), time_step)

        if self.options.compute_market_settlements:
            print("Solving day-ahead market")
            ruc_market = self.create_and_solve_day_ahead_pricing(
                self.options, ruc_plan)

        else:
            ruc_market = None

        # the RUC instance to simulate only exists to store the actual demand and renewables outputs
        # to be realized during the course of a day. it also serves to provide a concrete instance,
        # from which static data and topological features of the system can be extracted.
        # IMPORTANT: This instance should *not* be passed to any method involved in the creation of
        #            economic dispatch instances, as that would enable those instances to be
        #            prescient.
        print("")
        print("Extracting scenario to simulate")

        simulation_actuals = self.create_simulation_actuals(time_step)
        result = RucPlan(simulation_actuals, ruc_plan, ruc_market)

        return result

    def get_first_time_step(self) -> PrescientTime:
        """port of TimeManager.get_first_time_step"""
        t0 = datetime.datetime.combine(
            dateutil.parser.parse(self.options.start_date).date(),
            datetime.time(0)
            )

        return PrescientTime(t0, False, False)

    def initialize_oracle(self):
        """
        merge of OracleManager.call_initialization_oracle
             and OracleManager._generate_ruc
        """

        ruc = self.generate_ruc(self._time_manager.get_first_time_step())
        self._data_manager.set_pending_ruc_plan(self.options, ruc)
        self._data_manager.activate_pending_ruc(self.options)

        return ruc

    def call_planning_oracle(self, time_step):
        projected_state = self._get_projected_state(time_step)
        uc_hour, uc_date = self._get_uc_activation_time(time_step)

        ruc = self.generate_ruc(
            PrescientTime(datetime.datetime.combine(
                uc_date, datetime.time(hour=uc_hour)), False, False),
            projected_state
            )
        self._data_manager.set_pending_ruc_plan(self.options, ruc)

        return ruc

    def _get_projected_state(self,
                             time_step: PrescientTime) -> SimulationState:
        ''' Get the simulation state as we project it will appear after the RUC delay '''

        ruc_delay = self._get_ruc_delay()

        # If there is no RUC delay, use the current state as is
        if ruc_delay == 0:
            print("")
            print("Drawing UC initial conditions for date:", time_step.date,
                  "hour:", time_step.hour, "from prior SCED instance.")
            return self._data_manager.current_state

        uc_hour, uc_date = self._get_uc_activation_time(time_step)

        print("")
        print(
            "Creating and solving SCED to determine UC initial conditions for date:",
            str(uc_date), "hour:", uc_hour)

        # determine the SCED execution mode, in terms of how discrepancies between forecast and actuals are handled.
        # prescient processing is identical in the case of deterministic and stochastic RUC.
        # persistent processing differs, as there is no point forecast for stochastic RUC.
        sced_forecast_error_method = ForecastErrorMethod.PRESCIENT  # always the default
        if self.options.run_sced_with_persistent_forecast_errors:
            print(
                "Using persistent forecast error model when projecting demand and renewables in SCED")
            sced_forecast_error_method = ForecastErrorMethod.PERSISTENT
        else:
            print(
                "Using prescient forecast error model when projecting demand and renewables in SCED")
        print("")

        # NOTE: the projected sced probably doesn't have to be run for a full 24 hours - just enough
        #       to get you to midnight and a few hours beyond (to avoid end-of-horizon effects).
        #       But for now we run for 24 hours.
        current_state = self._data_manager.current_state.get_state_with_step_length(
            60)
        projected_sced_instance = create_sced_instance(
            self._data_provider,
            current_state,
            self.options,
            sced_horizon=min(24, self._data_manager.current_state.timestep_count),
            forecast_error_method=sced_forecast_error_method,
            )

        self._hours_in_objective = min(
            24, self._data_manager.current_state.timestep_count)

        projected_sced_instance, solve_time = self.solve_sced_instance(
            projected_sced_instance)

        future_state = StateWithOffset(current_state, projected_sced_instance,
                                       ruc_delay)

        return future_state

    def _get_ruc_delay(self):
        return -(self.options.ruc_execution_hour
                 % (-self.options.ruc_every_hours))

    def _get_uc_activation_time(self, time_step):
        ''' Get the hour and date that a RUC generated at the given time will be activated '''
        ruc_delay = self._get_ruc_delay()
        activation_time = time_step.datetime + timedelta(hours=ruc_delay)

        return activation_time.hour, activation_time.date()

    def call_oracle(self, time_step):
        """port of OracleManager.call_operation_oracle"""

        # determine the SCED execution mode, in terms of how discrepancies between forecast and actuals are handled.
        # prescient processing is identical in the case of deterministic and stochastic RUC.
        # persistent processing differs, as there is no point forecast for stochastic RUC.
        if self.options.run_sced_with_persistent_forecast_errors:
            forecast_error_method = ForecastErrorMethod.PERSISTENT
        else:
            forecast_error_method = ForecastErrorMethod.PRESCIENT

        lp_filename = None
        if self.options.write_sced_instances:
            lp_filename = self.options.output_directory + os.sep + str(
                time_step.date) + \
                          os.sep + "sced_hour_" + str(time_step.hour) + ".lp"

        print("")
        print("Solving SCED instance")

        sced_horizon_timesteps = self.options.sced_horizon
        current_sced_instance = create_sced_instance(
            self._data_provider,
            self._data_manager.current_state.get_state_with_step_length(
                self.options.sced_frequency_minutes),
            self.options,
            sced_horizon=sced_horizon_timesteps,
            forecast_error_method=forecast_error_method,
            )

        self._hours_in_objective = 1

        current_sced_instance, solve_time = self.solve_sced_instance(
            current_sced_instance)

        pre_quickstart_cache = None

        if self.options.enable_quick_start_generator_commitment:
            # Determine whether we are going to run a quickstart optimization
            # TODO: Why the "if True" here?
            if True or engine.operations_data_extractor.has_load_shedding(
                    current_sced_instance):
                # Yep, we're doing it.  Cache data we can use to compare results with and without quickstart
                pre_quickstart_cache = engine.operations_data_extractor.get_pre_quickstart_data(
                    current_sced_instance)

                # TODO: report solution/load shedding before unfixing Quick Start Generators
                # print("")
                # print("SCED Solution before unfixing Quick Start Generators")
                # print("")
                # self.report_sced_stats()

                # Set up the quickstart run, allowing quickstart generators to turn on
                print("Re-solving SCED after unfixing Quick Start Generators")
                current_sced_instance = self.engine.enable_quickstart_and_solve(
                    options, current_sced_instance)

        print("Solving for LMPs")
        lmp_sced = self.create_and_solve_lmp(current_sced_instance)

        self._data_manager.apply_sced(self.options, current_sced_instance)


        ops_stats = self._stats_manager.collect_operations(
            current_sced_instance, solve_time, lmp_sced, pre_quickstart_cache,
            self._sced_extractor
            )
        self._report_sced_stats(ops_stats)

        if self.options.compute_market_settlements:
            self.simulator.stats_manager.collect_market_settlement(
                current_sced_instance,
                self.engine.operations_data_extractor,
                self.simulator.data_manager.ruc_market_active,
                time_step.hour % options.ruc_every_hours)

        return current_sced_instance


    def create_deterministic_ruc(self, time_step: PrescientTime,
                                 output_init_conditions=False,
                                 current_state=None) -> ModelData:
        """
        merge of EgretEngine.create_deterministic_ruc
             and egret_plugin.create_deterministic_ruc
        """

        start_time = datetime.datetime.combine(
            time_step.date, datetime.time(hour=time_step.hour))

        # Create a new model
        ruc_model = self._data_provider.get_initial_model(
            self.options, self.options.ruc_horizon, minutes_per_timestep=60)

        # Populate the T0 data
        if current_state is None or current_state.timestep_count == 0:
            self._data_provider.populate_initial_state_data(self.options,
                                                            start_time.date(),
                                                            ruc_model)

        else:
            _copy_initial_state_into_model(self.options,
                                           current_state, ruc_model)

        # Populate forecasts
        copy_first_day = not self.options.run_ruc_with_next_day_data
        copy_first_day &= time_step.hour != 0

        forecast_request_count = 24
        if not copy_first_day:
            forecast_request_count = self.options.ruc_horizon

        self._data_provider.populate_with_forecast_data(
            self.options, start_time, forecast_request_count,
            time_period_length_minutes=60, model=ruc_model
            )

        # Make some near-term forecasts more accurate
        ruc_delay = -(self.options.ruc_execution_hour
                      % -self.options.ruc_every_hours)

        # TODO: when does this apply?
        if self.options.ruc_prescience_hour > ruc_delay + 1:
            improved_hour_count = self.options.ruc_prescience_hour - ruc_delay
            improved_hour_count -= 1
            future_actuals = current_state.get_future_actuals()

            for forecast, actuals in zip(get_forecastables(ruc_model),
                                         future_actuals):
                for t in range(improved_hour_count):
                    forecast_portion = (ruc_delay + t)
                    forecast_portion /= self.options.ruc_prescience_hour

                    actuals_portion = 1 - forecast_portion
                    forecast[t] = forecast_portion * forecast[t]
                    forecast[t] += actuals_portion * actuals[t]

        # Ensure the reserve requirement is satisfied
        _ensure_reserve_factor_honored(self.options, ruc_model,
                                       range(forecast_request_count))

        # Copy from first 24 to second 24, if necessary
        if copy_first_day:
            for vals, in get_forecastables(ruc_model):
                for t in range(24, self.options.ruc_horizon):
                    vals[t] = vals[t - 24]

        if output_init_conditions:
            report_initial_conditions_for_deterministic_ruc(ruc_model)
            report_demand_for_deterministic_ruc(ruc_model,
                                                self.options.ruc_every_hours)

        return ruc_model

    def solve_deterministic_ruc(self, ruc_model, time_step):
        """
        merge of EgretEngine.solve_deterministic_ruc
             and egret_plugin.solve_deterministic_ruc
        """

        self._ptdf_manager.mark_active(ruc_model)

        pyo_model = create_tight_unit_commitment_model(
            ruc_model, ptdf_options=self._ptdf_manager.ruc_ptdf_options,
            PTDF_matrix_dict=self._ptdf_manager.PTDF_matrix_dict
            )

        # update in case lines were taken out
        self._ptdf_manager.PTDF_matrix_dict = pyo_model._PTDFs

        # TODO: better error handling
        try:
            ruc_results, pyo_results, _ = self.call_solver(
                pyo_model, self._ruc_solver,
                self.options.deterministic_ruc_solver_options
                )

        except:
            print("Failed to solve deterministic RUC instance - likely "
                  "because no feasible solution exists!")

            output_filename = "bad_ruc.json"
            ruc_model.write(output_filename)
            print("Wrote failed RUC model to file=" + output_filename)
            raise

        self._ptdf_manager.update_active(ruc_results)
        # TODO: add the reporting stuff in
        #  egret_plugin._solve_deterministic_ruc

        return ruc_results

    def create_simulation_actuals(self, time_step):
        """
        merge of EgretEngine.create_simulation_actuals
             and egret_plugin.create_simulation_actuals

        Get an Egret model consisting of data to be treated as actuals,
        starting at a given time.

        """

        # Convert time string to time
        start_time = datetime.datetime.combine(
            time_step.date, datetime.time(hour=time_step.hour))

        # Pick whether we're getting actuals or forecasts
        if self.options.simulate_out_of_sample:
            get_data_func = self._data_provider.populate_with_actuals
        else:
            print("")
            print(
                "***WARNING: Simulating the forecast scenario when running deterministic RUC - "
                "time consistency across midnight boundaries is not guaranteed, and may lead to threshold events.")
            get_data_func = self._data_provider.populate_with_forecast_data

        # Get a new model
        total_step_count = self.options.ruc_horizon * 60
        total_step_count //= self._actuals_step_frequency

        model = self._data_provider.get_initial_model(
            self.options, total_step_count, self._actuals_step_frequency)

        # Fill it in with data
        self._data_provider.populate_initial_state_data(
            self.options, start_time.date(), model)

        if time_step.hour == 0:
            get_data_func(self.options, start_time, total_step_count,
                          self._actuals_step_frequency, model)

        else:
            # only get up to 24 hours of data, then copy it
            timesteps_per_day = 24 * 60 / self._actuals_step_frequency
            steps_to_request = min(timesteps_per_day, total_step_count)

            get_data_func(self.options, start_time, steps_to_request,
                          self._actuals_step_frequency, model)

            for vals, in get_forecastables(model):
                for t in range(timesteps_per_day, total_step_count):
                    vals[t] = vals[t - timesteps_per_day]

        return model

    def create_sced_instance(self, ):
        pass

    def solve_sced_instance(self, sced_instance):
        """
        merge of EgretEngine.solve_sced_instance
        and engine.create_sced_uc_model
        """

        if self._hours_in_objective > 10:
            ptdf_options = self._ptdf_manager.look_ahead_sced_ptdf_options
        else:
            ptdf_options = self._ptdf_manager.sced_ptdf_options

        self._ptdf_manager.mark_active(sced_instance)

        pyo_model = _get_uc_model(
            sced_instance, self.formulation_list, relax_binaries=False,
            ptdf_options=ptdf_options,
            PTDF_matrix_dict=self._ptdf_manager.PTDF_matrix_dict
            )

        # update in case lines were taken out
        self._ptdf_manager.PTDF_matrix_dict = pyo_model._PTDFs
        _zero_out_costs(pyo_model, self._hours_in_objective)

        try:
            sced_results, sced_time, pyo_solver = self.call_solver(
                pyo_model, self._sced_solver, self.options.sced_solver_options)

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
                infeasible_sced_filename = options.output_directory + os.sep + "FAILED_SCED.json"
            sced_instance.write(infeasible_sced_filename)
            print(
                "Problematic SCED instance written to file=" + infeasible_sced_filename)
            raise

        self._ptdf_manager.update_active(sced_results)
        self._last_sced_pyo_model = pyo_model
        self._last_sced_pyo_solver = pyo_solver

        return sced_results, sced_time

    def create_and_solve_lmp(self, sced_instance):
        lmp_sced_instance = sced_instance.clone()

        # In case of demand shortfall, the price skyrockets, so we threshold the value.
        if 'load_mismatch_cost' not in lmp_sced_instance.data['system'] or \
                lmp_sced_instance.data['system']['load_mismatch_cost'] > \
                    self.options.price_threshold:
            lmp_sced_instance.data['system']['load_mismatch_cost'] = self.options.price_threshold

        # In case of reserve shortfall, the price skyrockets, so we threshold the value.
        if 'reserve_shortfall_cost' not in lmp_sced_instance.data['system'] or \
                lmp_sced_instance.data['system']['reserve_shortfall_cost'] > \
                    self.options.reserve_price_threshold:
            lmp_sced_instance.data['system']['reserve_shortfall_cost'] = \
                    self.options.reserve_price_threshold

        if self._last_sced_pyo_model is None:
            self._ptdf_manager.mark_active(lmp_sced_instance)

            pyo_model = _get_uc_model(
                lmp_sced_instance, self.formulation_list, relax_binaries=True,
                ptdf_options=self._ptdf_manager.lmpsced_ptdf_options,
                PTDF_matrix_dict=self._ptdf_manager.PTDF_matrix_dict
                )

            pyo_solver = self._sced_solver
            _zero_out_costs(pyo_model, self._hours_in_objective)

        else:
            pyo_model = self._last_sced_pyo_model
            pyo_solver = self._last_sced_pyo_solver
            self._transform_for_lmp(pyo_model, pyo_solver, lmp_sced_instance)

        pyo_model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

        try:
            lmp_sced_results, _, _ = self.call_solver(
                pyo_model, pyo_solver, self.options.sced_solver_options,
                relaxed=True, set_instance=(self._last_sced_pyo_model is None)
                )

        except:
            print("Some issue with LMP SCED, writing instance")
            quickstart_uc_filename = self.options.output_directory+os.sep+"FAILED_LMP_SCED.json"
            lmp_sced_instance.write(quickstart_uc_filename)
            print(f"Problematic LMP SCED written to {quickstart_uc_filename}")
            raise

        return lmp_sced_results


    def _transform_for_lmp(self, pyo_model, pyo_solver, lmp_sced_instance):
        from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
        import math

        uc_instance_binary_relaxer(pyo_model, pyo_solver)

        ## reset the penalites
        system = lmp_sced_instance.data['system']

        update_obj = False

        new_load_penalty = system['baseMVA'] * system['load_mismatch_cost']
        if not math.isclose(new_load_penalty, pyo_model.LoadMismatchPenalty.value):
            pyo_model.LoadMismatchPenalty.value = new_load_penalty
            update_obj = True

        new_reserve_penalty =  system['baseMVA'] * system['reserve_shortfall_cost']
        if not math.isclose(new_reserve_penalty, pyo_model.ReserveShortfallPenalty.value):
            pyo_model.ReserveShortfallPenalty.value = new_reserve_penalty
            update_obj = True

        pyo_model.model_data = lmp_sced_instance

        if update_obj and isinstance(pyo_solver, PersistentSolver):
            pyo_solver.set_objective(pyo_model.TotalCostObjective)

    def call_solver(self, model, solver, solver_options,
                    relaxed=False, set_instance=True):
        if not self.options.output_solver_logs:
            egret_logger.setLevel(logging.WARNING)

        solver_options_list = [opt.split('=') for opt in solver_options]
        solver_options_dict = {option: literal_eval(val)
                               for option, val in solver_options_list}

        m, results, solver = _solve_unit_commitment(
            m=model, solver=solver, mipgap=self.options.ruc_mipgap,
            timelimit=None, solver_tee=self.options.output_solver_logs,
            symbolic_solver_labels=self.options.symbolic_solver_labels,
            solver_options=solver_options_dict, solve_method_options=None,
            #solve_method_options=self.options.solve_method_options,
            relaxed=relaxed, set_instance=set_instance
            )

        md = _save_uc_results(m, relaxed)

        if hasattr(results, 'egret_metasolver_status'):
            time = results.egret_metasolver_status['time']
        else:
            time = results.solver.wallclock_time

        return md, time, solver

    def _report_sced_stats(self, ops_stats):
        print("Fixed costs:    %12.2f" % ops_stats.fixed_costs)
        print("Variable costs: %12.2f" % ops_stats.variable_costs)
        print("")

        if ops_stats.load_shedding != 0.0:
            print("Load shedding reported at t=%d -     total=%12.2f" % (1, ops_stats.load_shedding))
        if ops_stats.over_generation!= 0.0:
            print("Over-generation reported at t=%d -   total=%12.2f" % (1, ops_stats.over_generation))

        if ops_stats.reserve_shortfall != 0.0:
            print("Reserve shortfall reported at t=%2d: %12.2f" % (1, ops_stats.reserve_shortfall))
            print("Quick start generation capacity available at t=%2d: %12.2f" % (1, ops_stats.available_quickstart))
            print("")

        if ops_stats.renewables_curtailment > 0:
            print("Renewables curtailment reported at t=%d - total=%12.2f" % (1, ops_stats.renewables_curtailment))
            print("")

        print("Number on/offs:       %12d" % ops_stats.on_offs)
        print("Sum on/off ramps:     %12.2f" % ops_stats.sum_on_off_ramps)
        print("Sum nominal ramps:    %12.2f" % ops_stats.sum_nominal_ramps)
        print("")
