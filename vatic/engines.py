
import os
import datetime
import time
import math
from pathlib import Path
import dill as pickle
from typing import Union, Tuple, Dict, Any, Callable

from .data_providers import PickleProvider
from .simulation_state import VaticSimulationState
from .managers.reporting_manager import ReportingManager
from .models import UCModel

from prescient.engine.egret.engine import EgretEngine
from egret.data.model_data import ModelData as EgretModel
from egret.common.lazy_ptdf_utils import uc_instance_binary_relaxer
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.environ import Suffix as PyomoSuffix

from prescient.simulator.time_manager import PrescientTime
from prescient.data.simulation_state.state_with_offset import StateWithOffset
from prescient.simulator import TimeManager, StatsManager
from prescient.engine.egret.ptdf_manager import PTDFManager
from prescient.engine.egret.data_extractors import ScedDataExtractor
from prescient.engine.modeling_engine import ForecastErrorMethod


class Simulator(EgretEngine):

    def __init__(self,
                 in_dir, out_dir, light_output,
                 init_ruc_file, save_init_ruc, verbosity, prescient_options):
        self.simulation_start_time = time.time()
        self.simulation_end_time = None
        self._current_time = None

        self.options = prescient_options
        self.verbosity = verbosity
        self._setup_solvers(prescient_options)
        self._hours_in_objective = None

        self._data_provider = PickleProvider(in_dir, prescient_options)
        self.init_ruc_file = init_ruc_file
        self.save_init_ruc = save_init_ruc

        self._ruc_market_active = None
        self._ruc_market_pending = None
        self._simulation_state = VaticSimulationState(prescient_options)
        self._prior_sced_instance = None

        self._time_manager = TimeManager()
        self._time_manager.initialize(prescient_options)

        self._sced_extractor = ScedDataExtractor()
        self._ptdf_manager = PTDFManager()

        self._stats_manager = StatsManager()
        self._stats_manager.initialize(prescient_options)
        self._reporting_manager = ReportingManager(
            out_dir, light_output, self._stats_manager, prescient_options)

        self._actuals_step_frequency = 60
        if prescient_options.simulate_out_of_sample:
            if self._data_provider.data_freq != prescient_options.sced_frequency_minutes:
                raise ValueError(
                    "Given SCED frequency of `{}` minutes doesn't match what "
                    "is available in the data!".format(
                        prescient_options.sced_frequency_minutes)
                    )

            self._actuals_step_frequency = prescient_options.sced_frequency_minutes

        self.network_constraints = 'ptdf_power_flow'

        self.ruc_model = UCModel(
            params_forml='default_params',
            #params_forml='renewable_cost_params',
            status_forml='garver_3bin_vars',
            power_forml='garver_power_vars',
            reserve_forml='garver_power_avail_vars',
            generation_forml='pan_guan_gentile_KOW_generation_limits',
            ramping_forml='damcikurt_ramping',
            production_forml='KOW_production_costs_tightened',
            #production_forml='KOW_Vatic_production_costs_tightened',
            updown_forml='rajan_takriti_UT_DT',
            startup_forml='KOW_startup_costs',
            network_forml=self.network_constraints
            )

        self.sced_model = UCModel(
            params_forml='default_params',
            status_forml='garver_3bin_vars',
            power_forml='garver_power_vars',
            reserve_forml='MLR_reserve_vars',
            generation_forml='MLR_generation_limits',
            ramping_forml='damcikurt_ramping',
            production_forml='CA_production_costs',
            updown_forml='rajan_takriti_UT_DT',
            startup_forml='MLR_startup_costs',
            network_forml=self.network_constraints
            )

    def simulate(self):
        self.initialize_oracle()

        for time_step in self._time_manager.time_steps():
            self._stats_manager.begin_timestep(time_step)
            self._current_time = time_step

            if time_step.is_planning_time:
                self.call_planning_oracle(time_step)

            if time_step.is_ruc_activation_time:
                self._ruc_market_active = self._ruc_market_pending
                self._ruc_market_pending = None

            self.call_oracle(time_step)

            self._stats_manager.end_timestep(time_step)

        self._stats_manager.end_simulation()

        if self.verbosity > 0:
            self.simulation_end_time = time.time()

            print("Simulation Complete")
            print("Total simulation time: {:.2f} seconds".format(
                self.simulation_end_time - self.simulation_start_time))

        self._reporting_manager.save_output()

    def initialize_oracle(self):
        """
        merge of OracleManager.call_initialization_oracle
             and OracleManager._generate_ruc
        """

        if self.init_ruc_file:
            sim_actuals = self.create_simulation_actuals(
                self._time_manager.get_first_time_step())

            with open(self.init_ruc_file, 'rb') as f:
                ruc = pickle.load(f)

            ruc_market = None

        else:
            sim_actuals, ruc, ruc_market = self.solve_ruc(
                self._time_manager.get_first_time_step())

        if self.save_init_ruc:
            if not isinstance(self.save_init_ruc, (str, Path)):
                ruc_file = "init_ruc.p"
            else:
                ruc_file = Path(self.save_init_ruc)

            with open(ruc_file, 'wb') as f:
                pickle.dump(ruc, f, protocol=-1)

        self._simulation_state.apply_initial_ruc(ruc, sim_actuals)
        self._ruc_market_active = ruc_market

    def call_planning_oracle(self, time_step):
        projected_state = self._get_projected_state(time_step)
        uc_hour, uc_date = self._get_uc_activation_time(time_step)

        sim_actuals, ruc, ruc_market = self.solve_ruc(
            PrescientTime(
                datetime.datetime.combine(uc_date,
                                          datetime.time(hour=uc_hour)),
                False, False
                ),
            projected_state
            )

        self._simulation_state.apply_planning_ruc(ruc, sim_actuals)
        self._ruc_market_pending = ruc_market

    def call_oracle(self, time_step):
        """port of OracleManager.call_operation_oracle"""

        # determine the SCED execution mode, in terms of how discrepancies
        # between forecast and actuals are handled.
        # prescient processing is identical in the case of deterministic and
        # stochastic RUC. persistent processing differs, as there is no point
        # forecast for stochastic RUC.

        if self.options.run_sced_with_persistent_forecast_errors:
            forecast_error_method = ForecastErrorMethod.PERSISTENT
        else:
            forecast_error_method = ForecastErrorMethod.PRESCIENT

        lp_filename = None
        if self.options.write_sced_instances:
            lp_filename = self.options.output_directory + os.sep + str(
                time_step.date) + \
                          os.sep + "sced_hour_" + str(time_step.hour) + ".lp"

        if self.verbosity > 0:
            print("\nSolving SCED instance")

        cur_state = self._simulation_state.get_state_with_step_length(
            self.options.sced_frequency_minutes)

        current_sced_instance, solve_time = self.solve_sced(
            cur_state,
            hours_in_objective=1, sced_horizon=self.options.sced_horizon,
            forecast_error_method=forecast_error_method,
            )

        pre_quickstart_cache = None
        if self.options.enable_quick_start_generator_commitment:
            # Determine whether we are going to run a quickstart optimization

            # TODO: Why the "if True" here?
            if True or engine.operations_data_extractor.has_load_shedding(
                    current_sced_instance):
                # Yep, we're doing it.  Cache data we can use to compare
                # results with and without quickstart
                pre_quickstart_cache = engine.operations_data_extractor \
                    .get_pre_quickstart_data(current_sced_instance)

                # TODO: report solution/load shedding before unfixing
                #  Quick Start Generators
                # print("")
                # print("SCED Solution before unfixing Quick Start Generators")
                # print("")
                # self.report_sced_stats()

                # Set up the quickstart run, allowing quickstart
                # generators to turn on
                print("Re-solving SCED after unfixing Quick Start Generators")

                current_sced_instance = self.engine \
                    .enable_quickstart_and_solve(options,
                                                 current_sced_instance)

        if self.verbosity > 0:
            print("Solving for LMPs")

        lmp_sced = self.solve_lmp(current_sced_instance)
        self._simulation_state.apply_sced(current_sced_instance)
        self._prior_sced_instance = current_sced_instance

        ops_stats = self._stats_manager.collect_operations(
            current_sced_instance, solve_time, lmp_sced, pre_quickstart_cache,
            self._sced_extractor
            )

        if self.verbosity > 0:
            self._report_sced_stats(ops_stats)

        if self.options.compute_market_settlements:
            self.simulator.stats_manager.collect_market_settlement(
                current_sced_instance,
                self.engine.operations_data_extractor,
                self.simulator.data_manager.ruc_market_active,
                time_step.hour % options.ruc_every_hours)

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
        # TODO: lol what
        self._ptdf_manager.PTDF_matrix_dict = self.ruc_model.pyo_instance._PTDFs

        # TODO: better error handling
        try:
            ruc_plan, solve_time = self.ruc_model.solve_model(
                self._ruc_solver,
                self.options.deterministic_ruc_solver_options,
                options=self.options
                )

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

        if self.options.compute_market_settlements:
            print("Solving day-ahead market")
            ruc_market = self.create_and_solve_day_ahead_pricing(
                self.options, ruc_plan)

        else:
            ruc_market = None

        # the RUC instance to simulate only exists to store the actual demand
        # and renewables outputs to be realized during the course of a day. it
        # also serves to provide a concrete instance, from which static data
        # and topological features of the system can be extracted.
        # IMPORTANT: This instance should *not* be passed to any method
        #            involved in the creation of economic dispatch instances,
        #            as that would enable those instances to be prescient.

        if self.verbosity > 0:
            print("\nExtracting scenario to simulate")

        return self.create_simulation_actuals(time_step), ruc_plan, ruc_market

    def solve_sced(self,
                   current_state, hours_in_objective,
                   sced_horizon, forecast_error_method):

        sced_model_data = self._data_provider.create_sced_instance(
            current_state, sced_horizon=sced_horizon,
            forecast_error_method=forecast_error_method,
            )

        self._hours_in_objective = hours_in_objective

        if self._hours_in_objective > 10:
            ptdf_options = self._ptdf_manager.look_ahead_sced_ptdf_options
        else:
            ptdf_options = self._ptdf_manager.sced_ptdf_options

        self._ptdf_manager.mark_active(sced_model_data)

        self.sced_model.generate_model(
            sced_model_data, relax_binaries=False, ptdf_options=ptdf_options,
            ptdf_matrix_dict=self._ptdf_manager.PTDF_matrix_dict,
            objective_hours=hours_in_objective
            )

        # update in case lines were taken out
        self._ptdf_manager.PTDF_matrix_dict = self.sced_model.pyo_instance._PTDFs

        try:
            sced_results, solve_time = self.sced_model.solve_model(
                self._sced_solver, self.options.sced_solver_options,
                options=self.options
                )

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

        return sced_results, solve_time

    def solve_lmp(self, sced_instance):
        lmp_sced_instance = sced_instance.clone()

        # In case of demand shortfall, the price skyrockets, so we
        # threshold the value.
        if ('load_mismatch_cost' not in lmp_sced_instance.data['system']
                or (lmp_sced_instance.data['system']['load_mismatch_cost']
                    > self.options.price_threshold)):
            lmp_sced_instance.data[
                'system']['load_mismatch_cost'] = self.options.price_threshold

        # In case of reserve shortfall, the price skyrockets, so we
        # threshold the value.
        if ('reserve_shortfall_cost' not in lmp_sced_instance.data['system']
                or (lmp_sced_instance.data['system']['reserve_shortfall_cost']
                    > self.options.reserve_price_threshold)):
            lmp_sced_instance.data['system']['reserve_shortfall_cost'] = \
                self.options.reserve_price_threshold

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

            ## reset the penalites
            system = lmp_sced_instance.data['system']
            update_obj = False
            new_load_penalty = system['baseMVA'] * system['load_mismatch_cost']

            if not math.isclose(
                    new_load_penalty,
                    self.sced_model.pyo_instance.LoadMismatchPenalty.value
                    ):
                self.sced_model.pyo_instance.LoadMismatchPenalty.value = new_load_penalty
                update_obj = True

            new_reserve_penalty = (system['baseMVA']
                                   * system['reserve_shortfall_cost'])

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
            lmp_sced_results, _ = self.sced_model.solve_model(
                solver_options=self.options.sced_solver_options,
                relaxed=True,
                set_instance=(self.sced_model.pyo_instance is None),
                options=self.options
                )

        except:
            print("Some issue with LMP SCED, writing instance")
            quickstart_uc_filename = (self.options.output_directory
                                      + os.sep + "FAILED_LMP_SCED.json")

            lmp_sced_instance.write(quickstart_uc_filename)
            print(f"Problematic LMP SCED written to {quickstart_uc_filename}")
            raise

        return lmp_sced_results

    def _get_projected_state(self,
                             time_step: PrescientTime) -> VaticSimulationState:
        """
        Get the simulation state as we project it will appear
        after the RUC delay.
        """

        # If there is no RUC delay, use the current state as is
        if self._simulation_state.ruc_delay == 0:
            print("")
            print("Drawing UC initial conditions for date:", time_step.date,
                  "hour:", time_step.hour, "from prior SCED instance.")
            return self._simulation_state

        uc_hour, uc_date = self._get_uc_activation_time(time_step)

        if self.verbosity > 0:
            print("Creating and solving SCED to determine UC initial "
                  "conditions for date:", str(uc_date), "hour:", uc_hour)

        # determine the SCED execution mode, in terms of how discrepancies
        # between forecast and actuals are handled. prescient processing is
        # identical in the case of deterministic and stochastic RUC.
        # persistent processing differs, as there is no point forecast
        # for stochastic RUC.

        # always the default
        sced_forecast_error_method = ForecastErrorMethod.PRESCIENT

        if self.options.run_sced_with_persistent_forecast_errors:
            sced_forecast_error_method = ForecastErrorMethod.PERSISTENT

            if self.verbosity > 0:
                print("Using persistent forecast error model when projecting "
                      "demand and renewables in SCED\n")

        else:
            if self.verbosity > 0:
                print("Using prescient forecast error model when projecting "
                      "demand and renewables in SCED\n")

        # NOTE: the projected sced probably doesn't have to be run for a full
        #       24 hours - just enough to get you to midnight and a few hours
        #       beyond (to avoid end-of-horizon effects).
        #       But for now we run for 24 hours.
        current_state = self._simulation_state.get_state_with_step_length(60)
        proj_hours = min(24, current_state.timestep_count)

        proj_sced_instance, solve_time = self.solve_sced(
            current_state,
            hours_in_objective=proj_hours, sced_horizon=proj_hours,
            forecast_error_method=sced_forecast_error_method,
            )

        return StateWithOffset(current_state, proj_sced_instance,
                               self._simulation_state.ruc_delay)

    def _get_uc_activation_time(self, time_step):
        """
        Get the hour and date that a RUC generated at the given
        time will be activated.
        """
        activation_time = (
                time_step.datetime
                + datetime.timedelta(hours=self._simulation_state.ruc_delay)
                )

        return activation_time.hour, activation_time.date()

    def create_simulation_actuals(self, time_step):
        """
        merge of EgretEngine.create_simulation_actuals
             and egret_plugin.create_simulation_actuals

        Get an Egret model consisting of data to be treated as actuals,
        starting at a given time.

        """

        # Convert time string to time
        start_time = datetime.datetime.combine(
            time_step.date, datetime.time(time_step.hour))

        # Pick whether we're getting actuals or forecasts
        if self.options.simulate_out_of_sample:
            use_actuals = True
        else:
            print("")
            print("***WARNING: Simulating the forecast scenario when running "
                  "deterministic RUC - time consistency across midnight "
                  "boundaries is not guaranteed, and may lead to "
                  "threshold events.")

            use_actuals = False

        # Get a new model
        total_step_count = self.options.ruc_horizon * 60
        total_step_count //= self._actuals_step_frequency

        if time_step.hour == 0:
            sim_model = self._data_provider.get_populated_model(
                use_actuals, start_time, total_step_count)

        else:
            # only get up to 24 hours of data, then copy it
            timesteps_per_day = 24 * 60 / self._actuals_step_frequency
            steps_to_request = min(timesteps_per_day, total_step_count)

            sim_model = self._data_provider.get_populated_model(
                use_actuals, start_time, steps_to_request)

            for vals, in sim_model.get_forecastables():
                for t in range(timesteps_per_day, total_step_count):
                    vals[t] = vals[t - timesteps_per_day]

        return sim_model.to_egret()

    def _report_sced_stats(self, ops_stats):
        print("Fixed costs:    %12.2f" % ops_stats.fixed_costs)
        print("Variable costs: %12.2f" % ops_stats.variable_costs)
        print("")

        if ops_stats.load_shedding != 0.0:
            print("Load shedding reported at t=%d -     total=%12.2f"
                  % (1, ops_stats.load_shedding))
        if ops_stats.over_generation!= 0.0:
            print("Over-generation reported at t=%d -   total=%12.2f"
                  % (1, ops_stats.over_generation))

        if ops_stats.reserve_shortfall != 0.0:
            print("Reserve shortfall reported at t=%2d: %12.2f"
                  % (1, ops_stats.reserve_shortfall))
            print("Quick start generation capacity available at t=%2d: %12.2f"
                  % (1, ops_stats.available_quickstart))
            print("")

        if ops_stats.renewables_curtailment > 0:
            print("Renewables curtailment reported at t=%d - total=%12.2f"
                  % (1, ops_stats.renewables_curtailment))
            print("")

        print("Number on/offs:       %12d" % ops_stats.on_offs)
        print("Sum on/off ramps:     %12.2f" % ops_stats.sum_on_off_ramps)
        print("Sum nominal ramps:    %12.2f" % ops_stats.sum_nominal_ramps)
        print("")
