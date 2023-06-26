"""Running a simulation of alternating UC and ED grid optimization steps."""

from __future__ import annotations

import dill as pickle
from pathlib import Path
import time
from typing import Optional
import datetime
import pandas as pd
from copy import deepcopy

from .data_providers import PickleProvider
from .formulations import RucModel, ScedModel
from .simulation_state import VaticSimulationState
from .ptdf_manager import VaticPTDFManager
from .stats_manager import StatsManager
from ..egret.time_manager import VaticTimeManager, VaticTime


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

    def __init__(self,
                 template_data: dict, gen_data: pd.DataFrame,
                 load_data: pd.DataFrame, out_dir: Path | str | None,
                 start_date: datetime.date, num_days: int,
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
                 renew_costs: Optional[dict | str | Path | list[float]],
                 save_to_csv, last_conditions_file) -> None:

        self.run_lmps = run_lmps #lmp: Locational Marginal Price
        self.mipgap = mipgap #Put mipgap in the initalization
        self.solver_options = solver_options
        self.sced_horizon = sced_horizon
        self.lmp_shortfall_costs = lmp_shortfall_costs

        self._hours_in_objective = None
        self._current_timestep = None
        #time dictionary for profiling
        self.simulation_times = {'Init': 0., 'Plan': 0., 'Sim': 0.}

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
                                           output_max_decimals, create_plots,
                                           save_to_csv, last_conditions_file)

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

            # clean gurobi envrionment to remove memory cache and run faster
        sim_time = time.time() - simulation_start_time

        print("Simulation Complete")
        print("Total simulation time: {:.1f} seconds".format(sim_time))

        print("Initialization time: {:.2f} seconds".format(
            self.simulation_times['Init']))
        print("Planning time: {:.2f} seconds".format(
            self.simulation_times['Plan']))
        print("Real-time sim time: {:.2f} seconds".format(
            self.simulation_times['Sim']))

        return self._stats_manager.consolidate_output(sim_time)

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
                ruc = pickle.load(f)

        # ...otherwise, solve the initial RUC
        else:
            ruc = self.solve_ruc(first_step)

        self._stats_manager.collect_ruc_solution(first_step, ruc)

        # if an initial RUC file has been given and it doesn't already exist
        # then save the solved RUC for future use
        if self.init_ruc_file and not self.init_ruc_file.exists():
            with open(self.init_ruc_file, 'wb') as f:
                pickle.dump(ruc, f, protocol=-1)

        sim_actuals = self._data_provider.get_forecastables(
            use_actuals=True, times_requested=self._data_provider.ruc_horizon)
        self._simulation_state.apply_initial_ruc(ruc, sim_actuals)

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

        sced_model = self.solve_sced(hours_in_objective=1,
                                     sced_horizon=self.sced_horizon)

        if self.run_lmps:
            if self.verbosity > 0:
                print("Solving for LMPs")

            lmp_sced = self.solve_lmp(hours_in_objective=1,
                                      sced_horizon=self.sced_horizon,
                                      sced_model=sced_model)
        else:
            lmp_sced = None

        self._simulation_state.apply_sced(sced_model)
        self._prior_sced_instance = sced_model

        self._stats_manager.collect_sced_solution(self._current_timestep,
                                                  sced_model, lmp_sced)

    def solve_ruc(self,
                  time_step: VaticTime,
                  sim_state: Optional[VaticSimulationState] = None) -> RucModel:
        ruc_model = self._data_provider.create_ruc(time_step, sim_state)

        ruc_model.generate(relax_binaries=False,
                           ptdf=self._ptdf_manager.ptdf_matrix,
                           ptdf_options=self._ptdf_manager.ruc_ptdf_options,
                           objective_hours=self._data_provider.ruc_horizon)

        ruc_model.solve(relaxed=False, mipgap=self.mipgap,
                        threads=self.solver_options['Threads'], outputflag=0)

        self._ptdf_manager.ptdf_matrix = ruc_model.PTDF
        self._ptdf_manager.update_active(ruc_model)

        return ruc_model

    def solve_sced(self,
                   hours_in_objective: int, sced_horizon: int) -> ScedModel:

        sced_model = self._data_provider.create_sced(self._current_timestep,
                                                     self._simulation_state,
                                                     sced_horizon=sced_horizon)

        sced_model.generate(relax_binaries=False,
                            ptdf=self._ptdf_manager.ptdf_matrix,
                            ptdf_options=self._ptdf_manager.sced_ptdf_options,
                            objective_hours=hours_in_objective)
        self._hours_in_objective = hours_in_objective

        sced_model.solve(relaxed=False, mipgap=self.mipgap,
                         threads=self.solver_options['Threads'], outputflag=0)

        return sced_model

    def _get_projected_state(self) -> VaticSimulationState:
        if self._simulation_state.ruc_delay == 0:
            proj_state = deepcopy(self._simulation_state)

        else:
            proj_hours = min(24, self._simulation_state.timestep_count)
            proj_sced = self.solve_sced(hours_in_objective=proj_hours,
                                        sced_horizon=proj_hours)

            proj_state = self._simulation_state.get_state_with_sced_offset(
                sced=proj_sced, offset=self._simulation_state.ruc_delay)

        return proj_state
