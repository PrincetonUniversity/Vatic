"""Collecting and reporting statistics on the states of the simulation."""

from __future__ import annotations

import os
from pathlib import Path
import bz2
import dill as pickle
from typing import Optional

import numpy as np
import pandas as pd

from ..egret.time_manager import VaticTime
from .models import RucModel, ScedModel


class StatsManager:
    """Processing statistics generated by solved simulation models.

    This class consolidates data produced by the unit commitment and economic
    dispatch models solved as part of a simulation run and produces reports
    in the form of data tables saved to file as well as plots.

    See prescient.simulator.reporting_manager
    and prescient.simulator.stats_manager for the original implementations of
    the functionalities included in this class.
    """

    def __init__(self,
                 write_dir: Path | str | None,
                 output_detail: int, verbosity: int, output_max_decimals: int,
                 create_plots: bool, save_to_csv: bool,
                 init_states: dict,
                 last_conditions_file: str | Path) -> None:
        self._sced_stats = dict()
        self._ruc_stats = dict()

        self.write_dir = write_dir
        self.output_detail = output_detail
        self.verbosity = verbosity
        self.max_decimals = output_max_decimals
        self.create_plots = create_plots
        self.save_to_csv = save_to_csv
        self.init_states = init_states

        if last_conditions_file:
            self.last_conditions_file = Path(last_conditions_file)
        else:
            self.last_conditions_file = None

        if self.write_dir is None and create_plots:
            raise ValueError("Cannot create plots without providing an "
                             "output directory for saving them!")

        if self.write_dir is not None:
            os.makedirs(self.write_dir, exist_ok=True)

            if self.create_plots:
                os.makedirs(Path(self.write_dir, "plots"), exist_ok=True)

    def _dict_to_frame(self, stats: dict[tuple[str, int], int | float]):
        return pd.Series(stats).unstack().round(self.max_decimals)

    def collect_ruc_solution(self,
                             time_step: VaticTime,
                             ruc: RucModel) -> None:
        """Gets the key statistics from a solved reliability unit commitment.

        Args
        ----
            time_step   The time in the simulation at which the RUC was solved.
            ruc         The solved RUC model.
        """

        self._ruc_stats[time_step] = {
            'runtime': ruc.solve_time,

            'fixed_cost': round(sum(ruc.results['commitment_cost'].values()),
                                self.max_decimals),
            'variable_cost': round(sum(
                ruc.results['production_cost'].values()), self.max_decimals),

            'generation': self._dict_to_frame(ruc.results['power_generated']),
            'commitments': self._dict_to_frame(ruc.results['commitment']),
            'reserves': self._dict_to_frame(ruc.results['reserves_provided']),

            'costs': pd.Series(ruc.gen_costs),
            }

        if self.verbosity > 0:
            print("RUC fixed costs: "
                  f"{self._ruc_stats[time_step]['fixed_cost']}"
                  "\tvariable costs: "
                  f"{self._ruc_stats[time_step]['variable_cost']}"
                  "\n")

    def collect_sced_solution(self,
                              time_step: VaticTime, sced: ScedModel,
                              lmp_sced: Optional[ScedModel] = None) -> None:
        """Gets the key statistics from a solved economic dispatch.

        Args
        ----
            time_step   The time in the simulation at which the RUC was solved.
            sced        The solved security-constrained economic dispatch model.

            lmp_sced    If applicable, an additional economic dispatch that was
                        solved to get locational marginal prices.

        """
        t1 = sced.InitialTime

        tgen = {g: pwr
                for (g, t), pwr in sced.results['power_generated'].items()
                if g in sced.ThermalGenerators and t == t1}
        rgen = {g: pwr
                for (g, t), pwr in sced.results['power_generated'].items()
                if g in sced.RenewableGenerators and t == t1}

        self._sced_stats[time_step] = {
            'runtime': sced.solve_time,

            'total_demand': round(sced.Demand[t1].sum(), self.max_decimals),

            'fixed_costs': round(sum(cost for (g, t), cost
                                     in sced.results['commitment_cost'].items()
                                     if t == t1),
                                 self.max_decimals),

            'variable_costs': round(
                sum(cost for (g, t), cost
                    in sced.results['production_cost'].items() if t == t1),
                self.max_decimals
                ),

            'thermal_generation': round(sum(tgen.values()), self.max_decimals),
            'renewable_generation': round(sum(
                rgen.values()), self.max_decimals),

            'load_shedding': round(
                sum(shed for (b, t), shed
                    in sced.results['load_shedding'].items() if t == t1),
                self.max_decimals
                ),

            'over_generation': round(
                sum(pwr for (b, t), pwr
                    in sced.results['over_generation'].items() if t == t1),
                self.max_decimals
                ),

            'reserve_shortfall': round(
                sum(shrt for t, shrt
                    in sced.results['reserve_shortfall'].items() if t == t1),
                self.max_decimals
                ),

            'available_reserve': round(sum(sced.headrooms.values()),
                                       self.max_decimals),

            'available_renewables': round(sum(
                sced.MaxPowerOutput[g, t1] for g in sced.RenewableGenerators),
                self.max_decimals
                ),

            'on_offs': sum((sced.is_generator_on(gen)
                            ^ sced.was_generator_on(gen))
                           for gen in sced.ThermalGenerators),

            'sum_on_off_ramps': round(sum(
                sced.results['power_generated'][gen, t1]
                for gen in sced.ThermalGenerators
                if sced.is_generator_on(gen) ^ sced.was_generator_on(gen)
                ), self.max_decimals),

            'sum_nominal_ramps': round(sum(
                abs(sced.results['power_generated'][gen, t1]
                    - sced.PowerGeneratedT0[gen])
                for gen in sced.ThermalGenerators
                if sced.is_generator_on(gen) == sced.was_generator_on(gen)
                ), self.max_decimals),

            'price': round(sced.system_price, self.max_decimals),

            'observed_thermal_dispatch_levels': pd.Series({
                gen: sced.results['power_generated'][gen, t1]
                for gen in sced.ThermalGenerators
                }).round(self.max_decimals),

            'observed_thermal_states': pd.Series({
                gen: sced.is_generator_on(gen)
                for gen in sced.ThermalGenerators
                }),

            'observed_thermal_headroom_levels': pd.Series(
                sced.headrooms).round(self.max_decimals),
            'observed_flow_levels': round(
                sced.flows.loc[t1], self.max_decimals),

            'observed_bus_mismatches': pd.Series({
                gen: v for (gen, t), v in sced.results['p_balances'].items()
                if t == t1
                }).round(self.max_decimals),

            'observed_costs': pd.Series({
                gen: (sced.results['commitment_cost'][gen, t1]
                      + sced.results['production_cost'][gen, t1])
                for gen in sced.ThermalGenerators}).round(self.max_decimals)
            }

        if lmp_sced:
            pass

        if self.verbosity > 0:
            print("SCED fixed costs: "
                  f"{self._sced_stats[time_step]['fixed_costs']}"
                  "\tvariable costs: "
                  f"{self._sced_stats[time_step]['variable_costs']}")

    def consolidate_output(self, sim_runtime=None) -> dict[str, pd.DataFrame]:
        """Creates tables storing outputs of all models this simulation ran."""

        report_dfs = {
            'hourly_summary': pd.DataFrame({
                time_step: {'FixedCosts': stats['fixed_costs'],
                            'VariableCosts': stats['variable_costs'],
                            'LoadShedding': stats['load_shedding'],
                            'OverGeneration': stats['over_generation'],
                            'AvailableReserves': stats['available_reserve'],
                            'ReserveShortfall': stats['reserve_shortfall'],
                            'RenewablesUsed': stats['renewable_generation'],
                            'RenewablesAvailable': stats[
                                'available_renewables'],
                            'Demand': stats['total_demand'],
                            'Price': stats['price'],
                            'Number on/offs': stats['on_offs'],
                            'Sum on/off ramps': stats['sum_on_off_ramps'],
                            'Sum nominal ramps': stats['sum_nominal_ramps'],
                            }
                for time_step, stats in self._sced_stats.items()
                }).T,
            }

        if self.output_detail > 0:
            report_dfs['thermal_detail'] = pd.DataFrame.from_records([
                {**time_step.labels(),
                 **{'Generator': gen,
                    'Dispatch': stats['observed_thermal_dispatch_levels'][gen],
                    'Headroom': stats['observed_thermal_headroom_levels'][gen],
                    'Unit State': gen_state,
                    'Unit Cost': stats['observed_costs'][gen]}}
                for time_step, stats in self._sced_stats.items()
                for gen, gen_state in stats['observed_thermal_states'].items()
                ]).drop('Minute', axis=1).set_index(
                ['Date', 'Hour', 'Generator'], verify_integrity=True)

        return report_dfs

    def save_output(self, sim_runtime=None) -> dict[str, pd.DataFrame]:
        """Writes simulation summary statistics and other output to file.

        This function collects the data pulled from UC and ED models that were
        solved during the course of the simulation and organizes them into
        pandas dataframes. These dataframes generally have rows corresponding
        to time steps of the simulation and columns corresponding to various
        model data fields. These dataframes are all stored in a single
        dictionary that is then serialized and saved as a compressed pickle
        object.

        Note that depending on the `output_detail` simulator option, this
        output dictionary may omit certain types of model outputs that are
        particularly space-intensive.

        """
        report_dfs = self.consolidate_output(sim_runtime)

        # if desired, save the final on/off states of the thermal generators
        # to use as initial states for a future simulation
        if self.last_conditions_file:
            tgen_gby = report_dfs['thermal_detail'].groupby('Generator')

            # get the final output for each generator
            final_dispatch = tgen_gby.apply(lambda x: round(x['Dispatch'][-1], 2))

            # get the final on/off state for each generator
            final_bool = tgen_gby.apply(
                lambda x: (x['Unit State'][-1] * 2 - 1))

            # find how long it has been since each generator was in a state not
            # matching its final state, combine this info with final on/off
            last_conds = tgen_gby.apply(
                lambda x: (x['Unit State']
                           != x['Unit State'][-1])[::-1].argmax()
                ) * final_bool

            # for generators which were on or off for the entire simulation
            last_conds[last_conds == 0] = len(self._sced_stats)
            last_conds[last_conds == len(self._sced_stats)] *= final_bool[
                last_conds[last_conds == len(self._sced_stats)].index]

            # save the final states to file, merging with initial states for
            # this sim for generators that were on/off the entire time
            df = pd.DataFrame.from_dict({
                gen: {
                    'UnitOnT0State': (
                        init_cond + last_conds[gen]
                        if (abs(last_conds[gen]) == len(self._sced_stats))
                           and (np.sign(init_cond) == np.sign(last_conds[gen]))
                        else last_conds[gen]),
                    'PowerGeneratedT0': final_dispatch[gen]
                    }
                for gen, init_cond in self.init_states.items()
                }, orient='index')

            df.index.name = 'GEN'
            df.to_csv(self.last_conditions_file)

        if self.write_dir:
            if self.save_to_csv:
                for report_lbl, report_df in report_dfs.items():
                    if report_lbl != 'total_runtime':
                        report_df.to_csv(
                            Path(self.write_dir, "{}.csv".format(report_lbl)))

            else:
                with bz2.BZ2File(Path(self.write_dir, "output.p.gz"),
                                 'w') as f:
                    pickle.dump(report_dfs, f, protocol=-1)

        return report_dfs