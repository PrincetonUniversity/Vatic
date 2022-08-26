"""Collecting and reporting statistics on the states of the simulation."""

import os
from pathlib import Path
import bz2
import dill as pickle
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from .model_data import VaticModelData
from .time_manager import VaticTime

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'


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
                 write_dir: Union[Path, str], output_detail: int,
                 verbosity: int, init_model: VaticModelData,
                 output_max_decimals: int, create_plots: bool,
                 save_to_csv: bool) -> None:
        """
        write_dir       Path to where output statistics will be saved.
        output_detail   How much information to include in the output saved to
                        file, with larger integers specifying more detail.
        verbosity       How much logging about simulation running stats to do.

        """
        self._sced_stats = dict()
        self._ruc_stats = dict()

        self.output_detail = output_detail
        self.verbosity = verbosity
        self.write_dir = write_dir
        self.save_to_csv = save_to_csv

        self._round = lambda entry: (
            round(entry, output_max_decimals)
            if isinstance(entry, (int, float))

            else {k: round(val, output_max_decimals)
                  for k, val in entry.items()}
            if isinstance(entry, dict) and isinstance(tuple(entry.values())[0],
                                                      (int, float))

            else {k: [round(val, output_max_decimals)
                      for val in vals]
                  for k, vals in entry.items()}
            )

        self.create_plots = create_plots
        if create_plots:
            os.makedirs(Path(self.write_dir, "plots"), exist_ok=True)

        # static information regarding the characteristics of the power grid
        self._grid_data = {
            'generator_fuels': init_model.fuels,
            'storage_types': init_model.storage_types,

            'thermal_fleet_capacity': self._round(
                init_model.thermal_fleet_capacity),
            'thermal_capacities': init_model.thermal_capacities,
            'thermal_min_outputs': init_model.thermal_minimum_outputs
            }

    def collect_ruc_solution(self,
                             time_step: VaticTime,
                             ruc: VaticModelData) -> None:
        """Gets the key statistics from a solved reliability unit commitment.

        Args
        ----
            time_step   The time in the simulation at which the RUC was solved.
            ruc         The solved RUC model.
        """

        #TODO: add generation values per thermal generator?
        new_ruc_data = {'runtime': self._round(ruc.model_runtime),
                        'duration_minutes': ruc.duration_minutes,
                        'fixed_costs': self._round(ruc.all_fixed_costs),
                        'variable_costs': self._round(ruc.all_variable_costs),
                        'commitments': ruc.commitments,
                        'generation': self._round(ruc.generation),
                        'reserves': self._round(ruc.reserves),
                        'costs': self._round(ruc.generator_total_prices)}

        if self.verbosity > 0:
            print("Fixed costs:    %12.2f" % new_ruc_data['fixed_costs'])
            print("Variable costs: %12.2f" % new_ruc_data['variable_costs'])
            print("")

        self._ruc_stats[time_step] = new_ruc_data

    def collect_sced_solution(self,
                              time_step: VaticTime, sced: VaticModelData,
                              lmp_sced: Optional[VaticModelData] = None,
                              pre_quickstart_cache = None) -> None:
        """Gets the key statistics from a solved economic dispatch.

        Args
        ----
        time_step   The time in the simulation at which the RUC was solved.
        sced        The solved security-constrained economic dispatch model.

        lmp_sced    If applicable, an additional economic dispatch that was
                    solved to get locational marginal prices.

        """
        new_sced_data = {
            'runtime': self._round(sced.model_runtime),
            'duration_minutes': sced.duration_minutes,

            'total_demand': self._round(sced.total_demand),
            'fixed_costs': self._round(sced.fixed_costs),
            'variable_costs': self._round(sced.variable_costs),
            'thermal_generation': self._round(
                sum(sced.thermal_generation.values())),
            'renewable_generation': self._round(
                sum(sced.renewable_generation.values())),

            'load_shedding': self._round(sced.load_shedding),
            'over_generation': self._round(sced.over_generation),
            'reserve_shortfall': self._round(sced.reserve_shortfall),
            'available_reserve': self._round(
                sum(sced.available_reserve.values())),
            'available_quickstart': sced.available_quickstart,
            'available_renewables': self._round(sced.available_renewables),
            'renewables_curtailment': self._round(
                sum(sced.curtailment.values())),

            'on_offs': sced.on_offs,
            'sum_on_off_ramps': sced.on_off_ramps,
            'sum_nominal_ramps': self._round(sced.nominal_ramps),
            'price': self._round(sced.price),
            }

        if pre_quickstart_cache is None:
            new_sced_data['quick_start_additional_costs'] = 0.
            new_sced_data['quick_start_additional_power_generated'] = 0.
            new_sced_data['used_as_quickstart'] = {
                gen: False for gen in sced.quickstart_generators}

        else:
            new_sced_data['quick_start_additional_costs'] = self._round(
                sced.total_costs - pre_quickstart_cache.total_cost)
            new_sced_data['quick_start_additional_power_generated'] \
                = self._round(sced.thermal_generation
                              - pre_quickstart_cache.power_generated)

            new_sced_data['used_as_quickstart'] = {
                gen: (gen in pre_quickstart_cache.quickstart_generators_off
                      and sced.is_generator_on(gen))
                for gen in sced.quickstart_generators
                }

        new_sced_data[
            'observed_thermal_dispatch_levels'] = sced.thermal_generation
        new_sced_data[
            'observed_thermal_headroom_levels'] = sced.available_reserve
        new_sced_data['observed_thermal_states'] = sced.thermal_states
        new_sced_data['previous_thermal_states'] = sced.previous_thermal_states

        new_sced_data['observed_costs'] = self._round(sced.generator_costs)
        new_sced_data['observed_renewables_levels'] = self._round(
            sced.renewable_generation)
        new_sced_data['observed_renewables_curtailment'] = self._round(
            sced.curtailment)

        new_sced_data['observed_flow_levels'] = self._round(sced.flows)
        new_sced_data['bus_demands'] = self._round(sced.bus_demands)
        new_sced_data['observed_bus_mismatches'] = self._round(
            sced.bus_mismatches)

        new_sced_data['storage_input_dispatch_levels'] = sced.storage_inputs
        new_sced_data['storage_output_dispatch_levels'] = sced.storage_outputs
        new_sced_data['storage_soc_dispatch_levels'] = sced.storage_states

        if lmp_sced:
            new_sced_data['observed_bus_LMPs'] = self._round(lmp_sced.bus_LMPs)
            new_sced_data['reserve_RT_price'] = self._round(
                lmp_sced.reserve_RT_price)

        else:
            new_sced_data['observed_bus_LMPs'] = {
                bus: np.nan for bus, _ in sced.elements('load')}

        if self.verbosity > 0:
            print("Fixed costs:    %12.2f" % new_sced_data['fixed_costs'])
            print("Variable costs: %12.2f" % new_sced_data['variable_costs'])
            print("")

            if new_sced_data['load_shedding'] != 0.0:
                print("Load shedding reported at t=%d -     total=%12.2f"
                      % (1, new_sced_data['load_shedding']))
            if new_sced_data['over_generation'] != 0.0:
                print("Over-generation reported at t=%d -   total=%12.2f"
                      % (1, new_sced_data['over_generation']))

            if new_sced_data['reserve_shortfall'] != 0.0:
                print("Reserve shortfall reported at t=%2d: %12.2f"
                      % (1, new_sced_data['reserve_shortfall']))
                print("Quick start generation capacity available at t=%2d: "
                      "%12.2f" % (1, new_sced_data['available_quickstart']))
                print("")

            if new_sced_data['renewables_curtailment'] > 0:
                print("Renewables curtailment reported at t=%d - total=%12.2f"
                      % (1, new_sced_data['renewables_curtailment']))
                print("")

            print("Number on/offs:       %12d" % new_sced_data['on_offs'])
            print("Sum on/off ramps:     %12.2f"
                  % new_sced_data['sum_on_off_ramps'])
            print("Sum nominal ramps:    %12.2f"
                  % new_sced_data['sum_nominal_ramps'])
            print("")

        self._sced_stats[time_step] = new_sced_data

    def consolidate_output(self, sim_runtime=None) -> Dict[str, pd.DataFrame]:
        report_dfs = {
            'hourly_summary': pd.DataFrame.from_records([
                {**time_step.labels(),
                 **{'FixedCosts': stats['fixed_costs'],
                    'VariableCosts': stats['variable_costs'],
                    'LoadShedding': stats['load_shedding'],
                    'OverGeneration': stats['over_generation'],
                    'AvailableReserves': stats['available_reserve'],
                    'ReserveShortfall': stats['reserve_shortfall'],
                    'RenewablesUsed': stats['renewable_generation'],
                    'RenewablesAvailable': stats['available_renewables'],
                    'RenewablesCurtailment': stats['renewables_curtailment'],
                    'Demand': stats['total_demand'], 'Price': stats['price'],
                    'Number on/offs': stats['on_offs'],
                    'Sum on/off ramps': stats['sum_on_off_ramps'],
                    'Sum nominal ramps': stats['sum_nominal_ramps']}}
                for time_step, stats in self._sced_stats.items()
                ]).drop('Minute', axis=1).set_index(
                ['Date', 'Hour'], verify_integrity=True)
            }

        if self.output_detail > 0:
            report_dfs['runtimes'] = pd.DataFrame.from_records([
                {**time_step.labels(),
                 **{'Type': 'SCED', 'Solve Time': stats['runtime']}}
                for time_step, stats in self._sced_stats.items()
                ]).drop('Minute', axis=1).set_index(
                ['Date', 'Hour', 'Type'], verify_integrity=True)

            if sim_runtime:
                report_dfs['total_runtime'] = self._round(sim_runtime)

            report_dfs['ruc_summary'] = pd.DataFrame.from_records([
                {**time_step.labels(),
                 **{'FixedCosts': stats['fixed_costs'],
                    'VariableCosts': stats['variable_costs']}}
                for time_step, stats in self._ruc_stats.items()
                ]).drop('Minute', axis=1).set_index(
                ['Date', 'Hour'], verify_integrity=True)

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

            report_dfs['renew_detail'] = pd.DataFrame.from_records([
                {**time_step.labels(),
                 **{'Generator': gen, 'Output': gen_output,
                    'Curtailment': stats['observed_renewables_curtailment'][
                        gen]}}
                for time_step, stats in self._sced_stats.items()
                for gen, gen_output in
                stats['observed_renewables_levels'].items()
                ]).drop('Minute', axis=1).set_index(
                ['Date', 'Hour', 'Generator'], verify_integrity=True)

        if self.output_detail > 1:
            report_dfs['daily_commits'] = pd.DataFrame.from_records([
                {**time_step.labels(),
                 **{'Generator': gen,
                    **{'Commit {}'.format(i + 1): val
                       for i, val in enumerate(cmt)},
                    **{'Output {}'.format(i + 1): val
                       for i, val in enumerate(stats['generation'][gen])},
                    **{'Reserve {}'.format(i + 1): val
                       for i, val in enumerate(stats['reserves'][gen])}}}
                for time_step, stats in self._ruc_stats.items()
                for gen, cmt in stats['commitments'].items()
                ]).drop('Minute', axis=1).set_index(
                ['Date', 'Hour', 'Generator'], verify_integrity=True)

            report_dfs['daily_commits'].columns = pd.MultiIndex.from_tuples(
                [tuple(x)
                 for x in report_dfs['daily_commits'].columns.str.split(' ')]
                )

            report_dfs['bus_detail'] = pd.DataFrame.from_records([
                {**time_step.labels(),
                 **{'Bus': bus, 'Demand': bus_demand,
                    'Mismatch': stats['observed_bus_mismatches'][bus],
                    'LMP': stats['observed_bus_LMPs'][bus]}}
                for time_step, stats in self._sced_stats.items()
                for bus, bus_demand in stats['bus_demands'].items()
                ]).drop('Minute', axis=1).set_index(
                ['Date', 'Hour', 'Bus'], verify_integrity=True)

            report_dfs['line_detail'] = pd.DataFrame.from_records([
                {**time_step.labels(), **{'Line': line, 'Flow': line_flow}}
                for time_step, stats in self._sced_stats.items()
                for line, line_flow in stats['observed_flow_levels'].items()
                ]).drop('Minute', axis=1).set_index(
                ['Date', 'Hour', 'Line'], verify_integrity=True)

        return report_dfs

    def save_output(self, sim_runtime=None) -> None:
        """Consolidate collected model stats into tables written to file.

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

        if self.save_to_csv:
            for report_lbl, report_df in report_dfs.items():
                if report_lbl != 'total_runtime':
                    report_df.to_csv(
                        Path(self.write_dir, "{}.csv".format(report_lbl)))

        else:
            with bz2.BZ2File(Path(self.write_dir, "output.p.gz"), 'w') as f:
                pickle.dump(report_dfs, f, protocol=-1)

        if self.create_plots:
            self.generate_stack_graph()
            self.generate_cost_graph()
            self.generate_commitment_heatmaps()
            self.plot_thermal_detail()

    def generate_stack_graph(self) -> None:
        """Stacked bar plots of power output by time and generator type."""

        # collect statistics from the simulation
        stack_data = pd.DataFrame({
            time_step.when: {'Demand': stats['total_demand'],
                             'Thermal': stats['thermal_generation'],
                             'Renewables': stats['renewable_generation'],
                             'Load Shedding': stats['load_shedding'],
                             'Over Generation': stats['over_generation']}
            for time_step, stats in self._sced_stats.items()
            }).transpose()

        # define the order bars are stacked in and the size of the figure
        plt_clrs = [('Thermal', '#C50000'), ('Renewables', '#009E00')]
        fig, ax = plt.subplots(figsize=(9, 5))

        # plot the bars, each on top of the one preceding it
        for i, (plt_lbl, plt_clr) in enumerate(plt_clrs):
            if i == 0:
                btm_loc = 0
            else:
                btm_loc = sum([stack_data[lbl] for lbl, _ in plt_clrs[:i]])

            ax.bar(stack_data.index, stack_data[plt_lbl], color=plt_clr,
                   width=1. / 29, bottom=btm_loc)

        # define the entries of the plot's legend
        lgnd_ptchs = [Patch(color=plt_clr, label=plt_lbl)
                      for plt_lbl, plt_clr in plt_clrs
                      if stack_data[plt_lbl].sum() > 0.]

        for hour, hour_data in stack_data.iterrows():
            ax.plot([hour - pd.Timedelta(hours=0.47),
                     hour + pd.Timedelta(hours=0.47)],
                    [hour_data.Demand, hour_data.Demand],
                    linewidth=1.9, c='black', alpha=0.67)

        # annotate bars with percentages of demand served by each category
        if stack_data.shape[0] < 25:
            for hour, hour_data in stack_data.iterrows():
                ax.text(hour, hour_data.Thermal * 0.99,
                        format(hour_data.Thermal / hour_data.Demand, '.1%'),
                        size=4.7, c='white', ha='center', va='top',
                        weight='semibold', transform=ax.transData)

                ax.text(hour, hour_data.Thermal * 1.005,
                        format(hour_data.Renewables / hour_data.Demand, '.1%'),
                        size=4.7, c='white', ha='center', va='bottom',
                        weight='semibold', transform=ax.transData)

                if hour_data['Load Shedding'] > 0.:
                    ax.text(hour, hour_data.Demand * 0.99,
                            format(hour_data['Load Shedding']
                                   / hour_data.Demand, '.1%'),
                            size=4.7, c='black', ha='center', va='top',
                            weight='semibold', transform=ax.transData)

                if hour_data['Over Generation'] > 0.:
                    ax.text(hour, hour_data.Demand * 1.005,
                            format(hour_data['Over Generation']
                                   / hour_data.Demand, '.1%'),
                            size=4.7, c='black', ha='center', va='bottom',
                            weight='semibold', transform=ax.transData)

        # create a legend for the load mismatch information
        ax.plot([0.08, 0.16], [0.88, 0.88],
                linewidth=2.7, c='black', alpha=0.83, transform=ax.transAxes)
        ax.text(0.17, 0.877, "Demand", size=9, c='black', style='italic',
                ha='left', va='center', transform=ax.transAxes)

        ax.text(0.12, 0.897, "Over Generation", size=10, c='black',
                ha='center', va='bottom', transform=ax.transAxes)
        ax.text(0.12, 0.858, "Load Shedding", size=10, c='black',
                ha='center', va='top', transform=ax.transAxes)

        ax.xaxis.set_major_formatter(DateFormatter("%m/%d\n%I%p"))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4, steps=[1, 2, 5]))
        ax.set_ylabel("MWh Generated", size=19, weight='semibold')

        ax.grid(lw=0.7, alpha=0.53)
        ax.axhline(0, c='black', lw=1.1)
        ax.legend(handles=lgnd_ptchs, frameon=False,
                  fontsize=17, ncol=2, handletextpad=0.61)

        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=12)
        ax.yaxis.get_offset_text().set_weight('semibold')
        ymax = stack_data.Demand.max() * 1.31
        ax.set_ylim(-ymax / 61, ymax)

        fig.savefig(Path(self.write_dir, "plots", "stack-graph.pdf"),
                    bbox_inches='tight', format='pdf')

        plt.close()

    def generate_cost_graph(self) -> None:
        """Line chart of various types of costs over time."""

        cost_data = pd.DataFrame({
            time_step.when: {'Fixed': stats['fixed_costs'],
                             'Variable': stats['variable_costs'],
                             'Shedding': stats['load_shedding'] * 1000}
            for time_step, stats in self._sced_stats.items()
            }).transpose()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(cost_data.Fixed, c='black', lw=4.7, alpha=0.53)
        ax.plot(cost_data.Variable, c='blue', lw=4.7, alpha=0.53)
        ax.plot(cost_data.Shedding, c='red', lw=4.7, alpha=0.53)

        lgnd_ptchs = [Patch(color='black', alpha=0.71, label="Fixed"),
                      Patch(color='blue', alpha=0.71, label="Variable"),
                      Patch(color='red', alpha=0.71, label="Load Shed")]

        ax.grid(lw=0.7, alpha=0.53)
        ax.axhline(0, c='black', lw=1.1)
        ax.legend(handles=lgnd_ptchs, frameon=False,
                  loc=8, bbox_to_anchor=(0.5, 1.),
                  fontsize=18, ncol=3, handletextpad=0.7)

        ax.xaxis.set_major_formatter(DateFormatter("%m/%d\n%H:%M"))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4, steps=[1, 2, 5]))
        ax.set_ylabel("Costs ($)", size=19, weight='semibold')

        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=12)
        ax.yaxis.get_offset_text().set_weight('semibold')

        fig.savefig(Path(self.write_dir, "plots", "costs.pdf"),
                    bbox_inches='tight', format='pdf')

        plt.close()

    def generate_commitment_heatmaps(self) -> None:
        """When are thermal generators planned to be turned on by each RUC?"""

        for ruc_time, ruc_data in self._ruc_stats.items():
            commits = pd.DataFrame.from_dict({
                gen: {
                    i: ((output - self._grid_data['thermal_min_outputs'][gen])
                        / (self._grid_data['thermal_capacities'][gen]
                           - self._grid_data['thermal_min_outputs'][gen])
                        if output > 0. else np.NaN)
                    for i, output in enumerate(outputs)
                    }

                for gen, outputs in ruc_data['generation'].items()
                if self._grid_data['generator_fuels'][gen] in {'C', 'O',
                                                               'G', 'N'}
                }, orient='index').iloc[:, :24]

            use_cmap = sns.cubehelix_palette(start=0.45, rot=0.43,
                                             light=0.87, dark=0.03,
                                             as_cmap=True)

            xlbls = [
                t.strftime('%m/%d\n%-I%p') if i % 6 == 0 else ""
                for i, t in enumerate(pd.date_range(start=ruc_time.when,
                                                    periods=commits.shape[1],
                                                    freq='H'))
                ]

            ylbls = [
                gen if pd.isnull(ruc_data['costs'][gen])
                else '   '.join([gen,
                                 "${:.2f}".format(ruc_data['costs'][gen])])
                for gen in commits.index
                ]

            fig, ax = plt.subplots(figsize=(4, 13))
            sns.heatmap(commits, cbar=False, vmin=0., vmax=1., ax=ax,
                        cmap=use_cmap, xticklabels=xlbls, yticklabels=ylbls)

            for i in range(commits.shape[1]):
                if i % 6 == 0:
                    ax.axvline(i, c='black',
                               lw=0.07, linestyle=':', alpha=0.61)

            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=7)
            ax.set_xlim(-0.23, ax.get_xlim()[1])

            date_lbl = ruc_time.labels()['Date']
            fig.savefig(Path(self.write_dir, "plots",
                             "{}_commits.pdf".format(date_lbl)),
                        bbox_inches='tight', format='pdf')

            plt.close()

    def plot_thermal_detail(self) -> None:
        """Stacked bar plots of thermal generators' production and ramping."""

        # collect statistics from the simulation
        thermal_data = pd.DataFrame.from_records([
            {**{'Time': time_step.when},
             **{'Generator': gen,
                'Dispatch': stats['observed_thermal_dispatch_levels'][gen],
                'Headroom': stats['observed_thermal_headroom_levels'][gen],
                'Unit State': gen_state,
                'Min Output': (self._grid_data['thermal_min_outputs'][gen]
                               if gen_state else 0.),
                'Last Unit State': stats['previous_thermal_states'][gen]}}

            for time_step, stats in self._sced_stats.items()
            for gen, gen_state in stats['observed_thermal_states'].items()
            ]).set_index(['Time', 'Generator'], verify_integrity=True)

        plot_data = pd.DataFrame({
            'Output': thermal_data.groupby('Time').Dispatch.sum(),
            'Headroom': thermal_data.groupby('Time').Headroom.sum(),
            'On': thermal_data.groupby('Time')['Unit State'].sum(),
            'MinOutput': thermal_data.groupby('Time')['Min Output'].sum(),

            'TurnedOn': (
                    thermal_data['Unit State']
                    & ~thermal_data['Last Unit State']
                    ).groupby('Time').sum(),
            'TurnedOff': (
                    ~thermal_data['Unit State']
                    & thermal_data['Last Unit State']
                    ).groupby('Time').sum()
            })

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(plot_data.index, plot_data.MinOutput,
               color='#9B0000', width=1. / 29)
        ax.bar(plot_data.index, plot_data.Output - plot_data.MinOutput,
               color='#C50000', width=1. / 29, bottom=plot_data.MinOutput)

        ax.bar(plot_data.index, plot_data.Headroom,
               color='white', edgecolor='#C50000', lw=1.14, width=1. / 31,
               bottom=plot_data.Output)

        ax.axhline(self._grid_data['thermal_fleet_capacity'],
                   c='black', lw=1.7, ls='--')
        ax.text(plot_data.index[0] + pd.Timedelta(minutes=11),
                self._grid_data['thermal_fleet_capacity'] * 1.005,
                "Thermal Capacity", size=12, ha='left', va='bottom',
                style='italic', transform=ax.transData)

        for hour, hour_data in plot_data.iterrows():
            ax.text(hour, hour_data.Output * 0.98, int(hour_data.On),
                    size=8, ha='center', va='top',
                    weight='semibold', transform=ax.transData)

            onoff_lbl = ""
            if hour_data.TurnedOff > 0:
                onoff_lbl += "\u2212{}".format(int(hour_data.TurnedOff))
            if hour_data.TurnedOn > 0:
                onoff_lbl += "\n\u002B{}".format(int(hour_data.TurnedOn))

            ax.text(hour, hour_data.Output * 1.005, onoff_lbl,
                    size=8, ha='center', va='bottom', weight='semibold',
                    transform=ax.transData)

        ax.xaxis.set_major_formatter(DateFormatter("%m/%d\n%I%p"))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4, steps=[1, 2, 5]))
        ax.set_ylabel("MWh Generated", size=19, weight='semibold')

        ax.grid(lw=0.7, alpha=0.53)
        ax.axhline(0, c='black', lw=1.1)

        ax.legend(handles=[Patch(color='#9B0000', label="Minimum\nGeneration"),
                           Patch(color='#C50000', label="Generation"),
                           Patch(facecolor='white', edgecolor='#C50000',
                                 lw=1.7, label="Headroom")],
                  loc=9, bbox_to_anchor=(0.5, -0.13),
                  fontsize=17, ncol=3, handletextpad=0.61, frameon=False)

        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=12)
        ax.yaxis.get_offset_text().set_weight('semibold')
        ymax = self._grid_data['thermal_fleet_capacity'] * 1.07
        ax.set_ylim(-ymax / 61, ymax)

        fig.savefig(Path(self.write_dir, "plots", "thermal-detail.pdf"),
                    bbox_inches='tight', format='pdf')

        plt.close()
