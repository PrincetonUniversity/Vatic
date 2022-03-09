
import os
from pathlib import Path
import bz2
import dill as pickle
import pandas as pd

from .model_data import VaticModelData
from .time_manager import VaticTime

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'


class StatsManager:

    def __init__(self,
                 write_dir, light_output, verbosity, options):
        self._sced_stats = dict()
        self._ruc_stats = dict()

        self.light_output = light_output
        self.verbosity = verbosity
        self.write_dir = write_dir

        self._round = lambda entry: (
            round(entry, options.output_max_decimal_places)
            if isinstance(entry, (int, float))

            else {k: round(val, options.output_max_decimal_places)
                  for k, val in entry.items()}
            if isinstance(entry, dict) and isinstance(tuple(entry.values())[0],
                                                      (int, float))

            else {k: [round(val, options.output_max_decimal_places)
                      for val in vals]
                  for k, vals in entry.items()}
            )

        if not options.disable_stackgraphs:
            os.makedirs(Path(self.write_dir, "plots"), exist_ok=True)

    def collect_ruc_solution(self,
                             time_step: VaticTime,
                             ruc: VaticModelData) -> None:
        new_ruc_data = {'runtime': self._round(ruc.model_runtime),
                        'duration_minutes': ruc.duration_minutes,
                        'fixed_costs': self._round(ruc.all_fixed_costs),
                        'variable_costs': self._round(ruc.all_variable_costs),
                        'commitments': ruc.commitments,
                        'generation': self._round(ruc.generation),
                        'reserves': self._round(ruc.reserves)}

        if self.verbosity > 0:
            print("Fixed costs:    %12.2f" % new_ruc_data['fixed_costs'])
            print("Variable costs: %12.2f" % new_ruc_data['variable_costs'])
            print("")

        self._ruc_stats[time_step] = new_ruc_data

    def collect_sced_solution(self,
                              time_step: VaticTime,
                              sced: VaticModelData, lmp_sced: VaticModelData,
                              pre_quickstart_cache) -> None:
        #TODO: fleet capacity is a constant, doesn't need to be recalculated,
        # keep it here until we decide on a better place to keep it
        new_sced_data = {
            'runtime': self._round(sced.model_runtime),
            'duration_minutes': sced.duration_minutes,

            'thermal_fleet_capacity': self._round(sced.thermal_fleet_capacity),
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

        new_sced_data['generator_fuels'] = sced.fuels
        new_sced_data[
            'observed_thermal_dispatch_levels'] = sced.thermal_generation
        new_sced_data[
            'observed_thermal_headroom_levels'] = sced.available_reserve
        new_sced_data['observed_thermal_states'] = sced.thermal_states

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
        new_sced_data['storage_types'] = sced.storage_types

        new_sced_data['observed_bus_LMPs'] = self._round(lmp_sced.bus_LMPs)
        new_sced_data['reserve_RT_price'] = self._round(
            lmp_sced.reserve_RT_price)

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

    def save_output(self):
        report_dfs = dict()

        report_dfs['runtimes'] = pd.DataFrame.from_records([
            {**time_step.labels(),
             **{'Type': 'SCED', 'Solve Time': stats['runtime']}}
            for time_step, stats in self._sced_stats.items()
            ]).drop('Minute', axis=1).set_index(
                ['Date', 'Hour', 'Type'], verify_integrity=True)

        report_dfs['ruc_summary'] = pd.DataFrame.from_records([
            {**time_step.labels(),
             **{'FixedCosts': stats['fixed_costs'],
                'VariableCosts': stats['variable_costs']}}
            for time_step, stats in self._ruc_stats.items()
            ]).drop('Minute', axis=1).set_index(
                ['Date', 'Hour'], verify_integrity=True)

        report_dfs['hourly_summary'] = pd.DataFrame.from_records([
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

        if not self.light_output:
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

        with bz2.BZ2File(Path(self.write_dir, "output.p.gz"), 'w') as f:
            pickle.dump(report_dfs, f, protocol=-1)

    def generate_stack_graph(self) -> None:
        stack_data = pd.DataFrame({
            time_step: {'Demand': stats['total_demand'],
                        'Thermal': stats['thermal_generation'],
                        'Renews': stats['renewable_generation'],
                        'Shedding': stats['load_shedding'],
                        'OverGen': stats['over_generation']}
            for time_step, stats in self._sced_stats.items()
            }).transpose()

        fig, ax = plt.subplots(figsize=(8, 5))
        ind = list(range(stack_data.shape[0]))

        ax.bar(ind, stack_data.Thermal, color='r', width=0.79)
        ax.bar(ind, stack_data.Renews, bottom=stack_data.Thermal,
               color='g', width=0.79)

        fig.savefig(Path(self.write_dir, "plots", "stack-graph.pdf"),
                    bbox_inches='tight', format='pdf')
