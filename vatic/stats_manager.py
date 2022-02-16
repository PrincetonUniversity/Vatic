#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import os
from pathlib import Path
import bz2
import dill as pickle
from typing import TypeVar, Callable, Iterable, Any, Union, Dict
import pandas as pd

from .model_data import VaticModelData
from .time_manager import VaticTime

from prescient.simulator.reporting_manager import (
    _collect_time, _collect_time_assert_equal,
    _add_timeseries_attribute_to_egret_dict
    )

from prescient.stats.overall_stats import OverallStats
from prescient.stats.daily_stats import DailyStats
from egret.data.model_data import ModelData
from egret.models.unit_commitment import _time_series_dict

# If appropriate back-ends for Matplotlib are not installed
# (e.g, gtk), then graphing will not be available.
try:
    from prescient.util import graphutils
    graphutils_functional = True
except ValueError:
    print("***Unable to load Gtk back-end for matplotlib - graphics generation is disabled")
    graphutils_functional = False


class StatsManager:

    def __init__(self,
                 write_dir, light_output, verbosity, options):
        self._sced_stats = dict()
        self._ruc_stats = dict()

        self.light_output = light_output
        self.verbosity = verbosity
        self.write_dir = write_dir

        self._round = lambda val: round(val, options.output_max_decimal_places)
        if not options.disable_stackgraphs:
            os.makedirs(Path(self.write_dir, "plots"), exist_ok=True)

    def collect_sced_solution(self,
                              time_step: VaticTime,
                              sced: VaticModelData, lmp_sced: VaticModelData,
                              pre_quickstart_cache) -> None:
        #TODO: fleet capacity is a constant, doesn't need to be recalculated,
        # keep it here until we decide on a better place to keep it
        new_sced_data = {
            'runtime': sced.model_runtime,
            'duration_minutes': sced.duration_minutes,
            'thermal_fleet_capacity': sced.thermal_fleet_capacity,
            'total_demand': sced.total_demand,
            'fixed_costs': sced.fixed_costs,
            'variable_costs': sced.variable_costs,
            'thermal_generation': sum(sced.thermal_generation.values()),
            'renewable_generation': sum(sced.renewable_generation.values()),
            'load_shedding': sced.load_shedding,
            'over_generation': sced.over_generation,
            'reserve_shortfall': sced.reserve_shortfall,
            'available_reserve': sum(sced.available_reserve.values()),
            'available_quickstart': sced.available_quickstart,
            'available_renewables': sced.available_renewables,
            'renewables_curtailment': sum(sced.curtailment.values()),
            'on_offs': sced.on_offs,
            'sum_on_off_ramps': sced.on_off_ramps,
            'sum_nominal_ramps': sced.nominal_ramps,
            'price': sced.price
            }

        if pre_quickstart_cache is None:
            new_sced_data['quick_start_additional_costs'] = 0.
            new_sced_data['quick_start_additional_power_generated'] = 0.
            new_sced_data['used_as_quickstart'] = {
                gen: False for gen in sced.quickstart_generators}

        else:
            new_sced_data['quick_start_additional_costs'] = (
                sced.total_costs - pre_quickstart_cache.total_cost)
            new_sced_data['quick_start_additional_power_generated'] = (
                sced.thermal_generation - pre_quickstart_cache.power_generated)

            new_sced_data['used_as_quickstart'] = {
                gen: (gen in pre_quickstart_cache.quickstart_generators_off
                      and sced.is_generator_on(gen))
                for gen in sced.quickstart_generators
                }

        new_sced_data['generator_fuels'] = sced.fuels
        new_sced_data['observed_thermal_dispatch_levels'] = sced.thermal_generation
        new_sced_data['observed_thermal_headroom_levels'] = sced.available_reserve
        new_sced_data['observed_thermal_states'] = sced.thermal_states
        new_sced_data['observed_costs'] = sced.generator_costs
        new_sced_data['observed_renewables_levels'] = sced.renewable_generation
        new_sced_data['observed_renewables_curtailment'] = sced.curtailment

        new_sced_data['observed_flow_levels'] = sced.flows
        new_sced_data['bus_demands'] = sced.bus_demands
        new_sced_data['observed_bus_mismatches'] = sced.bus_mismatches

        new_sced_data['storage_input_dispatch_levels'] = sced.storage_inputs
        new_sced_data['storage_output_dispatch_levels'] = sced.storage_outputs
        new_sced_data['storage_soc_dispatch_levels'] = sced.storage_states
        new_sced_data['storage_types'] = sced.storage_types

        new_sced_data['observed_bus_LMPs'] = lmp_sced.bus_LMPs
        new_sced_data['reserve_RT_price'] = lmp_sced.reserve_RT_price

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
            ])

        if not self.light_output:
            report_dfs['thermal_detail'] = pd.DataFrame.from_records([
                {**time_step.labels(),
                 **{'Generator': gen,
                    'Dispatch': stats['observed_thermal_dispatch_levels'][gen],
                    'Headroom': stats['observed_thermal_dispatch_levels'][gen],
                    'Unit State': gen_state,
                    'Unit Cost': stats['observed_costs'][gen]}}
                for time_step, stats in self._sced_stats.items()
                for gen, gen_state in stats['observed_thermal_states'].items()
                ])

            report_dfs['renew_detail'] = pd.DataFrame.from_records([
                {**time_step.labels(),
                 **{'Generator': gen, 'Output': gen_output,
                    'Curtailment': stats['observed_renewables_curtailment'][
                        gen]}}
                for time_step, stats in self._sced_stats.items()
                for gen, gen_output in
                stats['observed_renewables_levels'].items()
                ])

            report_dfs['bus_detail'] = pd.DataFrame.from_records([
                {**time_step.labels(),
                 **{'Bus': bus, 'Demand': bus_demand,
                    'Mismatch': stats['observed_bus_mismatches'][bus],
                    'LMP': stats['observed_bus_LMPs']}}
                for time_step, stats in self._sced_stats.items()
                for bus, bus_demand in stats['bus_demands'].items()
                ])

            report_dfs['line_detail'] = pd.DataFrame.from_records([
                {**time_step.labels(), **{'Line': line, 'Flow': line_flow}}
                for time_step, stats in self._sced_stats.items()
                for line, line_flow in stats['observed_flow_levels'].items()
                ])


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
            ]).drop('Minute', axis=1)

        with bz2.BZ2File(Path(self.write_dir, "output.p.gz"), 'w') as f:
            pickle.dump(report_dfs, f, protocol=-1)

    @staticmethod
    def generate_stack_graph(options, daily_stats: DailyStats):

        md_dict = ModelData.empty_model_data_dict()

        system = md_dict['system']

        # put just the HH:MM in the graph
        system['time_keys'] = [
            str(opstats.timestamp.time())[0:5]
            for opstats in daily_stats.operations_stats()
            ]

        system['reserve_requirement'] = _time_series_dict([
            opstats.reserve_requirement
            for opstats in daily_stats.operations_stats()
            ])
        system['reserve_shortfall'] = _time_series_dict([
            opstats.reserve_shortfall
            for opstats in daily_stats.operations_stats()
            ])

        elements = md_dict['elements']

        elements['load'] = {'system_load': {'p_load': _time_series_dict([
            opstats.total_demand
            for opstats in daily_stats.operations_stats()
            ])}}

        elements['bus'] = {'system_load_shed': {
            'p_balance_violation': _time_series_dict([
                opstats.load_shedding
                for opstats in daily_stats.operations_stats()])
            },

            'system_over_generation': {
                'p_balance_violation' : _time_series_dict([
                    -opstats.over_generation
                    for opstats in daily_stats.operations_stats()
                    ])
                }
            }

        ## load in generation, storage
        generator_fuels = {}
        thermal_quickstart = {}
        thermal_dispatch = {}
        thermal_headroom = {}
        thermal_states = {}
        renewables_dispatch = {}
        renewables_curtailment = {}
        storage_input_dispatch = {}
        storage_output_dispatch = {}
        storage_types = {}

        for opstats in daily_stats.operations_stats():
            _collect_time_assert_equal(opstats.generator_fuels, generator_fuels)
            _collect_time_assert_equal(opstats.quick_start_capable, thermal_quickstart)
            _collect_time_assert_equal(opstats.storage_types, storage_types)

            _collect_time(opstats.observed_thermal_dispatch_levels, thermal_dispatch)
            _collect_time(opstats.observed_thermal_headroom_levels, thermal_headroom)
            _collect_time(opstats.observed_thermal_states, thermal_states)

            _collect_time(opstats.observed_renewables_levels, renewables_dispatch)
            _collect_time(opstats.observed_renewables_curtailment, renewables_curtailment)

            _collect_time(opstats.storage_input_dispatch_levels, storage_input_dispatch)
            _collect_time(opstats.storage_output_dispatch_levels, storage_output_dispatch)

        # load generation
        gen_dict = {}
        for g, fuel in generator_fuels.items():
            gen_dict[g] = { 'fuel' : fuel , 'generator_type' : 'renewable' } # will get re-set below for thermal units
        for g, quickstart in thermal_quickstart.items():
            gen_dict[g]['fast_start'] = quickstart
            gen_dict[g]['generator_type'] = 'thermal'

        _add_timeseries_attribute_to_egret_dict(gen_dict, thermal_dispatch, 'pg')
        _add_timeseries_attribute_to_egret_dict(gen_dict, thermal_headroom, 'headroom')
        _add_timeseries_attribute_to_egret_dict(gen_dict, thermal_states, 'commitment')

        _add_timeseries_attribute_to_egret_dict(gen_dict, renewables_dispatch, 'pg')
        _add_timeseries_attribute_to_egret_dict(gen_dict, renewables_curtailment, 'curtailment')
        for g_dict in gen_dict.values():
            if g_dict['generator_type'] == 'renewable':
                pg = g_dict['pg']['values']
                curtailment = g_dict['curtailment']['values']
                g_dict['p_max'] = _time_series_dict([pg_val+c_val for pg_val, c_val in zip(pg, curtailment)])

        elements['generator'] = gen_dict

        # load storage
        storage_dict = {}
        for s, stype in storage_types.items():
            storage_dict[s] = { 'fuel' : stype }
        _add_timeseries_attribute_to_egret_dict(storage_dict, storage_input_dispatch, 'p_charge')
        _add_timeseries_attribute_to_egret_dict(storage_dict, storage_output_dispatch, 'p_discharge')

        elements['storage'] = storage_dict

        figure_path = os.path.join(options.output_directory, "plots","stackgraph_"+str(daily_stats.date)+".png")

        graphutils.generate_stack_graph(ModelData(md_dict),
                                        bar_width=1,
                                        x_tick_frequency=4*(60//options.sced_frequency_minutes),
                                        title=str(daily_stats.date),
                                        save_fig=figure_path)

    @staticmethod
    def generate_cost_summary_graph(options, overall_stats: OverallStats):
        daily_fixed_costs = [daily_stats.this_date_fixed_costs for daily_stats in overall_stats.daily_stats]
        daily_generation_costs = [daily_stats.this_date_variable_costs for daily_stats in overall_stats.daily_stats]
        daily_load_shedding = [daily_stats.this_date_load_shedding for daily_stats in overall_stats.daily_stats]
        daily_over_generation = [daily_stats.this_date_over_generation for daily_stats in overall_stats.daily_stats]
        daily_reserve_shortfall = [daily_stats.this_date_reserve_shortfall for daily_stats in overall_stats.daily_stats]
        daily_renewables_curtailment = [daily_stats.this_date_renewables_curtailment for daily_stats in overall_stats.daily_stats]

        graphutils.generate_cost_summary_graph(daily_fixed_costs, daily_generation_costs,
                                               daily_load_shedding, daily_over_generation,
                                               daily_reserve_shortfall,
                                               daily_renewables_curtailment,
                                               output_directory=os.path.join(options.output_directory, "plots"))
