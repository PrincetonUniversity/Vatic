"""Loading power grid datasets into a standardized interface.

This module contains a `GridLoader` class for each power grid system; these
classes contain methods for parsing the raw input data for a grid into formats
suitable for use within Vatic as well as downstream analyses.

The grid parsing logic implemented here was originally adapted from
rtsgmlc_to_dat.py and process_RTS_GMLC_data.py in
Prescient/prescient/downloaders/rts_gmlc_prescient.
"""

from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
import bz2
import dill as pickle

import math
import pandas as pd

from datetime import datetime
from typing import Tuple, Optional, Union


def load_input(data_dir: Path, start_date: Optional[datetime] = None,
               num_days: Optional[int] = None) -> Tuple[dict, pd.DataFrame,
                                                        pd.DataFrame]:
    """Gets grid data from an input folder; creates the data if necessary."""

    if data_dir.exists():
        template_file = Path(data_dir, "grid-template.p")
        gen_file = Path(data_dir, "gen-data.p")
        load_file = Path(data_dir, "load-data.p")

        # load input datasets, starting with static grid data (e.g. network
        # topology, thermal generator outputs)
        if template_file.exists() and gen_file.exists() and load_file.exists():
            with open(template_file, 'rb') as f:
                template: dict = pickle.load(f)

            # load renewable generator forecasted and actual outputs
            with open(gen_file, 'rb') as f:
                gen_data: pd.DataFrame = pickle.load(f)

            # load bus forecasted and actual outputs
            with open(load_file, 'rb') as f:
                load_data: pd.DataFrame = pickle.load(f)

        # if the input datasets have not yet been saved to file, generate
        # them from scratch
        else:
            start_date = pd.Timestamp(start_date, tz='utc')
            end_date = start_date + pd.Timedelta(days=num_days)

            loaders = {'RTS_Data': RtsLoader,
                       'texas-7k': T7kLoader, 'texas-7k_2030': T7k2030Loader}
            loader = loaders[data_dir.stem](data_dir)

            template = loader.create_template()
            gen_data, load_data = loader.create_timeseries(
                start_date, end_date)

    else:
        raise ValueError(
            "Input directory `{}` does not exist!".format(data_dir))

    return template, gen_data, load_data


class GridLoader(ABC):
    """Input dataset parsing logic common across all power grid networks.

    This abstract class contains utilities for loading and processing input
    datasets from any power grid. Utilities whose behaviour will be unique
    to each power grid are implemented as abstract methods; each grid thus
    defines its own child class specifying these processing steps (see below).
    In some cases there are "default" behaviours with concrete implementations
    in this root class that are overridden in classes whose grids exhibit
    unusual characteristics.
    """

    Generator = namedtuple(
        'Generator', ['ID', # integer
                      'Bus',
                      'UnitGroup', 'UnitType',
                      'Fuel',
                      'MinPower', 'MaxPower',
                      'MinDownTime', 'MinUpTime',

                      # units are MW/minute
                      'RampRate',

                      # units are hours
                      'StartTimeCold', 'StartTimeWarm', 'StartTimeHot',

                      # units are MBTU
                      'StartCostCold',
                      'StartCostWarm',
                      'StartCostHot',

                      # units are $
                      'NonFuelStartCost', # units are $

                      # units are $ / MMBTU
                      'FuelPrice',

                      'TotalCostPoints', 'TotalCostValues']
        )

    Bus = namedtuple(
        'Bus', ['ID',  # integer
                'Name',
                'BaseKV',
                'Type',
                'MWLoad',
                'Area', 'SubArea', 'Zone',
                'Lat', 'Long']
        )

    Branch = namedtuple(
        'Branch', ['ID',
                   'FromBus', 'ToBus',
                   'R', 'X',

                   # csv file is in PU, multiply by 100 to make consistent with MW
                   'B',

                   'ContRating']
        )

    thermal_gen_types = {'Nuclear': "N", 'NG': "G", 'Oil': "O", 'Coal': "C"}
    renew_gen_types = {'Wind': "W", 'Solar': "S", 'Hydro': "H"}

    def __init__(self, in_dir):
        self.in_dir = in_dir

        self.gen_df = pd.read_csv(Path(in_dir, "SourceData", "gen.csv"))
        self.branch_df = pd.read_csv(Path(in_dir, "SourceData", "branch.csv"))

        self.bus_df = pd.read_csv(Path(in_dir, "SourceData", "bus.csv"))
        self.bus_df.loc[pd.isnull(self.bus_df['MW Load']), 'MW Load'] = 0.

    @property
    @abstractmethod
    def grid_label(self):
        pass

    @property
    @abstractmethod
    def init_state_file(self):
        pass

    @property
    @abstractmethod
    def utc_offset(self):
        pass

    @property
    @abstractmethod
    def timeseries_cohorts(self):
        pass

    @property
    def no_scenario_renews(self):
        return {'Hydro'}

    @property
    def bus_id_field(self):
        return "Bus Name"

    @staticmethod
    @abstractmethod
    def get_dispatch_types(renew_types):
        pass

    @abstractmethod
    def get_generator_type(self, gen):
        pass

    @abstractmethod
    def get_generator_zone(self, gen):
        pass

    def map_wind_assets(self, asset_df):
        return asset_df

    def map_solar_assets(self, asset_df):
        return asset_df

    def get_asset_info(self) -> pd.DataFrame:
        type_dict = dict()

        for asset_type in self.timeseries_cohorts - self.no_scenario_renews:
            for asset in self.get_forecasts(asset_type).columns:
                type_dict[asset] = asset_type

        return pd.DataFrame({'Type': pd.Series(type_dict)}).merge(
            self.gen_df.set_index('GEN UID', verify_integrity=True),
            left_index=True, right_index=True
            ).rename({'Area Name of Gen': 'Area'}, axis='columns')

    @staticmethod
    def subset_dates(df: pd.DataFrame, start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Helper function to get the rows of a table matching a date range."""

        sub_df = df.copy()
        if start_date:
            sub_df = sub_df.loc[sub_df.index >= start_date]
        if end_date:
            sub_df = sub_df.loc[sub_df.index
                                < (end_date + pd.Timedelta(days=1))]

        return sub_df

    def process_forecasts(self,
                          forecasts_file: Union[str, Path],
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Parse forecasted load/generation values read from an input file."""

        fcst_df = pd.read_csv(forecasts_file)
        df_times = [
            pd.Timestamp(year=year, month=month, day=day, hour=hour, tz='utc')
            for year, month, day, hour in zip(fcst_df.Year, fcst_df.Month,
                                              fcst_df.Day, fcst_df.Period)
            ]

        use_df = fcst_df.drop(columns=['Year', 'Month', 'Day', 'Period'])
        use_df.index = df_times

        return self.subset_dates(use_df)

    @staticmethod
    @abstractmethod
    def process_actuals(actuals_file: Union[str, Path],
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Parse realized load/generation values read from an input file."""
        pass

    def get_forecasts(self, asset_type, start_date=None, end_date=None):
        data_dir = Path(self.in_dir, 'timeseries_data_files', asset_type)
        fcst_file = tuple(data_dir.glob("DAY_AHEAD*"))
        assert len(fcst_file) == 1

        return self.process_forecasts(fcst_file[0], start_date, end_date)

    def get_actuals(self, asset_type, start_date=None, end_date=None):
        data_dir = Path(self.in_dir, 'timeseries_data_files', asset_type)
        actl_file = tuple(data_dir.glob("REAL_TIME*"))
        assert len(actl_file) == 1

        return self.process_actuals(actl_file[0], start_date, end_date)

    def load_by_bus(self, start_date=None, end_date=None, load_actls=None):
        load_fcsts = self.get_forecasts('Load', start_date, end_date)

        if load_actls is None:
            load_actls = self.get_actuals(
                'Load', start_date, end_date).resample('H').mean()

        site_dfs = dict()
        for zone, zone_df in self.bus_df.groupby('Area'):
            area_total_load = zone_df['MW Load'].sum()

            for bus_name, bus_load in zip(zone_df[self.bus_id_field],
                                          zone_df['MW Load']):
                site_df = pd.DataFrame(
                    {'fcst': load_fcsts[str(zone)],
                     'actl': load_actls[str(zone)]}
                    ) * bus_load / area_total_load

                site_df.columns = pd.MultiIndex.from_tuples(
                    [(x, bus_name) for x in site_df.columns])
                site_dfs[bus_name] = site_df

        return pd.concat(site_dfs.values(), axis=1).sort_index(axis=1)

    @classmethod
    @abstractmethod
    def parse_generator(cls, gen_info):
        pass

    def parse_bus(self, bus_info):
        return self.Bus(
            int(bus_info["Bus ID"]), bus_info[self.bus_id_field],
            bus_info["BaseKV"], bus_info["Bus Type"],
            float(bus_info["MW Load"]),
            int(bus_info["Area"]), int(bus_info["Sub Area"]), bus_info["Zone"],
            bus_info["lat"], bus_info["lng"]
            )

    @classmethod
    def parse_branch(cls, branch_info):
        return cls.Branch(
            branch_info["UID"], branch_info["From Bus"], branch_info["To Bus"],
            float(branch_info["R"]),

            # nix per unit
            float(branch_info["X"]) / 100.0,

            float(branch_info["B"]), float(branch_info["Cont Rating"])
            )

    @classmethod
    def must_gen_run(cls, gen: Generator) -> bool:
        return gen.Fuel == 'Nuclear'

    def create_template(self) -> dict:
        init_states = pd.read_csv(self.init_state_file).iloc[0].to_dict()

        generators = [self.parse_generator(gen_info)
                      for _, gen_info in self.gen_df.iterrows()]

        buses = [self.parse_bus(bus_info)
                 for _, bus_info in self.bus_df.iterrows()]
        bus_name_mapping = {bus.ID: bus.Name for bus in buses}

        branches = [self.parse_branch(branch_info)
                    for _, branch_info in self.branch_df.iterrows()]

        mins_per_time_period = 60
        ## we'll bring the ramping down by this factor
        ramp_scaling_factor = 1.

        # remove duplicate buses while maintaining order for reference bus
        use_buses = list()
        for bus in buses:
            if bus.Name not in use_buses:
                use_buses += [bus.Name]

        template = {
            'NumTimePeriods': 48, 'TimePeriodLength': 1,
            'StageSet': ['Stage_1', 'Stage_2'], 'CopperSheet': False,

            'CommitmentTimeInStage': {'Stage_1': list(range(1, 49)),
                                      'Stage_2': list()},
            'GenerationTimeInStage': {'Stage_1': list(),
                                      'Stage_2': list(range(1, 49))},

            'Buses': use_buses,
            'TransmissionLines': [branch.ID for branch in branches],

            'BusFrom': {branch.ID: bus_name_mapping[branch.FromBus]
                        for branch in branches},
            'BusTo': {branch.ID: bus_name_mapping[branch.ToBus]
                      for branch in branches},

            'ThermalLimit': {branch.ID: round(branch.ContRating, 8)
                             for branch in branches},
            'Impedence': {branch.ID: round(branch.X, 8)
                          for branch in branches},

            'MustRun': [gen.ID for gen in generators if self.must_gen_run(gen)]
            }

        tgen_bus_map = {bus.Name: list() for bus in buses}
        rgen_bus_map = {bus.Name: list() for bus in buses}
        tgens = list()
        rgens = list()

        for gen in generators:
            if gen.ID in init_states:
                if gen.Fuel in self.thermal_gen_types:
                    tgens += [gen]
                    tgen_bus_map[bus_name_mapping[gen.Bus]] += [gen.ID]

                if gen.Fuel in self.renew_gen_types:
                    rgens += [gen]
                    rgen_bus_map[bus_name_mapping[gen.Bus]] += [gen.ID]

        template.update({
            'ThermalGenerators': [gen.ID for gen in tgens],
            'NondispatchableGenerators': [gen.ID for gen in rgens],
            'ThermalGeneratorsAtBus': tgen_bus_map,
            'NondispatchableGeneratorsAtBus': rgen_bus_map
            })

        template['ThermalGeneratorType'] = {
            gen.ID: self.thermal_gen_types[gen.Fuel] for gen in tgens}
        template['NondispatchableGeneratorType'] = {
            gen.ID: self.renew_gen_types[gen.Fuel] for gen in rgens}
        template.update(
            self.get_dispatch_types(template['NondispatchableGeneratorType']))

        template['MinimumPowerOutput'] = {gen.ID: round(gen.MinPower, 2)
                                          for gen in tgens}
        template['MaximumPowerOutput'] = {gen.ID: round(gen.MaxPower, 2)
                                          for gen in tgens}

        template['MinimumUpTime'] = {gen.ID: round(gen.MinUpTime, 2)
                                     for gen in tgens}
        template['MinimumDownTime'] = {gen.ID: round(gen.MinDownTime, 2)
                                       for gen in tgens}

        # ramp rates, given in MW/min, are converted to MW/hour; if no rate is
        # given we assume generator can ramp up/down in a single time period
        template['NominalRampUpLimit'] = {
            gen.ID: (round(gen.RampRate
                           * mins_per_time_period / ramp_scaling_factor, 2)
                     if gen.RampRate > 0.
                     else template['MinimumPowerOutput'][gen.ID])
            for gen in tgens
            }

        template['NominalRampDownLimit'] = {
            gen.ID: (round(gen.RampRate
                           * mins_per_time_period / ramp_scaling_factor, 2)
                     if gen.RampRate > 0.
                     else template['MinimumPowerOutput'][gen.ID])
            for gen in tgens
            }

        template['StartupRampLimit'] = {gen.ID: round(gen.MinPower, 2)
                                        for gen in tgens}
        template['ShutdownRampLimit'] = {gen.ID: round(gen.MinPower, 2)
                                         for gen in tgens}

        template['StartupLags'] = {
            gen.ID: (
                [gen.MinDownTime]
                if (gen.StartTimeCold <= gen.MinDownTime
                    or (gen.StartTimeCold == gen.StartTimeWarm
                        == gen.StartTimeHot))

                else [gen.MinDownTime, gen.StartTimeCold]
                if gen.StartTimeWarm <= gen.MinDownTime

                else [gen.MinDownTime, gen.StartTimeWarm, gen.StartTimeCold]
                )

            for gen in tgens
            }

        template['StartupCosts'] = {
            gen.ID: (
                [gen.StartCostCold * gen.FuelPrice + gen.NonFuelStartCost]
                if (gen.StartTimeCold <= gen.MinDownTime
                    or (gen.StartTimeCold == gen.StartTimeWarm
                        == gen.StartTimeHot))

                else [gen.StartCostWarm * gen.FuelPrice + gen.NonFuelStartCost,
                      gen.StartCostCold * gen.FuelPrice + gen.NonFuelStartCost]
                if gen.StartTimeWarm <= gen.MinDownTime

                else [gen.StartCostHot * gen.FuelPrice + gen.NonFuelStartCost,
                      gen.StartCostWarm * gen.FuelPrice + gen.NonFuelStartCost,
                      gen.StartCostCold * gen.FuelPrice + gen.NonFuelStartCost]
                )

            for gen in tgens
            }

        # according to Egret, startup costs cannot be duplicated, so we
        # introduce a small perturbation to avoid this
        for gen_id in template['StartupCosts']:
            if (len(set(template['StartupCosts'][gen_id]))
                    < len(template['StartupCosts'][gen_id])):
                template['StartupCosts'][gen_id] = [
                    round(cost, 2) + i * 0.01
                    for i, cost in enumerate(template['StartupCosts'][gen_id])
                    ]

            else:
                template['StartupCosts'][gen_id] = [
                    round(cost, 2)
                    for cost in template['StartupCosts'][gen_id]
                    ]

        template['CostPiecewisePoints'] = {
            gen.ID: [round(pnt, 2) for pnt in gen.TotalCostPoints]
            for gen in tgens
            }
        template['CostPiecewiseValues'] = {
            gen.ID: [round(cst, 4) for cst in gen.TotalCostValues]
            for gen in tgens
            }

        template['UnitOnT0State'] = {gen.ID: init_states[gen.ID]
                                     for gen in tgens if gen.ID in init_states}

        template['PowerGeneratedT0'] = {
            gen.ID: 0. if init_states[gen.ID] < 0 else round(gen.MinPower, 2)
            for gen in tgens if gen.ID in init_states
            }

        return template

    def create_timeseries(self, start_date=None, end_date=None):
        gen_dfs = list()

        for asset_type in self.timeseries_cohorts:
            gen_fcsts = self.get_forecasts(asset_type, start_date, end_date)
            gen_actls = self.get_actuals(
                asset_type, start_date, end_date).resample('H').mean()

            gen_fcsts.columns = pd.MultiIndex.from_tuples(
                [('fcst', asset_name) for asset_name in gen_fcsts.columns])
            gen_actls.columns = pd.MultiIndex.from_tuples(
                [('actl', asset_name) for asset_name in gen_actls.columns])

            gen_dfs += [gen_fcsts, gen_actls]

        gen_df = pd.concat(gen_dfs, axis=1).sort_index(axis=1)
        demand_df = self.load_by_bus(start_date, end_date)

        return gen_df, demand_df

    def create_scenario_timeseries(self,
                                   scen_dir, start_date, end_date, scenario):
        gen_df = pd.concat([self.get_forecasts(asset_type,
                                               start_date, end_date)
                            for asset_type in self.timeseries_cohorts],
                           axis=1)

        gen_df.columns = pd.MultiIndex.from_tuples(
            [('fcst', asset_name) for asset_name in gen_df.columns])

        year_scens = {'Load': dict(), 'Wind': dict(), 'Solar': dict()}
        for scen_day in pd.date_range(start=start_date, end=end_date,
                                      freq='D'):
            out_file = Path(scen_dir, "scens_{}.p.gz".format(scen_day.date()))

            with bz2.BZ2File(out_file, 'r') as f:
                day_scens = pickle.load(f)

            for asset_type in ['Load', 'Wind', 'Solar']:
                year_scens[asset_type][scen_day] = day_scens[
                    asset_type].iloc[scenario]

        wind_rts_scens = pd.concat(year_scens['Wind'].values()).unstack().T
        wind_rts_scens.index += self.utc_offset
        solar_rts_scens = pd.concat(year_scens['Solar'].values()).unstack().T
        solar_rts_scens.index += self.utc_offset

        gen_scens = pd.concat([self.map_wind_assets(wind_rts_scens),
                               self.map_solar_assets(solar_rts_scens)],
                              axis=1)

        gen_scens.columns = pd.MultiIndex.from_tuples(
            [('actl', asset_name) for asset_name in gen_scens.columns])
        gen_df = pd.concat([gen_df, gen_scens], axis=1)

        for asset_type in self.no_scenario_renews:
            new_actuals = self.get_actuals(
                asset_type, start_date, end_date).resample('H').mean()

            for asset_name, asset_actuals in new_actuals.iteritems():
                gen_df['actl', asset_name] = asset_actuals

        #TODO: investigate cases where forecasts are missing for assets
        #      with scenario values
        for gen in gen_df.actl.columns:
            if gen not in gen_df.fcst.columns:
                gen_df.drop(('actl', gen), axis=1, inplace=True)

        assert (sorted(gen_df['fcst'].columns)
                == sorted(gen_df['actl'].columns)), (
            "Mismatching sets of assets with forecasts and with actuals!")

        load_rts_scens = pd.concat(year_scens['Load'].values()).unstack().T
        load_rts_scens.index += self.utc_offset
        demand_df = self.load_by_bus(start_date, end_date,
                                     load_actls=load_rts_scens)

        return gen_df.sort_index(axis=1), demand_df


class RtsLoader(GridLoader):
    """The RTS-GMLC grid which was created for testing purposes.

    See github.com/GridMod/RTS-GMLC for the raw data files used for this grid.
    """

    @property
    def grid_label(self):
        return "RTS-GMLC"

    @property
    def init_state_file(self):
        return Path(self.in_dir, 'FormattedData', 'PLEXOS', 'PLEXOS_Solution',
                    'DAY_AHEAD Solution Files', 'noTX', 'on_time_7.12.csv')

    @property
    def utc_offset(self):
        return -pd.Timedelta(hours=8)

    @property
    def timeseries_cohorts(self):
        return {'WIND', 'PV', 'RTPV', 'Hydro'}

    @staticmethod
    def get_dispatch_types(renew_types):
        return {
            'DispatchRenewables': {gen
                                   for gen, gen_type in renew_types.items()
                                   if (gen_type != 'H'
                                       and gen.split('_')[1] != 'RTPV')},

            'NondispatchRenewables': {gen
                                      for gen, gen_type in renew_types.items()
                                      if (gen_type == 'H'
                                          or gen.split('_')[1] == 'RTPV')},

            'ForecastRenewables': {gen for gen, gen_type in renew_types.items()
                                   if gen_type != 'H' and '_CSP_' not in gen}
            }

    def get_generator_type(self, gen):
        gen_type = self.gen_df['Unit Group'][self.gen_df['GEN UID']
                                             == gen].iloc[0]

        if gen_type == 'WIND':
            gen_type = 'Wind'

        return gen_type

    def get_generator_zone(self, gen):
        return gen[0]

    @classmethod
    def parse_generator(cls, gen_info):
        # round the power points to the nearest 10kW
        # IMPT: These quantities are MW
        cost_points = [
            round(output_pct * gen_info['PMax MW'], 1)
            for output_pct in [gen_info.Output_pct_0, gen_info.Output_pct_1,
                               gen_info.Output_pct_2, gen_info.Output_pct_3]
            ]

        cost_vals = [gen_info['Fuel Price $/MMBTU']
                     * ((gen_info.HR_avg_0 * 1000.0 / 1000000.0)
                        * cost_points[0])]

        # NOTES:
        # 1) Fuel price is in $/MMBTU
        # 2) Heat Rate quantities are in BTU/KWH
        # 3) 1+2 => need to convert both from BTU->MMBTU and from KWH->MWH
        for i, rate_incr in enumerate(
                [gen_info.HR_incr_1, gen_info.HR_incr_2, gen_info.HR_incr_3]):
            cost_vals += [gen_info['Fuel Price $/MMBTU']
                          * ((cost_points[i + 1] - cost_points[i])
                             * (rate_incr * 1000.0 / 1000000.0))
                          + cost_vals[-1]]

        for i in range(len(cost_vals)):
            cost_vals[i] = round(cost_vals[i], 2)

        # PRESCIENT currently doesn't gracefully handle generators
        # with zero marginal cost
        if cost_vals[0] == cost_vals[-1]:
            cost_points = [cost_points[0], cost_points[-1]]
            cost_vals = [cost_vals[0], cost_vals[-1] + 0.01]

        else:
            convex = False

            # have to avoid cost curves becoming non-convex due to rounding
            # since Prescient also doesn't like non-convexity
            while not convex:
                convex = True

                cost_slopes = [((cost_vals[i + 1] - cost_vals[i])
                                / (cost_points[i + 1] - cost_points[i]))
                               for i in range(len(cost_vals) - 1)]

                for j in range(len(cost_vals) - 2):
                    if cost_slopes[j + 1] < cost_slopes[j]:
                        cost_vals[j + 2] += 0.01

                        convex = False
                        break

        return cls.Generator(
            gen_info["GEN UID"], int(gen_info["Bus ID"]),
            gen_info["Unit Group"], gen_info["Unit Type"], gen_info["Fuel"],
            float(gen_info["PMin MW"]), float(gen_info["PMax MW"]),

            # per Brendan, PLEXOS takes the ceiling at hourly resolution for up
            # and down times.
            int(math.ceil(gen_info["Min Down Time Hr"])),
            int(math.ceil(gen_info["Min Up Time Hr"])),

            gen_info["Ramp Rate MW/Min"],
            int(gen_info["Start Time Cold Hr"]),
            int(gen_info["Start Time Warm Hr"]),
            int(gen_info["Start Time Hot Hr"]),

            float(gen_info["Start Heat Cold MBTU"]),
            float(gen_info["Start Heat Warm MBTU"]),
            float(gen_info["Start Heat Hot MBTU"]),

            float(gen_info["Non Fuel Start Cost $"]),
            float(gen_info["Fuel Price $/MMBTU"]),

            cost_points, cost_vals
            )

    def process_forecasts(self,
                          forecasts_file, start_date=None, end_date=None):
        fcst_df = pd.read_csv(forecasts_file)

        df_times = [
            pd.Timestamp(year=year, month=month, day=day, hour=hour, tz='utc')
            for year, month, day, hour in zip(fcst_df.Year, fcst_df.Month,
                                              fcst_df.Day, fcst_df.Period - 1)
            ]

        use_df = fcst_df.drop(columns=['Year', 'Month', 'Day', 'Period'])
        use_df.index = df_times

        return self.subset_dates(use_df, start_date, end_date)

    def process_actuals(self, actuals_file, start_date=None, end_date=None):
        actl_df = pd.read_csv(actuals_file)

        actl_df['Hour'] = (actl_df.Period - 1) // 12
        actl_df['Min'] = (actl_df.Period - 1) % 12 * 5

        df_times = [
            pd.Timestamp(year=year, month=month, day=day, hour=hour,
                         minute=minute, tz='utc')
            for year, month, day, hour, minute in zip(
                actl_df.Year, actl_df.Month, actl_df.Day,
                actl_df.Hour, actl_df.Min
                )
            ]

        use_df = actl_df.drop(
            columns=['Year', 'Month', 'Day', 'Period', 'Hour', 'Min'])
        use_df.index = df_times

        return self.subset_dates(use_df, start_date, end_date)


class T7kLoader(GridLoader):
    """The Texas-7k grid modeling the ERCOT system, developed by Texas A&M."""

    thermal_gen_types = {
        'NUC (Nuclear)': 'N', 'NG (Natural Gas)': 'G',
        'LIG (Lignite Coal)': 'O', 'SUB (Subbituminous Coal)': 'C',
        'WH (Waste Heat)': 'A', 'WDS (Wood/Wood Waste Solids)': 'A',
        'PUR (Purchased Steam)': 'A'
        }

    renew_gen_types = {'WND (Wind)': "W", 'SUN (Solar)': "S",
                       'WAT (Water)': "H"}

    @property
    def grid_label(self):
        return "Texas-7k"

    @property
    def init_state_file(self):
        return Path(self.in_dir, 'FormattedData', 'PLEXOS', 'PLEXOS_Solution',
                    'DAY_AHEAD Solution Files', 'noTX', 'on_time_7.10.csv')

    @property
    def utc_offset(self):
        return -pd.Timedelta(hours=6)

    @property
    def timeseries_cohorts(self):
        return {'WIND', 'PV'}

    @property
    def bus_id_field(self):
        return "Sub Name"

    @staticmethod
    def get_dispatch_types(renew_types):
        return {
            'DispatchRenewables': set(),

            'NondispatchRenewables': {gen
                                      for gen, gen_type in renew_types.items()
                                      if gen_type != 'H'},

            'ForecastRenewables': {gen for gen, gen_type in renew_types.items()
                                   if gen_type != 'H'}
            }

    def map_wind_assets(self, asset_df):
        wind_maps = pd.read_csv(Path(self.in_dir, "Texas7k_NREL_wind_map.csv"),
                                index_col=0)
        wind_gens = self.gen_df.loc[self.gen_df.Fuel == 'WND (Wind)']

        mapped_vals = dict()
        for _, gen_info in wind_gens.iterrows():
            map_match = wind_maps['Texas7k BusNum'] == gen_info['Bus ID']
            assert map_match.sum() == 1

            nrel_name, t7k_max, nrel_capacity, dist_factor = wind_maps[
                map_match][['NREL Wind Site', 'Texas7k Max MW',
                            'NREL Capacity Proportion',
                            'Distribution Factor']].iloc[0]

            parsed_name = nrel_name.replace('_', ' ')
            if parsed_name == 'S Hills Wind':
                parsed_name = 'S_Hills Wind'

            mapped_vals[gen_info['GEN UID']] = (
                    asset_df[parsed_name]
                    * float(dist_factor / nrel_capacity * t7k_max)
                    )

        return pd.concat(mapped_vals, axis=1)

    def map_solar_assets(self, asset_df):
        solar_maps = pd.read_csv(Path(self.in_dir,
                                      "Texas7k_NREL_solar_map.csv"),
                                 index_col=0)
        solar_gens = self.gen_df.loc[self.gen_df.Fuel == 'SUN (Solar)']

        mapped_vals = dict()
        for _, gen_info in solar_gens.iterrows():
            map_match = solar_maps['BusNum'] == gen_info['Bus ID']
            assert map_match.sum() == 1

            mapped_vals[gen_info['GEN UID']] = (
                    asset_df[solar_maps[map_match]['Min_site'].iloc[0]]
                    * float(solar_maps[map_match]['dist_factor'].iloc[0])
                    )

        return pd.concat(mapped_vals, axis=1)

    @classmethod
    def parse_generator(cls, gen_info):
        break_cols = gen_info.index[
            gen_info.index.str.match('MW Break [0-9]*')]
        price_cols = gen_info.index[
            gen_info.index.str.match('MWh Price [0-9]*')]

        break_indxs = {int(col.split(' Break ')[1]): col for col in break_cols}
        price_indxs = {int(col.split(' Price ')[1]): col for col in price_cols}
        cost_indxs = sorted(set(break_indxs) & set(price_indxs))

        cost_values = [gen_info['Fixed Cost($/hr)']
                       + gen_info[price_indxs[cost_indxs[0]]]
                       * gen_info[break_indxs[cost_indxs[0]]]]

        for i in range(len(cost_indxs) - 1):
            cost_values += [cost_values[-1]
                            + gen_info[price_indxs[cost_indxs[i]]]
                            * (gen_info[break_indxs[cost_indxs[i + 1]]]
                               - gen_info[break_indxs[cost_indxs[i]]])]

        cost_values += [cost_values[-1]
                        + gen_info[price_indxs[cost_indxs[-1]]]
                        * (gen_info['PMax MW']
                           - gen_info[break_indxs[cost_indxs[-1]]])]

        return cls.Generator(
            gen_info["GEN UID"], int(gen_info["Bus ID"]),
            gen_info["Unit Group"], gen_info["Unit Type"], gen_info["Fuel"],
            float(gen_info["PMin MW"]), float(gen_info["PMax MW"]),

            # per Brendan, PLEXOS takes the ceiling at hourly resolution for
            # up and down times
            int(math.ceil(gen_info["Min Down Time Hr"])),
            int(math.ceil(gen_info["Min Up Time Hr"])),

            gen_info["Ramp Rate MW/Min"],
            int(gen_info["Start Time Cold Hr"]),
            int(gen_info["Start Time Warm Hr"]),
            int(gen_info["Start Time Hot Hr"]),

            float(gen_info["Start Heat Cold MBTU"]),
            float(gen_info["Start Heat Warm MBTU"]),
            float(gen_info["Start Heat Hot MBTU"]),

            float(gen_info["Non Fuel Start Cost $"]),
            float(gen_info["Fuel Price $/MMBTU"]),

            gen_info[break_cols].tolist() + [gen_info['PMax MW']], cost_values
            )

    def parse_bus(self, bus_info):
        return self.Bus(
            int(bus_info["Bus ID"]), bus_info[self.bus_id_field],
            bus_info["BaseKV"], bus_info["Bus Type"],
            float(bus_info["MW Load"]),
            bus_info["Area"], int(bus_info["Sub Area"]), bus_info["Zone"],
            bus_info["lat"], bus_info["lng"]
            )

    #TODO: why doesn't T7k have timeseries for the hydro plants in the system?
    @property
    def no_scenario_renews(self):
        return set()

    def get_generator_type(self, gen):
        return self.gen_df.Fuel[self.gen_df['GEN UID']
                                == gen].iloc[0].split('(')[1][:-1]

    def get_generator_zone(self, gen):
        return gen[0]

    @classmethod
    def must_gen_run(cls, gen: GridLoader.Generator) -> bool:
        return gen.Fuel == 'NUC (Nuclear)'

    def process_actuals(self, actuals_file, start_date=None, end_date=None):
        actl_df = pd.read_csv(actuals_file)

        df_times = [
            pd.Timestamp(year=year, month=month, day=day, hour=hour,
                         minute=0, tz='utc')
            for year, month, day, hour in zip(actl_df.Year, actl_df.Month,
                                              actl_df.Day, actl_df.Period)
            ]

        use_df = actl_df.drop(columns=['Year', 'Month', 'Day', 'Period'])
        use_df.index = df_times

        return self.subset_dates(use_df, start_date, end_date)


class T7k2030Loader(T7kLoader):
    """The Texas-7k grid projected into a future with more renewables."""

    @property
    def grid_label(self):
        return "Texas-7k(2030)"

    @property
    def timeseries_cohorts(self):
        return {'wind', 'solar'}

    @property
    def bus_id_field(self):
        return "Bus Name"

    def get_forecasts(self, asset_type, start_date=None, end_date=None):
        fcst_file = tuple(Path(self.in_dir, 'timeseries_data_files').glob(
            "{}_day_ahead_forecast_*".format(asset_type)))

        assert len(fcst_file) == 1
        fcst_df = pd.read_csv(fcst_file[0], parse_dates=['Forecast_time'])
        fcst_df.drop('Issue_time', axis=1, inplace=True)
        fcst_df.Forecast_time += self.utc_offset
        fcst_df.set_index('Forecast_time', inplace=True, verify_integrity=True)

        if start_date:
            fcst_df = fcst_df.loc[fcst_df.index >= start_date]
        if end_date:
            fcst_df = fcst_df.loc[fcst_df.index
                                  < (end_date + pd.Timedelta(days=1))]

        if asset_type == 'wind':
            fcst_df = self.map_wind_assets(fcst_df)
        elif asset_type == 'solar':
            fcst_df = self.map_solar_assets(fcst_df)

        return fcst_df

    def get_actuals(self, asset_type, start_date=None, end_date=None):
        actl_file = tuple(Path(self.in_dir, 'timeseries_data_files').glob(
            "{}_actual_1h_*".format(asset_type)))

        assert len(actl_file) == 1
        actl_df = pd.read_csv(actl_file[0], parse_dates=['Time'])
        actl_df.Time += self.utc_offset
        actl_df.set_index('Time', inplace=True, verify_integrity=True)

        if start_date:
            actl_df = actl_df.loc[actl_df.index >= start_date]
        if end_date:
            actl_df = actl_df.loc[actl_df.index
                                  < (end_date + pd.Timedelta(days=1))]

        if asset_type == 'wind':
            actl_df = self.map_wind_assets(actl_df)
        elif asset_type == 'solar':
            actl_df = self.map_solar_assets(actl_df)

        return actl_df

    def load_by_bus(self, start_date, end_date, load_actls=None):
        load_fcsts = self.get_forecasts('load', start_date, end_date)

        if load_actls is None:
            load_actls = self.get_actuals('load', start_date, end_date)

        site_dfs = dict()
        for zone, zone_df in self.bus_df.groupby('Area'):
            area_total_load = zone_df['MW Load'].sum()

            for bus_name, bus_load in zip(zone_df[self.bus_id_field],
                                          zone_df['MW Load']):
                site_df = pd.DataFrame(
                    {'fcst': load_fcsts[str(zone)],
                     'actl': load_actls[str(zone)]}
                    ) * bus_load / area_total_load

                site_df.columns = pd.MultiIndex.from_tuples(
                    [(x, bus_name) for x in site_df.columns])
                site_dfs[bus_name] = site_df

        return pd.concat(site_dfs.values(), axis=1).sort_index(axis=1)

    #TODO: should we standardize the mapping file format between the T7k and
    #      the T7k(2030) grids instead of doing this?
    def map_wind_assets(self, asset_df):
        wind_maps = pd.read_csv(Path(self.in_dir, "Texas7k_NREL_wind_map.csv"))
        wind_gens = self.gen_df.loc[self.gen_df.Fuel == 'WND (Wind)']

        mapped_vals = dict()
        for _, gen_info in wind_gens.iterrows():
            map_match = wind_maps['BUS UID'] == gen_info['BUS UID']
            assert map_match.sum() == 1

            nrel_name, t7k_max, nrel_capacity, dist_factor = wind_maps[
                map_match][['NREL Wind Site', 'Texas7k Max MW',
                            'NREL Capacity Proportion',
                            'Distribution Factor']].iloc[0]

            parsed_name = nrel_name.replace('_', ' ')
            if parsed_name == 'S Hills Wind':
                parsed_name = 'S_Hills Wind'

            mapped_vals[gen_info['GEN UID']] = (
                    asset_df[parsed_name]
                    * float(dist_factor / nrel_capacity * t7k_max)
                    )

        return pd.concat(mapped_vals, axis=1)

    def map_solar_assets(self, asset_df):
        solar_maps = pd.read_csv(Path(self.in_dir,
                                      "Texas7k_NREL_solar_map.csv"))
        solar_gens = self.gen_df.loc[self.gen_df.Fuel == 'SUN (Solar)']

        mapped_vals = dict()
        for _, gen_info in solar_gens.iterrows():
            map_match = solar_maps['BUS UID'] == gen_info['BUS UID']
            assert map_match.sum() == 1

            mapped_vals[gen_info['GEN UID']] = (
                    asset_df[solar_maps[map_match]['min_site'].iloc[0]]
                    * float(solar_maps[map_match]['dist_factor'].iloc[0])
                    )

        return pd.concat(mapped_vals, axis=1)
