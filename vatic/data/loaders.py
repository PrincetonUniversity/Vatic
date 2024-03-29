"""Loading power grid datasets into a standardized interface.

This module contains a `GridLoader` class for each power grid system; these
classes contain methods for parsing the raw input data for a grid into formats
suitable for use within Vatic as well as for downstream analyses.

In particular, these loaders are used to create the three primary grid data
input files used by Vatic:
    `template`  A dictionary of static grid elements including thermal
                generators, transmission lines, load buses, etc. and their
                characteristics, as well as the initial state of the grid.
    `gen_data`  A dataframe of forecasted and actual renewable generator power
                output over the timeframe to be simulated.
    `load_data` A dataframe of forecasted and actual load bus power demand over
                the timeframe to be simulated.

The grid parsing logic implemented here was originally adapted from
rtsgmlc_to_dat.py and process_RTS_GMLC_data.py in
Prescient/prescient/downloaders/rts_gmlc_prescient.
"""

from __future__ import annotations

import bz2
import dill as pickle
from pathlib import Path

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Mapping, Iterable, Optional

import math
import pandas as pd
from datetime import datetime

import os
_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_input(
        grid: str, start_date: str | pd.Timestamp, num_days: int,
        init_state_file: Optional[str | Path]
        ) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Uses a label to retrieve the input datasets for a given grid system."""

    if grid == 'RTS-GMLC':
        use_loader = RtsLoader(init_state_file)

    elif grid == 'Texas-7k':
        use_loader = T7kLoader(init_state_file)

    elif grid == 'Texas-7k_2030':
        use_loader = T7k2030Loader(init_state_file)

    else:
        raise ValueError(f"Unsupported grid `{grid}`!")

    start_date = pd.Timestamp(start_date, tz='utc')
    end_date = start_date + pd.Timedelta(days=num_days)
    gen_data, load_data = use_loader.create_timeseries(start_date, end_date)

    return use_loader.template, gen_data, load_data


class GridLoader(ABC):
    """Input dataset parsing logic common across all power grid networks.

    This abstract class contains utilities for loading and processing input
    datasets from any power grid. Utilities whose behaviour will be unique
    to each power grid are implemented as abstract methods; each grid thus
    defines its own child class specifying these processing steps (see below).
    In some cases there are "default" behaviours with concrete implementations
    in this root class that are overridden in classes whose grids exhibit
    unusual characteristics.

    Attributes
    ----------
    grid_lbl    the label used for this grid's data folder within
                vatic/data/grids, and in other contexts such as command line
                arguments, plot labels, etc.

    plot_lbl    a short name for the grid to be used in plots

    _data_dir   the name of the data folder *within* this grid's own data
                folder that by convention contains the grid data (as opposed
                to the metadata)

    """
    grid_lbl = None
    _data_dir = None

    # the three static elements of a power grid, with the properties of each
    # thermal generators produce power based on a partly pre-planned schedule
    Generator = namedtuple(
        'Generator', [
            'ID',   # usually the GEN UID
            'Bus',  # the Bus ID of the load bus this gen is located at

            # specifying the type of generator at various
            # granularities; e.g. Group: U20, Type: CT, Fuel: Oil
            'UnitGroup', 'UnitType', 'Fuel',

            'MinPower', 'MaxPower', # the operating range given in MW

            # how long the generator must be on/off before being
            # turned off/on, in hours
            'MinDownTime', 'MinUpTime',

            # how quickly the output of the generator can be changed, in MW/min
            'RampRate',

            # time needed after shutdown for various startup regimes, in hours
            'StartTimeCold', 'StartTimeWarm', 'StartTimeHot',

            # the cost of the above regimes in $/MBTU
            'StartCostCold', 'StartCostWarm', 'StartCostHot',

            # the cost of fuel used for generation, in $/MBTU
            'FuelPrice',

            # total cost curve for this generator, given as break points in MW
            # and corresponding total cost values in $/MWh
            'TotalCostPoints', 'TotalCostValues'
            ]
        )

    # load buses are where demand for electricity is measured (MWLoad)
    Bus = namedtuple(
        'Bus', [
            'ID',   # usually a six-digit integer
            'Name', 'BaseKV',

            'Type', # PQ vs. PV
            'MWLoad',

            # various ways of specifying the location of the bus
            'Area', 'SubArea', 'Zone',
            'Lat', 'Long'
            ]
        )

    # transmission lines are how power flows from generators to load buses,
    Branch = namedtuple(
        'Branch', [
            'ID',   # usually upper-case letter followed by integer, e.g. A31
            'FromBus', 'ToBus', # the IDs of the buses this line connects

            # resistance, reactance, and charging susceptance, usually given
            # in p.u.; reactance is divided by 100 to make consistent with MW
            'R', 'X', 'B',

            # the capacity of the line in MW
            'ContRating'
            ]
        )

    thermal_gen_types = {'Nuclear': "N", 'NG': "G", 'Oil': "O", 'Coal': "C"}
    renew_gen_types = {'Wind': "W", 'Solar': "S", 'Hydro': "H"}

    def __init__(self,
                 init_state_file: str | Path | None = None,
                 mins_per_time_period: int = 60) -> None:
        """Create the static characteristics of the grid.

        Initializing a grid loader entails parsing the grid metadata to get the
        characteristics of the system that remain constant across experiment
        runs. These include fields such as:

            ThermalGenerators
                Non-renewable power plants whose output can be manipulated by
                the operator subject to operating constraints.
            NondispatchableGenerators
                Power plants that rely on renewable sources of energy such as
                wind and sun whose output levels are determined exogenously.

            Buses
                The nodes of the grid at which load demand is measured, and
                where generators are located.
            TransmissionLines
                The power lines carrying electricity from generators to buses.

            BusFrom, BusTo
                The nodes of the grid that each power line connects.
            ThermalGeneratorsAtBus, NondispatchableGeneratorsAtBus
                Which generators are located at each bus; every generator is
                located at a bus.

            MinimumPowerOutput
                Some generators, especially thermals, must produce a certain
                baseline level of power once they are turned on.
            MaximumPowerOutput
                All generators have a maximum generation they are capable of.
            MinimumUpTime, MinimumDownTime
                Thermal generators must be kept on for a certain amount of time
                once they are turned on, and kept off once they are turned off.

            NominalRampUpLimit, NominalRampDownLimit
                How rapidly thermal generators' power output can be adjusted
                once they are turned on.
            StartupRampLimit, ShutdownRampLimit
                How rapidly thermal generators' power output can be brought
                online when turned off, and how quickly it can be shut off when
                the plant is turned on.

            CostPiecewisePoints, CostPiecewiseValues
                The marginal cost of producing a certain amount of power varies
                according to total power output for each thermal generator; we
                define a cost curve to capture this behaviour specifying total
                costs of power production at particular power output levels.
            StartupLags, StartupCosts
                Each thermal generator also has a specific cost associated with
                the process of being turned on, and this cost varies depending
                on how long the generator has been turned off.

            UnitOnT0State, PowerGeneratedT0
                For the sake of the simulation we need initial conditions for
                the thermal generators specifying how long they have been
                turned on or off and how much power they are producing.

        Args
        ----
            mins_per_time_period    How many minutes each time step in the
                                    simulation will take; used to normalize
                                    ramping rates.

        """
        self.mins_per_time_period = mins_per_time_period

        # read in metadata about static grid elements from file, clean data
        # where necessary, and parse grid elements to get key properties
        self.gen_df = pd.read_csv(Path(self.data_path,
                                       "SourceData", "gen.csv"))
        self.branch_df = pd.read_csv(Path(self.data_path,
                                          "SourceData", "branch.csv"))

        self.bus_df = pd.read_csv(Path(self.data_path,
                                       "SourceData", "bus.csv"))
        self.bus_df.loc[pd.isnull(self.bus_df['MW Load']), 'MW Load'] = 0.

        self.generators = [self.parse_generator(gen_info)
                           for _, gen_info in self.gen_df.iterrows()]

        self.buses = [self.parse_bus(bus_info)
                      for _, bus_info in self.bus_df.iterrows()]
        bus_name_mapping = {bus.ID: bus.Name for bus in self.buses}

        self.branches = [self.parse_branch(branch_info)
                         for _, branch_info in self.branch_df.iterrows()]

        # remove duplicate buses while maintaining order for reference bus
        use_buses = list()
        for bus in self.buses:
            if bus.Name not in use_buses:
                use_buses += [bus.Name]

        # initialize grid template with basic elements that do not require any
        # further data processing steps
        template = {
            'NumTimePeriods': 48, 'TimePeriodLength': 1,
            'StageSet': ['Stage_1', 'Stage_2'], 'CopperSheet': False,

            'CommitmentTimeInStage': {'Stage_1': list(range(1, 49)),
                                      'Stage_2': list()},
            'GenerationTimeInStage': {'Stage_1': list(),
                                      'Stage_2': list(range(1, 49))},

            'Buses': use_buses,
            'TransmissionLines': [branch.ID for branch in self.branches],

            'BusFrom': {branch.ID: bus_name_mapping[branch.FromBus]
                        for branch in self.branches},
            'BusTo': {branch.ID: bus_name_mapping[branch.ToBus]
                      for branch in self.branches},

            'ThermalLimit': {branch.ID: round(branch.ContRating, 8)
                             for branch in self.branches},
            'Impedence': {branch.ID: round(branch.X, 8)
                          for branch in self.branches},

            'MustRun': [gen.ID for gen in self.generators
                        if self.must_gen_run(gen)]
            }

        tgen_bus_map = {bus.Name: list() for bus in self.buses}
        rgen_bus_map = {bus.Name: list() for bus in self.buses}
        tgens = list()
        rgens = list()

        first_states = pd.read_csv(self.init_state_file,
                                   index_col='GEN').to_dict(orient='index')
        if init_state_file:
            init_states = pd.read_csv(init_state_file,
                                      index_col='GEN').to_dict(orient='index')
        else:
            init_states = first_states

        for gen in self.generators:
            if gen.ID in first_states:
                if gen.Fuel in self.thermal_gen_types:
                    tgens += [gen]
                    tgen_bus_map[bus_name_mapping[gen.Bus]] += [gen.ID]

                if gen.Fuel in self.renew_gen_types and gen:
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
            gen.ID: (round(gen.RampRate * mins_per_time_period, 2)
                     if gen.RampRate > 0.
                     else template['MinimumPowerOutput'][gen.ID])
            for gen in tgens
            }

        template['NominalRampDownLimit'] = {
            gen.ID: (round(gen.RampRate * mins_per_time_period, 2)
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
            gen.ID: ([gen.StartCostCold * gen.FuelPrice]
                     if (gen.StartTimeCold <= gen.MinDownTime
                         or (gen.StartTimeCold == gen.StartTimeWarm
                             == gen.StartTimeHot))

                     else [gen.StartCostWarm * gen.FuelPrice,
                           gen.StartCostCold * gen.FuelPrice]
            if gen.StartTimeWarm <= gen.MinDownTime

            else [gen.StartCostHot * gen.FuelPrice,
                  gen.StartCostWarm * gen.FuelPrice,
                  gen.StartCostCold * gen.FuelPrice])

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

        template['UnitOnT0State'] = {
            gen.ID: init_states[gen.ID]['UnitOnT0State']
            for gen in tgens if gen.ID in init_states
            }

        template['PowerGeneratedT0'] = {
            gen.ID: (0. if init_states[gen.ID]['UnitOnT0State'] < 0
                     else round(init_states[gen.ID]['PowerGeneratedT0'], 2))
            for gen in tgens if gen.ID in init_states
            }

        self.template = template

    @property
    def in_path(self) -> Path:
        """The folder containing this grid's data and metadata."""
        return Path(_ROOT, 'grids', self.grid_lbl)

    @property
    def data_path(self) -> Path:
        """The folder containing this grid's datasets."""
        return Path(self.in_path, self._data_dir)

    @property
    @abstractmethod
    def init_state_file(self) -> Path:
        """Location of initial conditions for the grid's thermal generators."""
        pass

    @property
    @abstractmethod
    def utc_offset(self) -> pd.Timedelta:
        """Difference between the grid's local time zone and UTC in hours."""
        pass

    @property
    @abstractmethod
    def timeseries_cohorts(self) -> set[str]:
        pass

    @property
    def no_scenario_renews(self) -> set[str]:
        return {'Hydro'}

    @staticmethod
    @abstractmethod
    def get_dispatch_types(renew_types):
        pass

    @abstractmethod
    def get_generator_type(self, gen: str) -> str:
        """The asset class of this generator, e.g. WIND, PV."""
        pass

    @abstractmethod
    def get_generator_zone(self, gen):
        """The geographical area within the grid this generator is in."""
        pass

    @property
    def renews_list(self) -> list[str]:
        """Gets the solar and wind assets in this grid."""

        return sorted({
            gen for gen, fuel in self.template[
                'NondispatchableGeneratorType'].items() if fuel in {'S', 'W'}
            })

    @property
    def renews_info(self) -> pd.DataFrame:
        """Gets the metadata on the solar and wind assets in this grid."""

        gen_df = self.gen_df.set_index('GEN UID')
        gen_df = gen_df.loc[gen_df.index.isin(self.renews_list)]
        gen_df['Type'] = [self.get_generator_type(gen) for gen in gen_df.index]
        gen_df['Zone'] = [self.get_generator_zone(gen) for gen in gen_df.index]

        return gen_df

    def map_wind_generators(self, asset_df: pd.DataFrame) -> pd.DataFrame:
        """Transform data for NREL wind farms to those used in the grid."""
        return asset_df

    def map_solar_generators(self, asset_df: pd.DataFrame) -> pd.DataFrame:
        """Transform data for NREL solar plants to those used in the grid."""
        return asset_df

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
                          forecasts_file: str | Path,
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

        return self.subset_dates(use_df, start_date, end_date)

    @staticmethod
    @abstractmethod
    def process_actuals(actuals_file: str | Path,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Parse realized load/generation values read from an input file."""
        pass

    def get_forecasts(self,
                      asset_type: str, start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Find and parse forecasted values for a particular asset type."""

        data_dir = Path(self.data_path, 'timeseries_data_files', asset_type)
        fcst_file = tuple(data_dir.glob("DAY_AHEAD_*.csv"))
        assert len(fcst_file) == 1

        return self.process_forecasts(fcst_file[0], start_date, end_date)

    def get_actuals(self,
                    asset_type: str, start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Find and parse realized values for a particular asset type."""

        data_dir = Path(self.data_path, 'timeseries_data_files', asset_type)
        actl_file = tuple(data_dir.glob("REAL_TIME_*.csv"))
        assert len(actl_file) == 1

        return self.process_actuals(actl_file[0], start_date, end_date)

    def load_by_bus(self,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    load_fcsts: Optional[pd.DataFrame] = None,
                    load_actls: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Parse forecast and actual load demands from zone to bus level."""
        if load_fcsts is None:
            load_fcsts = self.get_forecasts('Load', start_date, end_date)

        if load_actls is None:
            load_actls = self.get_actuals(
                'Load', start_date, end_date).resample('H').mean()

        site_dfs = dict()
        for zone, zone_df in self.bus_df.groupby('Area'):
            area_total_load = zone_df['MW Load'].sum()

            for bus_name, bus_load in zip(zone_df['Bus Name'],
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
    def parse_generator(cls, gen_info: pd.Series) -> Generator:
        """Read in relevant generator info fields from a single dataset row."""
        pass

    @classmethod
    def parse_bus(cls, bus_info: pd.Series) -> Bus:
        """Read in relevant bus info fields from a single dataset row."""

        return cls.Bus(
            int(bus_info["Bus ID"]), bus_info['Bus Name'],
            bus_info["BaseKV"], bus_info["Bus Type"], bus_info["MW Load"],
            bus_info["Area"], int(bus_info["Sub Area"]), bus_info["Zone"],
            bus_info["lat"], bus_info["lng"]
            )

    @classmethod
    def parse_branch(cls, branch_info: pd.Series) -> Branch:
        """Read in relevant branch info fields from a single dataset row."""

        return cls.Branch(
            branch_info["UID"], branch_info["From Bus"], branch_info["To Bus"],
            float(branch_info["R"]), float(branch_info["X"]) / 100.0,
            float(branch_info["B"]), float(branch_info["Cont Rating"])
            )

    @classmethod
    def must_gen_run(cls, gen: Generator) -> bool:
        """Must this generator be turned on at all times in the grid?"""
        return gen.Fuel == 'Nuclear'

    def create_timeseries(
            self,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get asset values using the historical sets of forecasts, actuals."""
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

    def load_scenarios(
            self,
            scen_dir: str | Path, dates: Iterable[datetime],
            scenarios: Iterable[int],
            asset_types: tuple[str] = ('Load', 'Wind', 'Solar')
            ) -> dict[str, pd.DataFrame]:
        """Parse a set of scenarios saved to file."""
        scens = {asset_type: dict() for asset_type in asset_types}

        for scen_day in dates:
            out_file = Path(scen_dir, "scens_{}.p.gz".format(scen_day.date()))

            with bz2.BZ2File(out_file, 'r') as f:
                day_scens = pickle.load(f)

            for asset_type in asset_types:
                scens[asset_type][scen_day] = day_scens[
                    asset_type].iloc[scenarios]

        scen_dfs = {asset_type: pd.concat(scens[asset_type].values(),
                                          axis=1).stack()
                    for asset_type in asset_types}

        for asset_type in asset_types:
            scen_dfs[asset_type].index = pd.MultiIndex.from_tuples(
                [(scenario, t + self.utc_offset)
                 for scenario, t in scen_dfs[asset_type].index],
                names=('Scenario', 'Time')
                )

            if asset_type == 'Wind':
                scen_dfs[asset_type] = self.map_wind_generators(
                    scen_dfs[asset_type])
            if asset_type == 'Solar':
                scen_dfs[asset_type] = self.map_solar_generators(
                    scen_dfs[asset_type])

        return scen_dfs

    def create_scenario_timeseries(
            self,
            scen_dfs: Mapping[str, pd.DataFrame],
            start_date: datetime, end_date: datetime, scenario: int
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get asset values corresponding to a scenario for actuals."""

        use_scens = {asset_type: scen_df.loc[scenario]
                     for asset_type, scen_df in scen_dfs.items()}

        use_scens = {asset_type: scen_df.loc[scen_df.index >= start_date]
                     for asset_type, scen_df in use_scens.items()}
        use_scens = {asset_type: scen_df.loc[scen_df.index <= end_date]
                     for asset_type, scen_df in use_scens.items()}

        # consolidate scenario data for the renewable generators as the actuals
        gen_scens = pd.concat([use_scens['Wind'], use_scens['Solar']], axis=1)
        gen_scens.columns = pd.MultiIndex.from_tuples(
            [('actl', asset_name) for asset_name in gen_scens.columns])

        # consolidate forecasted output values for the renewable generators
        gen_df = pd.concat([self.get_forecasts(asset_type,
                                               start_date, end_date)
                            for asset_type in self.timeseries_cohorts],
                           axis=1)

        # create one matrix with both forecasted and actual renewable outputs
        gen_df.columns = pd.MultiIndex.from_tuples(
            [('fcst', asset_name) for asset_name in gen_df.columns])
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

        demand_df = self.load_by_bus(start_date, end_date,
                                     load_actls=use_scens['Load'])

        return gen_df.sort_index(axis=1), demand_df


class RtsLoader(GridLoader):
    """The RTS-GMLC grid which was created for testing purposes.

    See github.com/GridMod/RTS-GMLC for the raw data files used for this grid.

    """
    grid_lbl = "RTS-GMLC"
    plot_lbl = "RTS-GMLC"
    _data_dir = "RTS_Data"

    @property
    def init_state_file(self) -> Path:
        return Path(_ROOT, "grids", "initial-state",
                    self.grid_lbl, "on_time_7.12.csv")

    @property
    def utc_offset(self) -> pd.Timedelta:
        return -pd.Timedelta(hours=8)

    @property
    def timeseries_cohorts(self) -> set[str]:
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

    def get_generator_type(self, gen: str) -> str:
        return self.gen_df['Unit Group'][self.gen_df['GEN UID'] == gen].iloc[0]

    def get_generator_zone(self, gen):
        return gen[0]

    @classmethod
    def parse_generator(cls, gen_info: pd.Series) -> GridLoader.Generator:
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
            # and down times
            int(math.ceil(gen_info["Min Down Time Hr"])),
            int(math.ceil(gen_info["Min Up Time Hr"])),

            gen_info["Ramp Rate MW/Min"],
            int(gen_info["Start Time Cold Hr"]),
            int(gen_info["Start Time Warm Hr"]),
            int(gen_info["Start Time Hot Hr"]),

            float(gen_info["Start Heat Cold MBTU"]),
            float(gen_info["Start Heat Warm MBTU"]),
            float(gen_info["Start Heat Hot MBTU"]),

            float(gen_info["Fuel Price $/MMBTU"]),
            cost_points, cost_vals
            )

    def process_forecasts(self,
                          forecasts_file: str | Path,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        fcst_df = pd.read_csv(forecasts_file)

        df_times = [
            pd.Timestamp(year=year, month=month, day=day, hour=hour, tz='utc')
            for year, month, day, hour in zip(fcst_df.Year, fcst_df.Month,
                                              fcst_df.Day, fcst_df.Period - 1)
            ]

        use_df = fcst_df.drop(columns=['Year', 'Month', 'Day', 'Period'])
        use_df.index = df_times

        return self.subset_dates(use_df, start_date, end_date)

    def process_actuals(self,
                        actuals_file: str | Path,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
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

    grid_lbl = "Texas-7k"
    plot_lbl = "Texas-7k"
    _data_dir = "TX_Data"

    thermal_gen_types = {
        'NUC (Nuclear)': 'N', 'NG (Natural Gas)': 'G',
        'LIG (Lignite Coal)': 'O', 'SUB (Subbituminous Coal)': 'C',
        'WH (Waste Heat)': 'A', 'WDS (Wood/Wood Waste Solids)': 'A',
        'PUR (Purchased Steam)': 'A'
        }

    renew_gen_types = {'WND (Wind)': "W", 'SUN (Solar)': "S",
                       'WAT (Water)': "H"}

    @property
    def init_state_file(self):
        return Path(_ROOT, "grids", "initial-state",
                    self.grid_lbl, "on_time_7.10.csv")

    @property
    def utc_offset(self) -> pd.Timedelta:
        return -pd.Timedelta(hours=6)

    @property
    def timeseries_cohorts(self):
        return {'WIND', 'PV'}

    @staticmethod
    def get_dispatch_types(renew_types):
        return {
            'DispatchRenewables': {gen for gen, gen_type in renew_types.items()
                                   if gen_type != 'H'},

            'NondispatchRenewables': set(),

            'ForecastRenewables': {gen for gen, gen_type in renew_types.items()
                                   if gen_type != 'H'}
            }

    def map_wind_generators(self, asset_df: pd.DataFrame) -> pd.DataFrame:
        wind_maps = pd.read_csv(Path(self.data_path,
                                     "Texas7k_NREL_wind_map.csv"))

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

    def map_solar_generators(self, asset_df: pd.DataFrame) -> pd.DataFrame:
        solar_maps = pd.read_csv(Path(self.data_path,
                                      "Texas7k_NREL_solar_map.csv"))

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
    def parse_generator(cls, gen_info: pd.Series) -> GridLoader.Generator:
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

            float(gen_info["Fuel Price $/MMBTU"]),
            gen_info[break_cols].tolist() + [gen_info['PMax MW']], cost_values
            )

    #TODO: why doesn't T7k have timeseries for the hydro plants in the system?
    @property
    def no_scenario_renews(self):
        return set()

    def get_generator_type(self, gen: str) -> str:
        gen_match = self.gen_df['GEN UID'] == gen

        assert gen_match.sum() == 1, (
            "Unable to find a unique record in the "
            "metadata for generator `{}`!".format(gen)
            )

        gen_type = self.gen_df.Fuel[gen_match].iloc[0].split(
            '(')[1][:-1].upper()

        if gen_type == 'SOLAR':
            gen_type = 'PV'

        return gen_type

    def get_generator_zone(self, gen: str) -> str:
        return self.bus_df.set_index('Bus ID').loc[
            int(self.gen_df.set_index('GEN UID').loc[
                    gen, 'BUS UID'].split('_')[0]), 'Zone']

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

    grid_lbl = "Texas-7k_2030"
    plot_lbl = "Texas-7k(2030)"
    _data_dir = "TX2030_Data"

    def get_forecasts(self, asset_type, start_date=None, end_date=None):
        fcst_file = tuple(Path(self.data_path, 'timeseries_data_files',
                               asset_type).glob("DAY_AHEAD_*.csv"))
        assert len(fcst_file) == 1

        fcst_df = pd.read_csv(fcst_file[0], parse_dates=['Forecast_time'])
        fcst_df.drop('Issue_time', axis=1, inplace=True)

        fcst_df.Forecast_time += self.utc_offset
        fcst_df.set_index('Forecast_time', inplace=True, verify_integrity=True)
        fcst_df = self.subset_dates(fcst_df, start_date, end_date)

        # TODO: should scenarios and T7k(2030) output values be "pre-mapped"?
        if asset_type == 'WIND':
            fcst_df = self.map_wind_generators(fcst_df)
        elif asset_type == 'PV':
            fcst_df = self.map_solar_generators(fcst_df)

        return fcst_df

    def get_actuals(self, asset_type, start_date=None, end_date=None):
        actl_file = tuple(Path(self.data_path, 'timeseries_data_files',
                               asset_type).glob("REAL_TIME_*.csv"))
        assert len(actl_file) == 1

        actl_df = pd.read_csv(actl_file[0], parse_dates=['Time'])
        actl_df.Time += self.utc_offset

        actl_df.set_index('Time', inplace=True, verify_integrity=True)
        actl_df = self.subset_dates(actl_df, start_date, end_date)

        if asset_type == 'WIND':
            actl_df = self.map_wind_generators(actl_df)
        elif asset_type == 'PV':
            actl_df = self.map_solar_generators(actl_df)

        return actl_df

    # TODO: should we standardize the mapping file format between the T7k and
    #       the T7k(2030) grids instead of doing this?
    def map_solar_generators(self, asset_df: pd.DataFrame) -> pd.DataFrame:
        solar_maps = pd.read_csv(Path(self.data_path,
                                      "Texas7k_NREL_solar_map.csv"))

        solar_gens = self.gen_df.loc[self.gen_df.Fuel == 'SUN (Solar)']
        mapped_vals = dict()

        for _, gen_info in solar_gens.iterrows():
            gen_bus = int(gen_info['BUS UID'].split('_')[0])
            map_match = solar_maps['BusNum'] == gen_bus
            assert map_match.sum() == 1

            mapped_vals[gen_info['GEN UID']] = (
                    asset_df[solar_maps[map_match]['Min_site'].iloc[0]]
                    * float(solar_maps[map_match]['dist_factor'].iloc[0])
                    )

        return pd.concat(mapped_vals, axis=1)
