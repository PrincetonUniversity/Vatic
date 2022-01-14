#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

# modified version of the scripts rtsgmlc_to_dat.py and
# process_RTS_GMLC_data.py from
# Prescient/prescient/downloaders/rts_gmlc_prescient

import os
import argparse
from pathlib import Path
import dill as pickle

import pandas as pd
import math
from collections import namedtuple


def process_fcst_df(rts_file: Path,
                    start_date: pd.Timestamp,
                    end_date: pd.Timestamp) -> pd.DataFrame:

    rts_df = pd.read_csv(rts_file)
    df_times = pd.to_datetime({'year': rts_df.Year, 'month': rts_df.Month,
                               'day': rts_df.Day, 'hour': rts_df.Period})

    use_df = rts_df.drop(columns=['Year', 'Month', 'Day', 'Period'])
    use_df.index = df_times

    return use_df.loc[(use_df.index >= start_date)
                      & (use_df.index < (end_date + pd.Timedelta(days=1)))]


def process_actual_df(rts_file: Path,
                      start_date: pd.Timestamp,
                      end_date: pd.Timestamp) -> pd.DataFrame:

    # TODO: are we making an off-by-one-hour error here?
    rts_df = pd.read_csv(rts_file)
    rts_df['Hour'] = (rts_df.Period - 1) // 12
    rts_df['Min'] = (rts_df.Period - 1) % 12 * 5

    df_times = pd.to_datetime({'year': rts_df.Year, 'month': rts_df.Month,
                               'day': rts_df.Day, 'hour': rts_df.Hour,
                               'minute': rts_df.Min})

    use_df = rts_df.drop(
        columns=['Year', 'Month', 'Day', 'Period', 'Hour', 'Min'])
    use_df.index = df_times

    return use_df.loc[(use_df.index >= start_date)
                      & (use_df.index < (end_date + pd.Timedelta(days=1)))]


def create_template(rts_gmlc_dir, copper_sheet=False, reserve_factor=None):

    data_dir = Path(rts_gmlc_dir, 'RTS_Data', 'SourceData')

    Generator = namedtuple('Generator',
                           ['ID', # integer
                            'Bus',
                            'UnitGroup',
                            'UnitType',
                            'Fuel',
                            'MinPower',
                            'MaxPower',
                            'MinDownTime',
                            'MinUpTime',
                            'RampRate',         # units are MW/minute
                            'StartTimeCold',    # units are hours
                            'StartTimeWarm',    # units are hours
                            'StartTimeHot',     # units are hours
                            'StartCostCold',    # units are MBTU
                            'StartCostWarm',    # units are MBTU
                            'StartCostHot',     # units are MBTU
                            'NonFuelStartCost', # units are $
                            'FuelPrice',        # units are $ / MMBTU
                            'OutputPct0',
                            'OutputPct1',
                            'OutputPct2',
                            'OutputPct3',
                            'HeatRateAvg0',
                            'HeatRateIncr1',
                            'HeatRateIncr2',
                            'HeatRateIncr3'],
                           )

    Bus = namedtuple('Bus',
                     ['ID', # integer
                      'Name',
                      'BaseKV',
                      'Type',
                      'MWLoad',
                      'Area',
                      'SubArea',
                      'Zone',
                      'Lat',
                      'Long'],
                     )

    Branch = namedtuple('Branch',
                        ['ID',
                         'FromBus',
                         'ToBus',
                         'R',
                         'X', # csv file is in PU, multiple by 100 to make consistent with MW
                         'B',
                         'ContRating'],
                       )


    gen_dict = {} # keys are ID
    bus_dict = {} # keys are ID
    branch_dict = {} # keys are ID
    timeseries_pointer_dict = {} # keys are (ID, simulation-type) pairs

    generator_df = pd.read_csv(Path(data_dir, "gen.csv"))
    bus_df = pd.read_csv(Path(data_dir, "bus.csv"))
    branch_df = pd.read_csv(Path(data_dir, "branch.csv"))

    for generator_index in generator_df.index.tolist():
        this_generator_dict = generator_df.loc[generator_index].to_dict()
        new_generator = Generator(this_generator_dict["GEN UID"],
                                  int(this_generator_dict["Bus ID"]),
                                  this_generator_dict["Unit Group"],
                                  this_generator_dict["Unit Type"],
                                  this_generator_dict["Fuel"],
                                  float(this_generator_dict["PMin MW"]),
                                  float(this_generator_dict["PMax MW"]),
                                  # per Brendan, PLEXOS takes the ceiling at hourly resolution for up and down times.
                                  int(math.ceil(this_generator_dict["Min Down Time Hr"])),
                                  int(math.ceil(this_generator_dict["Min Up Time Hr"])),
                                  this_generator_dict["Ramp Rate MW/Min"],
                                  int(this_generator_dict["Start Time Cold Hr"]),
                                  int(this_generator_dict["Start Time Warm Hr"]),
                                  int(this_generator_dict["Start Time Hot Hr"]),
                                  float(this_generator_dict["Start Heat Cold MBTU"]),
                                  float(this_generator_dict["Start Heat Warm MBTU"]),
                                  float(this_generator_dict["Start Heat Hot MBTU"]),
                                  float(this_generator_dict["Non Fuel Start Cost $"]),
                                  float(this_generator_dict["Fuel Price $/MMBTU"]),
                                  float(this_generator_dict["Output_pct_0"]),
                                  float(this_generator_dict["Output_pct_1"]),
                                  float(this_generator_dict["Output_pct_2"]),
                                  float(this_generator_dict["Output_pct_3"]),
                                  float(this_generator_dict["HR_avg_0"]),
                                  float(this_generator_dict["HR_incr_1"]),
                                  float(this_generator_dict["HR_incr_2"]),
                                  float(this_generator_dict["HR_incr_3"]))

        gen_dict[new_generator.ID] = new_generator

    bus_id_to_name_dict = {}

    for bus_index in bus_df.index.tolist():
        this_bus_dict = bus_df.loc[bus_index].to_dict()
        new_bus = Bus(int(this_bus_dict["Bus ID"]),
                      this_bus_dict["Bus Name"],
                      this_bus_dict["BaseKV"],
                      this_bus_dict["Bus Type"],
                      float(this_bus_dict["MW Load"]),
                      int(this_bus_dict["Area"]),
                      int(this_bus_dict["Sub Area"]),
                      this_bus_dict["Zone"],
                      this_bus_dict["lat"],
                      this_bus_dict["lng"])
        bus_dict[new_bus.Name] = new_bus
        bus_id_to_name_dict[new_bus.ID] = new_bus.Name

    for branch_index in branch_df.index.tolist():
        this_branch_dict = branch_df.loc[branch_index].to_dict()
        new_branch = Branch(this_branch_dict["UID"],
                            this_branch_dict["From Bus"],
                            this_branch_dict["To Bus"],
                            float(this_branch_dict["R"]),
                            float(this_branch_dict["X"]) / 100.0, # nix per unit
                            float(this_branch_dict["B"]),
                            float(this_branch_dict["Cont Rating"]))
        branch_dict[new_branch.ID] = new_branch

    unit_on_time_df = pd.read_csv(Path(
        data_dir, '..', 'FormattedData', 'PLEXOS', 'PLEXOS_Solution',
        'DAY_AHEAD Solution Files', 'noTX', 'on_time_7.12.csv'
        ))

    unit_on_time_df_as_dict = unit_on_time_df.to_dict(orient="split")
    unit_on_t0_state_dict = {}

    for i in range(0,len(unit_on_time_df_as_dict["columns"])):
        gen_id = unit_on_time_df_as_dict["columns"][i]
        unit_on_t0_state_dict[gen_id] = int(
            unit_on_time_df_as_dict["data"][0][i])

    #print("Writing Prescient template file")

    mins_per_time_period = 60
    ## we'll bring the ramping down by this factor
    ramp_scaling_factor = 1.

    template = {
        'NumTimePeriods': 48, 'TimePeriodLength': 1,
        'StageSet': ['Stage_1', 'Stage_2'],
        'CommitmentTimeInStage': {'Stage_1': list(range(1, 49)),
                                  'Stage_2': list()},
        'GenerationTimeInStage': {'Stage_1': list(),
                                  'Stage_2': list(range(1, 49))},
        'CopperSheet': copper_sheet,
        }

    if reserve_factor is not None:
        template['ReserveFactor'] = round(reserve_factor, 8)

    if not copper_sheet:
        template['Buses'] = list(bus_dict.keys())
        template['TransmissionLines'] = list(branch_dict.keys())

        template['BusFrom'] = {
            branch_id: bus_id_to_name_dict[branch_spec.FromBus]
            for branch_id, branch_spec in branch_dict.items()
            }

        template['BusTo'] = {
            branch_id: bus_id_to_name_dict[branch_spec.ToBus]
            for branch_id, branch_spec in branch_dict.items()
            }

        template['ThermalLimit'] = {
            branch_id: round(branch_spec.ContRating, 8)
            for branch_id, branch_spec in branch_dict.items()
            }

        template['Impedence'] = {
            branch_id: round(branch_spec.X, 8)
            for branch_id, branch_spec in branch_dict.items()
            }

    template['ThermalGenerators'] = [
        gen_id for gen_id, gen_spec in gen_dict.items()
        if gen_spec.Fuel in {'Oil', 'Coal', 'NG', 'Nuclear'}
        ]

    if copper_sheet:
        template['ThermalGeneratorsAtBus'] = {
            'CopperSheet': list(template['ThermalGenerators'])}

    else:
        gen_bus_map = {bus_spec.ID: list() for bus_spec in bus_dict.values()}

        for gen_id in template['ThermalGenerators']:
            gen_bus_map[gen_dict[gen_id].Bus] += [gen_id]

        template['ThermalGeneratorsAtBus'] = {
            bus_id: gen_bus_map[bus_spec.ID]
            for bus_id, bus_spec in bus_dict.items()
            }

    template['NondispatchableGenerators'] = [
        gen_id for gen_id, gen_spec in gen_dict.items()
        if gen_spec.Fuel in {'Solar', 'Wind', 'Hydro'}
        ]

    if copper_sheet:
        template['NondispatchableGeneratorsAtBus'] = {
            'CopperSheet': list(template['NondispatchableGenerators'])}

    else:
        gen_bus_map = {bus_spec.ID: list() for bus_spec in bus_dict.values()}

        for gen_id in template['NondispatchableGenerators']:
            gen_bus_map[gen_dict[gen_id].Bus] += [gen_id]

        template['NondispatchableGeneratorsAtBus'] = {
            bus_id: gen_bus_map[bus_spec.ID]
            for bus_id, bus_spec in bus_dict.items()
            }

    template['MustRun'] = [gen_id for gen_id, gen_spec in gen_dict.items()
                           if gen_spec.Fuel == 'Nuclear']

    tgen_types = {'Nuclear': "N", 'NG': "G", 'Oil': "O", 'Coal': "C"}
    template['ThermalGeneratorType'] = {
        gen_id: tgen_types[gen_spec.Fuel]
        for gen_id, gen_spec in gen_dict.items()
        if gen_id in template['ThermalGenerators']
        }

    rgen_types = {'Wind': "W", 'Solar': "S", 'Hydro': "H"}
    template['NondispatchableGeneratorType'] = {
        gen_id: rgen_types[gen_spec.Fuel]
        for gen_id, gen_spec in gen_dict.items()
        if gen_id in template['NondispatchableGenerators']
        }

    template['MinimumPowerOutput'] = {gen_id: round(gen_spec.MinPower, 2)
                                      for gen_id, gen_spec in gen_dict.items()
                                      if gen_spec.Fuel in {'Oil', 'Coal',
                                                           'NG', 'Nuclear'}}

    template['MaximumPowerOutput'] = {gen_id: round(gen_spec.MaxPower, 2)
                                      for gen_id, gen_spec in gen_dict.items()
                                      if gen_spec.Fuel in {'Oil', 'Coal',
                                                           'NG', 'Nuclear'}}

    template['MinimumUpTime'] = {gen_id: round(gen_spec.MinUpTime, 2)
                                 for gen_id, gen_spec in gen_dict.items()
                                 if gen_spec.Fuel in {'Oil', 'Coal',
                                                      'NG', 'Nuclear'}}

    template['MinimumDownTime'] = {gen_id: round(gen_spec.MinDownTime, 2)
                                   for gen_id, gen_spec in gen_dict.items()
                                   if gen_spec.Fuel in {'Oil', 'Coal',
                                                        'NG', 'Nuclear'}}

    template['NominalRampUpLimit'] = {
        gen_id: round((gen_spec.RampRate
                       * float(mins_per_time_period) / ramp_scaling_factor), 2)
        for gen_id, gen_spec in gen_dict.items()
        if gen_spec.Fuel in {'Oil', 'Coal', 'NG', 'Nuclear'}
        }

    template['NominalRampDownLimit'] = {
        gen_id: round((gen_spec.RampRate
                       * float(mins_per_time_period) / ramp_scaling_factor), 2)
        for gen_id, gen_spec in gen_dict.items()
        if gen_spec.Fuel in {'Oil', 'Coal', 'NG', 'Nuclear'}
        }

    template['StartupRampLimit'] = {gen_id: round(gen_spec.MinPower, 2)
                                    for gen_id, gen_spec in gen_dict.items()
                                    if gen_spec.Fuel in {'Oil', 'Coal',
                                                         'NG', 'Nuclear'}}

    template['ShutdownRampLimit'] = {gen_id: round(gen_spec.MinPower, 2)
                                     for gen_id, gen_spec in gen_dict.items()
                                     if gen_spec.Fuel in {'Oil', 'Coal',
                                                          'NG', 'Nuclear'}}

    template['StartupLags'] = {
        gen_id: ([gen_spec.MinDownTime]
                 if (gen_spec.StartTimeCold <= gen_spec.MinDownTime
                     or (gen_spec.StartTimeCold == gen_spec.StartTimeWarm
                         == gen_spec.StartTimeHot))

                 else [gen_spec.MinDownTime, gen_spec.StartTimeCold]
                 if gen_spec.StartTimeWarm <= gen_spec.MinDownTime

                 else [gen_spec.MinDownTime, gen_spec.StartTimeWarm,
                       gen_spec.StartTimeCold])

        for gen_id, gen_spec in gen_dict.items()
        if gen_spec.Fuel in {'Oil', 'Coal', 'NG', 'Nuclear'}
        }

    template['StartupCosts'] = {
        gen_id: ([gen_spec.StartCostCold
                  * gen_spec.FuelPrice + gen_spec.NonFuelStartCost]
                 if (gen_spec.StartTimeCold <= gen_spec.MinDownTime
                     or (gen_spec.StartTimeCold == gen_spec.StartTimeWarm
                         == gen_spec.StartTimeHot))

                 else [gen_spec.StartCostWarm
                       * gen_spec.FuelPrice + gen_spec.NonFuelStartCost,
                       gen_spec.StartCostCold
                       * gen_spec.FuelPrice + gen_spec.NonFuelStartCost]
                 if gen_spec.StartTimeWarm <= gen_spec.MinDownTime

                 else [gen_spec.StartCostHot
                       * gen_spec.FuelPrice + gen_spec.NonFuelStartCost,
                       gen_spec.StartCostWarm
                       * gen_spec.FuelPrice + gen_spec.NonFuelStartCost,
                       gen_spec.StartCostCold
                       * gen_spec.FuelPrice + gen_spec.NonFuelStartCost])

        for gen_id, gen_spec in gen_dict.items()
        if gen_spec.Fuel in {'Oil', 'Coal', 'NG', 'Nuclear'}
        }

    for gen_id in template['StartupCosts']:
        template['StartupCosts'][gen_id] = [
            round(cost, 2) for cost in template['StartupCosts'][gen_id]]

    template['CostPiecewisePoints'] = dict()
    template['CostPiecewiseValues'] = dict()

    for gen_id, gen_spec in gen_dict.items():
        if gen_spec.Fuel in {'Oil', 'Coal', 'NG', 'Nuclear'}:
            # round the power points to the nearest 10kW
            # IMPT: These quantities are MW
            x0 = round(gen_spec.OutputPct0 * gen_spec.MaxPower, 1)
            x1 = round(gen_spec.OutputPct1 * gen_spec.MaxPower, 1)
            x2 = round(gen_spec.OutputPct2 * gen_spec.MaxPower, 1)
            x3 = round(gen_spec.OutputPct3 * gen_spec.MaxPower, 1)

            # NOTES:
            # 1) Fuel price is in $/MMBTU
            # 2) Heat Rate quantities are in BTU/KWH
            # 3) 1+2 => need to convert both from BTU->MMBTU and from KWH->MWH
            y0 = (gen_spec.FuelPrice
                  * ((gen_spec.HeatRateAvg0 * 1000.0 / 1000000.0) * x0))

            y1 = (gen_spec.FuelPrice
                  * (((x1 - x0)
                      * (gen_spec.HeatRateIncr1 * 1000.0 / 1000000.0))) + y0)

            y2 = (gen_spec.FuelPrice
                  * (((x2 - x1)
                      * (gen_spec.HeatRateIncr2 * 1000.0 / 1000000.0))) + y1)

            y3 = (gen_spec.FuelPrice
                  * (((x3 - x2)
                      * (gen_spec.HeatRateIncr3 * 1000.0 / 1000000.0))) + y2)

            y0 = round(y0, 2)
            y1 = round(y1, 2)
            y2 = round(y2, 2)
            y3 = round(y3, 2)

            ## for the nuclear unit
            if y0 == y3:
                # PRESCIENT currently doesn't gracefully handle generators
                # with zero marginal cost
                template['CostPiecewisePoints'][gen_id] = [x0, x3]
                template['CostPiecewiseValues'][gen_id] = [y0, y3 + 0.01]

            else:
                template['CostPiecewisePoints'][gen_id] = [x0, x1, x2, x3]
                template['CostPiecewiseValues'][gen_id] = [y0, y1, y2, y3]

    template['UnitOnT0State'] = {
        gen_id: t0_val for gen_id, t0_val in unit_on_t0_state_dict.items()
        if gen_dict[gen_id].Fuel not in {'Sync_Cond', 'Hydro', 'Wind', 'Solar'}
        }

    template['PowerGeneratedT0'] = {
        gen_id: 0. if t0_val < 0 else round(gen_dict[gen_id].MinPower, 2)
        for gen_id, t0_val in unit_on_t0_state_dict.items()
        if gen_dict[gen_id].Fuel not in {'Sync_Cond', 'Hydro', 'Wind', 'Solar'}
        }

    return template


def gmlc_to_vatic(asset_type, tmsrs_dir, start_date, end_date,
                  forecast_error=True, aggregate=False):
    """
    This takes the RTS-GMLC time series data and
    puts it into the format required for prescient
    :param asset_type: options are WIND, PV, RTPV, or Hydro
    :param aggregate: Aggregate all sites within the file or not?
    :return: writes csv files of forecast/actual in prescient format
    """

    # Find the files you want, forecast and actual:
    data_dir = Path(tmsrs_dir, asset_type)

    fcst_file = tuple(data_dir.glob("DAY_AHEAD*"))
    assert len(fcst_file) == 1
    fcst_file = fcst_file[0]

    actual_file = tuple(data_dir.glob("REAL_TIME*"))
    assert len(actual_file) == 1
    actual_file = actual_file[0]

    # Read in forecast data, identify site names, collect datetime
    fcst_data = process_fcst_df(fcst_file, start_date, end_date)

    # Read in actual data, create 5-min datetime, re-sample to hourly
    if forecast_error:
        actual_data = process_actual_df(
            actual_file, start_date, end_date).resample('H').mean()

    else:
        actual_data = fcst_data.copy()

    asset_data = pd.concat([fcst_data, actual_data],
                           axis=1, keys=['fcst', 'actual'])

    # If you want to combine all sites (or regions, whatever),
    # write one file for all data:
    if aggregate:
        asset_data = asset_data.groupby(axis=1, level=[0]).sum()

    return asset_data


def load_by_bus(tmsrs_dir, source_dir, start_date, end_date,
                forecast_error=True):
    """
    This gets bus-level time series of load
    data based on the load participation factors
    calculated from the ratio of bus loads in the
    bus.csv file.
    This takes the RTS-GMLC time series data and
    puts it into the format required for prescient,
    aggregated by zone (1, 2, or 3 for RTS-GMLC)
    :param source: options are WIND, PV, RTPV, or Hydro
    :param aggregate: Aggregate all sites within the file or not?
    :return: writes csv files of forecast/actual in prescient format
    """

    load_data = gmlc_to_vatic('Load', tmsrs_dir, start_date, end_date,
                              forecast_error, aggregate=False)
    bus_data = pd.read_csv(Path(source_dir, "bus.csv"))
    site_dfs = list()

    for zone, zone_df in bus_data.groupby('Area'):
        area_total_load = zone_df['MW Load'].sum()

        for bus_name, bus_load in zip(zone_df['Bus Name'], zone_df['MW Load']):
            load_factor = bus_load / area_total_load
            site_df = load_data.loc[:, (slice(None), str(zone))] * load_factor

            site_df.columns = pd.MultiIndex.from_tuples(
                [(x, bus_name) for x, _ in site_df.columns])
            site_dfs += [site_df]

    return pd.concat(site_dfs, axis=1).sort_index(axis=1)


def create_timeseries(rts_dir, start_date, end_date,
                      template, forecast_error, aggregate=False):

    source_dir = Path(rts_dir, "RTS_Data", "SourceData")
    tmsrs_dir = Path(rts_dir, "RTS_Data", "timeseries_data_files")

    gen_df = pd.concat([
        gmlc_to_vatic(asset_type, tmsrs_dir, start_date, end_date,
                      forecast_error, aggregate)
        for asset_type in ['WIND', 'PV', 'RTPV', 'Hydro']
        ], axis=1).sort_index(axis=1)

    demand_df = load_by_bus(tmsrs_dir, source_dir, start_date, end_date,
                            forecast_error=forecast_error)

    return gen_df, demand_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("rts_gmlc_dir",
                        help="location of the RTS-GMLC data", type=str)
    parser.add_argument("out_dir",
                        help="specify the file name to write", type=str)

    parser.add_argument("start_date")
    parser.add_argument("end_date")

    parser.add_argument("--copper-sheet",
                        action='store_true', dest="copper_sheet",
                        help="don't create network")
    parser.add_argument("--no-forecast-error",
                        action='store_false', dest="fcst_err",
                        help="make forecasts equal actuals")

    args = parser.parse_args()
    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    os.makedirs(args.out_dir, exist_ok=True)

    template = create_template(args.rts_gmlc_dir, args.copper_sheet, )
    gen_df, demand_df = create_timeseries(
        args.rts_gmlc_dir, start_date, end_date, template, args.fcst_err)

    with open(Path(args.out_dir, "grid-template.p"), 'wb') as f:
        pickle.dump(template, f, protocol=-1)
    with open(Path(args.out_dir, "gen-data.p"), 'wb') as f:
        pickle.dump(gen_df, f, protocol=-1)
    with open(Path(args.out_dir, "load-data.p"), 'wb') as f:
        pickle.dump(demand_df, f, protocol=-1)


if __name__ == '__main__':
    main()
