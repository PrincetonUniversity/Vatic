
from datetime import datetime, date, timedelta
from copy import deepcopy
from pathlib import Path
import dill as pickle
import pandas as pd
import math
from typing import Optional

from prescient.engine.modeling_engine import ForecastErrorMethod
from prescient.simulator.options import Options
from prescient.engine.egret.reporting import (
    report_initial_conditions_for_deterministic_ruc,
    report_demand_for_deterministic_ruc
    )

from .model_data import VaticModelData
from .time_manager import VaticTime
from .simulation_state import VaticSimulationState


class ProviderError(Exception):
    pass


class PickleProvider:
    """Loading data from input datasets and generating UC and ED models.

    Args
    ----
        data_dir    The path to where the input datasets are stored.
        options     Miscelleanous properties of the simulation engine, some of
                    which also apply to data provision.

    """

    def __init__(self,
                 data_dir: str, start_date: datetime, num_days: int,
                 options: Options) -> None:
        self._time_period_mins = 60
        self._load_mismatch_cost = 1e4
        self._reserve_mismatch_cost = 1e3
        self._reserve_factor = options.reserve_factor

        self._ruc_execution_hour = options.ruc_execution_hour
        self._ruc_every_hours = options.ruc_every_hours
        self._ruc_delay = -(self._ruc_execution_hour
                            % -options.ruc_every_hours)

        self._run_ruc_with_next_day_data = options.run_ruc_with_next_day_data
        self._ruc_horizon = options.ruc_horizon
        self._ruc_prescience_hour = options.ruc_prescience_hour
        self._output_ruc_initial_conditions \
            = options.output_ruc_initial_conditions

        self._no_startup_shutdown_curves = options.no_startup_shutdown_curves
        self._enforce_sced_shutdown_ramprate \
            = options.enforce_sced_shutdown_ramprate

        # load input datasets, starting with static grid data (e.g. network
        # topology, thermal generator outputs)
        with open(Path(data_dir, "grid-template.p"), 'rb') as f:
            self.template: dict = pickle.load(f)

        # load renewable generator forecasted and actual outputs
        with open(Path(data_dir, "gen-data.p"), 'rb') as f:
            self.gen_data: pd.DataFrame = pickle.load(f).round(8)

        # load bus forecasted and actual outputs
        with open(Path(data_dir, "load-data.p"), 'rb') as f:
            self.load_data: pd.DataFrame = pickle.load(f)

        self._first_day = pd.Timestamp(start_date).date()
        self._final_day = (start_date + pd.Timedelta(days=num_days)).date()

        if not ((self.gen_data.index.date >= self._first_day)
                & (self.gen_data.index.date <= self._final_day)).all():
            raise ProviderError("The generator data in the input directory "
                                "does not match the given start/end dates!")

        if not ((self.load_data.index.date >= self._first_day)
                & (self.load_data.index.date <= self._final_day)).all():
            raise ProviderError("The bus demand data in the input directory "
                                "does not match the given start/end dates!")

        if not (self.gen_data.index == self.load_data.index).all():
            raise ProviderError("The generator and the bus demand datasets "
                                "have inconsistent time points!")

        data_freq = set(self.gen_data.index.to_series().diff()[1:].values)
        if len(data_freq) != 1:
            raise ProviderError("Inconsistent dataset time frequencies!")

        self.data_freq = int(
            tuple(data_freq)[0].astype('timedelta64[m]').astype('int'))
        self.date_cache = {'actual': dict(), 'fcst': dict()}

        # TODO: better generalize this across different power grids
        self.renewables = self.gen_data.columns.get_level_values(
            level=1).unique().tolist()

        rnwbl_info = {gen: gen.split('_')
                      for gen in self.template['NondispatchableGenerators']
                      if gen in self.renewables}

        self.dispatch_renewables = {gen for gen, gen_info in rnwbl_info.items()
                                    if gen_info[1] not in {'HYDRO', 'RTPV'}}
        self.nondisp_renewables = {gen for gen, gen_info in rnwbl_info.items()
                                   if gen_info[1] in {'HYDRO', 'RTPV'}}
        self.forecast_renewables = {gen for gen, gen_info in rnwbl_info.items()
                                    if gen_info[1] != 'HYDRO'}

        # create an empty template model
        self.init_model = self._get_model_for_date(self._first_day,
                                                   use_actuals=False)
        self.init_model.reset_timeseries()
        self.shutdown_curves = dict()

    def _get_initial_model(self,
                           num_time_steps: int,
                           minutes_per_timestep: int) -> VaticModelData:
        """Creates an empty model ready to be populated with timeseries data.

        This function creates a copy of the empty template model, which only
        contains static grid info (e.g. generator names and other metadata),
        and adds to it empty timeseries lists that will be filled with
        forecasted and actual data.

        Args
        ----
            num_time_steps  The length of the timeseries list for each asset.
            minutes_per_timestep    Metadata on the time between observing
                                    values in each timeseries.

        """
        new_model = deepcopy(self.init_model)
        new_model.set_time_steps(num_time_steps, minutes_per_timestep)

        return new_model

    def _get_initial_state_model(self,
                                 day: date, num_time_steps: int,
                                 minutes_per_timestep: int) -> VaticModelData:
        """Creates a model with initial generator states from input datasets.

        This function makes a copy of the empty template model, and adds to it
        empty timeseries of the requested lengths as well as generators' time
        since on/off (`initial_status`) and their initial power output
        (`initial_p_output`).

        Args
        ----
            day     Which day's initial generator states to use.

            num_time_steps  The length of the timeseries list for each asset.
            minutes_per_timestep    Metadata on the time between observing
                                    values in each timeseries.

        """
        if day < self._first_day:
            day = self._first_day
        elif day > self._final_day:
            day = self._final_day

        # get a model with empty timeseries and the model with initial states
        new_model = self._get_initial_model(num_time_steps,
                                            minutes_per_timestep)
        actuals_model = self._get_model_for_date(day, use_actuals=True)

        # copy initial states into the empty model
        new_model.copy_elements(actuals_model, 'storage',
                                ['initial_state_of_charge'], strict_mode=True)
        new_model.copy_elements(actuals_model, 'generator',
                                ['initial_status', 'initial_p_output'],
                                strict_mode=True, generator_type='thermal')

        return new_model

    def _copy_initial_state_into_model(
            self,
            use_state: VaticSimulationState, num_time_steps: int,
            minutes_per_timestep: int) -> VaticModelData:
        """Creates a model with initial generator states from a simulation.

        This function makes a copy of the empty template model and copies over
        the generator states from the current status of a simulation of a
        power grid's operation.

        Args
        ----
            use_state   The state of a running simulation created by an engine
                        such as :class:`vatic.engines.Simulator`.

            num_time_steps  The length of the timeseries list for each asset.
            minutes_per_timestep    Metadata on the time between observing
                                    values in each timeseries.

        """
        new_model = self._get_initial_model(num_time_steps,
                                            minutes_per_timestep)

        # copy over generator initial states
        for gen, g_data in new_model.elements('generator',
                                              generator_type='thermal'):
            g_data['initial_status'] = use_state.get_initial_generator_state(
                gen)
            g_data['initial_p_output'] = use_state.get_initial_power_generated(
                gen)

        # copy over energy storage initial states
        for store, store_data in new_model.elements('storage'):
            store_data['initial_state_of_charge'] \
                = use_state.get_initial_state_of_charge(store)

        return new_model

    def get_populated_model(
            self,
            use_actuals: bool, start_time: datetime, num_time_periods: int,
            use_state: Optional[VaticSimulationState] = None,
            ) -> VaticModelData:
        """Creates a model with all grid asset data for a given time period.

        Args
        ----
            use_actuals     Whether to use actual output/demand values to
                            populate timeseries or their forecasts.

            start_time          Which time point to pull model data from.
            num_time_periods    How many time steps' data to pull, including
                                the starting time step.

            use_state   The state of a running simulation created by an engine
                        such as :class:`vatic.engines.Simulator`. If given,
                        this will be used to populate initial generator states
                        instead of the initial states in the input datasets.

        """
        start_hour = start_time.hour
        start_day = start_time.date()
        assert (start_time.minute == 0)
        assert (start_time.second == 0)
        step_delta = timedelta(minutes=self.data_freq)

        # populate the initial generator outputs and times since on/off, using
        # the given simulation state or the input datasets as requested
        if use_state is None or use_state.timestep_count == 0:
            new_model = self._get_initial_state_model(
                day=start_time.date(), num_time_steps=num_time_periods,
                minutes_per_timestep=self.data_freq,
                )

        else:
            new_model = self._copy_initial_state_into_model(
                use_state, num_time_periods, self.data_freq)

        # get the data for this date from the input datasets
        day_model = self._get_model_for_date(start_day, use_actuals)

        # advance through the given number of time steps
        for step_index in range(num_time_periods):
            step_time = start_time + step_delta * step_index
            day = step_time.date()

            # if we have advanced beyond the last day, just repeat the
            # final day's values
            if day > self._final_day:
                day = self._final_day

            # if we cross midnight and we didn't start at midnight, we start
            # pulling data from the next day
            if day != start_day and start_hour != 0:
                day_model = self._get_model_for_date(day, use_actuals)
                src_step_index = step_index - 24

            # if we did start at midnight, pull tomorrow's data from today's
            # input datasets
            else:
                src_step_index = step_index

            # copy over timeseries data for the current timestep
            new_model.copy_forecastables(day_model, step_index, src_step_index)

            # set aside a proportion of the total demand as the model's reserve
            # requirement (if any) at each time point
            new_model.honor_reserve_factor(self._reserve_factor, step_index)

        return new_model

    def create_deterministic_ruc(
            self,
            time_step: VaticTime,
            current_state: Optional[VaticSimulationState] = None
            ) -> VaticModelData:
        """Generates a Reliability Unit Commitment model.

        This a merge of Prescient's EgretEngine.create_deterministic_ruc and
        egret_plugin.create_deterministic_ruc.

        Args
        ----
            time_step   Which time point to pull data from.
            current_state   If given, a simulation state used to get initial
                            states for the generators, which will otherwise be
                            pulled from the input datasets.

        """
        copy_first_day = not self._run_ruc_with_next_day_data
        copy_first_day &= time_step.hour() != 0

        forecast_request_count = 24
        if not copy_first_day:
            forecast_request_count = self._ruc_horizon

        # create a new model using the forecasts for the given time steps
        ruc_model = self.get_populated_model(
            use_actuals=False, start_time=time_step.when,
            num_time_periods=forecast_request_count, use_state=current_state,
            )

        # make some near-term forecasts more accurate if necessary
        if self._ruc_prescience_hour > self._ruc_delay + 1:
            improved_hour_count = self._ruc_prescience_hour - self._ruc_delay
            improved_hour_count -= 1
            future_actuals = current_state.get_future_actuals()

            for forecast, actuals in zip(ruc_model.get_forecastables(),
                                         future_actuals):
                for t in range(improved_hour_count):
                    forecast_portion = (self._ruc_delay + t)
                    forecast_portion /= self._ruc_prescience_hour
                    actuals_portion = 1 - forecast_portion

                    forecast[t] = forecast_portion * forecast[t]
                    forecast[t] += actuals_portion * actuals[t]

        # copy from the first 24 hours to the second 24 hours if necessary
        if copy_first_day:
            for vals in ruc_model.get_forecastables():
                for t in range(24, self._ruc_horizon):
                    vals[t] = vals[t - 24]

        if self._output_ruc_initial_conditions:
            report_initial_conditions_for_deterministic_ruc(ruc_model)
            report_demand_for_deterministic_ruc(ruc_model,
                                                self._ruc_every_hours)

        return ruc_model

    def create_sced_instance(
            self,
            current_state: VaticSimulationState, sced_horizon: int,
            forecast_error_method=ForecastErrorMethod.PRESCIENT
            ) -> VaticModelData:
        """Generates a Security Constrained Economic Dispatch model.

        This a merge of Prescient's EgretEngine.create_sced_instance and
        egret_plugin.create_sced_instance.

        Args
        ----
            current_state   The simulation state of a power grid which will
                            be used as the basis for the data included in this
                            model.

            sced_horizon    How many time steps this SCED will simulate over.
            forecast_error_method   How the forecasts used in this model will
                                    be adjusted to be closer to the actuals.

        """
        assert current_state is not None
        assert current_state.timestep_count >= sced_horizon

        # make a new model, starting with the simulation's initial asset states
        sced_model = self._copy_initial_state_into_model(
            current_state, sced_horizon, self.data_freq)

        # add the forecasted load demands and renewable generator outputs, and
        # adjust them to be closer to the corresponding actual values
        sced_forecastables = sced_model.get_forecastables()
        if forecast_error_method is ForecastErrorMethod.PRESCIENT:
            future_actuals = current_state.get_future_actuals()

            # this error method makes the forecasts equal to the actuals!
            for future, sced_data in zip(future_actuals, sced_forecastables):
                for t in range(sced_horizon):
                    sced_data[t] = future[t]

        else:
            current_actuals = current_state.get_current_actuals()
            forecasts = current_state.get_forecasts()

            # this error method adjusts future forecasts based on how much the
            # current forecast over/underestimated the current actual value
            for current_actual, forecast, sced_data in zip(current_actuals,
                                                           forecasts,
                                                           sced_forecastables):
                sced_data[0] = current_actual

                # find how much the first forecast was off from the actual as
                # a fraction of the forecast
                current_forecast = forecast[0]
                if current_forecast == 0.0:
                    forecast_error_ratio = 0.0
                else:
                    forecast_error_ratio = current_actual / forecast[0]

                # adjust the remaining forecasts based on the initial error
                for t in range(1, sced_horizon):
                    sced_data[t] = forecast[t] * forecast_error_ratio

        # set aside a proportion of the total demand as the model's reserve
        # requirement (if any) at each time point
        for t in range(sced_horizon):
            sced_model.honor_reserve_factor(self._reserve_factor, t)

        # pull generator commitments from the state of the simulation
        for gen, gen_data in sced_model.elements(element_type='generator',
                                                 generator_type='thermal'):
            gen_commits = [current_state.get_generator_commitment(gen, t)
                           for t in range(current_state.timestep_count)]

            gen_data['fixed_commitment'] = {
                'data_type': 'time_series',
                'values': gen_commits[:sced_horizon]
                }

            # look as far into the future as we can for startups if the
            # generator is committed to be off at the end of the model window
            future_status_time_steps = 0
            if gen_commits[sced_horizon - 1] == 0:
                for t in range(sced_horizon, current_state.timestep_count):
                    if gen_commits[t] == 1:
                        future_status_time_steps = (t - sced_horizon + 1)
                        break

            # same but for future shutdowns if the generator is committed to be
            # on at the end of the model's time steps
            elif gen_commits[sced_horizon - 1] == 1:
                for t in range(sced_horizon, current_state.timestep_count):
                    if gen_commits[t] == 0:
                        future_status_time_steps = -(t - sced_horizon + 1)
                        break

            gen_data['future_status'] = current_state.minutes_per_step / 60.
            gen_data['future_status'] *= future_status_time_steps

        # infer generator startup and shutdown curves
        if not self._no_startup_shutdown_curves:
            mins_per_step = current_state.minutes_per_step

            for gen, gen_data in sced_model.elements(element_type='generator',
                                                     generator_type='thermal'):
                if 'startup_curve' in gen_data:
                    continue

                ramp_up_rate_sced = gen_data['ramp_up_60min'] * mins_per_step
                ramp_up_rate_sced /= 60.

                sced_startup_capc = self._calculate_sced_ramp_capacity(
                    gen_data, ramp_up_rate_sced,
                    mins_per_step, 'startup_capacity'
                    )

                gen_data['startup_curve'] = [
                    sced_startup_capc - i * ramp_up_rate_sced
                    for i in range(1, int(math.ceil(sced_startup_capc
                                                    / ramp_up_rate_sced)))
                    ]

            for gen, gen_data in sced_model.elements(element_type='generator',
                                                     generator_type='thermal'):
                if 'shutdown_curve' in gen_data:
                    continue

                ramp_down_rate_sced = (
                        gen_data['ramp_down_60min'] * mins_per_step / 60.)

                # compute a new shutdown curve if we go from "on" to "off"
                if (gen_data['initial_status'] > 0
                        and gen_data['fixed_commitment']['values'][0] == 0):
                    power_t0 = gen_data['initial_p_output']
                    # if we end up using a historical curve, it's important
                    # for the time-horizons to match, particularly since this
                    # function is also used to create long-horizon look-ahead
                    # SCEDs for the unit commitment process
                    self.shutdown_curves[gen, mins_per_step] = [
                        power_t0 - i * ramp_down_rate_sced
                        for i in range(
                            1, int(math.ceil(power_t0 / ramp_down_rate_sced)))
                        ]

                if (gen, mins_per_step) in self.shutdown_curves:
                    gen_data['shutdown_curve'] = self.shutdown_curves[
                        gen, mins_per_step]

                else:
                    sced_shutdown_capc = self._calculate_sced_ramp_capacity(
                        gen_data, ramp_down_rate_sced,
                        mins_per_step, 'shutdown_capacity'
                        )

                    gen_data['shutdown_curve'] = [
                        sced_shutdown_capc - i * ramp_down_rate_sced
                        for i in range(
                            1, int(math.ceil(sced_shutdown_capc
                                             / ramp_down_rate_sced))
                            )
                        ]

        if not self._enforce_sced_shutdown_ramprate:
            for gen, gen_data in sced_model.elements(element_type='generator',
                                                     generator_type='thermal'):
                # make sure the generator can immediately turn off
                gen_data['shutdown_capacity'] = max(
                    gen_data['shutdown_capacity'],
                    (60. / current_state.minutes_per_step)
                    * gen_data['initial_p_output'] + 1.
                    )

        return sced_model

    # adapted from prescient.engine.egret.egret_plugin
    @staticmethod
    def _calculate_sced_ramp_capacity(gen_data: dict, ramp_rate_sced: float,
                                      minutes_per_step: int,
                                      capacity_key: str) -> float:
        if capacity_key in gen_data:
            susd_capacity_time_varying = isinstance(gen_data[capacity_key],
                                                    dict)
            p_min_time_varying = isinstance(gen_data['p_min'], dict)

            if p_min_time_varying and susd_capacity_time_varying:
                capacity = sum(
                    (susd - pm) * (minutes_per_step / 60.) + pm
                    for pm, susd in zip(gen_data['p_min']['values'],
                                        gen_data[capacity_key]['values'])
                    ) / len(gen_data['p_min']['values'])

            elif p_min_time_varying:
                capacity = sum(
                    (gen_data[capacity_key] - pm)
                    * (minutes_per_step / 60.) + pm
                    for pm in gen_data['p_min']['values']
                    ) / len(gen_data['p_min']['values'])

            elif susd_capacity_time_varying:
                capacity = sum(
                    (susd - gen_data['p_min']) * (minutes_per_step / 60.)
                    + gen_data['p_min']
                    for susd in gen_data[capacity_key]['values']
                    ) / len(gen_data[capacity_key]['values'])

            else:
                capacity = ((gen_data[capacity_key] - gen_data['p_min'])
                            * (minutes_per_step / 60.) + gen_data['p_min'])

        else:
            if isinstance(gen_data['p_min'], dict):
                capacity = sum(
                    pm + ramp_rate_sced / 2. for pm in
                    gen_data['p_min']['values']
                    ) / len(gen_data['p_min']['values'])

            else:
                capacity = gen_data['p_min'] + ramp_rate_sced / 2.

        return capacity

    def _get_model_for_date(self,
                            requested_date: date,
                            use_actuals: bool) -> VaticModelData:
        """Retrieves the data for a given day and creates a model template.

        Args
        ----
            requested_date  Which day's timeseries values to load.
            use_actuals     Whether to use the actual values for load demand
                            and renewable outputs or forecasted values.

        """
        if use_actuals:
            use_lbl = 'actual'
        else:
            use_lbl = 'fcst'

        # return cached model if we have already loaded it
        if requested_date in self.date_cache[use_lbl]:
            return VaticModelData(self.date_cache[use_lbl][requested_date])

        # get the static power grid characteristics such as thermal generator
        # properties and network topology
        day_data = deepcopy(self.template)
        del day_data['CopperSheet']

        # get the renewable generators' output values for this day
        gen_data = self.gen_data.loc[
            self.gen_data.index.date == requested_date, use_lbl]
        day_data['MaxNondispatchablePower'] = dict()
        day_data['MinNondispatchablePower'] = dict()

        # maximum renewable output is just the actual/forecasted output
        for gen in self.renewables:
            day_data['MaxNondispatchablePower'].update({
                (gen, i + 1): val for i, val in enumerate(gen_data[gen])})

        # we need a second day's worth of values for e.g. extended RUC horizons
        for gen in self.dispatch_renewables:
            if 'WIND' in gen:
                day_data['MaxNondispatchablePower'].update({
                    (gen, i): gen_data[gen][-1] for i in range(25, 49)})
            else:
                day_data['MaxNondispatchablePower'].update({
                    (gen, i + 25): val for i, val in enumerate(gen_data[gen])})

            # for dispatchable renewables, minimum output is always zero
            day_data['MinNondispatchablePower'].update({
                (gen, i + 1): 0 for i in range(48)})

        # for non-dispatchable renewables (RTPV), min output is equal to max
        # output, and we create a second day of values like we do for PV
        for gen in self.nondisp_renewables:
            day_data['MaxNondispatchablePower'].update({
                (gen, i + 25): val for i, val in enumerate(gen_data[gen])})
            day_data['MinNondispatchablePower'].update({
                (gen, i + 1): day_data['MaxNondispatchablePower'][gen, i + 1]
                for i in range(48)
                })

        # get the load demand values for this day
        load_data = self.load_data.loc[
            self.load_data.index.date == requested_date, use_lbl]
        day_data['Demand'] = dict()

        for bus in self.template['Buses']:
            day_data['Demand'].update({
                (bus, i + 1): val for i, val in enumerate(load_data[bus])})
            day_data['Demand'].update({
                (bus, i + 25): val for i, val in enumerate(load_data[bus])})

        # use the loaded data to create a model dictionary that is
        # interpretable by an Egret model formulation, save it to our cache
        model_dict = self.create_vatic_model_dict(day_data)
        self.date_cache[use_lbl][requested_date] = deepcopy(model_dict)

        return VaticModelData(model_dict)

    def create_vatic_model_dict(self, data: dict) -> dict:
        """Convert power grid data into an Egret model dictionary.

        Adapted from
        egret.parsers.prescient_dat_parser.create_model_data_dict_params

        """
        time_periods = range(1, data['NumTimePeriods'] + 1)

        loads = {
            bus: {'bus': bus, 'in_service': True,

                  'p_load': {
                      'data_type': 'time_series',
                      'values': [data['Demand'][bus, t] for t in time_periods]
                      }}

            for bus in data['Buses']
            }

        branches = {
            line: {'from_bus': data['BusFrom'][line],
                   'to_bus': data['BusTo'][line],

                   'reactance': data['Impedence'][line],
                   'rating_long_term': data['ThermalLimit'][line],
                   'rating_short_term': data['ThermalLimit'][line],
                   'rating_emergency': data['ThermalLimit'][line],

                   'in_service': True, 'branch_type': 'line',
                   'angle_diff_min': -90, 'angle_diff_max': 90,
                   }

            for line in data['TransmissionLines']
            }

        # how we create the model entries for generators depends on which model
        # formulation we will use and can thus be changed by children providers
        generators = {**self._create_thermals_model_dict(data),
                      **self._create_renewables_model_dict(data)}

        gen_buses = dict()
        for bus in data['Buses']:
            for gen in data['ThermalGeneratorsAtBus'][bus]:
                gen_buses[gen] = bus
            for gen in data['NondispatchableGeneratorsAtBus'][bus]:
                gen_buses[gen] = bus

        for gen in generators:
            generators[gen]['bus'] = gen_buses[gen]

        return {
            'system': {'time_keys': [str(t) for t in time_periods],
                       'time_period_length_minutes': self._time_period_mins,
                       'load_mismatch_cost': self._load_mismatch_cost,
                       'reserve_shortfall_cost': self._reserve_mismatch_cost,

                       'baseMVA': 1., 'reference_bus': data['Buses'][0],
                       'reference_bus_angle': 0.,

                       'reserve_requirement': {
                           'data_type': 'time_series',
                           'values': [0. for _ in time_periods]
                           }},

            'elements': {'bus': {bus: {'base_kv': 1e3}
                                 for bus in data['Buses']},

                         'load': loads, 'branch': branches,
                         'generator': generators,

                         'interface': dict(), 'zone': dict(), 'storage': dict()
                         }
            }

    def _create_thermals_model_dict(self, data: dict) -> dict:
        return {
            gen: {'generator_type': 'thermal',
                  'fuel': data['ThermalGeneratorType'][gen],
                  'fast_start': False,

                  'fixed_commitment': (1 if gen in data['MustRun'] else None),
                  'in_service': True, 'zone': 'None', 'failure_rate': 0.,

                  'p_min': data['MinimumPowerOutput'][gen],
                  'p_max': data['MaximumPowerOutput'][gen],

                  'ramp_up_60min': data['NominalRampUpLimit'][gen],
                  'ramp_down_60min': data['NominalRampDownLimit'][gen],
                  'startup_capacity': data['StartupRampLimit'][gen],
                  'shutdown_capacity': data['ShutdownRampLimit'][gen],
                  'min_up_time': data['MinimumUpTime'][gen],
                  'min_down_time': data['MinimumDownTime'][gen],
                  'initial_status': data['UnitOnT0State'][gen],
                  'initial_p_output': data['PowerGeneratedT0'][gen],
                  'startup_cost': list(zip(data['StartupLags'][gen],
                                           data['StartupCosts'][gen])),
                  'shutdown_cost': 0.,

                  'p_cost': {
                      'data_type': 'cost_curve',
                      'cost_curve_type': 'piecewise',
                      'values': list(zip(data['CostPiecewisePoints'][gen],
                                         data['CostPiecewiseValues'][gen]))
                      }}

            for gen in self.template['ThermalGenerators']
            }

    def _create_renewables_model_dict(self, data: dict) -> dict:
        gen_dict = {gen: {'generator_type': 'renewable',
                          'fuel': data['NondispatchableGeneratorType'][gen],
                          'in_service': True}
                    for gen in self.template['NondispatchableGenerators']}

        for gen in self.template['NondispatchableGenerators']:
            if gen in self.renewables:
                pmin_vals = [data['MinNondispatchablePower'][gen, t + 1]
                             for t in range(data['NumTimePeriods'])]
                pmax_vals = [data['MaxNondispatchablePower'][gen, t + 1]
                             for t in range(data['NumTimePeriods'])]

            # deal with cases like CSPs which show up in the model template but
            # for which there is no data
            else:
                pmin_vals = [0. for _ in range(data['NumTimePeriods'])]
                pmax_vals = [0. for _ in range(data['NumTimePeriods'])]

            gen_dict[gen]['p_min'] = {'data_type': 'time_series',
                                      'values': pmin_vals}
            gen_dict[gen]['p_max'] = {'data_type': 'time_series',
                                      'values': pmax_vals}

        return gen_dict


class AllocationPickleProvider(PickleProvider):

    def __init__(self, data_dir: str, options: Options) -> None:
        with open(Path(data_dir, "renew-costs.p"), 'rb') as f:
            self.renew_costs: dict = pickle.load(f)

        super().__init__(data_dir, options)

    def _create_renewables_model_dict(self, data: dict) -> dict:
        gen_dict = {gen: {'generator_type': 'renewable',
                          'fuel': data['NondispatchableGeneratorType'][gen],
                          'in_service': True}
                    for gen in self.template['NondispatchableGenerators']}

        for gen in self.template['NondispatchableGenerators']:
            if gen in self.forecast_renewables:
                pmin_vals = [0. for _ in range(data['NumTimePeriods'])]
                pmax_vals = [data['MaxNondispatchablePower'][gen, t + 1]
                             for t in range(data['NumTimePeriods'])]

            # renewables such as hydro which we don't allocate costs to
            elif gen in self.renewables:
                pmin_vals = [data['MinNondispatchablePower'][gen, t + 1]
                             for t in range(data['NumTimePeriods'])]
                pmax_vals = [data['MaxNondispatchablePower'][gen, t + 1]
                             for t in range(data['NumTimePeriods'])]

            # renewables such as CSP for which there is no data
            else:
                pmin_vals = [0. for _ in range(data['NumTimePeriods'])]
                pmax_vals = [0. for _ in range(data['NumTimePeriods'])]

            gen_dict[gen]['p_min'] = {'data_type': 'time_series',
                                      'values': pmin_vals}
            gen_dict[gen]['p_max'] = {'data_type': 'time_series',
                                      'values': pmax_vals}

        for gen, cost_vals in self.renew_costs.items():
            if gen not in self.forecast_renewables:
                raise ProviderError("Costs have been provided for generator "
                                    "`{}` which is not a forecastable (WIND, "
                                    "PV, RTPV) renewable!".format(gen))

            gen_dict[gen]['p_cost'] = deepcopy(cost_vals)

        return gen_dict


class AutoAllocationPickleProvider(PickleProvider):

    cost_vals = [(1., 0.)]

    def _create_renewables_model_dict(self, data: dict) -> dict:
        gen_dict = {gen: {'generator_type': 'renewable',
                          'fuel': data['NondispatchableGeneratorType'][gen],
                          'in_service': True}
                    for gen in self.template['NondispatchableGenerators']}

        for gen in self.template['NondispatchableGenerators']:
            if gen in self.forecast_renewables:
                pmin_vals = [0. for _ in range(data['NumTimePeriods'])]
                pmax_vals = [data['MaxNondispatchablePower'][gen, t + 1]
                             for t in range(data['NumTimePeriods'])]

            # renewables such as hydro which we don't allocate costs to
            elif gen in self.renewables:
                pmin_vals = [data['MinNondispatchablePower'][gen, t + 1]
                             for t in range(data['NumTimePeriods'])]
                pmax_vals = [data['MaxNondispatchablePower'][gen, t + 1]
                             for t in range(data['NumTimePeriods'])]

            # renewables such as CSP for which there is no data
            else:
                pmin_vals = [0. for _ in range(data['NumTimePeriods'])]
                pmax_vals = [0. for _ in range(data['NumTimePeriods'])]

            gen_dict[gen]['p_min'] = {'data_type': 'time_series',
                                      'values': pmin_vals}
            gen_dict[gen]['p_max'] = {'data_type': 'time_series',
                                      'values': pmax_vals}

        for gen in self.forecast_renewables:
            gen_dict[gen]['p_cost'] = {
                'data_type': 'time_series',

                'values': [{'data_type': 'cost_curve',
                            'cost_curve_type': 'piecewise',
                            'values': [(ratio * pmax, cost * ratio * pmax)
                                       for ratio, cost in self.cost_vals]}
                           for pmax in gen_dict[gen]['p_max']['values']]
                }

        return gen_dict
