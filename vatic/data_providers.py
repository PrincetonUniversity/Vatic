
from datetime import datetime, date, timedelta, time
from copy import deepcopy
from pathlib import Path
import dill as pickle
import pandas as pd
import math

from egret.parsers.prescient_dat_parser import get_uc_model, \
    create_model_data_dict_params
from egret.data.model_data import ModelData as EgretModel
from prescient.engine.modeling_engine import ForecastErrorMethod
from prescient.simulator.time_manager import PrescientTime

from prescient.simulator.options import Options
from typing import Union
from prescient.data.simulation_state import MutableSimulationState
from prescient.engine.egret.reporting import (
    report_initial_conditions_for_deterministic_ruc,
    report_demand_for_deterministic_ruc
    )

from .model_data import VaticModelData


class ProviderError(Exception):
    pass


class PickleProvider:

    def __init__(self,
                 data_dir: str, start_date: str, num_days: int,
                 init_ruc_file=None) -> None:
        self._uc_model_template = get_uc_model()

        with open(Path(data_dir, "grid-template.p"), 'rb') as f:
            self.template = pickle.load(f)
        with open(Path(data_dir, "gen-data.p"), 'rb') as f:
            self.gen_data = pickle.load(f).round(8)
        with open(Path(data_dir, "load-data.p"), 'rb') as f:
            self.load_data = pickle.load(f)

        self.init_ruc_file = init_ruc_file
        self._first_day = pd.Timestamp(start_date).date()
        self._final_day = self._first_day + pd.Timedelta(days=num_days)

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

        self.dispatch_renewables = [gen for gen, gen_info in rnwbl_info.items()
                                    if gen_info[1] not in {'HYDRO', 'RTPV'}]
        self.nondisp_renewables = [gen for gen, gen_info in rnwbl_info.items()
                                   if gen_info[1] in {'HYDRO', 'RTPV'}]

        self.init_model = self._get_model_for_date(self._first_day,
                                                   use_actuals=False)
        self.init_model.reset_timeseries()
        self.shutdown_curves = dict()

    def load_initial_model(self):
        with open(self.init_ruc_file, 'rb') as f:
            return pickle.load(f)

    def _get_initial_model(self,
                           num_time_steps: int,
                           minutes_per_timestep: int) -> VaticModelData:
        ''' Get a model ready to be populated with data

        Returns
        -------
        A model object populated with static system information, such as
        buses and generators, and with time series arrays that are large
        enough to hold num_time_steps entries.

        Initial values in time time series do not have meaning.
        '''

        new_model = deepcopy(self.init_model)
        new_model.set_time_steps(num_time_steps, minutes_per_timestep)

        return new_model

    def _get_initial_state_model(self,
                                 num_time_steps: int,
                                 minutes_per_timestep: int,
                                 day: date) -> VaticModelData:
        ''' Populate an existing model with initial state data for the requested day

        Sets T0 information from actuals:
          * initial_state_of_charge for each storage element
          * initial_status for each generator
          * initial_p_output for each generator

        Arguments
        ---------
        day:date
            The day whose initial state will be saved in the model
        model: EgretModel
            The model whose values will be modifed
        '''
        if day < self._first_day:
            day = self._first_day
        elif day > self._final_day:
            day = self._final_day

        new_model = self._get_initial_model(num_time_steps,
                                            minutes_per_timestep)
        actuals_model = self._get_model_for_date(day, use_actuals=True)

        new_model.copy_elements(actuals_model, 'storage',
                                ['initial_state_of_charge'], strict_mode=True)
        new_model.copy_elements(actuals_model, 'generator',
                                ['initial_status', 'initial_p_output'],
                                strict_mode=True, generator_type='thermal')

        return new_model

    def _copy_initial_state_into_model(
            self,
            use_state: MutableSimulationState, num_time_steps: int,
            minutes_per_timestep: int) -> VaticModelData:

        new_model = self._get_initial_model(num_time_steps,
                                            minutes_per_timestep)

        for gen, g_data in new_model.elements('generator',
                                              generator_type='thermal'):
            g_data['initial_status'] = use_state.get_initial_generator_state(
                gen)
            g_data['initial_p_output'] = use_state.get_initial_power_generated(
                gen)

        for store, store_data in new_model.elements('storage'):
            store_data['initial_state_of_charge'] \
                = use_state.get_initial_state_of_charge(store)

        return new_model

    def get_populated_model(
            self,
            use_actuals: bool, start_time: datetime, num_time_periods: int,
            use_state: Union[None, MutableSimulationState] = None,
            reserve_factor: float = 0.
            ) -> VaticModelData:

        start_hour = start_time.hour
        start_day = start_time.date()
        assert (start_time.minute == 0)
        assert (start_time.second == 0)
        step_delta = timedelta(minutes=self.data_freq)

        # Populate the T0 data
        if use_state is None or use_state.timestep_count == 0:
            new_model = self._get_initial_state_model(
                num_time_steps=num_time_periods,
                minutes_per_timestep=self.data_freq, day=start_time.date()
                )

        else:
            new_model = self._copy_initial_state_into_model(
                use_state, num_time_periods, self.data_freq)

        day_model = self._get_model_for_date(start_day, use_actuals)

        for step_index in range(0, num_time_periods):
            step_time = start_time + step_delta * step_index
            day = step_time.date()
            src_step_index = step_index

            # If request is beyond the last day, just repeat the
            # final day's values
            if day > self._final_day:
                day = self._final_day

            # How we handle crossing midnight depends on whether we
            # started at time 0
            if day != start_day:
                if start_hour == 0:
                    # For data starting at time 0, we collect tomorrow's
                    # data from today's dat file
                    day = start_day
                else:
                    # Otherwise we need to subtract off one day's worth of samples
                    src_step_index = step_index - 24
                    day_model = self._get_model_for_date(day, use_actuals)

            ### Note that we will never be asked to cross midnight more than once.
            ### That's because any data request that starts mid-day will only request
            ### 24 hours of data and then copy it as needed to fill out the horizon.
            ### If that ever changes, the code above will need to change.

            new_model.copy_forecastables(day_model, step_index, src_step_index)
            new_model.honor_reserve_factor(reserve_factor, step_index)

        return new_model

    def create_deterministic_ruc(self,
                                 time_step: PrescientTime, options: Options,
                                 current_state=None,
                                 output_init_conditions=False) -> EgretModel:
        """
        merge of EgretEngine.create_deterministic_ruc
             and egret_plugin.create_deterministic_ruc
        """

        start_time = datetime.combine(time_step.date, time(time_step.hour))
        copy_first_day = not options.run_ruc_with_next_day_data
        copy_first_day &= time_step.hour != 0

        forecast_request_count = 24
        if not copy_first_day:
            forecast_request_count = options.ruc_horizon

        # Populate forecasts
        ruc_model = self.get_populated_model(
            use_actuals=False, start_time=start_time,
            num_time_periods=forecast_request_count, use_state=current_state,
            reserve_factor=options.reserve_factor
            )

        # Make some near-term forecasts more accurate
        ruc_delay = -(options.ruc_execution_hour % -options.ruc_every_hours)
        if options.ruc_prescience_hour > ruc_delay + 1:
            improved_hour_count = options.ruc_prescience_hour - ruc_delay
            improved_hour_count -= 1
            future_actuals = current_state.get_future_actuals()

            for forecast, actuals in zip(ruc_model.get_forecastables(),
                                         future_actuals):
                for t in range(improved_hour_count):
                    forecast_portion = (ruc_delay + t)
                    forecast_portion /= options.ruc_prescience_hour
                    actuals_portion = 1 - forecast_portion

                    forecast[t] = forecast_portion * forecast[t]
                    forecast[t] += actuals_portion * actuals[t]

        # Copy from first 24 to second 24, if necessary
        if copy_first_day:
            for vals, in ruc_model.get_forecastables():
                for t in range(24, options.ruc_horizon):
                    vals[t] = vals[t - 24]

        if output_init_conditions:
            report_initial_conditions_for_deterministic_ruc(ruc_model)
            report_demand_for_deterministic_ruc(ruc_model,
                                                options.ruc_every_hours)

        return ruc_model.to_egret()

    def create_sced_instance(
            self,
            current_state: MutableSimulationState, options: Options,
            sced_horizon: int,
            forecast_error_method=ForecastErrorMethod.PRESCIENT
            ) -> EgretModel:
        assert current_state is not None

        sced_model = self._copy_initial_state_into_model(
            current_state, sced_horizon, self.data_freq)

        # initialize the demand and renewables data, based
        # on the forecast error model
        if forecast_error_method is ForecastErrorMethod.PRESCIENT:
            # Warning: This method can see into the future!
            future_actuals = current_state.get_future_actuals()
            sced_forecastables = sced_model.get_forecastables()

            for future, sced_data in zip(future_actuals, sced_forecastables):
                for t in range(sced_horizon):
                    sced_data[t] = future[t]

        else:  # persistent forecast error:
            current_actuals = current_state.get_current_actuals()
            forecasts = current_state.get_forecasts()
            sced_forecastables = sced_model.get_forecastables()

            # Go through each time series that can be forecasted
            for current_actual, forecast, sced_data in zip(current_actuals,
                                                           forecasts,
                                                           sced_forecastables):
                # the first value is, by definition, the actual.
                sced_data[0] = current_actual

                # Find how much the first forecast was off from the actual, as
                # a fraction of the forecast. For all subsequent times, adjust
                # the forecast by the same fraction.
                current_forecast = forecast[0]
                if current_forecast == 0.0:
                    forecast_error_ratio = 0.0
                else:
                    forecast_error_ratio = current_actual / forecast[0]

                for t in range(1, sced_horizon):
                    sced_data[t] = forecast[t] * forecast_error_ratio

        for t in range(sced_horizon):
            sced_model.honor_reserve_factor(options.reserve_factor, t)

        # Set generator commitments & future state, start by preparing an empty
        # array of the correct size for each generator
        for gen, gen_data in sced_model.elements(element_type='generator',
                                                 generator_type='thermal'):
            fixed_commitment = [current_state.get_generator_commitment(gen, t)
                                for t in range(sced_horizon)]
            gen_data['fixed_commitment'] = {'data_type': 'time_series',
                                            'values': fixed_commitment}

            # Look as far into the future as we can for future
            # startups / shutdowns
            last_commitment = fixed_commitment[-1]
            for t in range(sced_horizon, current_state.timestep_count):
                this_commitment = current_state.get_generator_commitment(
                    gen, t)

                # future startup
                if (this_commitment - last_commitment) > 0.5:
                    future_status_time_steps = (t - sced_horizon + 1)
                    break

                # future shutdown
                elif (last_commitment - this_commitment) > 0.5:
                    future_status_time_steps = -(t - sced_horizon + 1)
                    break

            # no break
            else:
                future_status_time_steps = 0

            gen_data['future_status'] = current_state.minutes_per_step / 60.
            gen_data['future_status'] *= future_status_time_steps

        if not options.no_startup_shutdown_curves:
            mins_per_step = current_state.minutes_per_step

            for gen, gen_data in sced_model.elements(element_type='generator',
                                                     generator_type='thermal'):
                if 'startup_curve' in gen_data:
                    continue

                ramp_up_rate_sced = gen_data['ramp_up_60min'] * mins_per_step
                ramp_up_rate_sced /= 60.

                sced_startup_capacity = _calculate_sced_ramp_capacity(
                    gen_data, ramp_up_rate_sced,
                    mins_per_step, 'startup_capacity'
                    )

                gen_data['startup_curve'] = [
                    sced_startup_capacity - i * ramp_up_rate_sced
                    for i in range(1, int(math.ceil(sced_startup_capacity
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
                    sced_shutdown_capacity = _calculate_sced_ramp_capacity(
                        gen_data, ramp_down_rate_sced,
                        mins_per_step, 'shutdown_capacity'
                        )

                    gen_data['shutdown_curve'] = [
                        sced_shutdown_capacity - i * ramp_down_rate_sced
                        for i in range(
                            1, int(math.ceil(sced_shutdown_capacity
                                             / ramp_down_rate_sced))
                            )
                        ]

        if not options.enforce_sced_shutdown_ramprate:
            for gen, gen_data in sced_model.elements(element_type='generator',
                                                     generator_type='thermal'):
                # make sure the generator can immediately turn off
                gen_data['shutdown_capacity'] = max(
                    gen_data['shutdown_capacity'],
                    (60. / current_state.minutes_per_step)
                    * gen_data['initial_p_output'] + 1.
                    )

        return sced_model.to_egret()

    def _get_model_for_date(self,
                            requested_date: date,
                            use_actuals) -> VaticModelData:
        if use_actuals:
            use_lbl = 'actual'
        else:
            use_lbl = 'fcst'

        # Return cached model, if we have it
        if requested_date in self.date_cache[use_lbl]:
            return self.date_cache[use_lbl][requested_date]

        model_dict = deepcopy(self.template)
        del model_dict['CopperSheet']

        gen_data = self.gen_data.loc[
            self.gen_data.index.date == requested_date, use_lbl]
        model_dict['MaxNondispatchablePower'] = dict()
        model_dict['MinNondispatchablePower'] = dict()

        for gen in self.renewables:
            model_dict['MaxNondispatchablePower'].update({
                (gen, i + 1): val for i, val in enumerate(gen_data[gen])})

        for gen in self.dispatch_renewables:
            if 'WIND' in gen:
                model_dict['MaxNondispatchablePower'].update({
                    (gen, i): gen_data[gen][-1] for i in range(25, 49)})
            else:
                model_dict['MaxNondispatchablePower'].update({
                    (gen, i + 25): val for i, val in enumerate(gen_data[gen])})

            model_dict['MinNondispatchablePower'].update({
                (gen, i + 1): 0 for i in range(48)})

        for gen in self.nondisp_renewables:
            model_dict['MaxNondispatchablePower'].update({
                (gen, i + 25): val for i, val in enumerate(gen_data[gen])})
            model_dict['MinNondispatchablePower'].update({
                (gen, i + 1): model_dict['MaxNondispatchablePower'][gen, i + 1]
                for i in range(48)
                })

        load_data = self.load_data.loc[
            self.load_data.index.date == requested_date, use_lbl]
        model_dict['Demand'] = dict()

        for bus in self.template['Buses']:
            model_dict['Demand'].update({
                (bus, i + 1): val for i, val in enumerate(load_data[bus])})
            model_dict['Demand'].update({
                (bus, i + 25): val for i, val in enumerate(load_data[bus])})

        namespace_ks = {'NumTimePeriods', 'TransmissionLines', 'StageSet',
                        'NondispatchableGenerators', 'ThermalGenerators',
                        'TimePeriodLength', 'Buses'}

        use_dict = {k: ({None: v} if k in namespace_ks else v)
                    for k, v in model_dict.items()}
        use_dict['MustRun'] = {k: 1 for k in use_dict['MustRun']}

        day_model = VaticModelData(create_model_data_dict_params(
            self._uc_model_template.create_instance(data={None: use_dict}),
            keep_names=True
            ))
        self.date_cache[use_lbl][requested_date] = deepcopy(day_model)

        return day_model


# adapted from prescient.engine.egret.egret_plugin
def _calculate_sced_ramp_capacity(gen_data, ramp_rate_sced,
                                  minutes_per_step, capacity_key):
    if capacity_key in gen_data:
        susd_capacity_time_varying = isinstance(gen_data[capacity_key], dict)
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
                pm + ramp_rate_sced / 2. for pm in gen_data['p_min']['values']
                ) / len(gen_data['p_min']['values'])

        else:
            capacity = gen_data['p_min'] + ramp_rate_sced / 2.

    return capacity
