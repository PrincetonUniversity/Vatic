#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import os.path
from datetime import datetime, date, timedelta
import copy
from pathlib import Path
import dill as pickle
import pandas as pd

from prescient.engine import forecast_helper
from egret.parsers.prescient_dat_parser import get_uc_model, \
    create_model_data_dict_params
from egret.data.model_data import ModelData as EgretModel

from prescient.simulator.options import Options
from typing import Dict, Any, Callable


class ProviderError(Exception):
    pass


class PickleProvider:

    def __init__(self, data_dir: str, start_date: str, num_days: int) -> None:
        self._uc_model_template = get_uc_model()

        with open(Path(data_dir, "grid-template.p"), 'rb') as f:
            self.template = pickle.load(f)
        with open(Path(data_dir, "gen-data.p"), 'rb') as f:
            self.gen_data = pickle.load(f).round(8)
        with open(Path(data_dir, "load-data.p"), 'rb') as f:
            self.load_data = pickle.load(f)

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

        self._data_freq = tuple(data_freq)[0].astype(
            'timedelta64[m]').astype('int')
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

    def negotiate_data_frequency(self, desired_frequency_minutes: int):
        ''' Get the number of minutes between each timestep of actuals data this provider will supply,
            given the requested frequency.

            Arguments
            ---------
            desired_frequency_minutes:int
                The number of minutes between actual values that the application would like to get
                from the data provider.

            Returns
            -------
            Returns the number of minutes between each timestep of data.

            The data provider may be able to provide data at different frequencies.  This method allows the
            data provider to select an appropriate frequency of data samples, given a requested data frequency.

            Note that the frequency indicated by this method only applies to actuals data; estimates are always
            hourly.
        '''
        # This provider can only return one value every 60 minutes.
        return int(self._data_freq)

    def get_initial_model(self, options: Options, num_time_steps: int,
                          minutes_per_timestep: int) -> EgretModel:
        ''' Get a model ready to be populated with data

        Returns
        -------
        A model object populated with static system information, such as
        buses and generators, and with time series arrays that are large
        enough to hold num_time_steps entries.

        Initial values in time time series do not have meaning.
        '''
        # Get data for the first simulation day
        first_day_model = self._get_egret_model_for_date(self._first_day,
                                                         use_actuals=False)

        # Copy it, making sure we've got the right number of time periods
        data = _recurse_copy_with_time_series_length(first_day_model.data,
                                                     num_time_steps)
        new_model = EgretModel(data)
        new_model.data['system']['time_keys'] = list(
            str(i) for i in range(1, num_time_steps + 1))
        new_model.data['system'][
            'time_period_length_minutes'] = minutes_per_timestep

        return new_model

    def populate_initial_state_data(self, options: Options,
                                    day: date,
                                    model: EgretModel) -> None:
        ''' Populate an existing model with initial state data for the requested day

        Sets T0 information from actuals:
          * initial_state_of_charge for each storage element
          * initial_status for each generator
          * initial_p_output for each generator

        Arguments
        ---------
        options:
            Option values
        day:date
            The day whose initial state will be saved in the model
        model: EgretModel
            The model whose values will be modifed
        '''
        if day < self._first_day:
            day = self._first_day
        elif day > self._final_day:
            day = self._final_day

        actuals = self._get_egret_model_for_date(day, use_actuals=True)

        for s, sdict in model.elements('storage'):
            soc = actuals.data['elements']['storage'][s][
                'initial_state_of_charge']
            sdict['initial_state_of_charge'] = soc

        for g, gdict in model.elements('generator', generator_type='thermal'):
            source = actuals.data['elements']['generator'][g]
            gdict['initial_status'] = source['initial_status']
            gdict['initial_p_output'] = source['initial_p_output']

    def populate_with_forecast_data(self, options: Options,
                                    start_time: datetime,
                                    num_time_periods: int,
                                    time_period_length_minutes: int,
                                    model: EgretModel
                                    ) -> None:
        ''' Populate an existing model with forecast data.

        Populates the following values for each requested time period:
            * demand for each bus
            * min and max non-dispatchable power for each non-dispatchable generator
            * reserve requirement

        Arguments
        ---------
        options:
            Option values
        start_time: datetime
            The time (day, hour, and minute) of the first time step for
            which forecast data will be provided
        num_time_periods: int
            The number of time steps for which forecast data will be provided.
        time_period_length_minutes: int
            The duration of each time step
        model: EgretModel
            The model where forecast data will be stored

        Notes
        -----
        This will store forecast data in the model's existing data arrays, starting
        at index 0.  If the model's arrays are not big enough to hold all the
        requested time steps, only those steps for which there is sufficient storage
        will be saved.  If arrays are larger than the number of requested time
        steps, the remaining array elements will be left unchanged.

        If start_time is midnight of any day, all data comes from the DAT file for
        the starting day.  Otherwise, forecast data is taken from the file matching
        the date of the time step.  In other words, if requesting data starting at
        midnight, all data in the first day's DAT file will be available, but otherwise
        only the first 24 hours of each DAT file will be used.

        Note that this method has the same signature as populate_with_actuals.
        '''
        self._populate_with_forecastable_data(
            options, start_time, num_time_periods, time_period_length_minutes,
            model, lambda day: self._get_egret_model_for_date(
                day, use_actuals=False)
            )

    def populate_with_actuals(self, options: Options,
                              start_time: datetime,
                              num_time_periods: int,
                              time_period_length_minutes: int,
                              model: EgretModel
                              ) -> None:
        ''' Populate an existing model with actuals data.

        Populates the following values for each requested time period:
            * demand for each bus
            * min and max non-dispatchable power for each non-dispatchable generator
            * reserve requirement

        Arguments
        ---------
        options:
            Option values
        start_time: datetime
            The time (day, hour, and minute) of the first time step for
            which data will be provided
        num_time_periods: int
            The number of time steps for which actuals data will be provided.
        time_period_length_minutes: int
            The duration of each time step
        model: EgretModel
            The model where actuals data will be stored

        Notes
        -----
        This will store actuals data in the model's existing data arrays, starting
        at index 0.  If the model's arrays are not big enough to hold all the
        requested time steps, only those steps for which there is sufficient storage
        will be saved.  If arrays are larger than the number of requested time
        steps, the remaining array elements will be left unchanged.

        If start_time is midnight of any day, all data comes from the DAT file for
        the starting day.  Otherwise, data is taken from the file matching
        the date of the time step.  In other words, if requesting data starting at
        midnight, all data in the first day's DAT file will be available, but otherwise
        only the first 24 hours of each DAT file will be used.

        Note that this method has the same signature as populate_with_forecast_data.
        '''
        self._populate_with_forecastable_data(
            options, start_time, num_time_periods, time_period_length_minutes,
            model, lambda day: self._get_egret_model_for_date(
                day, use_actuals=True)
            )

    def _populate_with_forecastable_data(self, options: Options,
                                         start_time: datetime,
                                         num_time_periods: int,
                                         time_period_length_minutes: int,
                                         model: EgretModel,
                                         identify_dat: Callable[
                                             [date], EgretModel]
                                         ) -> None:
        # For now, require the time period to always be 60 minutes
        assert (time_period_length_minutes == 60.0)
        step_delta = timedelta(minutes=time_period_length_minutes)

        # See if we have space to store all the requested data.
        # If not, only supply what we have space for
        if len(model.data['system']['time_keys']) < num_time_periods:
            num_time_periods = len(model.data['system']['time_keys'])

        start_hour = start_time.hour
        start_day = start_time.date()
        assert (start_time.minute == 0)
        assert (start_time.second == 0)

        # Find the ratio of native step length to requested step length
        src_step_length_minutes = identify_dat(start_day).data['system'][
            'time_period_length_minutes']
        step_ratio = int(time_period_length_minutes) // src_step_length_minutes

        # Loop through each time step
        for step_index in range(0, num_time_periods):
            step_time = start_time + step_delta * step_index
            day = step_time.date()

            # 0-based hour, useable as index into forecast arrays
            src_step_index = step_index * step_ratio

            # How we handle crossing midnight depends on whether we
            # started at time 0
            if day != start_day:
                if start_hour == 0:
                    # For data starting at time 0, we collect tomorrow's
                    # data from today's dat file
                    day = start_day
                else:
                    # Otherwise we need to subtract off one day's worth of samples
                    src_step_index -= 24 * 60 / src_step_length_minutes
            ### Note that we will never be asked to cross midnight more than once.
            ### That's because any data request that starts mid-day will only request
            ### 24 hours of data and then copy it as needed to fill out the horizon.
            ### If that ever changes, the code above will need to change.

            # If request is beyond the last day, just repeat the final day's values
            if day > self._final_day:
                day = self._final_day

            dat = identify_dat(day)

            for src, target in forecast_helper.get_forecastables(dat, model):
                target[step_index] = src[src_step_index]

    def _get_egret_model_for_date(self,
                                  requested_date: date,
                                  use_actuals) -> EgretModel:
        if use_actuals:
            use_lbl = 'actual'
        else:
            use_lbl = 'fcst'

        # Return cached model, if we have it
        if requested_date in self.date_cache[use_lbl]:
            return self.date_cache[use_lbl][requested_date]

        model_dict = copy.deepcopy(self.template)
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

        day_pyomo = self._uc_model_template.create_instance(
            data={None: use_dict})
        day_dict = create_model_data_dict_params(day_pyomo, True)

        day_model = EgretModel(day_dict)
        self.date_cache[use_lbl][requested_date] = day_model

        return day_model

def _recurse_copy_with_time_series_length(root: Dict[str, Any],
                                          time_count: int) -> Dict[str, Any]:
    new_node = {}
    for key, att in root.items():
        if isinstance(att, dict):
            if 'data_type' in att and att['data_type'] == 'time_series':
                val = att['values'][0]
                new_node[key] = {'data_type': 'time_series',
                                 'values': [val] * time_count}
            else:
                new_node[key] = _recurse_copy_with_time_series_length(att,
                                                                      time_count)
        else:
            new_node[key] = copy.deepcopy(att)
    return new_node
