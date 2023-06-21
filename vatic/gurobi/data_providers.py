"""Retrieving optimization model inputs from power grid datasets."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, date, timedelta
import dill as pickle
import pandas as pd
from typing import Optional

from ..egret.time_manager import VaticTime
from .simulation_state import VaticSimulationState
from .formulations import RucModel, ScedModel


class ProviderError(Exception):
    pass


class PickleProvider:
    """Loading data from input datasets and generating UC and ED models.

    This class' purpose is to store the parsed grid data created by
    vatic.data.loaders while a simulation is running, and to provide parsed
    slices of the grid data as required by various subroutines of the
    simulation. Most importantly, the methods `create_deterministic_ruc` and
    `create_sced_instance` are used to construct the input data structures used
    by a simulation's unit commitment and economic dispatch optimizations
    respectively. See vatic.model_data.VaticModelData for how these data
    structures are implemented.

    The core data retrieval functionality implemented here was based upon
    prescient.data.providers.dat_data_provider and adapted to use pickled input
    dataframes as opposed to .dat files. This class also includes model data
    creation methods originally included in prescient.egret.engine.egret_plugin
    """

    def __init__(
            self,
            template_data: dict, gen_data: pd.DataFrame,
            load_data: pd.DataFrame, load_shed_penalty: float,
            reserve_shortfall_penalty: float, reserve_factor: float,
            prescient_sced_forecasts: bool, ruc_prescience_hour: int,
            ruc_execution_hour: int, ruc_every_hours: int,
            ruc_horizon: int, enforce_sced_shutdown_ramprate: bool,
            no_startup_shutdown_curves: bool, verbosity: int,
            start_date: Optional[date] = None, num_days: Optional[int] = None,
            renew_costs: Optional[dict | str | Path | list[float]] = None
            ) -> None:

        if not (gen_data.index == load_data.index).all():
            raise ProviderError("The generator and the bus demand datasets "
                                "have inconsistent time points!")

        self.template = template_data
        self.gen_data = gen_data.sort_index().round(8)
        self.load_data = load_data.sort_index()

        self._time_period_mins = 60
        self._load_mismatch_cost = load_shed_penalty
        self._reserve_mismatch_cost = reserve_shortfall_penalty
        self._reserve_factor = reserve_factor

        self._ruc_execution_hour = ruc_execution_hour
        self._ruc_every_hours = ruc_every_hours
        self._ruc_delay = -(self._ruc_execution_hour % -ruc_every_hours)

        self.ruc_horizon = ruc_horizon
        self._ruc_prescience_hour = ruc_prescience_hour
        self._output_ruc_initial_conditions = verbosity > 3

        self.prescient_sced_forecasts = prescient_sced_forecasts
        self._enforce_sced_shutdown_ramprate = enforce_sced_shutdown_ramprate
        self._no_startup_shutdown_curves = no_startup_shutdown_curves

        # parse cost curves given for renewable generators as necessary
        if isinstance(renew_costs, dict):
            self.renew_costs = renew_costs

        elif isinstance(renew_costs, (str, Path)):
            with open(renew_costs, 'rb') as f:
                self.renew_costs = pickle.load(f)

        elif isinstance(renew_costs, list):
            ncosts = len(renew_costs) - 1

            if ncosts == 0:
                self.renew_costs = [(1., float(renew_costs[0]))]
            else:
                self.renew_costs = [(i / ncosts, float(c))
                                    for i, c in enumerate(renew_costs)]

        elif renew_costs is None:
            self.renew_costs = None

        else:
            raise TypeError("Unrecognized renewable "
                            "costs given: `{}`!".format(renew_costs))

        if start_date:
            self.first_day = start_date
        else:
            self.first_day = self.gen_data.index[0].date()

        if num_days:
            self.final_day = self.first_day + pd.Timedelta(days=num_days)
        else:
            self.final_day = self.gen_data.index[-1].date()

        for run_date in pd.date_range(self.first_day, self.final_day,
                                      freq='D'):
            if (self.gen_data.index.date == run_date.date()).sum() != 24:
                raise ProviderError(
                    "The generator data in the input directory does not have "
                    "the correct number of observed values for simulation day "
                    "{}!".format(run_date.strftime('%F'))
                    )

            if (self.load_data.index.date == run_date.date()).sum() != 24:
                raise ProviderError(
                    "The generator data in the input directory does not have "
                    "the correct number of observed values for simulation day "
                    "{}!".format(run_date.strftime('%F'))
                    )

        data_freq = set(self.gen_data.index.to_series().diff()[1:].values)
        if len(data_freq) != 1:
            raise ProviderError("Inconsistent dataset time frequencies!")

        self.data_freq = int(
            tuple(data_freq)[0].astype('timedelta64[m]').astype('int'))
        self.date_cache = {'actl': dict(), 'fcst': dict()}

        # get list of generators with available forecasted output values
        self.renewables = self.gen_data.columns.get_level_values(
            level=1).unique().tolist()

        # get generators with forecasts whose output can be adjusted (or not)
        self.template['DispatchRenewables'] = [
            gen for gen in self.template['DispatchRenewables']
            if gen in self.renewables
            ]
        self.template['NondispatchRenewables'] = [
            gen for gen in self.template['NondispatchRenewables']
            if gen in self.renewables
            ]

        self.shutdown_curves = dict()

    def create_ruc(
            self,
            time_step: VaticTime,
            current_state: VaticSimulationState | None = None,
            copy_first_day: bool = False
            ) -> RucModel:
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
        forecast_request_count = 24
        if not copy_first_day:
            forecast_request_count = self.ruc_horizon

        start_hour = time_step.when.hour
        start_day = time_step.when.date()
        assert (time_step.when.minute == 0)
        assert (time_step.when.second == 0)

        step_delta = timedelta(minutes=self.data_freq)
        start_dt = pd.Timestamp(time_step.when, tz='utc')
        ruc_times = list()

        for step_time in pd.date_range(
                start_dt, periods=forecast_request_count, freq='H'):
            step_day = step_time.date()

            # if we cross midnight and we didn't start at midnight, we start
            # pulling data from the next day
            if step_day != start_day and start_hour != 0:
                ruc_times += [step_time]

            elif step_day > self.final_day:
                ruc_times += [step_time - pd.Timedelta(days=1)]
            else:
                ruc_times += [step_time]

        if not all(pd.DatetimeIndex(ruc_times).isin(self.gen_data.index)):
            raise ProviderError(f"Cannot create model; missing data "
                                f"in renewable outputs!")

        if not all(pd.DatetimeIndex(ruc_times).isin(self.load_data.index)):
            raise ProviderError(f"Cannot create model; missing data "
                                f"in load demands!")

        # TODO: use `forecastables()` here the way we do in `create_sced`?
        # get the data for this date from the input datasets
        use_gen = self.gen_data.loc[ruc_times, 'fcst']
        use_load = self.load_data.loc[ruc_times, 'fcst']

        # make some near-term forecasts more accurate if necessary
        if self._ruc_prescience_hour > self._ruc_delay + 1:
            improved_hour_count = self._ruc_prescience_hour - self._ruc_delay
            improved_hour_count -= 1

            for fcst_key, fcst_vals in ruc_model.get_forecastables():
                for t in range(improved_hour_count):
                    forecast_portion = (self._ruc_delay + t)
                    forecast_portion /= self._ruc_prescience_hour
                    actuals_portion = 1 - forecast_portion
                    actl_val = current_state.get_future_actuals(fcst_key)[t]

                    fcst_vals[t] *= forecast_portion
                    fcst_vals[t] += actuals_portion * actl_val

        # copy from the first 24 hours to the second 24 hours if necessary
        if copy_first_day:
            for t in range(24, self.ruc_horizon):
                vals[t] = vals[t - 24]

        ruc_model = RucModel(self.template, use_gen, use_load,
                             self._reserve_factor,
                             sim_state=current_state)

        # TODO: add more reporting
        if self._output_ruc_initial_conditions:
            tgens = dict(ruc_model.elements('generator',
                                            generator_type='thermal'))
            gen_label_size = max((len(gen) for gen in tgens), default=0) + 1

            print("\nInitial generator conditions (gen-name t0-unit-on"
                  " t0-unit-on-state t0-power-generated):")

            for gen, gen_data in tgens.items():
                print(' '.join([
                    format(gen, '{}s'.format(gen_label_size)),
                    format(str(int(gen_data['initial_status'] > 0)), '5s'),
                    format(gen_data['initial_status'], '7d'),
                    format(gen_data['initial_p_output'], '12.2f')
                    ]))

        return ruc_model

    def sim_actuals(self,
                    start_time: VaticTime, num_steps: int) -> pd.DataFrame:
        start_dt = pd.Timestamp(start_time.when, tz='utc')

        start_hour = start_time.when.hour
        start_day = start_time.when.date()
        use_times = list()

        for step_time in pd.date_range(start_dt, periods=num_steps, freq='H'):
            step_day = step_time.date()

            # if we cross midnight and we didn't start at midnight, we start
            # pulling data from the next day
            if step_day != start_day and start_hour != 0:
                use_times += [step_time]

            elif step_day > self.final_day:
                use_times += [step_time - pd.Timedelta(days=1)]
            else:
                use_times += [step_time]

        return self.forecastables(use_actuals=True, time_steps=use_times)

    def forecastables(
            self,
            use_actuals: bool, time_steps: Optional[list[datetime]] = None
            ) -> pd.DataFrame:
        """Compare e.g. to formulations.forecastables."""

        if time_steps:
            if not all(pd.DatetimeIndex(time_steps).isin(
                    self.gen_data.index)):
                raise ProviderError("Missing renewables outputs for time "
                                    f"steps {time_steps[0]}...")

            if not all(pd.DatetimeIndex(time_steps).isin(
                    self.load_data.index)):
                raise ProviderError("Missing load demands for time "
                                    f"steps {time_steps[0]}...")

        if use_actuals:
            use_gen = self.gen_data.actl
            use_load = self.load_data.actl
        else:
            use_gen = self.gen_data.fcst
            use_load = self.load_data.fcst

        use_gen.columns = pd.MultiIndex.from_tuples(
            [('RenewGen', gen) for gen in use_gen.columns],
            names=('AssetType', 'Asset')
            )
        use_load.columns = pd.MultiIndex.from_tuples(
            [('LoadBus', bus) for bus in use_load.columns],
            names=('AssetType', 'Asset')
            )

        fcsts = pd.concat([use_gen, use_load], axis=1)
        if time_steps:
            fcsts = fcsts.loc[time_steps]

        return fcsts

    def create_sced(self,
                    time_step: VaticTime, current_state: VaticSimulationState,
                    sced_horizon: int) -> ScedModel:
        """Generates a Security Constrained Economic Dispatch model.

        This a merge of Prescient's EgretEngine.create_sced_instance and
        egret_plugin.create_sced_instance.

        Args
        ----
            current_state   The simulation state of a power grid which will
                            be used as the basis for the data included in this
                            model.
            sced_horizon    How many time steps this SCED will simulate over.

        """

        # assert current_state.timestep_count >= sced_horizon

        sced_times = [step_time for step_time in pd.date_range(
            time_step.when, periods=sced_horizon, freq='H', tz='utc')]

        # get the data for this date from the input datasets
        sced_data = self.forecastables(use_actuals=True, time_steps=sced_times)

        # add the forecasted load demands and renewable generator outputs, and
        # adjust them to be closer to the corresponding actual values
        if self.prescient_sced_forecasts:
            for k, sced_data in sced_model.get_forecastables():
                future = current_state.get_future_actuals(k)

                # this error method makes the forecasts equal to the actuals!
                for t in range(sced_horizon):
                    sced_data[t] = future[t]

        else:
            actuals = current_state.current_actuals
            forecasts = current_state.forecasts.iloc[:sced_horizon]

            # this error method adjusts future forecasts based on how much
            # the forecast over/underestimated the current actual value
            sced_data.iloc[0] = actuals

            # find how much the first forecast was off from the actual as
            # a fraction of the forecast
            if len(sced_times) > 1:
                fcst_error_ratios = pd.Series(
                    {k: (actuals[k] / cur_fcst if cur_fcst else 0.)
                     for k, cur_fcst in forecasts.iloc[0].items()
                     })

                # adjust the remaining forecasts based on the initial error
                sced_data.iloc[1:] = fcst_error_ratios * forecasts.iloc[1:]

        # look as far into the future as we can for startups if the
        # generator is committed to be off at the end of the model window
        horizon_cmts = current_state.timestep_commitments(sced_horizon)
        future_cmts = current_state.get_commitments(
            list(range(sced_horizon + 1, max(current_state.times) + 1)))
        future_status_times = {g: 0 for g in horizon_cmts.index}

        for off_gen, gen_cmts in future_cmts.loc[~horizon_cmts].iterrows():
            if gen_cmts.any():
                future_status_times[off_gen] = (gen_cmts[gen_cmts].index.max()
                                                - sced_horizon + 1)

        for on_gen, gen_cmts in future_cmts.loc[horizon_cmts].iterrows():
            if (~gen_cmts).any():
                future_status_times[on_gen] = -(gen_cmts[~gen_cmts].index.max()
                                                - sced_horizon + 1)

        sced_model = ScedModel(
            self.template, sced_data['RenewGen'], sced_data['LoadBus'],
            reserve_factor=self._reserve_factor,
            sim_state=current_state, future_status=future_status_times
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
