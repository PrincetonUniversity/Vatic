"""Storing the states of a grid's generation assets within a simulation run."""

# TODO: make this a set of attributes of simulation engines?

from __future__ import annotations

import pandas as pd
from typing import Optional
from .formulations import RucModel, ScedModel


class VaticStateError(Exception):
    pass


class VaticSimulationState:
    """A system state that can be updated with data from RUCs and SCEDs.

    Parameters
    ----------

    forecasts, actuals      Renewable outputs and load demands as forecasted
                            and realized at this state's current time.

    ruc_every_hours         How often to run unit commitments.
    sced_frequency_minutes  How often to run real-time economic dispatches.

    """

    def __init__(self,
                 ruc_execution_hour: int, ruc_every_hours: int,
                 sced_frequency_minutes: int) -> None:
        self.forecasts = None
        self.actuals = None
        self._commitments = pd.DataFrame()

        self._init_gen_state = dict()
        self._init_power_gen = dict()
        self._init_soc = dict()

        # timestep durations; how often a SCED is run
        self._minutes_per_forecast_step = 60
        self._minutes_per_actuals_step = 60

        # the current simulation minute and the next minute when forecasts
        # should be popped
        self._simulation_minute = 0
        self._next_forecast_pop_minute = 0
        self._next_actuals_pop_minute = 0

        # TODO: should this be here or in the time manager?
        self.ruc_delay = -(ruc_execution_hour % -ruc_every_hours)
        self._sced_frequency = sced_frequency_minutes

    @property
    def times(self) -> list[int]:
        """The timesteps we have data for."""

        return self._commitments.columns.tolist()

    @property
    def generators(self) -> list[str]:
        return list(self._init_gen_state.keys())

    @property
    def timestep_count(self) -> int:
        """The number of timesteps we have data for."""

        return len(self.times)

    def timestep_commitments(self, t: int) -> pd.Series[bool]:
        """Return commitments for a single time."""

        if self.actuals is None:
            raise VaticStateError("Cannot retrieve commitments "
                                  "from an empty state!")

        if not isinstance(t, int):
            raise TypeError("Timestep must be given as integer value "
                            f"in\n{self.times}")

        if t not in self.times:
            raise ValueError("Timestep must be given as integer value "
                             f"in\n{self.times}")

        return self._commitments.loc[:, t].astype(bool)

    def get_commitments(self,
                        ts: Optional[list[int]] = None) -> pd.DataFrame[bool]:
        """Return commitments across a number of times."""

        if self.actuals is None:
            raise VaticStateError("Cannot retrieve commitments "
                                  "from an empty state!")

        if ts is None:
            ts = self.times

        if not isinstance(ts, list):
            raise TypeError("Times must be given as a list of integers!")

        if len(ts) == 0:
            print("Give list of times is empty, returning empty dataframe!")

        # TODO: make this match retrieval of times in `timestep_commitments`?
        for t in ts:
            if not isinstance(t, int):
                raise ValueError(f"Given timestep {t} not an integer value "
                                 f"in\n{self.times}")

            if t not in self.times:
                raise ValueError(f"Given timestep {t} not an integer value "
                                 f"in\n{self.times}")

        return self._commitments.loc[:, ts].astype(bool)

    @property
    def current_actuals(self) -> pd.Series:
        """Get the current actual value for each forecastable.

        This is the actual value for the current time period (time index 0).
        Values are returned in the same order as
        BaseModel.get_forecastables, but instead of returning arrays it
        returns a single value.

        """
        return self.actuals.iloc[0]

    def get_future_actuals(self, k: tuple[str, str]) -> pd.Series:
        """Warning: Returns actual values for current time AND FUTURE TIMES.

        Be aware that this function returns information that is not yet known!
        The function lets you peek into the future.  Future actuals may be used
        by some (probably unrealistic) algorithm options, such as

        """
        return self.actuals[k].iloc[1:]

    def apply_initial_ruc(self,
                          ruc: RucModel, sim_actuals: pd.DataFrame) -> None:
        """This is the first RUC; save initial state."""

        self._init_gen_state = ruc.UnitOnT0State
        self._init_power_gen = ruc.PowerGeneratedT0
        self._commitments = pd.Series(ruc.results['commitment']).unstack()

        # usually assumed to be empty
        for store, store_data in ruc.Storage:
            self._init_soc[store] = store_data['initial_state_of_charge']

        # hard-code these for now
        self._minutes_per_forecast_step = 60
        self._next_forecast_pop_minute = 60
        self._minutes_per_actuals_step = 60
        self._next_actuals_pop_minute = 60

        self.forecasts = ruc.forecastables
        self.actuals = sim_actuals

    def apply_planning_ruc(self,
                           ruc: RucModel, sim_actuals: dict) -> None:
        """Incorporate a RUC instance into the current state.

        This will save the ruc's forecasts, and for the very first ruc
        this will also save initial state info.

        If there is a ruc delay, as indicated by options.ruc_execution_hour and
        options.ruc_every_hours, then the RUC is applied to future time
        periods, offset by the ruc delay.  This does not apply to the very
        first RUC, which is used to set up the initial simulation state with no
        offset.
        """

        # Now save all generator commitments
        # Keep the first "ruc_delay" commitments from the prior ruc
        for gen, gen_data in ruc.elements('generator',
                                          generator_type='thermal'):
            self._commitments[gen] = self._commitments[gen][:self.ruc_delay]
            self._commitments[gen] += tuple(gen_data['commitment']['values'])

        for k, new_ruc_vals in ruc.get_forecastables():
            self.forecasts[k] = self.forecasts[k][:self.ruc_delay]
            self.forecasts[k] += tuple(new_ruc_vals)

        for k, new_ruc_vals in sim_actuals.get_forecastables():
            self.actuals[k] = self.actuals[k][:self.ruc_delay]
            self.actuals[k] += tuple(new_ruc_vals)

    def apply_sced(self,
                   sced: ScedModel, sced_time: Optional[int] = 1) -> None:
        """Merge a sced into the current state, and move to next time period.

        This saves the sced's first time period of data as initial state
        information, and advances the current time forward by one time period.

        """
        if sorted(sced.ThermalGenerators) == self.generators:
            raise VaticStateError("Incompatible SCED model with a different "
                                  "set of thermal generators than this state!")

        if sced_time not in sced.TimePeriods:
            raise VaticStateError("Cannot apply the state from time "
                                  f"{sced_time}; this SCED model only has "
                                  f"states {sced.TimePeriods}")

        sced_indx = 1 + sced.TimePeriods.index(sced_time)

        for gen, init_state in self._init_gen_state.items():
            state_duration = abs(init_state)
            unit_on = init_state > 0

            # March forward, counting how long the state is on or off
            for t in sced.TimePeriods[:sced_indx]:
                new_on = int(round(sced.results['commitment'][gen, t])) > 0

                if new_on == unit_on:
                    state_duration += 1
                else:
                    state_duration = 1
                    unit_on = new_on

                if t == sced_time:
                    break

            if not unit_on:
                state_duration = -state_duration

            # Convert duration back into hours
            # state_duration *= hours_per_period

            # get how much power was generated, within bounds
            gen_power = sced.results['power_generated'][gen, sced_time]
            pmin = sced.MinPowerOutput[gen, sced_time]
            pmax = sced.MaxPowerOutput[gen, sced_time]

            if unit_on == 0:
                assert gen_power == 0.

            elif (pmin - 1e-5) <= gen_power < pmin:
                gen_power = pmin
            elif pmax < gen_power <= (pmax + 1e-5):
                gen_power = pmax

            elif gen_power < pmin:
                raise VaticStateError(f"Invalid power generation {gen_power} "
                                      f"for {gen} which is less than "
                                      f"pmin={pmin}!")

            elif gen_power > pmax:
                raise VaticStateError(f"Invalid power generation {gen_power} "
                                      f"for {gen} which is greater than "
                                      f"pmin={pmax}!")

            self._init_gen_state[gen] = state_duration
            self._init_power_gen[gen] = gen_power

        # Advance time, dropping data if necessary
        self._simulation_minute += self._sced_frequency

        while self._next_forecast_pop_minute <= self._simulation_minute:
            for k in self.forecasts:
                self.forecasts[k] = self.forecasts[k][1:]

            for gen in self._commitments:
                self._commitments[gen] = self._commitments[gen][1:]

            self._next_forecast_pop_minute += self._minutes_per_forecast_step

        while self._simulation_minute >= self._next_actuals_pop_minute:
            for k in self.actuals:
                self.actuals[k] = self.actuals[k][1:]

            self._next_actuals_pop_minute += self._minutes_per_actuals_step
