"""Storing the states of a grid's generation assets within a simulation run."""

from __future__ import annotations

import math
import itertools
from typing import Iterator, Sequence
from .model_data import VaticModelData


class VaticSimulationState:
    """A system state that can be updated with data from RUCs and SCEDs."""

    def __init__(self,
                 ruc_execution_hour: int, ruc_every_hours: int,
                 sced_frequency_minutes: int):
        self._forecasts = None
        self._actuals = None
        self._commits = dict()

        self._init_gen_state = dict()
        self._init_power_gen = dict()
        self._init_soc = dict()

        # timestep durations; how often a SCED is run
        self._minutes_per_forecast_step = 60
        self._minutes_per_actuals_step = 60
        self._sced_frequency = 60

        # the current simulation minute and the next minute when forecasts
        # should be popped
        self._simulation_minute = 0
        self._next_forecast_pop_minute = 0
        self._next_actuals_pop_minute = 0

        # TODO: should this be here or in the time manager?
        self.ruc_delay = -(ruc_execution_hour % -ruc_every_hours)
        self._sced_frequency = sced_frequency_minutes

    @property
    def timestep_count(self) -> int:
        """The number of timesteps we have data for."""

        if len(self._forecasts) > 0:
            steps = len(tuple(self._forecasts.values())[0])
        else:
            steps = 0

        return steps

    @property
    def minutes_per_step(self) -> int:
        """The duration of each time step in minutes."""
        return self._minutes_per_forecast_step

    def get_generator_commitment(self, g: str, time_index: int) -> int:
        """Is the gen committed to be on (1) or off (0) for a time step?"""
        return self._commits[g][time_index]

    def get_initial_generator_state(self, g: str) -> float:
        """Get the generator's state in the previous time period."""
        return self._init_gen_state[g]

    def get_initial_power_generated(self, g: str) -> float:
        """Get how much power was generated in the previous period."""
        return self._init_power_gen[g]

    def get_initial_state_of_charge(self, s: str) -> float:
        """Get state of charge in the previous time period."""
        return self._init_soc[s]

    def get_current_actuals(self, k: tuple[str, str]) -> float:
        """Get the current actual value for each forecastable.

        This is the actual value for the current time period (time index 0).
        Values are returned in the same order as
        VaticModelData.get_forecastables, but instead of returning arrays it
        returns a single value.

        """
        return self._actuals[k][0]

    def get_forecasts(self, k: tuple[str, str]) -> Sequence[float]:
        """Get the forecast values for each forecastable.

        This is very similar to VaticModelData.get_forecastables(); the
        function yields an array per forecastable, in the same order as
        get_forecastables().

        Note that the value at index 0 is the forecast for the current time,
        not the actual value for the current time.

        """
        return self._forecasts[k]

    def get_future_actuals(self, k: tuple[str, str]) -> Sequence[float]:
        """Warning: Returns actual values for current time AND FUTURE TIMES.

        Be aware that this function returns information that is not yet known!
        The function lets you peek into the future.  Future actuals may be used
        by some (probably unrealistic) algorithm options, such as

        """
        return self._actuals[k]

    def apply_initial_ruc(self,
                          ruc: VaticModelData,
                          sim_actuals: VaticModelData) -> None:
        """This is the first RUC; save initial state."""

        for gen, gen_data in ruc.elements('generator',
                                          generator_type='thermal'):
            self._init_gen_state[gen] = gen_data['initial_status']
            self._init_power_gen[gen] = gen_data['initial_p_output']
            self._commits[gen] = tuple(gen_data['commitment']['values'])

        for store, store_data in ruc.elements('storage'):
            self._init_soc[store] = store_data['initial_state_of_charge']

        # if this is first RUC, also save data to indicate when to pop
        # RUC-related state
        self._minutes_per_forecast_step = ruc.get_system_attr(
            'time_period_length_minutes')
        self._next_forecast_pop_minute = self._minutes_per_forecast_step

        self._minutes_per_actuals_step = sim_actuals.get_system_attr(
            'time_period_length_minutes')
        self._next_actuals_pop_minute = self._minutes_per_actuals_step

        self._forecasts = dict(ruc.get_forecastables())
        self._actuals = dict(sim_actuals.get_forecastables())

    def apply_planning_ruc(self,
                           ruc: VaticModelData,
                           sim_actuals: VaticModelData) -> None:
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
            self._commits[gen] = self._commits[gen][:self.ruc_delay]
            self._commits[gen] += tuple(gen_data['commitment']['values'])

        for k, new_ruc_vals in ruc.get_forecastables():
            self._forecasts[k] = self._forecasts[k][:self.ruc_delay]
            self._forecasts[k] += tuple(new_ruc_vals)

        for k, new_ruc_vals in sim_actuals.get_forecastables():
            self._actuals[k] = self._actuals[k][:self.ruc_delay]
            self._actuals[k] += tuple(new_ruc_vals)

    def apply_sced(self, sced: VaticModelData) -> None:
        """Merge a sced into the current state, and move to next time period.

        This saves the sced's first time period of data as initial state
        information, and advances the current time forward by one time period.

        """
        for gen, status, generated in self.get_generator_states_at_sced_offset(
                sced, 0):
            self._init_gen_state[gen] = status
            self._init_power_gen[gen] = generated

        for store, store_data in sced.elements('storage'):
            self._init_soc[store] = store_data['state_of_charge']['values'][0]

        # Advance time, dropping data if necessary
        self._simulation_minute += self._sced_frequency

        while self._next_forecast_pop_minute <= self._simulation_minute:
            for k in self._forecasts:
                self._forecasts[k] = self._forecasts[k][1:]

            for gen in self._commits:
                self._commits[gen] = self._commits[gen][1:]

            self._next_forecast_pop_minute += self._minutes_per_forecast_step

        while self._simulation_minute >= self._next_actuals_pop_minute:
            for k in self._actuals:
                self._actuals[k] = self._actuals[k][1:]

            self._next_actuals_pop_minute += self._minutes_per_actuals_step

    def get_generator_states_at_sced_offset(
            self, sced: VaticModelData, sced_index: int) -> tuple:
        # We'll be converting between time periods and hours.
        # Make the data type of hours_per_period an int if it's an integer
        # number of hours, float if fractional

        minutes_per_period = sced.get_system_attr('time_period_length_minutes')
        hours_per_period = minutes_per_period // 60 if minutes_per_period % 60 == 0 \
            else minutes_per_period / 60

        for g, g_dict in sced.elements('generator', generator_type='thermal'):
            ### Get generator state (whether on or off, and for how long) ###
            init_state = self.get_initial_generator_state(g)
            # state is in hours, convert to integer number of time periods
            init_state = round(init_state / hours_per_period)
            state_duration = abs(init_state)
            unit_on = init_state > 0
            g_commit = g_dict['commitment']['values']

            # March forward, counting how long the state is on or off
            for t in range(0, sced_index + 1):
                new_on = (int(round(g_commit[t])) > 0)
                if new_on == unit_on:
                    state_duration += 1
                else:
                    state_duration = 1
                    unit_on = new_on

            if not unit_on:
                state_duration = -state_duration

            # Convert duration back into hours
            state_duration *= hours_per_period

            ### Get how much power was generated, within bounds ###
            power_generated = g_dict['pg']['values'][sced_index]

            # the validators are rather picky, in that tolerances are not
            # acceptable. given that the average power generated comes from an
            # optimization problem solve, the average power generated can wind
            # up being less than or greater than the bounds by a small epsilon.
            # touch-up in this case.
            if isinstance(g_dict['p_min'], dict):
                min_power_output = g_dict['p_min']['values'][sced_index]
            else:
                min_power_output = g_dict['p_min']
            if isinstance(g_dict['p_max'], dict):
                max_power_output = g_dict['p_max']['values'][sced_index]
            else:
                max_power_output = g_dict['p_max']

            # TBD: Eventually make the 1e-5 an user-settable option.
            if unit_on == 0:
                # if the unit is off, then the power generated at
                # t0 must be greater than or equal to 0 (Egret #219)
                power_generated = max(power_generated, 0.0)
            elif math.isclose(min_power_output, power_generated, rel_tol=0,
                              abs_tol=1e-5):
                power_generated = min_power_output
            elif math.isclose(max_power_output, power_generated, rel_tol=0,
                              abs_tol=1e-5):
                power_generated = max_power_output

            ### Yield the results for this generator ###
            yield g, state_duration, power_generated

    @staticmethod
    def get_storage_socs_at_sced_offset(sced: VaticModelData,
                                        sced_index: int) -> Iterator:
        for store, store_data in sced.elements('storage'):
            yield store, store_data['state_of_charge']['values'][sced_index]


class VaticStateWithOffset:
    """Get expected state some number of time steps from the current state.

    The offset state is identical to the state being offset, except that time
    periods before the offset time are skipped.
    """

    def __init__(self, parent_state: VaticSimulationState, offset: int):
        self._parent = parent_state
        self._offset = offset

    @property
    def timestep_count(self) -> int:
        """The number of timesteps we have data for."""
        return self._parent.timestep_count - self._offset

    def get_generator_commitment(self, g: str, time_index: int) -> int:
        """Is the gen committed to be on (1) or off (0) for a time step?"""
        return self._parent.get_generator_commitment(g,
                                                     time_index + self._offset)

    def get_current_actuals(self, k: tuple[str, str]) -> float:
        """Get the current actual value for each forecastable.

        This is the actual value for the current time period (time index 0).
        Values are returned in the same order as
        VaticModelData.get_forecastables, but instead of returning arrays it
        returns a single value.

        """
        return self._parent.get_future_actuals(k)[self._offset]

    def get_forecasts(self, k: tuple[str, str]) -> Sequence[float]:
        """Get the forecast values for each forecastable.

        This is very similar to VaticModelData.get_forecastables(); the
        function yields an array per forecastable, in the same order as
        get_forecastables().

        Note that the value at index 0 is the forecast for the current time,
        not the actual value for the current time.

        """
        return list(itertools.islice(self._parent.get_forecasts(k),
                                     self._offset, None))

    def get_future_actuals(self, k: tuple[str, str]) -> Sequence[float]:
        """Warning: Returns actual values for current time AND FUTURE TIMES.

        Be aware that this function returns information that is not yet known!
        The function lets you peek into the future.  Future actuals may be used
        by some (probably unrealistic) algorithm options, such as

        """
        return list(itertools.islice(self._parent.get_future_actuals(k),
                                     self._offset, None))


class VaticStateWithScedOffset(VaticStateWithOffset, VaticSimulationState):
    """Get the future expected state, using a SCED for the initial state.

    The offset state is identical to the state being offset, except that time
    periods before the offset time are skipped, and the initial state of
    generators and storage is provided by a sced instance.  The sced instance
    is also offset, so that the initial state comes from the Nth time period
    of the sced.

    Args
    ----
        parent_state    The state to project into the future.
        sced    A sced instance whose state after the offset is used as
                the initial state.
        offset  The number of time periods into the future this state
                should reflect.
    """

    def __init__(self,
                 parent_state: VaticSimulationState, sced: VaticModelData,
                 offset: int) -> None:
        self._init_gen_state = dict()
        self._init_power_gen = dict()
        self._init_soc = dict()

        for gen, status, generated in parent_state\
                .get_generator_states_at_sced_offset(sced, offset - 1):
            self._init_gen_state[gen] = status
            self._init_power_gen[gen] = generated

        for store, soc in self.get_storage_socs_at_sced_offset(sced,
                                                               offset - 1):
            self._init_soc[store] = soc

        super().__init__(parent_state, offset)
