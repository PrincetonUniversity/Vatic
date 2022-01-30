
from prescient.engine.forecast_helper import get_forecastables
from prescient.data.simulation_state.time_interpolated_state import TimeInterpolatedState
from egret.data.model_data import ModelData as EgretModel

from typing import Iterable, Sequence, Tuple
import math

from prescient.engine.abstract_types import G, S


class VaticSimulationState:
    ''' A simulation state that can be updated with data pulled from RUCs and sceds.
    '''

    def __init__(self, options):
        self._forecasts = list()
        self._actuals = list()
        self._commits = dict()

        self._init_gen_state = dict()
        self._init_power_gen = dict()
        self._init_soc = dict()

        # Timestep durations
        self._minutes_per_forecast_step = 60
        self._minutes_per_actuals_step = 60
        # How often a SCED is run
        self._sced_frequency = 60

        # The current simulation minute
        self._simulation_minute = 0
        # Next simulation minute when forecasts should be popped
        self._next_forecast_pop_minute = 0
        self._next_actuals_pop_minute = 0

        self.ruc_delay = -(options.ruc_execution_hour
                           % (-options.ruc_every_hours))
        self._sced_frequency = options.sced_frequency_minutes

    @property
    def timestep_count(self) -> int:
        ''' The number of timesteps we have data for '''
        return len(self._forecasts[0]) if len(self._forecasts) > 0 else 0

    @property
    def minutes_per_step(self) -> int:
        ''' The duration of each time step in minutes '''
        return self._minutes_per_forecast_step

    def get_generator_commitment(self, g:G, time_index:int) -> Sequence[int]:
        ''' Get whether the generator is committed to be on (1) or off (0) for each time period
        '''
        return self._commits[g][time_index]

    def get_initial_generator_state(self, g:G) -> float:
        ''' Get the generator's state in the previous time period '''
        return self._init_gen_state[g]

    def get_initial_power_generated(self, g:G) -> float:
        ''' Get how much power was generated in the previous time period '''
        return self._init_power_gen[g]

    def get_initial_state_of_charge(self, s:S) -> float:
        ''' Get state of charge in the previous time period '''
        return self._init_soc[s]

    def get_current_actuals(self) -> Iterable[float]:
        ''' Get the current actual value for each forecastable.

        This is the actual value for the current time period (time index 0).
        Values are returned in the same order as forecast_helper.get_forecastables,
        but instead of returning arrays it returns a single value.
        '''
        for forecastable in self._actuals:
            yield forecastable[0]

    def get_forecasts(self) -> Iterable[Sequence[float]]:
        ''' Get the forecast values for each forecastable

        This is very similar to forecast_helper.get_forecastables(); the
        function yields an array per forecastable, in the same order as
        get_forecastables().

        Note that the value at index 0 is the forecast for the current time,
        not the actual value for the current time.
        '''
        for forecastable in self._forecasts:
            yield forecastable

    def get_future_actuals(self) -> Iterable[Sequence[float]]:
        ''' Warning: Returns actual values for the current time AND FUTURE TIMES.

        Be aware that this function returns information that is not yet known!
        The function lets you peek into the future.  Future actuals may be used
        by some (probably unrealistic) algorithm options, such as
        '''
        for forecastable in self._actuals:
            yield forecastable

    def apply_initial_ruc(self, ruc, sim_actuals) -> None:
        # The is the first RUC, save initial state

        for gen, gen_data in ruc.elements('generator',
                                          generator_type='thermal'):
            self._init_gen_state[gen] = gen_data['initial_status']
            self._init_power_gen[gen] = gen_data['initial_p_output']
            self._commits[gen] = tuple(gen_data['commitment']['values'])

        for store, store_data in ruc.elements('storage'):
            self._init_soc[store] = store_data['initial_state_of_charge']

        # If this is first RUC, also save data to indicate when to pop RUC-related state
        self._minutes_per_forecast_step = ruc.data['system'][
            'time_period_length_minutes']
        self._next_forecast_pop_minute = self._minutes_per_forecast_step

        self._minutes_per_actuals_step = sim_actuals.data['system'][
            'time_period_length_minutes']
        self._next_actuals_pop_minute = self._minutes_per_actuals_step

        self._forecasts = [new_ruc_vals
                           for (new_ruc_vals, ) in get_forecastables(ruc)]
        self._actuals = [new_ruc_vals
                         for (new_ruc_vals, ) in get_forecastables(sim_actuals)]

    def apply_planning_ruc(self, ruc, sim_actuals) -> None:
        ''' Incorporate a RUC instance into the current state.

        This will save the ruc's forecasts, and for the very first ruc
        this will also save initial state info.

        If there is a ruc delay, as indicated by options.ruc_execution_hour and
        options.ruc_every_hours, then the RUC is applied to future time periods,
        offset by the ruc delay.  This does not apply to the very first RUC, which
        is used to set up the initial simulation state with no offset.
        '''


        # Now save all generator commitments
        # Keep the first "ruc_delay" commitments from the prior ruc
        for gen, gen_data in ruc.elements('generator',
                                          generator_type='thermal'):
            self._commits[gen] = self._commits[gen][:self.ruc_delay]
            self._commits[gen] += tuple(gen_data['commitment']['values'])

        for i, (new_ruc_vals, ) in enumerate(get_forecastables(ruc)):
            self._forecasts[i] = self._forecasts[i][:self.ruc_delay]
            self._forecasts[i] += tuple(new_ruc_vals)

        for i, (new_ruc_vals, ) in enumerate(get_forecastables(sim_actuals)):
            self._actuals[i] = self._actuals[i][:self.ruc_delay]
            self._actuals[i] += tuple(new_ruc_vals)

    def apply_sced(self, sced) -> None:
        ''' Incorporate a sced's results into the current state, and move to the next time period.

        This saves the sced's first time period of data as initial state information,
        and advances the current time forward by one time period.
        '''
        for gen, status, generated in self.get_generator_states_at_sced_offset(
                sced, 0):
            self._init_gen_state[gen] = status
            self._init_power_gen[gen] = generated

        for store, store_data in sced.elements('storage'):
            self._init_soc[store] = store_data['state_of_charge']['values'][0]

        # Advance time, dropping data if necessary
        self._simulation_minute += self._sced_frequency

        while self._next_forecast_pop_minute <= self._simulation_minute:
            for i in range(len(self._forecasts)):
                self._forecasts[i] = self._forecasts[i][1:]

            for gen in self._commits:
                self._commits[gen] = self._commits[gen][1:]

            self._next_forecast_pop_minute += self._minutes_per_forecast_step

        while self._simulation_minute >= self._next_actuals_pop_minute:
            for i in range(len(self._actuals)):
                self._actuals[i] = self._actuals[i][1:]

            self._next_actuals_pop_minute += self._minutes_per_actuals_step

    def get_state_with_step_length(self, minutes_per_step:int):
        # If our data matches what's stored here, no need to create an interpolated view
        if self._minutes_per_forecast_step == minutes_per_step and \
           self._minutes_per_actuals_step == minutes_per_step and \
           self._sced_frequency == minutes_per_step:
            return self

        # Found out what fraction past the first step of each type we currently are
        minutes_past_forecast = self._simulation_minute - self._next_forecast_pop_minute + self._minutes_per_forecast_step
        minutes_past_actuals = self._simulation_minute - self._next_actuals_pop_minute + self._minutes_per_actuals_step
        return TimeInterpolatedState(self, self._minutes_per_forecast_step, minutes_past_forecast,
                                     self._minutes_per_actuals_step, minutes_past_actuals,
                                     minutes_per_step)

    def get_generator_states_at_sced_offset(
            self, sced: EgretModel, sced_index: int) -> Tuple:
        # We'll be converting between time periods and hours.
        # Make the data type of hours_per_period an int if it's an integer number of hours, float if fractional

        minutes_per_period = sced.data['system']['time_period_length_minutes']
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

            # the validators are rather picky, in that tolerances are not acceptable.
            # given that the average power generated comes from an optimization
            # problem solve, the average power generated can wind up being less
            # than or greater than the bounds by a small epsilon. touch-up in this
            # case.
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
