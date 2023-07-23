"""Classes for representing and producing the time steps of a simulation."""

from datetime import timedelta, datetime, time, date
from typing import Iterator


class GridTimeStep:
    """A single time step of a simulation.

    This class represents a time point within a run of a simulation engine and
    consists of a datetime object along with annotations regarding whether the
    engine is doing something other than just running a SCED at this time. Note
    that the engine usually runs an "initialization" RUC at the very first time
    point which is not included here.

    See prescient.simulator.time_manager.PrescientTime for the original
    implementation of this class.

    Args
    ----
        when    The date and the time this instance represents.
        is_planning_time    Is a day-ahead RUC being run at this time point?
        is_ruc_activation_time      Are the commitments made by the previous
                                    day-ahead RUC being loaded into the
                                    simulation state at this time point?
    """

    def __init__(self,
                 when: datetime,
                 is_planning_time: bool, is_ruc_activation_time: bool) -> None:
        self.when = when
        self.is_planning_time = is_planning_time
        self.is_ruc_activation_time = is_ruc_activation_time

    def date(self) -> date:
        return self.when.date()

    def time(self) -> datetime.time:
        return self.when.time()

    def hour(self) -> int:
        return self.when.hour

    def labels(self) -> dict:
        return {'Date': self.when.date().isoformat(),
                'Hour': self.when.hour, 'Minute': self.when.minute}

    def __str__(self):
        time_strs = [self.when.isoformat(sep=' ', timespec='minutes')]

        if self.is_planning_time:
            time_strs += ["(planning)"]
        if self.is_ruc_activation_time:
            time_strs += ["(ruc-activation)"]

        return ' '.join(time_strs)

    def __repr__(self):
        time_strs = [repr(self.when)]

        if self.is_planning_time:
            time_strs += ["(planning)"]
        else:
            time_strs += ["(no planning)"]

        if self.is_ruc_activation_time:
            time_strs += ["(ruc-activation)"]
        else:
            time_strs += ["(no ruc-activation)"]

        return ' '.join(time_strs)


class VaticTimeManager:
    """The clock of a simulation run, governing time steps and planning RUCs.

    This class keeps track of the time points over which a run of the
    simulation engine will iterate, as well as determining at which points the
    day-ahead RUCs will run and become active.

    See prescient.simulator.time_manager.TimeManager for the original
    implementation of this class.

    Args
    ----
        start_date  The first day of the simulation run.
        num_days    How many days the simulation will run for.
    """

    def __init__(self,
                 start_date: date, end_date: date, ruc_execution_hour: int,
                 ruc_every_hours: int, ruc_horizon: int,
                 sced_frequency_minutes: int) -> None:
        self.start_date = start_date
        self.end_date = end_date

        if 60 % sced_frequency_minutes != 0:
            raise ValueError(
                "Given SCED frequency ({} minutes) does not divide evenly "
                "into an hour!".format(sced_frequency_minutes)
                )

        if 24 % ruc_every_hours != 0:
            raise ValueError(
                "Given RUC frequency (every {} hours) does not divide evenly "
                "into a day!".format(ruc_every_hours)
                )

        if not ruc_every_hours <= ruc_horizon <= 48:
            raise ValueError(
                "Given RUC horizon ({} hours) is not between the given "
                "RUC frequency interval ({} hours) "
                "and 2 days!".format(ruc_horizon, ruc_every_hours)
                )

        self.sced_interval = timedelta(minutes=sced_frequency_minutes)
        self.ruc_delay = timedelta(
            hours=-(ruc_execution_hour % -ruc_every_hours))
        self.ruc_interval = timedelta(hours=ruc_every_hours)

        self._current_time = None

    def time_steps(self) -> Iterator[GridTimeStep]:
        """Produces the time points over which a simulation run iterates."""

        current_time = datetime.combine(self.start_date, time(0))
        end_time = datetime.combine(self.end_date, time(0))

        # figure out when the initial RUC will be activated and when the next
        # planning RUC is to be created
        next_activation_time = current_time + self.ruc_interval
        next_planning_time = next_activation_time - self.ruc_delay

        while current_time < end_time:
            is_planning_time = current_time == next_planning_time

            # if the next plan won't be activated until after the simulation
            # has finished, don't bother generating the plan
            is_planning_time &= (current_time + self.ruc_delay) < end_time
            if is_planning_time:
                next_planning_time += self.ruc_interval

            is_activation_time = current_time == next_activation_time
            if is_activation_time:
                next_activation_time += self.ruc_interval

            # produce a time point and update the time state of the simulation
            time_step = GridTimeStep(current_time,
                                     is_planning_time, is_activation_time)

            self._current_time = time_step
            yield time_step

            current_time += self.sced_interval

    def get_first_timestep(self) -> GridTimeStep:
        """Gets e.g. when the simulation's initialization RUC will run."""

        return GridTimeStep(datetime.combine(self.start_date, time(0)),
                            False, False)

    def get_uc_activation_time(self, time_step: GridTimeStep) -> datetime:
        """When will a RUC generated at this time point be activated?"""

        return time_step.when + self.ruc_delay
