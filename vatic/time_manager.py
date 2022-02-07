
from datetime import timedelta, datetime, time, date
from typing import Iterator


class VaticTime:

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

    def __str__(self):
        return self.when.isoformat(sep=' ', timespec='minutes')


class VaticTimeManager:

    def __init__(self, start_date, num_days, options):
        self.start_date = start_date
        self.end_date = self.start_date + timedelta(days=num_days)

        if 60 % options.sced_frequency_minutes != 0:
            raise ValueError(
                "Given SCED frequency ({} minutes) does not divide evenly "
                "into an hour!".format(options.sced_frequency_minutes)
                )

        if 24 % options.ruc_every_hours != 0:
            raise ValueError(
                "Given RUC frequency (every {} hours) does not divide evenly "
                "into a day!".format(options.ruc_every_hours)
                )

        if not options.ruc_every_hours <= options.ruc_horizon <= 48:
            raise ValueError(
                "Given RUC horizon ({} hours) is not between the given "
                "RUC frequency interval ({} hours) "
                "and 2 days!".format(options.ruc_horizon,
                                     options.ruc_every_hours)
                )

        self.sced_interval = timedelta(minutes=options.sced_frequency_minutes)
        self.ruc_delay = timedelta(hours=-(options.ruc_execution_hour
                                            % -options.ruc_every_hours))
        self.ruc_interval = timedelta(hours=options.ruc_every_hours)

        self._current_time = None

    def time_steps(self) -> Iterator[VaticTime]:
        current_time = datetime.combine(self.start_date, time(0))
        end_time = datetime.combine(self.end_date, time(0))

        next_activation_time = current_time + self.ruc_interval
        next_planning_time = next_activation_time - self.ruc_delay

        while current_time < end_time:
            is_planning_time = current_time == next_planning_time

            if is_planning_time:
                next_planning_time += self.ruc_interval

                if (next_planning_time + self.ruc_delay) >= end_time:
                    next_planning_time = end_time

            is_activation_time = current_time == next_activation_time
            if is_activation_time:
                next_activation_time += self.ruc_interval

            time_step = VaticTime(current_time,
                                  is_planning_time, is_activation_time)

            self._current_time = time_step
            yield time_step

            current_time += self.sced_interval

    def get_first_timestep(self) -> VaticTime:
        return VaticTime(datetime.combine(self.start_date, time(0)),
                         False, False)

    def get_uc_activation_time(self, time_step: VaticTime) -> datetime:
        """
        Get the hour and date that a RUC generated at the given
        time will be activated.
        """
        return time_step.when + self.ruc_delay
