
from copy import deepcopy
import pandas as pd
from datetime import datetime
from typing import Dict, Union
from .engines import Simulator


class RewindSimulator(Simulator):
    """A grid simulation engine that can save and reload its past states."""

    def __init__(self,
                 template_data: dict, gen_data: pd.DataFrame,
                 load_data: pd.DataFrame, **siml_args) -> None:
        super().__init__(template_data, gen_data, load_data, **siml_args)

        self._simulation_states = dict()

    def call_oracle(self) -> None:
        self._simulation_states[self._current_timestep] = deepcopy(
            self._simulation_state)

        super().call_oracle()

    def perturb_timestep(
            self,
            perturb_dict: Dict[str, float], datehour: datetime,
            run_lmps: bool = False
            ) -> Dict[str, Union[float, dict]]:
        """Rewind the simulation to a previous time and run a perturbation."""
        orig_state = deepcopy(self._simulation_state)

        # find the simulation timestep matching the given time
        for timestep in self._simulation_states:
            if timestep.when == datehour:
                self._simulation_state = self._simulation_states[timestep]
                break

        else:
            raise ValueError("No simulation states available for the "
                             "given timestep `{}`!".format(datehour))

        # run the perturbation using the given timestep and then return the
        # state of the simulation to its original state
        sced_stats = super().perturb_oracle(perturb_dict, run_lmps)
        self._simulation_state = orig_state

        return sced_stats
