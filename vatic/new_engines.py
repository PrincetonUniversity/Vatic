
import pandas as pd
from datetime import datetime
from typing import Dict, Union
from .engines import Simulator


class RewindSimulator(Simulator):

    def __init__(self,
                 template_data: dict, gen_data: pd.DataFrame,
                 load_data: pd.DataFrame, **siml_args) -> None:
        super().__init__(template_data, gen_data, load_data, **siml_args)

        self._simulation_states = dict()

    def call_oracle(self) -> None:
        self._simulation_states[
            self._current_timestep] = self._simulation_state

        super().call_oracle()

    def perturb_timestep(self,
                         perturb_dict: Dict[str, float],
                         datehour: datetime) -> Dict[str, Union[float, dict]]:
        """Rewind the simulation to a previous time and run a perturbation."""
        orig_state = self._simulation_state

        for timestep in self._simulation_states:
            if timestep.when == datehour:
                self._simulation_state = self._simulation_states[timestep]
                break

        sced_stats = super().perturb_oracle(perturb_dict)
        self._simulation_state = orig_state

        return sced_stats
