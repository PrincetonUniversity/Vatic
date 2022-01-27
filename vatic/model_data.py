
import dill as pickle
from pathlib import Path
from copy import copy, deepcopy

from typing import (TypeVar, Union, Optional, Iterable, Iterator,
                    List, Tuple, Dict)
VModelData = TypeVar('VModelData', bound='VaticModelData')

from egret.data.model_data import ModelData as EgretModel


class ModelError(Exception):
    pass


class VaticModelData(object):
    """Simplified version of egret.data.model_data.ModelData"""

    def __init__(self,
                 source: Union[dict, VModelData, str, Path,
                               None] = None) -> None:
        if isinstance(source, dict):
            self._data = source

        elif isinstance(source, VaticModelData):
            self._data = deepcopy(source._data)

        elif isinstance(source, (str, Path)):
            if not Path(source).is_file():
                raise ModelError(
                    "Cannot find model input file `{}`!".format(source))

            with open(source, 'rb') as f:
                self._data = pickle.load(f)

        elif source is None:
            self._data = {'system': dict(), 'elements': dict()}

        else:
            raise ValueError("Unrecognized source for ModelData")

    def __copy__(self):
        return VaticModelData(copy(self._data))

    def __deepcopy__(self, memo):
        return VaticModelData(deepcopy(self._data))

    def get_reserve_requirement(self, time_index):
        return self._data[
            'system']['reserve_requirement']['values'][time_index]

    def elements(self,
                 element_type: str,
                 **element_args) -> Iterator[Tuple[str, Dict]]:
        if element_type not in self._data['elements']:
            raise ModelError("This model does not include the element "
                             "type `{}`!".format(element_type))

        for name, elem in self._data['elements'][element_type].items():
            if all(k in elem and elem[k] == v
                   for k, v in element_args.items()):
                yield name, elem

    def get_forecastables(self) -> Iterator[List[float]]:
        for gen, gen_data in self.elements('generator',
                                           generator_type='renewable'):
            yield gen_data['p_min']['values']
            yield gen_data['p_max']['values']

        for bus, bus_data in self.elements('load'):
            yield bus_data['p_load']['values']

        yield self._data['system']['reserve_requirement']['values']

    def time_series(self,
                    element_types: Optional[Iterable[str]] = None,
                    include_reserves=True, **element_args) -> Iterator[Tuple]:

        if element_types is None:
            element_types = list(self._data['elements'])

        else:
            for element_type in element_types:
                if element_type not in self._data['elements']:
                    raise ModelError("This model does not include the element "
                                     "type `{}`!".format(element_type))

        for element_type in element_types:
            for name, elem in self.elements(element_type, **element_args):
                for k, v in elem.items():
                    if (isinstance(v, dict) and 'data_type' in v
                            and v['data_type'] == 'time_series'):
                        yield (name, k), v

        if include_reserves:
            yield (('system', 'reserve_requirement'),
                   self._data['system']['reserve_requirement'])

    def copy_elements(self,
                      other: VModelData, element_type, attrs=None,
                      strict_mode=False, **element_args) -> None:

        for name, elem in other.elements(element_type, **element_args):
            if name in self._data['elements'][element_type]:
                if attrs:
                    for attr in attrs:
                        self._data['elements'][element_type][name][attr] \
                            = deepcopy(elem[attr])
                else:
                    self._data['elements'][element_type][name] = deepcopy(elem)

            elif strict_mode:
                raise ModelError("Could not copy from other VaticModelData "
                                 "object which contains the missing "
                                 "element `{}`!".format(name))

    def copy_forecastables(self, other: VModelData,
                           time_index, other_time_index) -> None:

        for gen, gen_data in other.elements('generator',
                                            generator_type='renewable'):
            for k in ['p_min', 'p_max']:
                self._data[
                    'elements']['generator'][gen][k]['values'][time_index] \
                        = gen_data[k]['values'][other_time_index]

        for bus, bus_data in other.elements('load'):
            self._data[
                'elements']['load'][bus]['p_load']['values'][time_index] \
                    = bus_data['p_load']['values'][other_time_index]

        self._data['system']['reserve_requirement']['values'][time_index] \
            = other.get_reserve_requirement(other_time_index)

    def reset_timeseries(self) -> None:
        self._data['system']['time_keys'] = list()
        self._data['system']['time_period_length_minutes'] = None

        for _, time_entry in self.time_series():
            time_entry['values'] = list()

    def set_time_steps(self,
                       time_steps: Union[int, List[int]],
                       period_minutes: int) -> None:

        if isinstance(time_steps, int):
            self._data['system']['time_keys'] = [
                str(i + 1) for i in range(time_steps)]

        else:
            self._data['system']['time_keys'] = time_steps

        self._data['system']['time_period_length_minutes'] = period_minutes

        for _, time_entry in self.time_series():
            time_entry['values'] = [None
                                    for _ in self._data['system']['time_keys']]

    def honor_reserve_factor(self, reserve_factor: float, time_index: int):
        if reserve_factor > 0:
            total_load = sum(bus_data['p_load']['values'][time_index]
                             for bus, bus_data in self.elements('load'))

            cur_req = self._data[
                'system']['reserve_requirement']['values'][time_index]
            self._data['system']['reserve_requirement']['values'][time_index] \
                = max(reserve_factor * total_load, cur_req)

    def to_egret(self) -> EgretModel:
        return EgretModel(deepcopy(self._data))
