
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
                 **element_args: str) -> Iterator[Tuple[str, Dict]]:
        """Retrieves grid elements that match a set of criteria.

        Args
        ----
            element_type    Which type of element to search within, e.g. 'bus',
                            'generator', 'load', 'branch', etc.
            element_args    A set of element property values, all of which must
                            be present in an element's data entry and equal to
                            the given value for the element's entry to be
                            returned. e.g. generator_type='renewable'
                                           bus='Chifa'

        """
        if element_type not in self._data['elements']:
            raise ModelError("This model does not include the element "
                             "type `{}`!".format(element_type))

        for name, elem in self._data['elements'][element_type].items():
            if all(k in elem and elem[k] == v
                   for k, v in element_args.items()):
                yield name, elem

    def get_forecastables(self) -> Iterator[List[float]]:
        """Retrieves grid elements' timeseries that can be forecast."""

        for gen, gen_data in self.elements('generator',
                                           generator_type='renewable'):
            yield gen_data['p_min']['values']
            yield gen_data['p_max']['values']

        for bus, bus_data in self.elements('load'):
            yield bus_data['p_load']['values']

        yield self._data['system']['reserve_requirement']['values']

    def time_series(self,
                    element_types: Optional[Iterable[str]] = None,
                    include_reserves=True,
                    **element_args: str) -> Iterator[Tuple]:
        """Retrieves timeseries for grid elements that match a set of criteria.

        Args
        ----
            element_types   Which types of element to search within,
                            e.g. 'storage', 'load', generator', 'bus', etc.
            include_reserves    Whether to return reserve requirements, which
                                will never be returned otherwise.
            element_args    A set of element property values, all of which must
                            be present in an element's data entry and equal to
                            the given value for the element's entry to be
                            returned. e.g. generator_type='thermal'
                                           in_service=True

        """
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
                      other: VModelData, element_type: str,
                      attrs: Optional[Iterable[str]] = None,
                      strict_mode=False, **element_args: str) -> None:
        """Replaces grid elements with those in another model data object.

        Args
        ----
            other   The model data object we will be copying from.
            element_type    Which type of element to copy from, e.g. 'bus',
                            'generator', 'load', 'branch', etc.
            attrs       If given, only copy data from within these element
                        entry fields, e.g. 'p_min', 'generator_type', etc.
            strict_mode     Raise an error if an element meeting the criteria
                            for copying in the other model data object is
                            not present in this model data object.

            element_args    A set of element property values, all of which must
                            be present in an element's data entry and equal to
                            the given value for the element's entry to be
                            copied. e.g. generator_type='thermal'
                                         in_service=False

        """
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

    def copy_forecastables(self,
                           other: VModelData,
                           time_index: int, other_time_index: int) -> None:
        """Replaces forecastable values with those from another model data.

        Args
        ----
            other   The model data object we will be copying from.
            time_index      Which time step to replace in this model data.
            other_time_index    Which time step to copy from in the other data.

        """
        for gen, gen_data in other.elements('generator',
                                            generator_type='renewable'):
            for k in ['p_min', 'p_max', 'p_cost']:
                if k in gen_data:
                    self._data['elements']['generator'][gen][k]['values'][
                        time_index] = gen_data[k]['values'][other_time_index]

        for bus, bus_data in other.elements('load'):
            self._data[
                'elements']['load'][bus]['p_load']['values'][time_index] \
                    = bus_data['p_load']['values'][other_time_index]

        self._data['system']['reserve_requirement']['values'][time_index] \
            = other.get_reserve_requirement(other_time_index)

    def reset_timeseries(self) -> None:
        """Replaces all timeseries entries with empty lists."""

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
