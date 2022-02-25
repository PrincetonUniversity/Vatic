
import dill as pickle
from pathlib import Path
from copy import copy, deepcopy

from typing import (Any, TypeVar, Union, Optional, Iterable, Iterator,
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
            self._data = deepcopy(source)

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

    def clone_in_service(self) -> VModelData:
        """Returns a version of this grid without out-of-service elements."""

        new_dict = {'system': self._data['system'], 'elements': dict()}
        for element_type, elements in self._data['elements'].items():
            new_dict['elements'][element_type] = dict()

            # only copy elements which are in service or for which service
            # status is not defined in the first place
            for name, elem in elements.items():
                if 'in_service' not in elem or elem['in_service']:
                    new_dict['elements'][element_type][name] = elem

        return VaticModelData(new_dict)

    def get_reserve_requirement(self, time_index):
        return self._data[
            'system']['reserve_requirement']['values'][time_index]

    def attributes(self, element_type: str, **element_args) -> dict:
        if element_type not in self._data['elements']:
            raise ModelError("This model does not include the element "
                             "type `{}`!".format(element_type))

        attr_dict = {'names': list()}
        for name, elem in self.elements(element_type, **element_args):
            attr_dict['names'].append(name)

            for attr, value in elem.items():
                if attr not in attr_dict:
                    attr_dict[attr] = {name: value}
                else:
                    attr_dict[attr][name] = value

        return attr_dict

    def elements(self,
                 element_type: str,
                 **element_args) -> Iterator[Tuple[str, Dict]]:
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

    def get_system_attr(self, attr: str, default: Optional[Any] = None) -> Any:
        if attr in self._data['system']:
            return self._data['system'][attr]

        elif default is None:
            raise ModelError("This model does not include the system-level "
                             "attribute `{}`!".format(attr))

        else:
            return default

    def set_system_attr(self, attr: str, value: Any) -> None:
        self._data['system'][attr] = value

    def get_forecastables(self) -> Iterator[Tuple[Tuple[str, str], List[float]]]:
        """Retrieves grid elements' timeseries that can be forecast."""

        for gen, gen_data in self.elements('generator',
                                           generator_type='renewable'):
            yield ('p_min', gen), gen_data['p_min']['values']
            yield ('p_max', gen), gen_data['p_max']['values']

        for bus, bus_data in self.elements('load'):
            yield ('p_load', bus), bus_data['p_load']['values']

        yield ('req', 'reserve'), self._data[
            'system']['reserve_requirement']['values']

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

    @property
    def model_runtime(self):
        return self._data['system']['solver_runtime']

    @property
    def duration_minutes(self):
        return self._data['system']['time_period_length_minutes']

    def get_max_power_output(self, gen):
        gen_data = self._data['elements']['generator'][gen]

        if isinstance(gen_data['p_max'], dict):
            return gen_data['p_max']['values'][0]
        else:
            return gen_data['p_max']

    @property
    def thermal_fleet_capacity(self):
        return sum(self.get_max_power_output(gen)
                   for gen, _ in self.elements(element_type='generator',
                                               generator_type='thermal'))

    @property
    def total_demand(self):
        return sum(load_data['p_load']['values'][0]
                   for load_data in self._data['elements']['load'].values())

    @property
    def fixed_costs(self):
        return sum(gen_data['commitment_cost']['values'][0]
                   for _, gen_data in self.elements(element_type='generator',
                                                    generator_type='thermal'))

    #TODO: allow for renewables to also have variable costs?
    @property
    def variable_costs(self):
        return sum(gen_data['production_cost']['values'][0]
                   for _, gen_data in self.elements(element_type='generator',
                                                    generator_type='thermal'))

    @property
    def total_costs(self):
        return self.fixed_costs + self.variable_costs

    @property
    def thermal_generation(self):
        return {gen: gen_data['pg']['values'][0]
                for gen, gen_data in self.elements(element_type='generator',
                                                   generator_type='thermal')}

    @property
    def renewable_generation(self):
        return {
            gen: gen_data['pg']['values'][0]
            for gen, gen_data in self.elements(element_type='generator',
                                               generator_type='renewable')
            }

    @property
    def load_shedding(self):
        return sum(-bus_data['p_balance_violation']['values'][0]
                   for bus_data in self._data['elements']['bus'].values()
                   if bus_data['p_balance_violation']['values'][0] < 0.)

    @property
    def over_generation(self):
        return sum(bus_data['p_balance_violation']['values'][0]
                   for bus_data in self._data['elements']['bus'].values()
                   if bus_data['p_balance_violation']['values'][0] > 0.)

    @property
    def reserve_shortfall(self):
        if 'reserve_shortfall' in self._data['system']:
            return self._data['system']['reserve_shortfall']['values'][0]
        else:
            return 0.

    @property
    def reserve_requirement(self):
        if 'reserve_requirement' in self._data['system']:
            return self._data['system']['reserve_requirement']['values'][0]
        else:
            return 0.

    @property
    def reserve_RT_price(self):
        if 'reserve_price' in self._data['system']:
            return self._data['system']['reserve_price']['values'][0]
        else:
            return 0.

    @property
    def available_reserve(self):
        return {gen: (gen_data['headroom']['values'][0]
                      + gen_data['rg']['values'][0])
                for gen, gen_data in self.elements(element_type='generator',
                                                   generator_type='thermal')}

    @property
    def available_quickstart(self):
        return sum(
            min(self._data['elements']['generator'][gen]['startup_capacity'],
                self.get_max_power_output(gen))
            for gen, gen_data in self.elements(element_type='generator',
                                               fast_start=True)

            if (not self.is_generator_on(gen)
                and not self.was_generator_on(gen)
                and self._data['elements']['generator'][gen][
                    'min_up_time'] <= 1)
            )

    def is_generator_on(self, gen: str) -> bool:
        if gen not in self._data['elements']['generator']:
            raise ValueError(
                "Generator `{}` is not part of this model!".format(gen))

        gen_dict = self._data['elements']['generator'][gen]
        if 'fixed_commitment' in self._data['elements']['generator'][gen]:
            return gen_dict['fixed_commitment']['values'][0] > 0
        elif 'commitment' in gen_dict:
            return gen_dict['commitment']['values'][0] > 0

        else:
            raise ModelError("Cannot find commitment status "
                             "for generator `{}`!".format(gen))

    def was_generator_on(self, gen: str) -> bool:
        return self._data['elements']['generator'][gen]['initial_status'] > 0

    @property
    def available_renewables(self):
        return sum(self.get_max_power_output(gen)
                   for gen, _ in self.elements(element_type='generator',
                                               generator_type='renewable'))

    @property
    def on_offs(self):
        return sum(self.is_generator_on(gen) != self.was_generator_on(gen)
                   for gen, _ in self.elements(element_type='generator',
                                               generator_type='thermal'))

    @property
    def on_off_ramps(self):
        return sum(gen_data['pg']['values'][0]
                   for gen, gen_data in self.elements(element_type='generator',
                                                      generator_type='thermal')
                   if self.is_generator_on(gen) != self.was_generator_on(gen))

    @property
    def nominal_ramps(self):
        return sum(abs(gen_data['pg']['values'][0]
                       - gen_data['initial_p_output'])
                   for gen, gen_data in self.elements(element_type='generator',
                                                      generator_type='thermal')
                   if self.is_generator_on(gen) == self.was_generator_on(gen))

    @property
    def price(self):
        if self.total_demand > 0:
            return ((self.total_costs / self.total_demand)
                    * (60 / self.duration_minutes))

        else:
            return 0.

    @property
    def quickstart_generators(self):
        return [gen for gen, _ in self.elements(element_type='generator',
                                                fast_start=True)]

    @property
    def quickstart_capable(self):
        quick_starts = set(self.quickstart_generators)

        return {gen: gen in quick_starts
                for gen, _ in self.elements(element_type='generator',
                                            generator_type='thermal')}

    @property
    def fuels(self):
        return {gen: gen_data['fuel']
                for gen, gen_data in self._data['elements']['generator'].items()}

    @property
    def thermal_states(self):
        return {gen: self.is_generator_on(gen)
                for gen, _ in self.elements(element_type='generator',
                                            generator_type='thermal')}

    def get_generator_cost(self, gen):
        gen_data = self._data['elements']['generator'][gen]

        return (gen_data['commitment_cost']['values'][0]
                + gen_data['production_cost']['values'][0])

    @property
    def generator_costs(self):
        return {gen: self.get_generator_cost(gen)
                for gen, _ in self.elements(element_type='generator',
                                            generator_type='thermal')}

    @property
    def curtailment(self):
        return {gen: (self.get_max_power_output(gen)
                      - gen_data['pg']['values'][0])
                for gen, gen_data in self.elements(element_type='generator',
                                                   generator_type='renewable')}

    @property
    def flows(self):
        return {line: line_data['pf']['values'][0]
                for line, line_data in self.elements('branch')}

    @property
    def bus_demands(self):
        return {bus: bus_data['p_load']['values'][0]
                for bus, bus_data in self._data['elements']['load'].items()}

    @property
    def bus_mismatches(self):
        return {bus: bus_data['p_balance_violation']['values'][0]
                for bus, bus_data in self._data['elements']['bus'].items()}

    @property
    def bus_LMPs(self):
        return {bus: bus_data['lmp']['values'][0]
                for bus, bus_data in self._data['elements']['bus'].items()}

    @property
    def storage_inputs(self):
        return {
            store: store_data['p_charge']['values'][0]
            for store, store_data in self._data['elements']['storage'].items()
            }

    @property
    def storage_outputs(self):
        return {
            store: store_data['p_discharge']['values'][0]
            for store, store_data in self._data['elements']['storage'].items()
            }

    @property
    def storage_states(self):
        return {
            store: store_data['state_of_charge']['values'][0]
            for store, store_data in self._data['elements']['storage'].items()
            }

    @property
    def storage_types(self):
        for store, store_data in self._data['elements']['storage'].items():
            pass

        return

    def to_egret(self) -> EgretModel:
        return EgretModel(deepcopy(self._data))
