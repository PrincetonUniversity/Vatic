#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

## loads and validates input unit commitment data

import math
import pyomo.environ as pe

from egret.model_library.transmission import tx_utils
from egret.common.log import logger
from egret.model_library.unit_commitment.uc_utils import (
    add_model_attr, uc_time_helper)

from egret.model_library.unit_commitment.params import (
    _verify_must_run_t0_state_consistency,
    _add_initial_time_periods_on_off_line
    )

from ._utils import ModelError

from typing import Optional
component_name = 'data_loader'


def load_base_params(
        model: pe.ConcreteModel,
        model_data=None, renew_costs: bool = False
        ) -> pe.ConcreteModel:

    if model_data is None:
        model_data = model.model_data

    warn_neg_load = False
    time_keys = model_data.get_system_attr('time_keys')

    ## NOTE: generator, bus, and load should be in here for
    # a well-defined problem
    loads = dict(model_data.elements(element_type='load'))
    gens = dict(model_data.elements(element_type='generator'))

    thermal_gens = dict(model_data.elements(element_type='generator',
                                            generator_type='thermal'))
    renewable_gens = dict(model_data.elements(element_type='generator',
                                              generator_type='renewable'))

    buses = dict(model_data.elements(element_type='bus'))
    shunts = dict()
    branches = dict(model_data.elements(element_type='branch'))
    interfaces = dict(model_data.elements(element_type='interface'))
    contingencies = dict()
    storage = dict(model_data.elements(element_type='storage'))
    dc_branches = dict()

    thermal_gen_attrs = model_data.attributes(element_type='generator',
                                              generator_type='thermal')
    renewable_gen_attrs = model_data.attributes(element_type='generator',
                                                generator_type='renewable')

    bus_attrs = model_data.attributes(element_type='bus')
    branch_attrs = model_data.attributes(element_type='branch')
    interface_attrs = model_data.attributes(element_type='interface')

    storage_attrs = model_data.attributes(element_type='storage')
    storage_by_bus = tx_utils.gens_by_bus(buses, storage)

    dc_branch_attrs = dict(names=list())

    inlet_branches_by_bus, outlet_branches_by_bus = \
        tx_utils.inlet_outlet_branches_by_bus(branches, buses)
    dc_inlet_branches_by_bus, dc_outlet_branches_by_bus = \
        tx_utils.inlet_outlet_branches_by_bus(dc_branches, buses)

    thermal_gens_by_bus = tx_utils.gens_by_bus(buses, thermal_gens)
    renewable_gens_by_bus = tx_utils.gens_by_bus(buses, renewable_gens)

    ### get the fixed shunts at the buses
    bus_bs_fixed_shunts, bus_gs_fixed_shunts = \
        tx_utils.dict_of_bus_fixed_shunts(buses, shunts)

    ## attach some of these to the model object for ease/speed later
    # model._loads = loads
    model._buses = buses
    model._branches = branches
    model._shunts = shunts
    model._bus_gs_fixed_shunts = bus_gs_fixed_shunts
    model._interfaces = interfaces
    model._contingencies = contingencies
    model._dc_branches = dc_branches

    #
    # Parameters
    #

    ##############################################
    # string indentifiers for the set of busses. #
    ##############################################

    model.Buses = pe.Set(initialize=bus_attrs['names'])

    ref_bus = model_data.get_system_attr('reference_bus', '')
    if not ref_bus or ref_bus not in model.Buses:
        ref_bus = list(sorted(model.Buses))[0]

    model.ReferenceBus = pe.Param(within=model.Buses, initialize=ref_bus)

    ref_angle = model_data.get_system_attr('reference_bus_angle', 0.)
    model.ReferenceBusAngle = pe.Param(within=pe.Reals, initialize=ref_angle)

    ################################

    ## in minutes, assert that this must be a positive integer
    model.TimePeriodLengthMinutes = pe.Param(
        default=60, within=pe.PositiveIntegers,
        initialize=model_data.get_system_attr('time_period_length_minutes')
        )

    ## IN HOURS, assert athat this must be a positive number
    model.TimePeriodLengthHours = pe.Param(
        default=pe.value(model.TimePeriodLengthMinutes) / 60.,
        within=pe.PositiveReals
        )

    model.NumTimePeriods = pe.Param(within=pe.PositiveIntegers,
                                    initialize=len(time_keys))

    model.InitialTime = pe.Param(within=pe.PositiveIntegers, default=1)
    model.TimePeriods = pe.RangeSet(model.InitialTime, model.NumTimePeriods)
    TimeMapper = uc_time_helper(model.TimePeriods)

    ## For now, hard code these. Probably need to be able to specify in model_data
    model.StageSet = pe.Set(ordered=True, initialize=['Stage_1', 'Stage_2'])

    # the following sets must must come from the data files or from an initialization function that uses
    # a parameter that tells you when the stages end (and that thing needs to come from the data files)

    model.CommitmentTimeInStage = pe.Set(
        model.StageSet, within=model.TimePeriods,
        initialize={'Stage_1': model.TimePeriods, 'Stage_2': list()}
        )
    model.GenerationTimeInStage = pe.Set(
        model.StageSet, within=model.TimePeriods,
        initialize={'Stage_1': list(), 'Stage_2': model.TimePeriods}
        )

    ##############################################
    # Network definition (S)
    ##############################################

    model.TransmissionLines = pe.Set(initialize=branch_attrs['names'])
    model.HVDCLines = pe.Set(initialize=dc_branch_attrs['names'])

    model.BusFrom = pe.Param(model.TransmissionLines, within=model.Buses,
                             initialize=branch_attrs.get('from_bus', dict()))
    model.BusTo = pe.Param(model.TransmissionLines, within=model.Buses,
                           initialize=branch_attrs.get('to_bus', dict()))

    model.HVDCBusFrom = pe.Param(
        model.HVDCLines, within=model.Buses,
        initialize=dc_branch_attrs.get('from_bus', dict())
        )
    model.HVDCBusTo = pe.Param(
        model.HVDCLines, within=model.Buses,
        initialize=dc_branch_attrs.get('to_bus', dict())
        )

    model.LinesTo = pe.Set(model.Buses, within=model.TransmissionLines,
                           initialize=inlet_branches_by_bus)
    model.LinesFrom = pe.Set(model.Buses, within=model.TransmissionLines,
                             initialize=outlet_branches_by_bus)

    model.HVDCLinesTo = pe.Set(model.Buses, within=model.HVDCLines,
                               initialize=dc_inlet_branches_by_bus)
    model.HVDCLinesFrom = pe.Set(model.Buses, within=model.HVDCLines,
                                 initialize=dc_outlet_branches_by_bus)

    def load_base_params(model: pe.ConcreteModel,
                         model_data=None, renew_costs: bool = False) -> pe.ConcreteModel:

        if model_data is None:
            model_data = model.model_data

        warn_neg_load = False
        time_keys = model_data.get_system_attr('time_keys')

        ## NOTE: generator, bus, and load should be in here for
        # a well-defined problem
        loads = dict(model_data.elements(element_type='load'))
        gens = dict(model_data.elements(element_type='generator'))

        thermal_gens = dict(model_data.elements(element_type='generator',
                                                generator_type='thermal'))
        renewable_gens = dict(model_data.elements(element_type='generator',
                                                  generator_type='renewable'))

        buses = dict(model_data.elements(element_type='bus'))
        shunts = dict()
        branches = dict(model_data.elements(element_type='branch'))
        interfaces = dict(model_data.elements(element_type='interface'))
        contingencies = dict()
        storage = dict(model_data.elements(element_type='storage'))
        dc_branches = dict()

        thermal_gen_attrs = model_data.attributes(element_type='generator',
                                                  generator_type='thermal')
        renewable_gen_attrs = model_data.attributes(element_type='generator',
                                                    generator_type='renewable')

        bus_attrs = model_data.attributes(element_type='bus')
        branch_attrs = model_data.attributes(element_type='branch')
        interface_attrs = model_data.attributes(element_type='interface')

        storage_attrs = model_data.attributes(element_type='storage')
        storage_by_bus = tx_utils.gens_by_bus(buses, storage)

        dc_branch_attrs = dict(names=list())

        inlet_branches_by_bus, outlet_branches_by_bus = \
            tx_utils.inlet_outlet_branches_by_bus(branches, buses)
        dc_inlet_branches_by_bus, dc_outlet_branches_by_bus = \
            tx_utils.inlet_outlet_branches_by_bus(dc_branches, buses)

        thermal_gens_by_bus = tx_utils.gens_by_bus(buses, thermal_gens)
        renewable_gens_by_bus = tx_utils.gens_by_bus(buses, renewable_gens)

        ### get the fixed shunts at the buses
        bus_bs_fixed_shunts, bus_gs_fixed_shunts = \
            tx_utils.dict_of_bus_fixed_shunts(buses, shunts)

        ## attach some of these to the model object for ease/speed later
        # model._loads = loads
        model._buses = buses
        model._branches = branches
        model._shunts = shunts
        model._bus_gs_fixed_shunts = bus_gs_fixed_shunts
        model._interfaces = interfaces
        model._contingencies = contingencies
        model._dc_branches = dc_branches

        #
        # Parameters
        #

        ##############################################
        # string indentifiers for the set of busses. #
        ##############################################

        model.Buses = pe.Set(initialize=bus_attrs['names'])

        ref_bus = model_data.get_system_attr('reference_bus', '')
        if not ref_bus or ref_bus not in model.Buses:
            ref_bus = list(sorted(model.Buses))[0]

        model.ReferenceBus = pe.Param(within=model.Buses, initialize=ref_bus)

        ref_angle = model_data.get_system_attr('reference_bus_angle', 0.)
        model.ReferenceBusAngle = pe.Param(within=pe.Reals, initialize=ref_angle)

        ################################

        ## in minutes, assert that this must be a positive integer
        model.TimePeriodLengthMinutes = pe.Param(
            default=60, within=pe.PositiveIntegers,
            initialize=model_data.get_system_attr('time_period_length_minutes')
        )

        ## IN HOURS, assert athat this must be a positive number
        model.TimePeriodLengthHours = pe.Param(
            default=pe.value(model.TimePeriodLengthMinutes) / 60.,
            within=pe.PositiveReals
        )

        model.NumTimePeriods = pe.Param(within=pe.PositiveIntegers,
                                        initialize=len(time_keys))

        model.InitialTime = pe.Param(within=pe.PositiveIntegers, default=1)
        model.TimePeriods = pe.RangeSet(model.InitialTime, model.NumTimePeriods)
        TimeMapper = uc_time_helper(model.TimePeriods)

        ## For now, hard code these. Probably need to be able to specify in model_data
        model.StageSet = pe.Set(ordered=True, initialize=['Stage_1', 'Stage_2'])

        # the following sets must must come from the data files or from an initialization function that uses
        # a parameter that tells you when the stages end (and that thing needs to come from the data files)

        model.CommitmentTimeInStage = pe.Set(
            model.StageSet, within=model.TimePeriods,
            initialize={'Stage_1': model.TimePeriods, 'Stage_2': list()}
        )
        model.GenerationTimeInStage = pe.Set(
            model.StageSet, within=model.TimePeriods,
            initialize={'Stage_1': list(), 'Stage_2': model.TimePeriods}
        )

        ##############################################
        # Network definition (S)
        ##############################################

        model.TransmissionLines = pe.Set(initialize=branch_attrs['names'])
        model.HVDCLines = pe.Set(initialize=dc_branch_attrs['names'])

        model.BusFrom = pe.Param(model.TransmissionLines, within=model.Buses,
                                 initialize=branch_attrs.get('from_bus', dict()))
        model.BusTo = pe.Param(model.TransmissionLines, within=model.Buses,
                               initialize=branch_attrs.get('to_bus', dict()))

        model.HVDCBusFrom = pe.Param(
            model.HVDCLines, within=model.Buses,
            initialize=dc_branch_attrs.get('from_bus', dict())
        )
        model.HVDCBusTo = pe.Param(
            model.HVDCLines, within=model.Buses,
            initialize=dc_branch_attrs.get('to_bus', dict())
        )

        model.LinesTo = pe.Set(model.Buses, within=model.TransmissionLines,
                               initialize=inlet_branches_by_bus)
        model.LinesFrom = pe.Set(model.Buses, within=model.TransmissionLines,
                                 initialize=outlet_branches_by_bus)

        model.HVDCLinesTo = pe.Set(model.Buses, within=model.HVDCLines,
                                   initialize=dc_inlet_branches_by_bus)
        model.HVDCLinesFrom = pe.Set(model.Buses, within=model.HVDCLines,
                                     initialize=dc_outlet_branches_by_bus)

    def _warn_neg_impedence(m, v, l):
        if v == 0.:
            logger.error(f"Found zero reactance for line {l}")

            return False

        elif v < 0.:
            # We allow negative reactance, as it just reverses the
            # direction of the line. But we do print a warning.
            logger.warning(f"WARNING: found negative reactance for line {l}")
            return True

        return True

    model.Impedence = pe.Param(
        model.TransmissionLines, within=pe.Reals,
        initialize=branch_attrs.get('reactance', dict()),
        validate=_warn_neg_impedence
        )

    model.ThermalLimit = pe.Param(
        model.TransmissionLines,
        initialize=branch_attrs.get('rating_long_term', dict())
        )  # max flow across the line

    model.HVDCThermalLimit = pe.Param(
        model.HVDCLines,
        initialize=dc_branch_attrs.get('rating_long_term', dict())
        )  # max flow across the line

    model.LineOutOfService = pe.Param(
        model.TransmissionLines, model.TimePeriods,
        within=pe.Boolean, default=False,
        initialize=TimeMapper(branch_attrs.get('planned_outage', dict()))
        )

    model.HVDCLineOutOfService = pe.Param(
        model.HVDCLines, model.TimePeriods, within=pe.Boolean, default=False,
        initialize=TimeMapper(dc_branch_attrs.get('planned_outage', dict()))
        )

    _branch_penalties = dict()
    _md_violation_penalties = branch_attrs.get('violation_penalty')

    if _md_violation_penalties is not None:
        for i, val in _md_violation_penalties.items():
            if val is not None:
                _branch_penalties[i] = val

                if val <= 0:
                    logger.warning(
                        "Branch {} has a non-positive penalty {}, this will "
                        "cause its limits to be ignored!".format(i, val)
                        )

    model.BranchesWithSlack = pe.Set(within=model.TransmissionLines,
                                     initialize=_branch_penalties.keys())

    model.BranchLimitPenalty = pe.Param(model.BranchesWithSlack,
                                        within=pe.NonNegativeReals,
                                        initialize=_branch_penalties)

    ## Interfaces
    model.Interfaces = pe.Set(initialize=interface_attrs['names'])
    model.InterfaceLines = pe.Set(
        model.Interfaces, within=model.TransmissionLines,
        initialize=interface_attrs.get('lines', dict()), ordered=True
        )

    model.InterfaceMinFlow = pe.Param(
        model.Interfaces, within=pe.Reals,
        initialize=interface_attrs.get('minimum_limit', dict())
        )
    model.InterfaceMaxFlow = pe.Param(
        model.Interfaces, within=pe.Reals,
        initialize=interface_attrs.get('maximum_limit', dict())
        )

    def check_min_less_max_interface_flow_limits(m):
        for intfc in m.Interfaces:
            if (pe.value(m.InterfaceMinFlow[intfc])
                    > pe.value(m.InterfaceMaxFlow[intfc])):
                raise ModelError(
                    "Interface {} has a minimum_limit which is greater than "
                    "the maximum_limit".format(intfc)
                    )

    model.CheckInterfaceFlowLimits = pe.BuildAction(
        rule=check_min_less_max_interface_flow_limits)

    def get_interface_line_pairs(m):
        for intfc in m.Interfaces:
            for line in m.InterfaceLines[intfc]:
                yield intfc, line

    model.InterfaceLinePairs = pe.Set(initialize=get_interface_line_pairs,
                                      dimen=2)

    _interface_line_orientation_dict = dict()
    for intfc, interface in interfaces.items():
        for line, sign in zip(interface['lines'],
                              interface['line_orientation']):
            _interface_line_orientation_dict[intfc, line] = sign

    model.InterfaceLineOrientation = pe.Param(
        model.InterfaceLinePairs,
        initialize=_interface_line_orientation_dict, within={-1, 0, 1}
        )

    _interface_penalties = dict()
    _md_violation_penalties = interface_attrs.get('violation_penalty')

    if _md_violation_penalties is not None:
        for intfc, val in _md_violation_penalties.items():
            if val is not None:
                _interface_penalties[intfc] = val

                if val <= 0:
                    logger.warning(
                        "Interface {} has a non-positive penalty {}, this "
                        "will cause its limits to be ignored!".format(
                            intfc, val)
                        )

    model.InterfacesWithSlack = pe.Set(within=model.Interfaces,
                                       initialize=_interface_penalties.keys())
    model.InterfaceLimitPenalty = pe.Param(model.InterfacesWithSlack,
                                           within=pe.NonNegativeReals,
                                           initialize=_interface_penalties)

    ##########################################################
    # string indentifiers for the set of thermal generators. #
    # and their locations. (S)                               #
    ##########################################################

    model.ThermalGenerators = pe.Set(initialize=thermal_gen_attrs['names'])
    model.ThermalGeneratorsAtBus = pe.Set(model.Buses,
                                          initialize=thermal_gens_by_bus)

    model.ThermalGeneratorType = pe.Param(
        model.ThermalGenerators,
        within=pe.Any, default='C',
        initialize=thermal_gen_attrs.get('fuel', dict())
        )

    def verify_thermal_generator_buses_rule(m):
        assert set(m.ThermalGenerators) == {
            gen for bus in m.Buses for gen in m.ThermalGeneratorsAtBus[bus]}

    model.VerifyThermalGeneratorBuses = pe.BuildAction(
        rule=verify_thermal_generator_buses_rule)

    model.QuickStart = pe.Param(
        model.ThermalGenerators,
        within=pe.Boolean, default=False,
        initialize=thermal_gen_attrs.get('fast_start', dict())
        )

    def init_quick_start_generators(m):
        return [g for g in m.ThermalGenerators
                if pe.value(m.QuickStart[g]) == 1]

    model.QuickStartGenerators = pe.Set(within=model.ThermalGenerators,
                                        initialize=init_quick_start_generators)

    # optionally force a unit to be on/off
    model.FixedCommitmentTypes = pe.Set(initialize=[0, 1, None])

    model.FixedCommitment = pe.Param(
        model.ThermalGenerators, model.TimePeriods,
        within=model.FixedCommitmentTypes, default=None,
        initialize=TimeMapper(
            thermal_gen_attrs.get('fixed_commitment', dict()))
        )

    model.AllNondispatchableGenerators = pe.Set(
        initialize=renewable_gen_attrs['names'])
    model.NondispatchableGeneratorsAtBus = pe.Set(
        model.Buses, initialize=renewable_gens_by_bus)

    model.NondispatchableGeneratorType = pe.Param(
        model.AllNondispatchableGenerators,
        within=pe.Any, default='W',
        initialize=renewable_gen_attrs.get('fuel', dict())
        )

    def verify_renew_generator_buses_rule(m):
        assert set(m.AllNondispatchableGenerators) == {
            gen for bus in m.Buses
            for gen in m.NondispatchableGeneratorsAtBus[bus]
            }

    model.VerifyRenewGeneratorBuses = pe.BuildAction(
        rule=verify_renew_generator_buses_rule)

    #################################################################
    # the global system demand, for each time period. units are MW. #
    # demand as at busses (S) so total demand is derived            #
    #################################################################

    # at the moment, we allow for negative demand. this is probably
    # not a good idea, as "stuff" representing negative demand - including
    # renewables, interchange schedules, etc. - should probably be modeled
    # explicitly.

    bus_loads = {(b, t): 0
                 for b in bus_attrs['names'] for t in model.TimePeriods}

    for lname, load in loads.items():
        load_time = TimeMapper(load['p_load'])
        bus = load['bus']

        if isinstance(bus, dict):
            assert bus['data_type'] == 'load_distribution_factor'

            for bn, multi in bus['values'].items():
                for t in model.TimePeriods:
                    bus_loads[bn, t] += multi * load_time[t]

        else:
            for t in model.TimePeriods:
                bus_loads[bus, t] += load_time[t]

    model.Demand = pe.Param(model.Buses, model.TimePeriods,
                            initialize=bus_loads, mutable=True)

    def calculate_total_demand(m, t):
        return sum(pe.value(m.Demand[b,t]) for b in sorted(m.Buses))

    model.TotalDemand = pe.Param(model.TimePeriods,
                                 initialize=calculate_total_demand)

    # at this point, a user probably wants to see if they have negative demand.
    def warn_about_negative_demand_rule(m, b, t):
        this_demand = pe.value(m.Demand[b, t])

        if this_demand < 0.0:
            logger.warning(
                "***WARNING: The demand at bus `{}` for time period <{}> is "
                "negative - value {:.4f}}; model={}".format(
                    b, t, this_demand, m.name)
                )

    if warn_neg_load:
        model.WarnAboutNegativeDemand = pe.BuildAction(
            model.Buses, model.TimePeriods,
            rule=warn_about_negative_demand_rule
            )

    ##################################################################
    # the global system reserve, for each time period. units are MW. #
    ##################################################################

    reserve_req = TimeMapper(
        model_data.get_system_attr('reserve_requirement', 0.))

    model.ReserveRequirement = pe.Param(model.TimePeriods,
                                        within=pe.NonNegativeReals,
                                        initialize=reserve_req, mutable=True)

    ##########################################################################################################
    # the minimum number of time periods that a generator must be on-line (off-line) once brought up (down). #
    ##########################################################################################################

    model.MinimumUpTime = pe.Param(
        model.ThermalGenerators,
        within=pe.NonNegativeReals, default=0,
        initialize=thermal_gen_attrs['min_up_time']
        )
    model.MinimumDownTime = pe.Param(
        model.ThermalGenerators,
        within=pe.NonNegativeReals, default=0,
        initialize=thermal_gen_attrs['min_down_time']
        )

    ## Assert that MUT and MDT are at least 1 in the time units of the model.
    ## Otherwise, turn on/offs may not be enforced correctly.
    def scale_min_uptime(m, g):
        scaled_up_time = int(math.ceil(m.MinimumUpTime[g]
                                       / m.TimePeriodLengthHours))

        return min(max(scaled_up_time, 1), pe.value(m.NumTimePeriods))

    model.ScaledMinimumUpTime = pe.Param(model.ThermalGenerators,
                                         within=pe.NonNegativeIntegers,
                                         initialize=scale_min_uptime)

    def scale_min_downtime(m, g):
        scaled_down_time = int(math.ceil(m.MinimumDownTime[g]
                                         / m.TimePeriodLengthHours))

        return min(max(scaled_down_time, 1), pe.value(m.NumTimePeriods))

    model.ScaledMinimumDownTime = pe.Param(model.ThermalGenerators,
                                           within=pe.NonNegativeIntegers,
                                           initialize=scale_min_downtime)

    ####################################################################################
    # minimum and maximum generation levels, for each thermal generator. units are MW. #
    # could easily be specified on a per-time period basis, but are not currently.     #
    ####################################################################################

    # you can enter generator limits either once for the generator or for each period (or just take 0)

    if renew_costs:
        cost_gens = (model.ThermalGenerators
                     | model.AllNondispatchableGenerators)
        cost_attrs = model_data.attributes(element_type='generator')

    else:
        cost_gens = model.ThermalGenerators
        cost_attrs = thermal_gen_attrs

    model.MinimumPowerOutput = pe.Param(
        cost_gens, model.TimePeriods, within=pe.NonNegativeReals,
        initialize=TimeMapper(cost_attrs['p_min']), default=0.0
        )

    def maximum_power_output_validator(m, v, g, t):
        return v >= pe.value(m.MinimumPowerOutput[g, t])

    model.MaximumPowerOutput = pe.Param(
        cost_gens, model.TimePeriods, within=pe.NonNegativeReals, default=0.,
        validate=maximum_power_output_validator,
        initialize=TimeMapper(cost_attrs['p_max']),
        )

    model.MinimumReactivePowerOutput = pe.Param(
        model.ThermalGenerators, model.TimePeriods,
        within=pe.Reals, default=0.,
        initialize=TimeMapper(thermal_gen_attrs.get('q_min', dict())),
        )

    def maximum_reactive_output_validator(m, v, g, t):
        return v >= pe.value(m.MinimumReactivePowerOutput[g, t])

    model.MaximumReactivePowerOutput = pe.Param(
        model.ThermalGenerators, model.TimePeriods,
        within=pe.Reals, default=0.,
        initialize=TimeMapper(thermal_gen_attrs.get('q_max', dict())),
        )

    # wind is similar, but max and min will be equal for non-dispatchable wind

    model.MinNondispatchablePower = pe.Param(
        model.AllNondispatchableGenerators, model.TimePeriods,
        within=pe.Reals,  # more permissive; e.g. CSP
        default=0.0, mutable=True,
        initialize=TimeMapper(renewable_gen_attrs.get('p_min', dict()))
        )

    def maximum_nd_output_validator(m, v, g, t):
        return v >= pe.value(m.MinNondispatchablePower[g, t])

    model.MaxNondispatchablePower = pe.Param(
        model.AllNondispatchableGenerators, model.TimePeriods,
        within=pe.Reals,  # more permissive; e.g. CSP
        default=0.0, mutable=True, validate=maximum_nd_output_validator,
        initialize=TimeMapper(renewable_gen_attrs.get('p_max', dict()))
        )

    #################################################
    # generator ramp up/down rates. units are MW/h. #
    # IMPORTANT: Generator ramp limits can exceed   #
    # the maximum power output, because it is the   #
    # ramp limit over an hour. If the unit can      #
    # fully ramp in less than an hour, then this    #
    # will occur.                                   #
    #################################################

    ## be sure the generator can ramp
    ## between all the p_min/p_max values
    def ramp_up_validator(m, v, g):
        t1 = m.InitialTime

        for t in m.TimePeriods:
            if t == t1:
                continue

            diff = pe.value(m.MinimumPowerOutput[g, t]
                            - m.MaximumPowerOutput[g, t - 1])

            if v * m.TimePeriodLengthHours < diff:
                logger.error("Generator {} has an infeasible ramp up between "
                             "time periods {} and {}".format(g, t - 1, t))

                return False

        return True

    ## be sure the generator can ramp
    ## between all the p_min/p_max values
    def ramp_down_validator(m, v, g):
        t1 = m.InitialTime

        for t in m.TimePeriods:
            if t == t1:
                continue

            diff = pe.value(m.MinimumPowerOutput[g, t - 1]
                            - m.MaximumPowerOutput[g, t])

            if v * m.TimePeriodLengthHours < diff:
                logger.error(
                    "Generator {} has an infeasible ramp down between time "
                    "periods {} and {}".format(g, t - 1, t)
                    )

                return False

        return True

    # limits for normal time periods
    model.NominalRampUpLimit = pe.Param(
        model.ThermalGenerators,
        within=pe.NonNegativeReals, mutable=True,
        initialize=thermal_gen_attrs['ramp_up_60min'],
        validate=ramp_up_validator
        )

    model.NominalRampDownLimit = pe.Param(
        model.ThermalGenerators,
        within=pe.NonNegativeReals, mutable=True,
        initialize=thermal_gen_attrs['ramp_down_60min'],
        validate=ramp_down_validator
        )

    #############################################
    # unit on state at t=0 (initial condition). #
    #############################################

    # if positive, the number of hours prior to (and including) t=0 that the unit has been on.
    # if negative, the number of hours prior to (and including) t=0 that the unit has been off.
    # the value cannot be 0, by definition.

    def t0_state_nonzero_validator(m, v, g):
        return v != 0.

    model.UnitOnT0State = pe.Param(
        model.ThermalGenerators,
        within=pe.Reals, validate=t0_state_nonzero_validator, mutable=True,
        initialize=thermal_gen_attrs['initial_status']
        )

    def t0_unit_on_rule(m, g):
        return int(pe.value(m.UnitOnT0State[g]) > 0.)

    model.UnitOnT0 = pe.Param(model.ThermalGenerators,
                              within=pe.Binary, initialize=t0_unit_on_rule,
                              mutable=True)

    _add_initial_time_periods_on_off_line(model)
    _verify_must_run_t0_state_consistency(model)

    # For future shutdowns/startups beyond the time-horizon
    # Like UnitOnT0State, a postive quantity means the generator
    # *will start* in 'future_status' hours, and a negative quantity
    # means the generator *will stop* in -('future_status') hours.
    # The default of 0 means we have no information
    model.FutureStatus = pe.Param(
        model.ThermalGenerators,
        within=pe.Reals, mutable=True, default=0.,
        initialize=thermal_gen_attrs.get('future_status', dict())
        )

    def time_periods_since_last_shutdown_rule(m, g):
        if pe.value(m.UnitOnT0[g]):
            # longer than any time-horizon we'd consider
            return 10000
        else:
            return int(math.ceil(-pe.value(m.UnitOnT0State[g])
                                 / pe.value(m.TimePeriodLengthHours)))

    model.TimePeriodsSinceShutdown = pe.Param(
        model.ThermalGenerators,
        within=pe.PositiveIntegers, mutable=True,
        initialize=time_periods_since_last_shutdown_rule
        )

    def time_periods_before_startup_rule(m, g):
        if pe.value(m.FutureStatus[g]) <= 0:
            # longer than any time-horizon we'd consider
            return 10000
        else:
            return int(math.ceil(pe.value(m.FutureStatus[g])
                                 / pe.value(m.TimePeriodLengthHours)))

    model.TimePeriodsBeforeStartup = pe.Param(
        model.ThermalGenerators,
        within=pe.PositiveIntegers, mutable=True,
        initialize=time_periods_before_startup_rule
        )

    ###############################################
    # startup/shutdown curves for each generator. #
    # These are specified in the same time scales #
    # as 'time_period_length_minutes' and other   #
    # time-vary quantities.                       #
    ###############################################

    def startup_curve_init_rule(m, g):
        startup_curve = thermal_gens[g].get('startup_curve')

        if startup_curve is None:
            return ()

        min_down_time = int(math.ceil(m.MinimumDownTime[g]
                                      / m.TimePeriodLengthHours))

        if len(startup_curve) > min_down_time:
            logger.warn(
                "Truncating startup_curve longer than scaled minimum down "
                "time {} for generator {}".format(min_down_time, g)
                )

        return startup_curve[0:min_down_time]

    model.StartupCurve = pe.Set(model.ThermalGenerators,
                                within=pe.NonNegativeReals, ordered=True,
                                initialize=startup_curve_init_rule)

    def shutdown_curve_init_rule(m, g):
        shutdown_curve = thermal_gens[g].get('shutdown_curve')

        if shutdown_curve is None:
            return ()

        min_down_time = int(math.ceil(m.MinimumDownTime[g]
                                      / m.TimePeriodLengthHours))

        if len(shutdown_curve) > min_down_time:
            logger.warn(
                "Truncating shutdown_curve longer than scaled minimum down "
                "time {} for generator {}".format(min_down_time, g)
                )

        return shutdown_curve[0:min_down_time]

    model.ShutdownCurve = pe.Set(model.ThermalGenerators,
                                 within=pe.NonNegativeReals, ordered=True,
                                 initialize=shutdown_curve_init_rule)

    ####################################################################
    # generator power output at t=0 (initial condition). units are MW. #
    ####################################################################

    def power_generated_t0_validator(m, v, g):
        t = m.TimePeriods.first()

        if pe.value(m.UnitOnT0[g]):
            v_less_max = v <= pe.value(m.MaximumPowerOutput[g, t]
                                       + m.NominalRampDownLimit[g]
                                       * m.TimePeriodLengthHours)

            if not v_less_max:
                logger.error("Generator {} has more output at T0 than is "
                             "feasible to ramp down to".format(g))
                return False

            # TODO: double-check that `if not v_less_max` in original is wrong
            v_greater_min = v >= pe.value(m.MinimumPowerOutput[g, t]
                                          - m.NominalRampUpLimit[g]
                                          * m.TimePeriodLengthHours)

            if not v_less_max:
                logger.error("Generator {} has less output at T0 than is "
                             "feasible to ramp up to".format(g))
                return False

            return True

        else:
            # Generator was off, but could have residual power due to
            # start-up/shut-down curve. Therefore, do not be too picky
            # as the value doesn't affect any constraints directly
            return True

    model.PowerGeneratedT0 = pe.Param(
        model.ThermalGenerators,
        within=pe.NonNegativeReals, validate=power_generated_t0_validator,
        mutable=True, initialize=thermal_gen_attrs['initial_p_output']
        )

    # limits for time periods in which generators are brought on or off-line.
    # must be no less than the generator minimum output.
    def ramp_limit_validator(m, v, g, t):
        return v >= m.MinimumPowerOutput[g, t]

    ## These defaults follow what is in most market manuals
    ## We scale this for the time period below
    def startup_ramp_default(m, g, t):
        return m.MinimumPowerOutput[g, t] + m.NominalRampUpLimit[g] / 2.

    ## shutdown is based on the last period *on*
    def shutdown_ramp_default(m, g, t):
        return m.MinimumPowerOutput[g, t] + m.NominalRampDownLimit[g] / 2.

    model.StartupRampLimit = pe.Param(
        model.ThermalGenerators, model.TimePeriods,
        within=pe.NonNegativeReals, default=startup_ramp_default,
        validate=ramp_limit_validator, mutable=True,
        initialize=TimeMapper(
            thermal_gen_attrs.get('startup_capacity', dict()))
        )

    model.ShutdownRampLimit = pe.Param(
        model.ThermalGenerators, model.TimePeriods,
        within=pe.NonNegativeReals, default=shutdown_ramp_default,
        validate=ramp_limit_validator, mutable=True,
        initialize=TimeMapper(
            thermal_gen_attrs.get('shutdown_capacity', dict()))
        )

    ## These get used in the basic UC constraints, which implicity assume RU, RD <= Pmax
    ## Ramping constraints look backward, so these will accordingly as well
    ## NOTES: can't ramp up higher than the current pmax from the previous value
    ##        can't ramp down more than the pmax from the prior time period
    def scale_ramp_up(m, g, t):
        temp = m.NominalRampUpLimit[g] * m.TimePeriodLengthHours

        if pe.value(temp) > pe.value(m.MaximumPowerOutput[g, t]):
            return m.MaximumPowerOutput[g, t]
        else:
            return temp

    model.ScaledNominalRampUpLimit = pe.Param(
        model.ThermalGenerators, model.TimePeriods,
        within=pe.NonNegativeReals, initialize=scale_ramp_up, mutable=True
        )

    def scale_ramp_down(m, g, t):
        temp = m.NominalRampDownLimit[g] * m.TimePeriodLengthHours

        if t == m.InitialTime:
            param = max(pe.value(m.PowerGeneratedT0[g]),
                        pe.value(m.MaximumPowerOutput[g, t]))
        else:
            param = m.MaximumPowerOutput[g, t - 1]

        if pe.value(temp) > pe.value(param):
            return param
        else:
            return temp

    model.ScaledNominalRampDownLimit = pe.Param(
        model.ThermalGenerators, model.TimePeriods,
        within=pe.NonNegativeReals, initialize=scale_ramp_down, mutable=True
        )

    def scale_startup_limit(m, g, t):
        ## temp now has the "running room" over Pmin. This will be scaled for the time period length,
        ## most market models do not have this notion, so this is set-up so that the defaults
        ## will be scaled as they would be in most market models
        temp = m.StartupRampLimit[g, t] - m.MinimumPowerOutput[g, t]
        temp *= m.TimePeriodLengthHours

        if pe.value(temp) > pe.value(m.MaximumPowerOutput[g, t]
                                     - m.MinimumPowerOutput[g, t]):
            return m.MaximumPowerOutput[g, t]
        else:
            return temp + m.MinimumPowerOutput[g, t]

    model.ScaledStartupRampLimit = pe.Param(
        model.ThermalGenerators, model.TimePeriods,
        within=pe.NonNegativeReals, validate=ramp_limit_validator,
        initialize=scale_startup_limit, mutable=True
        )

    def scale_shutdown_limit(m, g, t):
        ## temp now has the "running room" over Pmin. This will be scaled for the time period length
        ## most market models do not have this notion, so this is set-up so that the defaults
        ## will be scaled as they would be in most market models
        temp = m.ShutdownRampLimit[g, t] - m.MinimumPowerOutput[g, t]
        temp *= m.TimePeriodLengthHours

        if pe.value(temp) > (pe.value(m.MaximumPowerOutput[g, t]
                                      - m.MinimumPowerOutput[g, t])):
            return m.MaximumPowerOutput[g, t]
        else:
            return temp + m.MinimumPowerOutput[g, t]

    model.ScaledShutdownRampLimit = pe.Param(
        model.ThermalGenerators, model.TimePeriods,
        within=pe.NonNegativeReals, validate=ramp_limit_validator,
        initialize=scale_shutdown_limit, mutable=True
        )

    ## Some additional ramping parameters to
    ## deal with shutdowns at time=1

    def _init_p_min_t0(m, g):
        if 'initial_p_min' in thermal_gen_attrs and \
                g in thermal_gen_attrs['initial_p_min']:
            return thermal_gen_attrs['initial_p_min'][g]
        else:
            return m.MinimumPowerOutput[g, m.InitialTime]

    model.MinimumPowerOutputT0 = pe.Param(model.ThermalGenerators,
                                          within=pe.NonNegativeReals,
                                          mutable=True,
                                          initialize=_init_p_min_t0)

    def _init_sd_t0(m, g):
        if 'initial_shutdown_capacity' in thermal_gen_attrs and \
                g in thermal_gen_attrs['initial_shutdown_capacity']:
            return thermal_gen_attrs['initial_shutdown_capacity'][g]

        return m.ShutdownRampLimit[g, m.InitialTime]

    model.ShutdownRampLimitT0 = pe.Param(model.ThermalGenerators,
                                         within=pe.NonNegativeReals,
                                         mutable=True, initialize=_init_sd_t0)

    def scale_shutdown_limit_t0(m, g):
        ## temp now has the "running room" over Pmin. This will be scaled for the time period length
        ## most market models do not have this notion, so this is set-up so that the defaults
        ## will be scaled as they would be in most market models
        temp = m.ShutdownRampLimitT0[g] - m.MinimumPowerOutputT0[g]
        temp *= m.TimePeriodLengthHours

        if pe.value(temp) > pe.value(m.PowerGeneratedT0[g]
                                     - m.MinimumPowerOutputT0[g]):
            return m.PowerGeneratedT0[g]
        else:
            return temp + m.MinimumPowerOutputT0[g]

    model.ScaledShutdownRampLimitT0 = pe.Param(
        model.ThermalGenerators,
        within=pe.NonNegativeReals, initialize=scale_shutdown_limit_t0,
        mutable=True
        )

    ###############################################
    # startup cost parameters for each generator. #
    ###############################################

    # startup costs are conceptually expressed as pairs (x, y), where x represents the number of hours that a unit has been off and y represents
    # the cost associated with starting up the unit after being off for x hours. these are broken into two distinct ordered sets, as follows.

    def _get_startup_lag(startup,default):
        try:
            iter(startup)
        except TypeError:
            return [default]
        else:
            return [i[0] for i in startup]

    def startup_lags_init_rule(m, g):
        startup_cost = thermal_gens[g].get('startup_cost')
        startup_fuel = thermal_gens[g].get('startup_fuel')

        if startup_cost is not None and startup_fuel is not None:
            logger.warning("WARNING: found startup_fuel for generator {}, "
                           "ignoring startup_cost".format(g))

        if startup_fuel is None and startup_cost is None:
            return [pe.value(m.MinimumDownTime[g])]

        elif startup_cost is None:
            return _get_startup_lag(startup_fuel,
                                    pe.value(m.MinimumDownTime[g]))

        else:
            return _get_startup_lag(startup_cost,
                                    pe.value(m.MinimumDownTime[g]))

    # units are hours / time periods.
    model.StartupLags = pe.Set(model.ThermalGenerators,
                               within=pe.NonNegativeReals, ordered=True,
                               initialize=startup_lags_init_rule)

    def _get_startup_cost(startup, fixed_adder, multiplier):
        try:
            iter(startup)

        except TypeError:
            return [fixed_adder + multiplier * startup]
        else:
            return [fixed_adder + multiplier * i[1] for i in startup]

    def startup_costs_init_rule(m, g):
        startup_cost = thermal_gens[g].get('startup_cost')
        startup_fuel = thermal_gens[g].get('startup_fuel')
        fixed_startup_cost = thermal_gens[g].get('non_fuel_startup_cost')

        if fixed_startup_cost is None:
            fixed_startup_cost = 0.

        if startup_fuel is None and startup_cost is None:
            return [fixed_startup_cost]

        elif startup_cost is None:
            fuel_cost = thermal_gens[g].get('fuel_cost')

            if fuel_cost is None:
                raise ModelError("No fuel cost for generator {}, but data is "
                                 "provided for fuel tracking".format(g))

            return _get_startup_cost(startup_fuel,
                                     fixed_startup_cost, fuel_cost)

        else:
            return _get_startup_cost(startup_cost, fixed_startup_cost, 1.)

    # units are $.
    model.StartupCosts = pe.Set(model.ThermalGenerators,
                                within=pe.NonNegativeReals, ordered=True,
                                initialize=startup_costs_init_rule)

    # startup lags must be monotonically increasing...
    def validate_startup_lags_rule(m, g):
        startup_lags = list(m.StartupLags[g])

        if len(startup_lags) == 0:
            raise ModelError("DATA ERROR: The number of startup lags for "
                             "thermal generator `{}` must be >= 1!".format(g))

        if startup_lags[0] != pe.value(m.MinimumDownTime[g]):
            raise ModelError(
                "DATA ERROR: The first startup lag for thermal generator `{}` "
                "must be equal the minimum down time {}!".format(
                    g, pe.value(m.MinimumDownTime[g]))
                )

        for i in range(0, len(startup_lags) - 1):
           if startup_lags[i] >= startup_lags[i + 1]:
              raise ModelError(
                  "DATA ERROR: Startup lags for thermal generator `{}` must "
                  "be monotonically increasing!".format(g)
                  )

    model.ValidateStartupLags = pe.BuildAction(model.ThermalGenerators,
                                               rule=validate_startup_lags_rule)

    # while startup costs must be monotonically non-decreasing!
    def validate_startup_costs_rule(m, g):
       startup_costs = m.StartupCosts[g]

       for i in range(1, len(startup_costs) - 1):
           if startup_costs[i] > startup_costs[i + 1]:
              raise ModelError(
                  "DATA ERROR: Startup costs for thermal generator `{}` must "
                  "be monotonically non-decreasing!".format(g)
                  )

    model.ValidateStartupCosts = pe.BuildAction(
        model.ThermalGenerators, rule=validate_startup_costs_rule)

    def validate_startup_lag_cost_cardinalities(m, g):
        if len(m.StartupLags[g]) != len(m.StartupCosts[g]):
            raise ModelError(
                "DATA ERROR: The number of startup lag entries ({}) for "
                "thermal generator `{}` must equal the number of startup cost "
                "entries ({})".format(
                    len(m.StartupLags[g]), g, len(m.StartupCosts[g]))
                )

    model.ValidateStartupLagCostCardinalities = pe.BuildAction(
        model.ThermalGenerators, rule=validate_startup_lag_cost_cardinalities)

    # for purposes of defining constraints, it is useful to have a set to index the various startup costs parameters.
    # entries are 1-based indices, because they are used as indicies into Pyomo sets - which use 1-based indexing.

    def startup_cost_indices_init_rule(m, g):
       return range(1, len(m.StartupLags[g]) + 1)

    model.StartupCostIndices = pe.Set(
        model.ThermalGenerators,
        within=pe.NonNegativeIntegers,
        initialize=startup_cost_indices_init_rule
        )

    ## scale the startup lags
    ## Again, assert that this must be at least one in the time units of the model
    def scaled_startup_lags_rule(m, g):
        return [max(int(round(this_lag / m.TimePeriodLengthHours)), 1)
                for this_lag in m.StartupLags[g]]

    model.ScaledStartupLags = pe.Set(model.ThermalGenerators,
                                     within=pe.NonNegativeIntegers,
                                     ordered=True,
                                     initialize=scaled_startup_lags_rule)

    ##################################################################################
    # shutdown cost for each generator. in the literature, these are often set to 0. #
    ##################################################################################

    # units are $.
    model.ShutdownFixedCost = pe.Param(
        model.ThermalGenerators,
        within=pe.NonNegativeReals, default=0.0,
        initialize=thermal_gen_attrs.get('shutdown_cost', dict())
        )

    ## FUEL-SUPPLY Sets
    def fuel_supply_gens_init(m):
        if 'fuel_supply' not in thermal_gen_attrs:
            thermal_gen_attrs['fuel_supply'] = dict()

        if 'aux_fuel_supply' not in thermal_gen_attrs:
            thermal_gen_attrs['aux_fuel_supply'] = dict()

        fuel_supply = thermal_gen_attrs['fuel_supply']
        for g in fuel_supply:
            yield g

        for g in thermal_gen_attrs['aux_fuel_supply']:
            if g not in fuel_supply:
                yield g

    def gen_cost_fuel_validator(m,g):
        # validators may get called once
        # with key None for empty sets
        if g is None:
            return True

        if 'p_fuel' in thermal_gen_attrs and g in thermal_gen_attrs['p_fuel']:
            pass

        else:
            raise ModelError("All fuel-constrained generators must have the "
                             "<p_fuel> attribute which tracks their fuel "
                             "consumption, could not find such an attribute "
                             "for generator `{}`!'".format(g))

        return True

    model.FuelSupplyGenerators = pe.Set(within=model.ThermalGenerators,
                                        initialize=fuel_supply_gens_init,
                                        validate=gen_cost_fuel_validator)

    ## DUAL-FUEL Sets

    def dual_fuel_init(m):
        for gen, g_data in thermal_gens.items():
            if 'aux_fuel_capable' in g_data and g_data['aux_fuel_capable']:
                yield gen

    model.DualFuelGenerators = pe.Set(within=model.ThermalGenerators,
                                      initialize=dual_fuel_init)

    ## This set is for modeling elements that are exhanged
    ## in whole for the dual-fuel model
    model.SingleFuelGenerators = (model.ThermalGenerators
                                  - model.DualFuelGenerators)

    ## BEGIN PRODUCTION COST
    ## NOTE: For better or worse, we handle scaling this to the time period length in the objective function.
    ##       In particular, this is done in objective.py.

    def _check_curve(m, g, curve, curve_type):
        for i, t in enumerate(m.TimePeriods):
            ## first, get a cost_curve out of time series
            if curve['data_type'] == 'time_series':
                curve_t = curve['values'][i]
                _t = t

            else:
                curve_t = curve
                _t = None

            tx_utils.validate_and_clean_cost_curve(
                curve_t, curve_type=curve_type,
                p_min=pe.value(m.MinimumPowerOutput[g, t]),
                p_max=pe.value(m.MaximumPowerOutput[g, t]),
                gen_name=g, t=_t
                )

            # if no curve_type+'_type' is specified, we assume piecewise
            # (for backwards compatibility with no 'fuel_curve_type')
            if (curve_type + '_type' in curve
                    and curve_t[curve_type + '_type'] == 'polynomial'):
                if not _check_curve.warn_piecewise_approx:
                    logger.warning("WARNING: Polynomial cost curves will be "
                                   "approximated using piecewise segments")
                    _check_curve.warn_piecewise_approx = True

            if curve['data_type'] != 'time_series':
                break

    ## set "static" variable for this function
    _check_curve.warn_piecewise_approx = False

    def validate_cost_rule(m, g):
        gen_dict = thermal_gens[g]
        cost = gen_dict.get('p_cost')
        fuel = gen_dict.get('p_fuel')
        fuel_cost = gen_dict.get('fuel_cost')

        if cost is None and fuel is None:
            logger.warning("WARNING: Generator {} has no cost information "
                           "associated with it".format(g))
            return True

        if cost is not None and fuel is not None:
            logger.warning("WARNING: ignoring provided p_cost and using fuel "
                           "cost data from p_fuel for generator {}".format(g))

        ## look at p_cost through time
        if fuel is None:
            _check_curve(m, g, cost, 'cost_curve')

        else:
            if fuel_cost is None:
                raise ModelError("Found fuel_curve but not fuel_cost "
                                 "for generator {}".format(g))

            _check_curve(m, g, fuel, 'fuel_curve')
            for i, t in enumerate(m.TimePeriods):
                if fuel_cost is dict:
                    if fuel_cost['data_type'] != 'time_series':
                        raise ModelError("fuel_cost must be either numeric "
                                         "or time_series")

                    fuel_cost_t = fuel_cost['values'][i]

                else:
                    fuel_cost_t = fuel_cost

                if fuel_cost_t < 0:
                    raise ModelError(
                        "fuel_cost must be non-negative, found negative "
                        "fuel_cost for generator `{}`!".format(g)
                        )

                if fuel_cost_t == fuel_cost:
                    break

        return True

    model.ValidateGeneratorCost = pe.BuildCheck(model.ThermalGenerators,
                                                rule=validate_cost_rule)

    ##############################################################################################
    # number of pieces in the linearization of each generator's quadratic cost production curve. #
    ##############################################################################################
    ## TODO: option-drive with Egret, either globally or per-generator

    model.NumGeneratorCostCurvePieces = pe.Param(within=pe.PositiveIntegers,
                                                 default=2, mutable=True)

    #######################################################################
    # points for piecewise linearization of power generation cost curves. #
    #######################################################################

    # BK -- changed to reflect that the generator's power output variable is always above minimum in the ME model
    #       this simplifies things quite a bit..

    # maps a (generator, time-index) pair to a list of points defining the piecewise cost linearization breakpoints.
    # the time index is redundant, but required - in the current implementation of the Piecewise construct, the
    # breakpoints must be indexed the same as the Piecewise construct itself.

    # the points are expected to be on the interval [0, maxpower-minpower], and must contain both endpoints.
    # power generated can always be 0, and piecewise expects the entire variable domain to be represented.
    model.PowerGenerationPiecewisePoints = {}

    # NOTE: the values are relative to the minimum production cost, i.e., the values represent
    # incremental costs relative to the minimum production cost.
    model.PowerGenerationPiecewiseCostValues = {}

    # NOTE; these values are relative to the minimum fuel conumption
    model.PowerGenerationPiecewiseFuelValues = {}

    _minimum_production_cost = {}
    _minimum_fuel_consumption = {}

    def _eliminate_piecewise_duplicates(input_func):
        if len(input_func) <= 1:
            return input_func

        new = [input_func[0]]
        for (o1, c1), (o2, c2) in zip(input_func, input_func[1:]):
            if not math.isclose(o1, o2) and not math.isclose(c1, c2):
                new.append((o2, c2))

        return new

    def _much_less_than(v1, v2):
        return v1 < v2 and not math.isclose(v1,v2)

    def _piecewise_adjustment_helper(m, p_min, p_max, input_func):
        minimum_val = 0.
        new_points = []
        new_vals = []

        input_func = _eliminate_piecewise_duplicates(input_func)
        set_p_min = False

        # NOTE: this implicitly inserts a (0.,0.)
        #       into every cost array
        prior_output, prior_cost = 0., 0.

        for output, cost in input_func:
            ## catch this case
            if math.isclose(output, p_min) and math.isclose(output, p_max):
                new_points.append(0.)
                new_vals.append(0.)
                minimum_val = cost
                break

            ## output < p_min
            elif _much_less_than(output, p_min):
                pass

            ## p_min == output
            elif math.isclose(output, p_min):
                assert set_p_min is False
                new_points.append(0.)
                new_vals.append(0.)
                minimum_val = cost
                set_p_min = True

            ## p_min < output
            elif (_much_less_than(p_min, output)
                  and _much_less_than(output, p_max)):
                if not set_p_min:
                    new_points.append(0.)
                    new_vals.append(0.)

                    price = (cost - prior_cost) / (output - prior_output)
                    minimum_val = (p_min - prior_output) * price + prior_cost

                    new_points.append(output - p_min)
                    new_vals.append((output - p_min) * price)
                    set_p_min = True

                else:
                    new_points.append(output - p_min)
                    new_vals.append(cost - minimum_val)

            elif math.isclose(output, p_max) or _much_less_than(p_max, output):
                if not set_p_min:
                    new_points.append(0.)
                    new_vals.append(0.)

                    price = (cost - prior_cost) / (output - prior_output)
                    minimum_val = (p_min - prior_output) * price + prior_cost
                    new_points.append(p_max - p_min)

                    if math.isclose(output, p_max):
                        new_vals.append(cost - minimum_val)
                    else:
                        new_vals.append((p_max - p_min) * price)

                    set_p_min = True

                else:
                    new_points.append(p_max - p_min)

                    if math.isclose(output, p_max):
                        new_vals.append(cost - minimum_val)
                    else:
                        price = (cost - prior_cost) / (output - prior_output)
                        new_vals.append((p_max - prior_output)
                                        * price + prior_cost - minimum_val)

                break

            else:
                raise ModelError(
                    "Unexpected case in _piecewise_adjustment_helper, "
                    "p_min={}, p_max={}, output={}".format(
                        p_min, p_max, output)
                    )

            prior_output, prior_cost = output, cost

        return new_points, new_vals, minimum_val

    def _polynomial_to_piecewise_helper(m, p_min, p_max, input_func):
        segment_max = pe.value(m.NumGeneratorCostCurvePieces)

        for key in {0, 1, 2}:
            if key not in input_func:
                input_func[key] = 0.

        poly_func = lambda x : (input_func[0] + input_func[1] * x
                                + input_func[2]* x * x)

        if p_min >= p_max:
            minimum_val = poly_func(p_min)
            new_points = [0.]
            new_vals = [0.]

            return new_points, new_vals, minimum_val

        elif input_func[2] == 0.: ## not actually quadratic
            minimum_val = poly_func(p_min)
            new_points = [0., p_max - p_min]
            new_vals = [0., poly_func(p_max) - minimum_val]

            return new_points, new_vals, minimum_val

        ## actually quadratic
        width = (p_max - p_min) / float(segment_max)

        new_points = [i * width for i in range(0, segment_max+1)]

        ## replace the last with (p_max - p_min)
        new_points[-1] = p_max - p_min

        minimum_val = poly_func(p_min)
        new_vals = [poly_func(pnt + p_min) - minimum_val for pnt in new_points]

        return new_points, new_vals, minimum_val

    def _piecewise_helper(m, p_min, p_max, curve, curve_type):
        if (curve_type not in curve
                or curve[curve_type] == 'piecewise'):
            return _piecewise_adjustment_helper(m, p_min, p_max,
                                                curve['values'])

        else:
            assert curve[curve_type] == 'polynomial'
            return _polynomial_to_piecewise_helper(m, p_min, p_max,
                                                   curve['values'])

    def power_generation_piecewise_points_rule(m, g):

        ## NOTE: it is often (usually) the case that cost curves
        ##       are the same in every time period, This function
        ##       is optimized to avoid data redunancy and recalculation
        ##       for that case

        gen_dict = gens[g]
        fuel_curve = gen_dict.get('p_fuel')
        cost_curve = gen_dict.get('p_cost')
        fuel_cost = gen_dict.get('fuel_cost', 0.)
        no_load_cost = gen_dict.get('non_fuel_no_load_cost', 0.)

        if isinstance(fuel_cost,dict):
            fuel_costs = fuel_cost['values']
        else:
            fuel_costs = (fuel_cost for t in m.TimePeriods)

        if isinstance(no_load_cost,dict):
            no_load_costs = no_load_cost['values']
        else:
            no_load_costs = (no_load_cost for _ in m.TimePeriods)

        _curve_cache = dict()
        if fuel_curve is not None:
            g_in_fuel_supply_generators = g in m.FuelSupplyGenerators
            g_in_single_fuel_generators = g in m.SingleFuelGenerators

            if (isinstance(fuel_curve,dict)
                    and fuel_curve['data_type'] == 'time_series'):
                fuel_curves = fuel_curve['values']
                one_fuel_curve = False

            else:
                fuel_curves = ( fuel_curve for _ in m.TimePeriods)
                one_fuel_curve = True

            for fuel_curve, fuel_cost, nlc, t in zip(fuel_curves, fuel_costs,
                                                     no_load_costs,
                                                     m.TimePeriods):
                p_min = pe.value(m.MinimumPowerOutput[g,t])
                p_max = pe.value(m.MaximumPowerOutput[g,t])

                if (p_min, p_max, fuel_cost, nlc) in _curve_cache:
                    curve = _curve_cache[p_min, p_max, fuel_cost, nlc]

                    if one_fuel_curve or curve['fuel_curve'] == fuel_curve:
                        m.PowerGenerationPiecewisePoints[g,t] = curve['points']

                        if g_in_fuel_supply_generators:
                            _minimum_fuel_consumption[g, t] = curve[
                                'min_fuel_consumption']
                            m.PowerGenerationPiecewiseFuelValues[g, t] = curve[
                                'fuel_values']

                        if g_in_single_fuel_generators:
                            _minimum_production_cost[g, t] = curve[
                                'min_production_cost']
                            m.PowerGenerationPiecewiseCostValues[g, t] = curve[
                                'cost_values']

                        continue

                points, values, minimum_val = _piecewise_helper(
                    m, p_min, p_max, fuel_curve, 'fuel_curve_type')
                curve = { 'points' : points }

                if not one_fuel_curve:
                    curve['fuel_curve'] = fuel_curve

                m.PowerGenerationPiecewisePoints[g, t] = points
                if g_in_fuel_supply_generators:
                    _minimum_fuel_consumption[g, t] = minimum_val
                    curve['min_fuel_consumption'] = minimum_val

                    m.PowerGenerationPiecewiseFuelValues[g, t] = values
                    curve['fuel_values'] = values

                if g_in_single_fuel_generators:
                    min_production_cost = (minimum_val
                                           * fuel_cost + no_load_cost)
                    _minimum_production_cost[g, t] = min_production_cost
                    curve['min_production_cost'] = min_production_cost

                    cost_values = [fuel_cost * val for val in values]
                    m.PowerGenerationPiecewiseCostValues[g, t] = cost_values
                    curve['cost_values'] = cost_values

                _curve_cache[p_min, p_max, fuel_cost, nlc] = curve

            return ## we can assume below that we don't have a fuel curve

        if (isinstance(cost_curve, dict)
                and cost_curve['data_type'] == 'time_series'):
            cost_curves = cost_curve['values']
            one_cost_curve = False

        else:
            cost_curves = (cost_curve for _ in m.TimePeriods )
            one_cost_curve = True

        for cost_curve, nlc, t in zip(cost_curves, no_load_costs,
                                      m.TimePeriods):
            p_min = pe.value(m.MinimumPowerOutput[g, t])
            p_max = pe.value(m.MaximumPowerOutput[g, t])

            if (p_min, p_max, nlc) in _curve_cache:
                curve = _curve_cache[p_min, p_max, nlc]

                if one_cost_curve or curve['cost_curve'] == cost_curve:
                    m.PowerGenerationPiecewisePoints[g,t] = curve['points']
                    m.PowerGenerationPiecewiseCostValues[g,t] = curve[
                        'cost_values']
                    _minimum_production_cost[g,t] = curve['min_production']

                    continue

            if cost_curve is None:
                if p_min >= p_max: ## only one point
                    points = [0.]
                    values = [0.]

                else:
                    points = [0., p_max - p_min]
                    values = [0., 0.]

                min_production = nlc

            else:
                points, values, minimum_val = _piecewise_helper(
                    m, p_min, p_max, cost_curve, 'cost_curve_type')
                min_production = minimum_val + nlc

            curve = {'points': points, 'cost_values': values,
                     'min_production': min_production}

            if not one_cost_curve:
                curve['cost_curve'] = cost_curve

            _curve_cache[p_min, p_max, nlc] = curve

            m.PowerGenerationPiecewisePoints[g,t] = points
            m.PowerGenerationPiecewiseCostValues[g,t] = values
            _minimum_production_cost[g,t] = min_production

    model.CreatePowerGenerationPiecewisePoints = pe.BuildAction(
        cost_gens, rule=power_generation_piecewise_points_rule)

    # Minimum production cost (needed because Piecewise constraint on
    # ProductionCost has to have lower bound of 0, so the unit can cost 0 when
    # off -- this is added back in to the objective if a unit is on

    if renew_costs:
        model.MinimumProductionCost = pe.Param(
            model.SingleFuelGenerators | model.AllNondispatchableGenerators, #the union of two different sets
            model.TimePeriods,
            within=pe.NonNegativeReals, initialize=_minimum_production_cost,
            mutable=True
            )

        model.MinimumFuelConsumption = pe.Param(
            model.FuelSupplyGenerators | model.AllNondispatchableGenerators,
            model.TimePeriods,
            within=pe.NonNegativeReals, initialize=_minimum_fuel_consumption,
            mutable=True
            )

    else:
        model.MinimumProductionCost = pe.Param(
            model.SingleFuelGenerators, model.TimePeriods,
            within=pe.NonNegativeReals, initialize=_minimum_production_cost,
            mutable=True
            )

        model.MinimumFuelConsumption = pe.Param(
            model.FuelSupplyGenerators, model.TimePeriods,
            within=pe.NonNegativeReals, initialize=_minimum_fuel_consumption,
            mutable=True
            )

    ## END PRODUCTION COST CALCULATIONS

    #########################################
    # penalty costs for constraint violation #
    #########################################

    kinda_big_penalty = (model_data.get_system_attr('reserve_shortfall_cost')
                         * model_data.get_system_attr('baseMVA'))
    big_penalty = (model_data.get_system_attr('load_mismatch_cost')
                   * model_data.get_system_attr('baseMVA'))

    model.ReserveShortfallPenalty = pe.Param(
        within=pe.NonNegativeReals, mutable=True,
        initialize=model_data.get_system_attr('reserve_shortfall_cost',
                                              kinda_big_penalty)
        )

    model.LoadMismatchPenalty = pe.Param(
        within=pe.NonNegativeReals, mutable=True,
        initialize=model_data.get_system_attr('load_mismtch_cost',
                                              big_penalty)
        )
    model.LoadMismatchPenaltyReactive = pe.Param(
        within=pe.NonNegativeReals, mutable=True,
        initialize=model_data.get_system_attr('q_load_mismatch_cost',
                                              big_penalty / 2.)
        )

    model.Contingencies = pe.Set(initialize=contingencies.keys())

    # leaving this unindexed for now for simpility
    model.ContingencyLimitPenalty = pe.Param(
        within=pe.NonNegativeReals,
        initialize=model_data.get_system_attr(
            'contingency_flow_violation_cost', big_penalty / 2.),
        mutable=True
        )

    #
    # STORAGE parameters
    #

    model.Storage = pe.Set(initialize=storage_attrs['names'])
    model.StorageAtBus = pe.Set(model.Buses, initialize=storage_by_bus)

    def verify_storage_buses_rule(m):
        assert set(m.Storage) == {store for bus in m.Buses
                                  for store in m.StorageAtBus[bus]}

    model.VerifyStorageBuses = pe.BuildAction(rule=verify_storage_buses_rule)

    ####################################################################################
    # minimum and maximum power ratings, for each storage unit. units are MW.          #
    # could easily be specified on a per-time period basis, but are not currently.     #
    ####################################################################################

    # Storage power output >0 when discharging

    model.MinimumPowerOutputStorage = pe.Param(
        model.Storage,
        within=pe.NonNegativeReals, default=0.0,
        initialize=storage_attrs.get('min_discharge_rate', dict())
        )

    def maximum_power_output_validator_storage(m, v, s):
        return v >= pe.value(m.MinimumPowerOutputStorage[s])

    model.MaximumPowerOutputStorage = pe.Param(
        model.Storage,
        within=pe.NonNegativeReals,
        validate=maximum_power_output_validator_storage, default=0.0,
        initialize=storage_attrs.get('max_discharge_rate', dict())
        )

    #Storage power input >0 when charging

    model.MinimumPowerInputStorage = pe.Param(
        model.Storage,
        within=pe.NonNegativeReals, default=0.0,
        initialize=storage_attrs.get('min_charge_rate', dict())
        )

    def maximum_power_input_validator_storage(m, v, s):
        return v >= pe.value(m.MinimumPowerInputStorage[s])

    model.MaximumPowerInputStorage = pe.Param(
        model.Storage,
        within=pe.NonNegativeReals,
        validate=maximum_power_input_validator_storage, default=0.0,
        initialize=storage_attrs.get('max_charge_rate', dict())
        )

    ###############################################
    # storage ramp up/down rates. units are MW/h. #
    ###############################################

    # ramp rate limits when discharging
    model.NominalRampUpLimitStorageOutput    = pe.Param(
        model.Storage,
        within=pe.NonNegativeReals,
        initialize=storage_attrs.get('ramp_up_output_60min', dict())
        )
    model.NominalRampDownLimitStorageOutput  = pe.Param(
        model.Storage,
        within=pe.NonNegativeReals,
        initialize=storage_attrs.get('ramp_down_output_60min', dict())
        )

    # ramp rate limits when charging
    model.NominalRampUpLimitStorageInput     = pe.Param(
        model.Storage,
        within=pe.NonNegativeReals,
        initialize=storage_attrs.get('ramp_up_input_60min', dict())
        )
    model.NominalRampDownLimitStorageInput   = pe.Param(
        model.Storage,
        within=pe.NonNegativeReals,
        initialize=storage_attrs.get('ramp_down_input_60min', dict())
        )

    def scale_storage_ramp_up_out(m, s):
        return m.NominalRampUpLimitStorageOutput[s] * m.TimePeriodLengthHours

    model.ScaledNominalRampUpLimitStorageOutput = pe.Param(
        model.Storage,
        within=pe.NonNegativeReals, initialize=scale_storage_ramp_up_out
        )

    def scale_storage_ramp_down_out(m, s):
        return m.NominalRampDownLimitStorageOutput[s] * m.TimePeriodLengthHours

    model.ScaledNominalRampDownLimitStorageOutput = pe.Param(
        model.Storage,
        within=pe.NonNegativeReals, initialize=scale_storage_ramp_down_out
        )

    def scale_storage_ramp_up_in(m, s):
        return m.NominalRampUpLimitStorageInput[s] * m.TimePeriodLengthHours

    model.ScaledNominalRampUpLimitStorageInput = pe.Param(
        model.Storage,
        within=pe.NonNegativeReals, initialize=scale_storage_ramp_up_in
        )

    def scale_storage_ramp_down_in(m, s):
        return m.NominalRampDownLimitStorageInput[s] * m.TimePeriodLengthHours

    model.ScaledNominalRampDownLimitStorageInput = pe.Param(
        model.Storage,
        within=pe.NonNegativeReals, initialize=scale_storage_ramp_down_in
        )

    ####################################################################################
    # minimum state of charge (SOC) and maximum energy ratings, for each storage unit. #
    # units are MWh for energy rating and p.u. (i.e. [0,1]) for SOC     #
    ####################################################################################

    # you enter storage energy ratings once for each storage unit

    model.MaximumEnergyStorage = pe.Param(
        model.Storage,
        within=pe.NonNegativeReals, default=0.0,
        initialize=storage_attrs.get('energy_capacity', dict())
        )
    model.MinimumSocStorage = pe.Param(
        model.Storage,
        within=pe.PercentFraction, default=0.0,
        initialize=storage_attrs.get('minimum_state_of_charge', dict())
        )

    ################################################################################
    # round trip efficiency for each storage unit given as a fraction (i.e. [0,1]) #
    ################################################################################

    model.InputEfficiencyEnergy  = pe.Param(
        model.Storage,
        within=pe.PercentFraction, default=1.0,
        initialize=storage_attrs.get('charge_efficiency', dict())
        )
    model.OutputEfficiencyEnergy = pe.Param(
        model.Storage,
        within=pe.PercentFraction, default=1.0,
        initialize=storage_attrs.get('discharge_efficienty', dict())
        )

    ## assumed to be %/hr
    model.RetentionRate          = pe.Param(
        model.Storage,
        within=pe.PercentFraction, default=1.0,
        initialize=storage_attrs.get('retention_rate_60min', dict())
        )

    model.ChargeCost = pe.Param(
        model.Storage,
        within=pe.Reals, default=0.0,
        initialize=storage_attrs.get('charge_cost', dict())
        )
    model.DischargeCost = pe.Param(
        model.Storage,
        within=pe.Reals, default=0.0,
        initialize=storage_attrs.get('discharge_cost', dict())
        )

    # this will be multiplied by itself 1/m.TimePeriodLengthHours times, so
    # this is the scaling to get us back to %/hr
    def scaled_retention_rate(m,s):
        return (pe.value(m.RetentionRate[s])
                ** pe.value(m.TimePeriodLengthHours))

    model.ScaledRetentionRate = pe.Param(model.Storage,
                                         within=pe.PercentFraction,
                                         initialize=scaled_retention_rate)

    ########################################################################
    # end-point SOC for each storage unit. units are in p.u. (i.e. [0,1])  #
    ########################################################################

    # end-point values are the SOC targets at the final time period. With no
    # end-point constraints storage units will always be empty at the final
    # time period.
    def _end_point_soc(m, s):
        if s is None:
            return

        s_dict = storage[s]
        if 'end_state_of_charge' in s_dict:
            return s_dict['end_state_of_charge']
        if 'initial_state_of_charge' in s_dict:
            return s_dict['initial_state_of_charge']

        return 0.5

    model.EndPointSocStorage = pe.Param(model.Storage,
                                        within=pe.PercentFraction, default=0.5,
                                        initialize=_end_point_soc)

    ############################################################
    # storage initial conditions: SOC, power output and input  #
    ############################################################

    def t0_storage_power_input_validator(m, v, s):
        return (pe.value(m.MinimumPowerInputStorage[s])
                <= v <= pe.value(m.MaximumPowerInputStorage[s]))

    def t0_storage_power_output_validator(m, v, s):
        return (pe.value(m.MinimumPowerOutputStorage[s])
                <= v <= pe.value(m.MaximumPowerOutputStorage[s]))

    model.StoragePowerOutputOnT0 = pe.Param(
        model.Storage,
        within=pe.NonNegativeReals, validate=t0_storage_power_output_validator,
        default=0.0,
        initialize=storage_attrs.get('initial_discharge_rate', dict())
        )
    model.StoragePowerInputOnT0  = pe.Param(
        model.Storage,
        within=pe.NonNegativeReals, validate=t0_storage_power_input_validator,
        default=0.0,
        initialize=storage_attrs.get('initial_charge_rate', dict())
        )

    model.StorageSocOnT0         = pe.Param(
        model.Storage,
        within=pe.PercentFraction, default=0.5,
        initialize=storage_attrs.get('initial_state_of_charge', dict())
        )

    return model


@add_model_attr(component_name)
def default_params(model: pe.ConcreteModel,
                   model_data=None) -> pe.ConcreteModel:
    """This loads unit commitment params from a GridModel object."""
    return load_base_params(model, model_data, renew_costs=False)


@add_model_attr(component_name)
def renew_cost_params(model: pe.ConcreteModel,
                      model_data=None) -> pe.ConcreteModel:
    """This loads unit commitment params from a GridModel object."""
    return load_base_params(model, model_data, renew_costs=True)


# add model parameters
