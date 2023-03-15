from gurobipy import Var

from math import degrees
from egret.models.unit_commitment import _time_series_dict, _preallocated_list
from egret.common.log import logger


def _get_value_general_gurobi(input):
    if type(input) == Var :
        return input.X
    else:
        return input.getValue()

def _save_uc_results(m, relaxed):
    # dual suffix is on top-level model
    if relaxed:
        dual = m.dual

    md = m._model_data

    # save results data to ModelData object
    thermal_gens = dict(
        md.elements(element_type='generator', generator_type='thermal'))
    renewable_gens = dict(
        md.elements(element_type='generator', generator_type='renewable'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    interfaces = dict(md.elements(element_type='interface'))
    contingencies = dict(md.elements(element_type='contingency'))
    storage = dict(md.elements(element_type='storage'))
    zones = dict(md.elements(element_type='zone'))
    areas = dict(md.elements(element_type='area'))
    pg_security_constraints = dict(
        md.elements(element_type='security_constraint',
                    security_constraint_type='pg'))
    dc_branches = dict(md.elements(element_type='dc_branch'))

    data_time_periods = md.data['system']['time_keys']
    reserve_requirement = ('reserve_requirement' in md.data['system'])

    regulation = bool(m._regulation_service)
    spin = bool(m._spinning_reserve)
    nspin = bool(m._non_spinning_reserve)
    supp = bool(m._supplemental_reserve)
    flex = bool(m._flexible_ramping)

    fs = bool(m._fuel_supply)
    fc = bool(m._fuel_consumption)

    ## All prices are in $/(MW*time_period) by construction
    ## time_period_length_hours == hours/time_period, so
    ##    $/(MW*time_period)/time_period_length_hours
    ## == $/(MW*time_period) * (time_period/hours)
    ## == $/(MW*hours) == $/MWh.
    ## All dual values are divided by this quantity
    ## so as to report out $/MWh.
    time_period_length_hours = m._TimePeriodLengthHours

    ## all of the potential constraints that could limit maximum output
    ## Not all unit commitment models have these constraints, so first
    ## we need check if they're on the model object
    ramp_up_avail_potential_constrs = [
        '_EnforceMaxAvailableRampUpRates',
        '_AncillaryServiceRampUpLimit',
        '_power_limit_from_start',
        '_power_limit_from_stop',
        '_power_limit_from_start_stop',
        '_power_limit_from_start_stops',
        '_max_power_limit_from_starts',
        '_EnforceMaxAvailableRampDownRates',
        '_EnforceMaxCapacity',
        '_OAVUpperBound',
        '_EnforceGenerationLimits',
    ]
    ramp_up_avail_constrs = []

    for constr in ramp_up_avail_potential_constrs:
        if hasattr(m, constr):
            ramp_up_avail_constrs.append(getattr(m, constr))

        for g, g_dict in thermal_gens.items():
            pg_dict = _preallocated_list(data_time_periods)
            if reserve_requirement:
                rg_dict = _preallocated_list(data_time_periods)
            commitment_dict = _preallocated_list(data_time_periods)
            commitment_cost_dict = _preallocated_list(data_time_periods)
            production_cost_dict = _preallocated_list(data_time_periods)
            ramp_up_avail_dict = _preallocated_list(data_time_periods)

            if regulation:
                reg_prov = _preallocated_list(data_time_periods)
                reg_up_supp = _preallocated_list(data_time_periods)
                reg_dn_supp = _preallocated_list(data_time_periods)
            if spin:
                spin_supp = _preallocated_list(data_time_periods)
            if nspin:
                nspin_supp = _preallocated_list(data_time_periods)
            if supp:
                supp_supp = _preallocated_list(data_time_periods)
            if flex:
                flex_up_supp = _preallocated_list(data_time_periods)
                flex_dn_supp = _preallocated_list(data_time_periods)
            gfs = (fs and (g in m.FuelSupplyGenerators))
            if gfs:
                fuel_consumed = _preallocated_list(data_time_periods)
            gdf = (fc and (g in m.DualFuelGenerators))
            if gdf:
                aux_fuel_consumed = _preallocated_list(data_time_periods)
            gdsf = (gdf and (g in m.SingleFireDualFuelGenerators))
            if gdsf:
                aux_fuel_indicator = _preallocated_list(data_time_periods)

            for dt, mt in enumerate(m._TimePeriods):
                pg_dict[dt] = m._PowerGeneratedStartupShutdown[g, mt].getValue()
                if reserve_requirement:
                    rg_dict[dt] = _get_value_general_gurobi(m._ReserveProvided[g, mt])
                if relaxed:
                    commitment_dict[dt] = m._UnitOn[g, mt].x
                else:
                    commitment_dict[dt] = int(round(m._UnitOn[g, mt].x))
                commitment_cost_dict[dt] = m._ShutdownCost[g, mt].x
                if g in m._DualFuelGenerators:
                    commitment_cost_dict[dt] += m._DualFuelCommitmentCost[g, mt].x
                    production_cost_dict[dt] = m._DualFuelProductionCost[g, mt].x
                else:
                    commitment_cost_dict[dt] += m._NoLoadCost[g, mt].getValue() + m._StartupCost[g, mt].x
                    production_cost_dict[dt] = m._ProductionCost[g, mt].x

                if regulation:
                    if g in m._AGC_Generators:
                        if relaxed:
                            reg_prov[dt] = m._RegulationOn[g, mt].x
                        else:
                            reg_prov[dt] = int(round(m._RegulationOn[g, mt].x))
                        reg_up_supp[dt] = m._RegulationReserveUp[g, mt].x
                        reg_dn_supp[dt] = m._RegulationReserveDn[g, mt].x
                        commitment_cost_dict[dt] += m._RegulationCostCommitment[g, mt].x
                        production_cost_dict[dt] += m._RegulationCostGeneration[g, mt].x
                    else:
                        reg_prov[dt] = 0.
                        reg_up_supp[dt] = 0.
                        reg_dn_supp[dt] = 0.

                if spin:
                    spin_supp[dt] = m._SpinningReserveDispatched[g, mt].x
                    production_cost_dict[dt] += m._SpinningReserveCostGeneration[g, mt].x

                if nspin:
                    if g in m._NonSpinGenerators:
                        nspin_supp[dt] = m._NonSpinningReserveDispatched[g, mt].x
                        production_cost_dict[dt] += m._NonSpinningReserveCostGeneration[g, mt].x
                    else:
                        nspin_supp[dt] = 0.
                if supp:
                    supp_supp[dt] = m._SupplementalReserveDispatched[g, mt].x
                    production_cost_dict[dt] += m._SupplementalReserveCostGeneration[g, mt].x
                if flex:
                    flex_up_supp[dt] = m._FlexUpProvided[g, mt].x
                    flex_dn_supp[dt] = m._FlexDnProvided[g, mt].x
                if gfs:
                    fuel_consumed[dt] = m._PrimaryFuelConsumed[g, mt].x
                if gdsf:
                    aux_fuel_indicator[dt] = m._UnitOnAuxFuel[g, mt].x
                if gdf:
                    aux_fuel_consumed[dt] = m._AuxiliaryFuelConsumed[g, mt].x

                ## pyomo doesn't add constraints that are skiped to the index set, so we also
                ## need check here if the index exists.
                slack_list = []
                for constr in ramp_up_avail_constrs:
                    if (g, mt) in constr:
                        slack_list.append(constr[g, mt].slack)
                if slack_list != []:
                    ramp_up_avail_dict[dt] = min(slack_list)

            g_dict['pg'] = _time_series_dict(pg_dict)
            if reserve_requirement:
                g_dict['rg'] = _time_series_dict(rg_dict)
            g_dict['commitment'] = _time_series_dict(commitment_dict)
            g_dict['commitment_cost'] = _time_series_dict(commitment_cost_dict)
            g_dict['production_cost'] = _time_series_dict(production_cost_dict)
            if regulation:
                g_dict['reg_provider'] = _time_series_dict(reg_prov)
                g_dict['reg_up_supplied'] = _time_series_dict(reg_up_supp)
                g_dict['reg_down_supplied'] = _time_series_dict(reg_dn_supp)
            if spin:
                g_dict['spinning_supplied'] = _time_series_dict(spin_supp)
            if nspin:
                g_dict['non_spinning_supplied'] = _time_series_dict(nspin_supp)
            if supp:
                g_dict['supplemental_supplied'] = _time_series_dict(supp_supp)
            if flex:
                g_dict['flex_up_supplied'] = _time_series_dict(flex_up_supp)
                g_dict['flex_down_supplied'] = _time_series_dict(flex_dn_supp)
            if gfs:
                g_dict['fuel_consumed'] = _time_series_dict(fuel_consumed)
            if gdsf:
                g_dict['aux_fuel_status'] = _time_series_dict(aux_fuel_indicator)
            if gdf:
                g_dict['aux_fuel_consumed'] = _time_series_dict(aux_fuel_consumed)
            g_dict['headroom'] = _time_series_dict(ramp_up_avail_dict)

        for g, g_dict in renewable_gens.items():
            pg_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m._TimePeriods):
                pg_dict[dt] = m._NondispatchablePowerUsed[g, mt].x
            g_dict['pg'] = _time_series_dict(pg_dict)

        for s, s_dict in storage.items():
            state_of_charge_dict = _preallocated_list(data_time_periods)
            p_discharge_dict = _preallocated_list(data_time_periods)
            p_charge_dict = _preallocated_list(data_time_periods)
            operational_cost_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m._TimePeriods):
                p_discharge_dict[dt] = m._PowerOutputStorage[s, mt].x
                p_charge_dict[dt] = m._PowerInputStorage[s, mt].x
                operational_cost_dict[dt] = m._StorageCost[s, mt].x
                state_of_charge_dict[dt] = m._SocStorage[s, mt].x

            s_dict['p_discharge'] = _time_series_dict(p_discharge_dict)
            s_dict['p_charge'] = _time_series_dict(p_charge_dict)
            s_dict['operational_cost'] = _time_series_dict(operational_cost_dict)
            s_dict['state_of_charge'] = _time_series_dict(state_of_charge_dict)

        for sc, sc_dict in pg_security_constraints.items():
            sc_violation = None
            sc_flow = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m._TimePeriods):
                b = m._TransmissionBlock[mt]
                sc_flow[dt] = b._pgSecurityExpression[sc].x
                if sc in b._pgRelaxedSecuritySet:
                    if sc_violation is None:
                        sc_violation = _preallocated_list(data_time_periods)
                    sc_violation[dt] = b._pgSecuritySlackPos[sc].x - b._pgSecuritySlackNeg[sc].x
            sc_dict['pf'] = _time_series_dict(sc_flow)
            if sc_violation is not None:
                sc_dict['pf_violation'] = _time_series_dict(sc_violation)

        ## NOTE: UC model currently has no notion of separate loads

        if m._power_balance == 'btheta_power_flow':
            for l, l_dict in branches.items():
                pf_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m._TimePeriods):
                    pf_dict[dt] = m._TransmissionBlock[mt]._pf[l].x
                l_dict['pf'] = _time_series_dict(pf_dict)
                if l in m._BranchesWithSlack:
                    pf_violation_dict = _preallocated_list(data_time_periods)
                    for dt, (mt, b) in enumerate(m._TransmissionBlock.items()):
                        if l in b._pf_slack_pos:
                            pf_violation_dict[dt] = b._pf_slack_pos[l].x - b._pf_slack_neg[l].x
                        else:
                            pf_violation_dict[dt] = 0.
                    l_dict['pf_violation'] = _time_series_dict(pf_violation_dict)

            for b, b_dict in buses.items():
                va_dict = _preallocated_list(data_time_periods)
                p_balance_violation_dict = _preallocated_list(data_time_periods)
                pl_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m._TimePeriods):
                    va_dict[dt] = m._TransmissionBlock[mt]._va[b].x
                    p_balance_violation_dict[dt] = m._LoadGenerateMismatch[b, mt].x
                    pl_dict[dt] = m._TransmissionBlock[mt]._pl[b].x
                print(va_dict)
                b_dict['va'] = _time_series_dict([degrees(v) for v in va_dict])
                b_dict['p_balance_violation'] = _time_series_dict(
                    p_balance_violation_dict)
                b_dict['pl'] = _time_series_dict(pl_dict)
                if relaxed:
                    lmp_dict = _preallocated_list(data_time_periods)
                    for dt, mt in enumerate(m._TimePeriods):
                        lmp_dict[dt] = dual[m._TransmissionBlock[
                            mt]._eq_p_balance[b]] / time_period_length_hours
                    b_dict['lmp'] = _time_series_dict(lmp_dict)

            for i, i_dict in interfaces.items():
                pf_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m._TimePeriods):
                    pf_dict[dt] = m._TransmissionBlock[mt]._pfi[i]
                i_dict['pf'] = _time_series_dict(pf_dict)
                if i in m._InterfacesWithSlack:
                    pf_violation_dict = _preallocated_list(data_time_periods)
                    for dt, (mt, b) in enumerate(m._TransmissionBlock.items()):
                        if i in b._pfi_slack_pos:
                            pf_violation_dict[dt] = b._pfi_slack_pos[i].x - b._pfi_slack_neg[i].x
                        else:
                            pf_violation_dict[dt] = 0.
                    i_dict['pf_violation'] = _time_series_dict(pf_violation_dict)

            for k, k_dict in dc_branches.items():
                pf_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m._TimePeriods):
                    pf_dict[dt] = m._HVDCLinePower[k, mt]
                k_dict['pf'] = _time_series_dict(pf_dict)

        elif m._power_balance == 'ptdf_power_flow':
            flows_dict = dict()
            interface_flows_dict = dict()
            voltage_angle_dict = dict()
            if relaxed:
                lmps_dict = dict()
            for mt in m._TimePeriods:
                b = m._TransmissionBlock[mt]
                PTDF = b._PTDF

                branches_idx = PTDF.branches_keys
                PFV, PFV_I, VA = PTDF.calculate_PFV(b)

                flows_dict[mt] = dict()
                for i, bn in enumerate(branches_idx):
                    flows_dict[mt][bn] = PFV[i]

                interface_idx = PTDF.interface_keys
                interface_flows_dict[mt] = dict()
                for i, i_n in enumerate(interface_idx):
                    interface_flows_dict[mt][i_n] = PFV_I[i]

                buses_idx = PTDF.buses_keys
                voltage_angle_dict[mt] = dict()
                for i, bn in enumerate(buses_idx):
                    voltage_angle_dict[mt][bn] = VA[i]

                if relaxed:
                    LMP = PTDF.calculate_LMP(b, dual, b._eq_p_balance)
                    lmps_dict[mt] = dict()
                    for i, bn in enumerate(buses_idx):
                        lmps_dict[mt][bn] = LMP[i]

            for i, i_dict in interfaces.items():
                pf_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m._TimePeriods):
                    pf_dict[dt] = interface_flows_dict[mt][i]
                i_dict['pf'] = _time_series_dict(pf_dict)
                if i in m._InterfacesWithSlack:
                    pf_violation_dict = _preallocated_list(data_time_periods)
                    for dt, (mt, b) in enumerate(m._TransmissionBlock.items()):
                        if i in b._pfi_slack_pos:
                            pf_violation_dict[dt] = b._pfi_slack_pos[i].x - b._pfi_slack_neg[i].x
                        else:
                            pf_violation_dict[dt] = 0.
                    i_dict['pf_violation'] = _time_series_dict(pf_violation_dict)

            if contingencies:
                for dt, (mt, b) in enumerate(m._TransmissionBlock.items()):
                    contingency_flows = PTDF.calculate_monitored_contingency_flows(
                        b)
                    for (cn, bn), flow in contingency_flows.items():
                        c_dict = contingencies[cn]
                        if 'monitored_branches' not in c_dict:
                            c_dict['monitored_branches'] = _time_series_dict(
                                [{} for _ in data_time_periods])
                        monitored_branches = \
                        c_dict['monitored_branches']['values'][dt]
                        monitored_branches[bn] = {'pf': flow}
                        if (cn, bn) in b._pfc_slack_pos:
                            monitored_branches[bn]['pf_violation'] = b._pfc_slack_pos[cn, bn].x - b._pfc_slack_neg[cn, bn].x

            for l, l_dict in branches.items():
                pf_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m._TimePeriods):
                    ## if the key doesn't exist, it is because that line was out
                    pf_dict[dt] = flows_dict[mt].get(l, 0.)
                l_dict['pf'] = _time_series_dict(pf_dict)
                if l in m._BranchesWithSlack:
                    pf_violation_dict = _preallocated_list(data_time_periods)
                    for dt, (mt, b) in enumerate(m._TransmissionBlock.items()):
                        if l in b._pf_slack_pos:
                            pf_violation_dict[dt] = b._pf_slack_pos[l].x - b._pf_slack_neg[l].x
                        else:
                            pf_violation_dict[dt] = 0.
                    l_dict['pf_violation'] = _time_series_dict(pf_violation_dict)

            for b, b_dict in buses.items():
                va_dict = _preallocated_list(data_time_periods)
                p_balance_violation_dict = _preallocated_list(data_time_periods)
                pl_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m._TimePeriods):
                    p_balance_violation_dict[dt] = m._LoadGenerateMismatch[b, mt].getValue()
                    pl_dict[dt] = m._TransmissionBlock[mt]._pl[b]
                    va_dict[dt] = voltage_angle_dict[mt][b]
                b_dict['p_balance_violation'] = _time_series_dict(
                    p_balance_violation_dict)
                b_dict['pl'] = _time_series_dict(pl_dict)
                if all(x is None for x in va_dict):
                    b_dict['va'] = _time_series_dict([degrees(v) for v in va_dict])
                if relaxed:
                    lmp_dict = _preallocated_list(data_time_periods)
                    for dt, mt in enumerate(m._TimePeriods):
                        lmp_dict[dt] = lmps_dict[mt][b] / time_period_length_hours
                    b_dict['lmp'] = _time_series_dict(lmp_dict)

            for k, k_dict in dc_branches.items():
                pf_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m._TimePeriods):
                    pf_dict[dt] = m._HVDCLinePower[k, mt].x
                k_dict['pf'] = _time_series_dict(pf_dict)

        elif m._power_balance == 'power_balance_constraints':
            for l, l_dict in branches.items():
                pf_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m._TimePeriods):
                    pf_dict[dt] = m._LinePower[l, mt].x
                l_dict['pf'] = _time_series_dict(pf_dict)
                if l in m._BranchesWithSlack:
                    pf_violation_dict = _preallocated_list(data_time_periods)
                    for dt, mt in enumerate(m._TimePeriods):
                        pf_violation_dict[dt] = m._BranchSlackPos[l, mt].x - m._BranchSlackNeg[l, mt].x
                    l_dict['pf_violation'] = _time_series_dict(pf_violation_dict)

            for b, b_dict in buses.items():
                va_dict = _preallocated_list(data_time_periods)
                p_balance_violation_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m._TimePeriods):
                    va_dict[dt] = m._Angle[b, mt].x
                    p_balance_violation_dict[dt] = m._LoadGenerateMismatch[b, mt].x
                b_dict['va'] = _time_series_dict([degrees(v) for v in va_dict])
                b_dict['p_balance_violation'] = _time_series_dict(
                    p_balance_violation_dict)
                if relaxed:
                    lmp_dict = _preallocated_list(data_time_periods)
                    for dt, mt in enumerate(m._TimePeriods):
                        lmp_dict[dt] = dual[m._PowerBalance[b, mt]].x/ time_period_length_hours
                    b_dict['lmp'] = _time_series_dict(lmp_dict)

            for i, i_dict in interfaces.items():
                pf_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m._TimePeriods):
                    pf_dict[dt] = m._InterfaceFlow[i, mt]
                i_dict['pf'] = _time_series_dict(pf_dict)
                if i in m._InterfacesWithSlack:
                    pf_violation_dict = _preallocated_list(data_time_periods)
                    for dt, mt in enumerate(m._TimePeriods):
                        pf_violation_dict[dt] = m._InterfaceSlackPos[i, mt].x - m._InterfaceSlackNeg[
                                i, mt].x
                    i_dict['pf_violation'] = _time_series_dict(pf_violation_dict)

            for k, k_dict in dc_branches.items():
                pf_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m._TimePeriods):
                    pf_dict[dt] = m._HVDCLinePower[k, mt].x
                k_dict['pf'] = _time_series_dict(pf_dict)

        elif m._power_balance in ['copperplate_power_flow',
                                 'copperplate_relaxed_power_flow']:
            sys_dict = md.data['system']
            p_viol_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m._TimePeriods):
                p_viol_dict[dt] = sum(
                   m._LoadGenerateMismatch[b, mt].x for b in m._Buses)
            sys_dict['p_balance_violation'] = _time_series_dict(p_viol_dict)
            if relaxed:
                p_price_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m._TimePeriods):
                    p_price_dict[dt] =dual[m._TransmissionBlock[
                        mt]._eq_p_balance].x / time_period_length_hours
                sys_dict['p_price'] = _time_series_dict(p_price_dict)
        else:
            raise Exception("Unrecongized network type " + m._power_balance)

        if reserve_requirement:
            ## populate the system attributes
            sys_dict = md.data['system']
            sr_s_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m._TimePeriods):
                sr_s_dict[dt] = m._ReserveShortfall[mt].x
            sys_dict['reserve_shortfall'] = _time_series_dict(sr_s_dict)
            if relaxed:
                sr_p_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m._TimePeriods):
                    ## TODO: if the 'relaxed' flag is set, we should automatically
                    ##       pick a formulation which uses the MLR reserve constraints
                    sr_p_dict[dt] = dual[m._EnforceReserveRequirements[
                        mt]].x / time_period_length_hours
                sys_dict['reserve_price'] = _time_series_dict(sr_p_dict)

        ## TODO: Can the code above this be re-factored in a similar way?
        ## as we add more zonal reserve products, they can be added here
        _zonal_reserve_map = dict()
        _system_reserve_map = dict()
        if spin:
            _zonal_reserve_map['spinning_reserve_requirement'] = {
                'shortfall': 'spinning_reserve_shortfall',
                'price': 'spinning_reserve_price',
                'shortfall_m': m._ZonalSpinningReserveShortfall,
                'balance_m': m._EnforceZonalSpinningReserveRequirement,
                }
            _system_reserve_map['spinning_reserve_requirement'] = {
                'shortfall': 'spinning_reserve_shortfall',
                'price': 'spinning_reserve_price',
                'shortfall_m': m._SystemSpinningReserveShortfall,
                'balance_m': m._EnforceSystemSpinningReserveRequirement,
                }
        if nspin:
            _zonal_reserve_map['non_spinning_reserve_requirement'] = {
                'shortfall': 'non_spinning_reserve_shortfall',
                'price': 'non_spinning_reserve_price',
                'shortfall_m': m._ZonalNonSpinningReserveShortfall,
                'balance_m': m._EnforceNonSpinningZonalReserveRequirement,
                }
            _system_reserve_map['non_spinning_reserve_requirement'] = {
                'shortfall': 'non_spinning_reserve_shortfall',
                'price': 'non_spinning_reserve_price',
                'shortfall_m': m._SystemNonSpinningReserveShortfall,
                'balance_m': m._EnforceSystemNonSpinningReserveRequirement,
                }
        if regulation:
            _zonal_reserve_map['regulation_up_requirement'] = {
                'shortfall': 'regulation_up_shortfall',
                'price': 'regulation_up_price',
                'shortfall_m': m._ZonalRegulationUpShortfall,
                'balance_m': m._EnforceZonalRegulationUpRequirements,
                }
            _system_reserve_map['regulation_up_requirement'] = {
                'shortfall': 'regulation_up_shortfall',
                'price': 'regulation_up_price',
                'shortfall_m': m._SystemRegulationUpShortfall,
                'balance_m': m._EnforceSystemRegulationUpRequirement,
                }
            _zonal_reserve_map['regulation_down_requirement'] = {
                'shortfall': 'regulation_down_shortfall',
                'price': 'regulation_down_price',
                'shortfall_m': m._ZonalRegulationDnShortfall,
                'balance_m': m._EnforceZonalRegulationDnRequirements,
                }
            _system_reserve_map['regulation_down_requirement'] = {
                'shortfall': 'regulation_down_shortfall',
                'price': 'regulation_down_price',
                'shortfall_m': m._SystemRegulationDnShortfall,
                'balance_m': m._EnforceSystemRegulationDnRequirement,
                }
        if flex:
            _zonal_reserve_map['flexible_ramp_up_requirement'] = {
                'shortfall': 'flexible_ramp_up_shortfall',
                'price': 'flexible_ramp_up_price',
                'shortfall_m': m._ZonalFlexUpShortfall,
                'balance_m': m._ZonalFlexUpRequirementConstr,
                }
            _system_reserve_map['flexible_ramp_up_requirement'] = {
                'shortfall': 'flexible_ramp_up_shortfall',
                'price': 'flexible_ramp_up_price',
                'shortfall_m': m._SystemFlexUpShortfall,
                'balance_m': m._SystemFlexUpRequirementConstr,
                }
            _zonal_reserve_map['flexible_ramp_down_requirement'] = {
                'shortfall': 'flexible_ramp_down_shortfall',
                'price': 'flexible_ramp_down_price',
                'shortfall_m': m._ZonalFlexDnShortfall,
                'balance_m': m._ZonalFlexDnRequirementConstr,
                }
            _system_reserve_map['flexible_ramp_down_requirement'] = {
                'shortfall': 'flexible_ramp_down_shortfall',
                'price': 'flexible_ramp_down_price',
                'shortfall_m': m._SystemFlexDnShortfall,
                'balance_m': m._SystemFlexDnRequirementConstr,
                }
        if supp:
            _zonal_reserve_map['supplemental_reserve_requirement'] = {
                'shortfall': 'supplemental_shortfall',
                'price': 'supplemental_price',
                'shortfall_m': m._ZonalSupplementalReserveShortfall,
                'balance_m': m._EnforceZonalSupplementalReserveRequirement,
                }

            _system_reserve_map['supplemental_reserve_requirement'] = {
                'shortfall': 'supplemental_shortfall',
                'price': 'supplemental_price',
                'shortfall_m': m._SystemSupplementalReserveShortfall,
                'balance_m': m._EnforceSystemSupplementalReserveRequirement,
                }

        def _populate_zonal_reserves(elements_dict, string_handle):
            for e, e_dict in elements_dict.items():
                me = string_handle + e
                for req, req_dict in _zonal_reserve_map.items():
                    if req in e_dict:
                        req_shortfall_dict = _preallocated_list(data_time_periods)
                        for dt, mt in enumerate(m._TimePeriods):
                            req_shortfall_dict[dt] = req_dict['shortfall_m'][me, mt].x
                        e_dict[req_dict['shortfall']] = _time_series_dict(
                            req_shortfall_dict)
                        if relaxed:
                            req_price_dict = _preallocated_list(data_time_periods)
                            for dt, mt in enumerate(m._TimePeriods):
                                req_price_dict[dt] = dual[req_dict[
                                    'balance_m'][
                                    me, mt]].x / time_period_length_hours
                            e_dict[req_dict['price']] = _time_series_dict(
                                req_price_dict)

        def _populate_system_reserves(sys_dict):
            for req, req_dict in _system_reserve_map.items():
                if req in sys_dict:
                    req_shortfall_dict = _preallocated_list(data_time_periods)
                    for dt, mt in enumerate(m._TimePeriods):
                        req_shortfall_dict[dt] = req_dict['shortfall_m'][mt].x
                    sys_dict[req_dict['shortfall']] = _time_series_dict(
                        req_shortfall_dict)
                    if relaxed:
                        req_price_dict = _preallocated_list(data_time_periods)
                        for dt, mt in enumerate(m._TimePeriods):
                            req_price_dict[dt] = dual[req_dict['balance_m'][
                                mt]].x/ time_period_length_hours
                        sys_dict[req_dict['price']] = _time_series_dict(
                            req_price_dict)

        _populate_zonal_reserves(areas, 'area_')
        _populate_zonal_reserves(zones, 'zone_')

        _populate_system_reserves(md.data['system'])

        if fs:
            fuel_supplies = dict(md.elements(element_type='fuel_supply'))
            for f, f_dict in fuel_supplies.items():
                fuel_consumed = _preallocated_list(data_time_periods)
                fuel_supply_type = f_dict['fuel_supply_type']
                if fuel_supply_type == 'instantaneous':
                    for dt, mt in enumerate(m._TimePeriods):
                        fuel_consumed[dt] = m._TotalFuelConsumedAtInstFuelSupply[f, mt].x
                else:
                    logger.warning(
                        'WARNING: unrecongized fuel_supply_type {} for fuel_supply {}'.format(
                            fuel_supply_type, f))
                f_dict['fuel_consumed'] = _time_series_dict(fuel_consumed)

        md.data['system']['total_cost'] = m.ObjVal
    return md