
from math import degrees
from pyomo.environ import value as pe_value
from egret.models.unit_commitment import _time_series_dict, _preallocated_list
from egret.common.log import logger


class ModelError(Exception):
    pass


def _save_uc_results(m, relaxed):
    # dual suffix is on top-level model
    if relaxed:
        dual = m.model().dual

    md = m.model_data

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

    regulation = bool(m.regulation_service)
    spin = bool(m.spinning_reserve)
    nspin = bool(m.non_spinning_reserve)
    supp = bool(m.supplemental_reserve)
    flex = bool(m.flexible_ramping)

    fs = bool(m.fuel_supply)
    fc = bool(m.fuel_consumption)

    ## All prices are in $/(MW*time_period) by construction
    ## time_period_length_hours == hours/time_period, so
    ##    $/(MW*time_period)/time_period_length_hours
    ## == $/(MW*time_period) * (time_period/hours)
    ## == $/(MW*hours) == $/MWh.
    ## All dual values are divided by this quantity
    ## so as to report out $/MWh.
    time_period_length_hours = pe_value(m.TimePeriodLengthHours)

    ## all of the potential constraints that could limit maximum output
    ## Not all unit commitment models have these constraints, so first
    ## we need check if they're on the model object
    ramp_up_avail_potential_constrs = [
        'EnforceMaxAvailableRampUpRates',
        'AncillaryServiceRampUpLimit',
        'power_limit_from_start',
        'power_limit_from_stop',
        'power_limit_from_start_stop',
        'power_limit_from_start_stops',
        'max_power_limit_from_starts',
        'EnforceMaxAvailableRampDownRates',
        'EnforceMaxCapacity',
        'OAVUpperBound',
        'EnforceGenerationLimits',
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

        for dt, mt in enumerate(m.TimePeriods):
            pg_dict[dt] = pe_value(m.PowerGeneratedStartupShutdown[g, mt])
            if reserve_requirement:
                rg_dict[dt] = pe_value(m.ReserveProvided[g, mt])
            if relaxed:
                commitment_dict[dt] = pe_value(m.UnitOn[g, mt])
            else:
                commitment_dict[dt] = int(round(pe_value(m.UnitOn[g, mt])))
            commitment_cost_dict[dt] = pe_value(m.ShutdownCost[g, mt])
            if g in m.DualFuelGenerators:
                commitment_cost_dict[dt] += pe_value(
                    m.DualFuelCommitmentCost[g, mt])
                production_cost_dict[dt] = pe_value(
                    m.DualFuelProductionCost[g, mt])
            else:
                commitment_cost_dict[dt] += pe_value(
                    m.NoLoadCost[g, mt] + m.StartupCost[g, mt])
                production_cost_dict[dt] = pe_value(m.ProductionCost[g, mt])

            if regulation:
                if g in m.AGC_Generators:
                    if relaxed:
                        reg_prov[dt] = pe_value(m.RegulationOn[g, mt])
                    else:
                        reg_prov[dt] = int(round(value(m.RegulationOn[g, mt])))
                    reg_up_supp[dt] = pe_value(m.RegulationReserveUp[g, mt])
                    reg_dn_supp[dt] = pe_value(m.RegulationReserveDn[g, mt])
                    commitment_cost_dict[dt] += pe_value(
                        m.RegulationCostCommitment[g, mt])
                    production_cost_dict[dt] += pe_value(
                        m.RegulationCostGeneration[g, mt])
                else:
                    reg_prov[dt] = 0.
                    reg_up_supp[dt] = 0.
                    reg_dn_supp[dt] = 0.

            if spin:
                spin_supp[dt] = pe_value(m.SpinningReserveDispatched[g, mt])
                production_cost_dict[dt] += pe_value(
                    m.SpinningReserveCostGeneration[g, mt])

            if nspin:
                if g in m.NonSpinGenerators:
                    nspin_supp[dt] = pe_value(
                        m.NonSpinningReserveDispatched[g, mt])
                    production_cost_dict[dt] += pe_value(
                        m.NonSpinningReserveCostGeneration[g, mt])
                else:
                    nspin_supp[dt] = 0.
            if supp:
                supp_supp[dt] = pe_value(m.SupplementalReserveDispatched[g, mt])
                production_cost_dict[dt] += pe_value(
                    m.SupplementalReserveCostGeneration[g, mt])
            if flex:
                flex_up_supp[dt] = pe_value(m.FlexUpProvided[g, mt])
                flex_dn_supp[dt] = pe_value(m.FlexDnProvided[g, mt])
            if gfs:
                fuel_consumed[dt] = pe_value(m.PrimaryFuelConsumed[g, mt])
            if gdsf:
                aux_fuel_indicator[dt] = pe_value(m.UnitOnAuxFuel[g, mt])
            if gdf:
                aux_fuel_consumed[dt] = pe_value(m.AuxiliaryFuelConsumed[g, mt])

            ## pyomo doesn't add constraints that are skiped to the index set, so we also
            ## need check here if the index exists.
            slack_list = []
            for constr in ramp_up_avail_constrs:
                if (g, mt) in constr:
                    slack_list.append(constr[g, mt].slack())

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
        for dt, mt in enumerate(m.TimePeriods):
            pg_dict[dt] = pe_value(m.NondispatchablePowerUsed[g, mt])
        g_dict['pg'] = _time_series_dict(pg_dict)

    for s, s_dict in storage.items():
        state_of_charge_dict = _preallocated_list(data_time_periods)
        p_discharge_dict = _preallocated_list(data_time_periods)
        p_charge_dict = _preallocated_list(data_time_periods)
        operational_cost_dict = _preallocated_list(data_time_periods)
        for dt, mt in enumerate(m.TimePeriods):
            p_discharge_dict[dt] = pe_value(m.PowerOutputStorage[s, mt])
            p_charge_dict[dt] = pe_value(m.PowerInputStorage[s, mt])
            operational_cost_dict[dt] = pe_value(m.StorageCost[s, mt])
            state_of_charge_dict[dt] = pe_value(m.SocStorage[s, mt])

        s_dict['p_discharge'] = _time_series_dict(p_discharge_dict)
        s_dict['p_charge'] = _time_series_dict(p_charge_dict)
        s_dict['operational_cost'] = _time_series_dict(operational_cost_dict)
        s_dict['state_of_charge'] = _time_series_dict(state_of_charge_dict)

    for sc, sc_dict in pg_security_constraints.items():
        sc_violation = None
        sc_flow = _preallocated_list(data_time_periods)
        for dt, mt in enumerate(m.TimePeriods):
            b = m.TransmissionBlock[mt]
            sc_flow[dt] = pe_value(b.pgSecurityExpression[sc])
            if sc in b.pgRelaxedSecuritySet:
                if sc_violation is None:
                    sc_violation = _preallocated_list(data_time_periods)
                sc_violation[dt] = pe_value(
                    b.pgSecuritySlackPos[sc] - b.pgSecuritySlackNeg[sc])
        sc_dict['pf'] = _time_series_dict(sc_flow)
        if sc_violation is not None:
            sc_dict['pf_violation'] = _time_series_dict(sc_violation)

    ## NOTE: UC model currently has no notion of separate loads

    if m.power_balance == 'btheta_power_flow':
        for l, l_dict in branches.items():
            pf_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m.TimePeriods):
                pf_dict[dt] = pe_value(m.TransmissionBlock[mt].pf[l])
            l_dict['pf'] = _time_series_dict(pf_dict)
            if l in m.BranchesWithSlack:
                pf_violation_dict = _preallocated_list(data_time_periods)
                for dt, (mt, b) in enumerate(m.TransmissionBlock.items()):
                    if l in b.pf_slack_pos:
                        pf_violation_dict[dt] = pe_value(
                            b.pf_slack_pos[l] - b.pf_slack_neg[l])
                    else:
                        pf_violation_dict[dt] = 0.
                l_dict['pf_violation'] = _time_series_dict(pf_violation_dict)

        for b, b_dict in buses.items():
            va_dict = _preallocated_list(data_time_periods)
            p_balance_violation_dict = _preallocated_list(data_time_periods)
            pl_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m.TimePeriods):
                va_dict[dt] = pe_value(m.TransmissionBlock[mt].va[b])
                p_balance_violation_dict[dt] = pe_value(
                    m.LoadGenerateMismatch[b, mt])
                pl_dict[dt] = pe_value(m.TransmissionBlock[mt].pl[b])
            b_dict['va'] = _time_series_dict([degrees(v) for v in va_dict])
            b_dict['p_balance_violation'] = _time_series_dict(
                p_balance_violation_dict)
            b_dict['pl'] = _time_series_dict(pl_dict)
            if relaxed:
                lmp_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m.TimePeriods):
                    lmp_dict[dt] = pe_value(dual[m.TransmissionBlock[
                        mt].eq_p_balance[b]]) / time_period_length_hours
                b_dict['lmp'] = _time_series_dict(lmp_dict)

        for i, i_dict in interfaces.items():
            pf_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m.TimePeriods):
                pf_dict[dt] = pe_value(m.TransmissionBlock[mt].pfi[i])
            i_dict['pf'] = _time_series_dict(pf_dict)
            if i in m.InterfacesWithSlack:
                pf_violation_dict = _preallocated_list(data_time_periods)
                for dt, (mt, b) in enumerate(m.TransmissionBlock.items()):
                    if i in b.pfi_slack_pos:
                        pf_violation_dict[dt] = pe_value(
                            b.pfi_slack_pos[i] - b.pfi_slack_neg[i])
                    else:
                        pf_violation_dict[dt] = 0.
                i_dict['pf_violation'] = _time_series_dict(pf_violation_dict)

        for k, k_dict in dc_branches.items():
            pf_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m.TimePeriods):
                pf_dict[dt] = pe_value(m.HVDCLinePower[k, mt])
            k_dict['pf'] = _time_series_dict(pf_dict)

    elif m.power_balance == 'ptdf_power_flow':
        flows_dict = dict()
        interface_flows_dict = dict()
        voltage_angle_dict = dict()
        if relaxed:
            lmps_dict = dict()
        for mt in m.TimePeriods:
            b = m.TransmissionBlock[mt]
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
                LMP = PTDF.calculate_LMP(b, dual, b.eq_p_balance)
                lmps_dict[mt] = dict()
                for i, bn in enumerate(buses_idx):
                    lmps_dict[mt][bn] = LMP[i]

        for i, i_dict in interfaces.items():
            pf_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m.TimePeriods):
                pf_dict[dt] = interface_flows_dict[mt][i]
            i_dict['pf'] = _time_series_dict(pf_dict)
            if i in m.InterfacesWithSlack:
                pf_violation_dict = _preallocated_list(data_time_periods)
                for dt, (mt, b) in enumerate(m.TransmissionBlock.items()):
                    if i in b.pfi_slack_pos:
                        pf_violation_dict[dt] = pe_value(
                            b.pfi_slack_pos[i] - b.pfi_slack_neg[i])
                    else:
                        pf_violation_dict[dt] = 0.
                i_dict['pf_violation'] = _time_series_dict(pf_violation_dict)

        if contingencies:
            for dt, (mt, b) in enumerate(m.TransmissionBlock.items()):
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
                    if (cn, bn) in b.pfc_slack_pos:
                        monitored_branches[bn]['pf_violation'] = pe_value(
                            b.pfc_slack_pos[cn, bn] - b.pfc_slack_neg[cn, bn])

        for l, l_dict in branches.items():
            pf_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m.TimePeriods):
                ## if the key doesn't exist, it is because that line was out
                pf_dict[dt] = flows_dict[mt].get(l, 0.)
            l_dict['pf'] = _time_series_dict(pf_dict)
            if l in m.BranchesWithSlack:
                pf_violation_dict = _preallocated_list(data_time_periods)
                for dt, (mt, b) in enumerate(m.TransmissionBlock.items()):
                    if l in b.pf_slack_pos:
                        pf_violation_dict[dt] = pe_value(
                            b.pf_slack_pos[l] - b.pf_slack_neg[l])
                    else:
                        pf_violation_dict[dt] = 0.
                l_dict['pf_violation'] = _time_series_dict(pf_violation_dict)

        for b, b_dict in buses.items():
            va_dict = _preallocated_list(data_time_periods)
            p_balance_violation_dict = _preallocated_list(data_time_periods)
            pl_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m.TimePeriods):
                p_balance_violation_dict[dt] = pe_value(
                    m.LoadGenerateMismatch[b, mt])
                pl_dict[dt] = pe_value(m.TransmissionBlock[mt].pl[b])
                va_dict[dt] = voltage_angle_dict[mt][b]
            b_dict['p_balance_violation'] = _time_series_dict(
                p_balance_violation_dict)
            b_dict['pl'] = _time_series_dict(pl_dict)
            b_dict['va'] = _time_series_dict([degrees(v) for v in va_dict])
            if relaxed:
                lmp_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m.TimePeriods):
                    lmp_dict[dt] = lmps_dict[mt][b] / time_period_length_hours
                b_dict['lmp'] = _time_series_dict(lmp_dict)

        for k, k_dict in dc_branches.items():
            pf_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m.TimePeriods):
                pf_dict[dt] = pe_value(m.HVDCLinePower[k, mt])
            k_dict['pf'] = _time_series_dict(pf_dict)

    elif m.power_balance == 'power_balance_constraints':
        for l, l_dict in branches.items():
            pf_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m.TimePeriods):
                pf_dict[dt] = pe_value(m.LinePower[l, mt])
            l_dict['pf'] = _time_series_dict(pf_dict)
            if l in m.BranchesWithSlack:
                pf_violation_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m.TimePeriods):
                    pf_violation_dict[dt] = pe_value(
                        m.BranchSlackPos[l, mt] - m.BranchSlackNeg[l, mt])
                l_dict['pf_violation'] = _time_series_dict(pf_violation_dict)

        for b, b_dict in buses.items():
            va_dict = _preallocated_list(data_time_periods)
            p_balance_violation_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m.TimePeriods):
                va_dict[dt] = pe_value(m.Angle[b, mt])
                p_balance_violation_dict[dt] = pe_value(
                    m.LoadGenerateMismatch[b, mt])
            b_dict['va'] = _time_series_dict([degrees(v) for v in va_dict])
            b_dict['p_balance_violation'] = _time_series_dict(
                p_balance_violation_dict)
            if relaxed:
                lmp_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m.TimePeriods):
                    lmp_dict[dt] = pe_value(
                        dual[m.PowerBalance[b, mt]]) / time_period_length_hours
                b_dict['lmp'] = _time_series_dict(lmp_dict)

        for i, i_dict in interfaces.items():
            pf_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m.TimePeriods):
                pf_dict[dt] = pe_value(m.InterfaceFlow[i, mt])
            i_dict['pf'] = _time_series_dict(pf_dict)
            if i in m.InterfacesWithSlack:
                pf_violation_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m.TimePeriods):
                    pf_violation_dict[dt] = pe_value(
                        m.InterfaceSlackPos[i, mt] - m.InterfaceSlackNeg[
                            i, mt])
                i_dict['pf_violation'] = _time_series_dict(pf_violation_dict)

        for k, k_dict in dc_branches.items():
            pf_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m.TimePeriods):
                pf_dict[dt] = pe_value(m.HVDCLinePower[k, mt])
            k_dict['pf'] = _time_series_dict(pf_dict)

    elif m.power_balance in ['copperplate_power_flow',
                             'copperplate_relaxed_power_flow']:
        sys_dict = md.data['system']
        p_viol_dict = _preallocated_list(data_time_periods)
        for dt, mt in enumerate(m.TimePeriods):
            p_viol_dict[dt] = sum(
                pe_value(m.LoadGenerateMismatch[b, mt]) for b in m.Buses)
        sys_dict['p_balance_violation'] = _time_series_dict(p_viol_dict)
        if relaxed:
            p_price_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m.TimePeriods):
                p_price_dict[dt] = pe_value(dual[m.TransmissionBlock[
                    mt].eq_p_balance]) / time_period_length_hours
            sys_dict['p_price'] = _time_series_dict(p_price_dict)
    else:
        raise Exception("Unrecongized network type " + m.power_balance)

    if reserve_requirement:
        ## populate the system attributes
        sys_dict = md.data['system']
        sr_s_dict = _preallocated_list(data_time_periods)
        for dt, mt in enumerate(m.TimePeriods):
            sr_s_dict[dt] = pe_value(m.ReserveShortfall[mt])
        sys_dict['reserve_shortfall'] = _time_series_dict(sr_s_dict)
        if relaxed:
            sr_p_dict = _preallocated_list(data_time_periods)
            for dt, mt in enumerate(m.TimePeriods):
                ## TODO: if the 'relaxed' flag is set, we should automatically
                ##       pick a formulation which uses the MLR reserve constraints
                sr_p_dict[dt] = pe_value(dual[m.EnforceReserveRequirements[
                    mt]]) / time_period_length_hours
            sys_dict['reserve_price'] = _time_series_dict(sr_p_dict)

    ## TODO: Can the code above this be re-factored in a similar way?
    ## as we add more zonal reserve products, they can be added here
    _zonal_reserve_map = dict()
    _system_reserve_map = dict()
    if spin:
        _zonal_reserve_map['spinning_reserve_requirement'] = {
            'shortfall': 'spinning_reserve_shortfall',
            'price': 'spinning_reserve_price',
            'shortfall_m': m.ZonalSpinningReserveShortfall,
            'balance_m': m.EnforceZonalSpinningReserveRequirement,
            }
        _system_reserve_map['spinning_reserve_requirement'] = {
            'shortfall': 'spinning_reserve_shortfall',
            'price': 'spinning_reserve_price',
            'shortfall_m': m.SystemSpinningReserveShortfall,
            'balance_m': m.EnforceSystemSpinningReserveRequirement,
            }
    if nspin:
        _zonal_reserve_map['non_spinning_reserve_requirement'] = {
            'shortfall': 'non_spinning_reserve_shortfall',
            'price': 'non_spinning_reserve_price',
            'shortfall_m': m.ZonalNonSpinningReserveShortfall,
            'balance_m': m.EnforceNonSpinningZonalReserveRequirement,
            }
        _system_reserve_map['non_spinning_reserve_requirement'] = {
            'shortfall': 'non_spinning_reserve_shortfall',
            'price': 'non_spinning_reserve_price',
            'shortfall_m': m.SystemNonSpinningReserveShortfall,
            'balance_m': m.EnforceSystemNonSpinningReserveRequirement,
            }
    if regulation:
        _zonal_reserve_map['regulation_up_requirement'] = {
            'shortfall': 'regulation_up_shortfall',
            'price': 'regulation_up_price',
            'shortfall_m': m.ZonalRegulationUpShortfall,
            'balance_m': m.EnforceZonalRegulationUpRequirements,
            }
        _system_reserve_map['regulation_up_requirement'] = {
            'shortfall': 'regulation_up_shortfall',
            'price': 'regulation_up_price',
            'shortfall_m': m.SystemRegulationUpShortfall,
            'balance_m': m.EnforceSystemRegulationUpRequirement,
            }
        _zonal_reserve_map['regulation_down_requirement'] = {
            'shortfall': 'regulation_down_shortfall',
            'price': 'regulation_down_price',
            'shortfall_m': m.ZonalRegulationDnShortfall,
            'balance_m': m.EnforceZonalRegulationDnRequirements,
            }
        _system_reserve_map['regulation_down_requirement'] = {
            'shortfall': 'regulation_down_shortfall',
            'price': 'regulation_down_price',
            'shortfall_m': m.SystemRegulationDnShortfall,
            'balance_m': m.EnforceSystemRegulationDnRequirement,
            }
    if flex:
        _zonal_reserve_map['flexible_ramp_up_requirement'] = {
            'shortfall': 'flexible_ramp_up_shortfall',
            'price': 'flexible_ramp_up_price',
            'shortfall_m': m.ZonalFlexUpShortfall,
            'balance_m': m.ZonalFlexUpRequirementConstr,
            }
        _system_reserve_map['flexible_ramp_up_requirement'] = {
            'shortfall': 'flexible_ramp_up_shortfall',
            'price': 'flexible_ramp_up_price',
            'shortfall_m': m.SystemFlexUpShortfall,
            'balance_m': m.SystemFlexUpRequirementConstr,
            }
        _zonal_reserve_map['flexible_ramp_down_requirement'] = {
            'shortfall': 'flexible_ramp_down_shortfall',
            'price': 'flexible_ramp_down_price',
            'shortfall_m': m.ZonalFlexDnShortfall,
            'balance_m': m.ZonalFlexDnRequirementConstr,
            }
        _system_reserve_map['flexible_ramp_down_requirement'] = {
            'shortfall': 'flexible_ramp_down_shortfall',
            'price': 'flexible_ramp_down_price',
            'shortfall_m': m.SystemFlexDnShortfall,
            'balance_m': m.SystemFlexDnRequirementConstr,
            }
    if supp:
        _zonal_reserve_map['supplemental_reserve_requirement'] = {
            'shortfall': 'supplemental_shortfall',
            'price': 'supplemental_price',
            'shortfall_m': m.ZonalSupplementalReserveShortfall,
            'balance_m': m.EnforceZonalSupplementalReserveRequirement,
            }

        _system_reserve_map['supplemental_reserve_requirement'] = {
            'shortfall': 'supplemental_shortfall',
            'price': 'supplemental_price',
            'shortfall_m': m.SystemSupplementalReserveShortfall,
            'balance_m': m.EnforceSystemSupplementalReserveRequirement,
            }

    def _populate_zonal_reserves(elements_dict, string_handle):
        for e, e_dict in elements_dict.items():
            me = string_handle + e
            for req, req_dict in _zonal_reserve_map.items():
                if req in e_dict:
                    req_shortfall_dict = _preallocated_list(data_time_periods)
                    for dt, mt in enumerate(m.TimePeriods):
                        req_shortfall_dict[dt] = pe_value(
                            req_dict['shortfall_m'][me, mt])
                    e_dict[req_dict['shortfall']] = _time_series_dict(
                        req_shortfall_dict)
                    if relaxed:
                        req_price_dict = _preallocated_list(data_time_periods)
                        for dt, mt in enumerate(m.TimePeriods):
                            req_price_dict[dt] = pe_value(dual[req_dict[
                                'balance_m'][
                                me, mt]]) / time_period_length_hours
                        e_dict[req_dict['price']] = _time_series_dict(
                            req_price_dict)

    def _populate_system_reserves(sys_dict):
        for req, req_dict in _system_reserve_map.items():
            if req in sys_dict:
                req_shortfall_dict = _preallocated_list(data_time_periods)
                for dt, mt in enumerate(m.TimePeriods):
                    req_shortfall_dict[dt] = pe_value(req_dict['shortfall_m'][mt])
                sys_dict[req_dict['shortfall']] = _time_series_dict(
                    req_shortfall_dict)
                if relaxed:
                    req_price_dict = _preallocated_list(data_time_periods)
                    for dt, mt in enumerate(m.TimePeriods):
                        req_price_dict[dt] = pe_value(dual[req_dict['balance_m'][
                            mt]]) / time_period_length_hours
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
                for dt, mt in enumerate(m.TimePeriods):
                    fuel_consumed[dt] = pe_value(
                        m.TotalFuelConsumedAtInstFuelSupply[f, mt])
            else:
                logger.warning(
                    'WARNING: unrecongized fuel_supply_type {} for fuel_supply {}'.format(
                        fuel_supply_type, f))
            f_dict['fuel_consumed'] = _time_series_dict(fuel_consumed)

    md.data['system']['total_cost'] = pe_value(m.TotalCostObjective)

    return md
