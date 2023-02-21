b._PowerGeneratedStartupShutdown
b._ThermalGeneratorsAtBus
b._PowerOutputStorage
b._StorageAtBus
b._PowerInputStorage
b._NondispatchablePowerUsed
b._NondispatchableGeneratorsAtBus
b._HVDCLinePower
b._HVDCLinesTo
b._HVDCLinePower
b._HVDCLinesFrom
b._LoadGenerateMismatch

b._ReactivePowerGenerated
b._LoadGenerateMismatchReactive
b._ptdf_options
b._Buses
b._Contingencies
b._TransmissionLines
b._PTDFs
b._ReferenceBus
b._BranchViolationCost
b._InterfaceViolationCost
b._gens_by_bus
b._Demand
b._bus_gs_fixed_shunts
b._branches
b._interfaces
b._LineOutOfService
b._InterfacesWithSlack
b._BranchesWithSlack

m.addConstr((m._p_nw[b] == (bus_gs_fixed_shunts[b]
                            + (m._pl[b] if bus_p_loads[b] != 0.0 else 0.0)
                            - quicksum(m._pg[g] for g in gens_by_bus[b])
                            + quicksum(m._dcpf[branch_name] for branch_name
                                       in dc_outlet_branches_by_bus[b])
                            - quicksum(m._dcpf[branch_name] for branch_name
                                       in dc_inlet_branches_by_bus[b]))))