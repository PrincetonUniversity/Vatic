
from abc import ABC, abstractmethod
import math
import numpy as np
import egret.model_library.transmission.tx_calc as tx_calc
from gurobipy import quicksum
from pyomo.environ import value
from enum import Enum
from typing import Sized


class FlowType(Enum):
    CURRENT = 1
    POWER = 2

class CoordinateType(Enum):
    POLAR = 1
    RECTANGULAR = 2

class ApproximationType(Enum):
    BTHETA = 1
    BTHETA_LOSSES = 2
    PTDF = 3
    PTDF_LOSSES = 4

class BasePointType(Enum):
    FLATSTART = 1
    SOLUTION = 2

class RelaxationType(Enum):
    NONE = 1
    SOC = 2


class _PTDFManagerBase(ABC):
    @abstractmethod
    def get_branch_ptdf_iterator(self, branch_name):
        pass

    @abstractmethod
    def get_branch_ptdf_abs_max(self, branch_name):
        pass

    @abstractmethod
    def get_branch_const(self, branch_name):
        pass

    @abstractmethod
    def get_interface_const(self, interface_name):
        pass

    @abstractmethod
    def get_interface_ptdf_abs_max(self, interface_name):
        pass

    @abstractmethod
    def get_interface_ptdf_iterator(self, interface_name):
        pass

    @abstractmethod
    def calculate_PFV(self, mb):
        pass


class VirtualPTDFMatrix(_PTDFManagerBase):
    '''
    Helper class which *looks like* a PTDF matrix, but doesn't
    actually calculate/store a PTDF matrix. Keeps a factorization
    of A^T B_d A instead (or A^T J in EGRET internal parlance) and
    computes flows and needed PTDF rows on-the-fly.

    Already calculated PTDF rows are cached for easy retreval on
    subsequent calls
    '''

    def __init__(self, branches, buses, reference_bus, base_point,
                 ptdf_options, branches_keys=None, buses_keys=None,
                 interfaces=None, contingencies=None):
        '''
        Creates a new VirtualPTDFMatrix object to provide
        some useful methods for interfacing with Egret pyomo models

        Parameters
        ----------
        branches (dict) :
            dictionary of branches and their attributes
        buses (dict) :
            dictionary of buses and their attributes
        reference_bus (str) :
            name of reference bus
        base_point (egret.model_library.defn.BasePointType) :
            whether or not to linearize around a given AC solution
        ptdf_options (dict) :
            dictionary of ptdf_options (see
            egret.common.lazy_ptdf_utils.populate_default_ptdf_options)
        branches_keys (list, tuple, etc) (optional) :
            When indexing into numpy/scipy arrays, the branches will be
            indexed by this list. E.g., the name at position i will be
            the i th entry in branch matrices/arrays
        buses_keys (list, tuple, etc) (optional) :
            Similar to branches_keys, though the reference bus will be
            eliminated. (Uses buses_keys_no_ref for direct indexing)
        interfaces (dict) (optional) :
            dictionary of interfaces and their attributes
        contingencies (dict) (optional) :
            dictionary of contingencies and their attributes
        '''
        self._branches = branches
        self._buses = buses
        self._reference_bus = reference_bus
        self._base_point = base_point
        if branches_keys is None:
            self.branches_keys = tuple(branches.keys())
        else:
            self.branches_keys = tuple(branches_keys)
        if buses_keys is None:
            self.buses_keys = tuple(buses.keys())
        else:
            self.buses_keys = tuple(buses_keys)
        self._branchname_to_index_map = {branch_n: i for i, branch_n in
                                         enumerate(self.branches_keys)}
        self._busname_to_index_map = {bus_n: j for j, bus_n in
                                      enumerate(self.buses_keys)}

        # the PTDF factorization code eliminates the reference bus from the associated
        # matrices, so we need to handle that fact smoothly
        self.buses_keys_no_ref = tuple(
            bus for bus in self.buses_keys if bus != reference_bus)

        self.branch_limits_array = np.fromiter(
            (branches[branch]['rating_long_term'] for branch in
             self.branches_keys), float, count=len(self.branches_keys))
        self.branch_limits_array.flags.writeable = False

        if interfaces is None:
            interfaces = dict()
        self.interfaces = interfaces
        self._calculate_interface_limits()

        if contingencies is None:
            contingencies = dict()
        self.contingencies = contingencies
        self._calculate_contingency_limits()

        self._base_point = base_point
        self._calculate_ptdf_factorization()

        self._set_lazy_limits(ptdf_options)

        # we'll cache the PTDF rows
        # we've calculated thus far
        # to prevent doing extra work
        # (esp. important for UC)
        self._ptdf_rows = dict()
        self._interface_rows = dict()
        self._contingency_rows = dict()

        # dense array write buffer
        self._bus_sensi_buffer = np.empty((1, len(self.buses_keys_no_ref)),
                                          dtype=np.float64)

    def _calculate_ptdf_factorization(self):
        print("Calculating PTDF Matrix Factorization")

        (MLU, B_dA, ref_bus_mask,
                contg_comps, B_dA_I, I) = tx_calc.calculate_ptdf_factorization(
            self._branches, self._buses, self.branches_keys,
            self.buses_keys, self._reference_bus, self._base_point,
            contingencies=self.contingencies,
            mapping_bus_to_idx=self._busname_to_index_map,
            mapping_branch_to_idx=self._branchname_to_index_map,
            interfaces=self.interfaces,
            index_set_interface=self.interface_keys
            )

        self.MLU = MLU
        self.B_dA = B_dA
        self.ref_bus_mask = ref_bus_mask
        self.contingency_compensators = contg_comps
        self.B_dA_I = B_dA_I

        self._calculate_phase_shift_flow_adjuster()
        self._calculate_phi_adjust(ref_bus_mask)

        self.phase_shift_flow_adjuster_array_interface = (
                I @ self.phase_shift_flow_adjuster_array)

    def _calculate_interface_limits(self):
        self.interface_keys = tuple(self.interfaces.keys())

        self.interfacename_to_index_map = \
            {i_n: idx for idx, i_n in enumerate(self.interface_keys)}

        def _interface_limit_iter(limit, inf):
            for i_n in self.interface_keys:
                interface = self.interfaces[i_n]
                if limit in interface and interface[limit] is not None:
                    yield interface[limit]
                else:
                    yield inf

        self.interface_max_limits = np.fromiter(
            _interface_limit_iter('maximum_limit', np.inf), float,
            count=len(self.interface_keys))
        self.interface_min_limits = np.fromiter(
            _interface_limit_iter('minimum_limit', -np.inf), float,
            count=len(self.interface_keys))

    def _calculate_contingency_limits(self):
        if self.contingencies:
            def _contingency_limit_iter():
                for bn in self.branches_keys:
                    branch = self._branches[bn]
                    if 'rating_emergency' in branch:
                        yield branch['rating_emergency']
                    else:
                        yield np.inf

            self.contingency_limits_array = np.fromiter(
                _contingency_limit_iter(), float,
                count=len(self.branches_keys))
        else:
            self.contingency_limits_array = np.empty(shape=(
            len(self.branches_keys),
            0))  # create an empty array for slicing on
        self.branch_limits_array.flags.writeable = False

    def _calculate_phi_adjust(self, ref_bus_mask):
        phi_adjust_array = tx_calc.calculate_phi_adjust(
            self._branches, self.branches_keys, self.buses_keys,
            ApproximationType.PTDF,
            mapping_bus_to_idx=self._busname_to_index_map
            )

        self.phi_adjust_array = phi_adjust_array[ref_bus_mask]

    def _calculate_phase_shift_flow_adjuster(self):
        self.phase_shift_flow_adjuster_array = \
            tx_calc.calculate_phase_shift_flow_adjuster(self._branches,
                                                        self.branches_keys)

    def _get_filtered_lines(self, ptdf_options):
        self.branch_limits_array_masked = self.branch_limits_array

        if ptdf_options['branch_kv_threshold'] is None:
            ## Nothing to do
            self.branch_mask = np.arange(len(self.branch_limits_array))
            self.branches_keys_masked = self.branches_keys
            self.branchname_to_index_masked_map = self._branchname_to_index_map
            self.B_dA_masked = self.B_dA
            self.phase_shift_flow_adjuster_array_masked = self.phase_shift_flow_adjuster_array
            self.contingency_limits_array_masked = self.contingency_limits_array

            return None

        branches = self._branches
        buses = self._buses
        branch_mask = list()
        one = (ptdf_options['kv_threshold_type'] == 'one')
        kv_limit = ptdf_options['branch_kv_threshold']

        for i, bn in enumerate(self.branches_keys):
            branch = branches[bn]
            fb = buses[branch['from_bus']]

            fbt = True
            ## NOTE: this warning will be printed only once if we just check the from_bus
            if 'base_kv' not in fb:
                logger.warning(
                    "WARNING: did not find 'base_kv' for bus {}, considering it large for the purposes of filtering".format(
                        branch['from_bus']))
            elif fb['base_kv'] < kv_limit:
                fbt = False

            if fbt and one:
                branch_mask.append(i)
                continue

            tb = buses[branch['to_bus']]
            tbt = False
            if ('base_kv' not in tb) or tb['base_kv'] >= kv_limit:
                tbt = True

            if fbt and tbt:
                branch_mask.append(i)
            elif one and tbt:
                branch_mask.append(i)

        self.branch_mask = np.array(branch_mask)
        self.branches_keys_masked = tuple(
            self.branches_keys[i] for i in self.branch_mask)
        self.branchname_to_index_masked_map = {bn: i for i, bn in enumerate(
            self.branches_keys_masked)}
        self.B_dA_masked = self.B_dA[branch_mask]
        self.phase_shift_flow_adjuster_array_masked = \
        self.phase_shift_flow_adjuster_array[branch_mask]
        self.branch_limits_array_masked = self.branch_limits_array[branch_mask]
        self.contingency_limits_array_masked = self.contingency_limits_array[
            branch_mask]

    def _set_lazy_limits(self, ptdf_options):
        if ptdf_options['lazy']:
            self._get_filtered_lines(ptdf_options)
            ## add / reset the relative limits based on the current options
            branch_limits = self.branch_limits_array_masked
            interface_max_limits = self.interface_max_limits
            interface_min_limits = self.interface_min_limits
            contingency_limits = self.contingency_limits_array_masked
            rel_flow_tol = ptdf_options['rel_flow_tol']
            abs_flow_tol = ptdf_options['abs_flow_tol']
            lazy_flow_tol = ptdf_options['lazy_rel_flow_tol']

            ## only enforce the relative and absolute, within tollerance
            self.enforced_branch_limits = np.maximum(
                branch_limits * (1 + rel_flow_tol),
                branch_limits + abs_flow_tol)
            self.enforced_contingency_limits = np.maximum(
                contingency_limits * (1 + rel_flow_tol),
                contingency_limits + abs_flow_tol)
            ## make sure the lazy limits are a superset of the enforce limits
            self.lazy_branch_limits = np.minimum(
                branch_limits * (1 + lazy_flow_tol),
                self.enforced_branch_limits)
            self.lazy_contingency_limits = np.minimum(
                contingency_limits * (1 + lazy_flow_tol),
                self.enforced_contingency_limits)
            abs_max_limits_i = np.abs(interface_max_limits)
            abs_min_limits_i = np.abs(interface_min_limits)

            self.enforced_interface_max_limits = \
                np.maximum(
                    interface_max_limits + rel_flow_tol * abs_max_limits_i,
                    interface_max_limits + abs_flow_tol)

            self.enforced_interface_min_limits = \
                np.minimum(
                    interface_min_limits - rel_flow_tol * abs_min_limits_i,
                    interface_min_limits - abs_flow_tol)

            self.lazy_interface_max_limits = \
                np.minimum(
                    interface_max_limits + lazy_flow_tol * abs_max_limits_i,
                    self.enforced_interface_max_limits)

            self.lazy_interface_min_limits = \
                np.maximum(
                    interface_min_limits - lazy_flow_tol * abs_min_limits_i,
                    self.enforced_interface_min_limits)

    def _get_ptdf_row(self, branch_name):
        if branch_name in self._ptdf_rows:
            PTDF_row = self._ptdf_rows[branch_name]
        else:
            # calculate row
            branch_idx = self._branchname_to_index_map[branch_name]
            PTDF_row = self.MLU.solve(
                self.B_dA[branch_idx].toarray(out=self._bus_sensi_buffer)[0],
                trans='T')
            self._ptdf_rows[branch_name] = PTDF_row
        return PTDF_row

    def _get_contingency_row(self, contingency_name, branch_name):
        if (contingency_name, branch_name) in self._contingency_rows:
            cont_PTDF_row = self._contingency_rows[
                contingency_name, branch_name]

        else:
            ## TODO: should we be using post-compensation if the PTDF row is already calculated?
            branch_idx = self._branchname_to_index_map[branch_name]
            cc = self.contingency_compensators[contingency_name]
            hatF = cc.U.solve((self.B_dA[branch_idx] @ cc.Pc).toarray(
                out=self._bus_sensi_buffer)[0], 'T')
            delF = cc.Wbar * ((-cc.c) * (cc.W.T @ hatF))
            cont_PTDF_row = cc.Pr.T @ cc.L.solve(hatF + delF, 'T')
            # print(f"contingency row: {cont_PTDF_row}")
            self._contingency_rows[
                contingency_name, branch_name] = cont_PTDF_row
        return cont_PTDF_row

    def _get_interface_row(self, interface_name):
        if interface_name in self._interface_rows:
            I_row = self._interface_rows[interface_name]
        else:
            interface_idx = self.interfacename_to_index_map[interface_name]
            I_row = self.MLU.solve(
                self.B_dA_I[interface_idx].toarray(out=self._bus_sensi_buffer)[
                    0], trans='T')
            self._interface_rows[interface_name] = I_row
        return I_row

    def get_branch_ptdf_iterator(self, branch_name):
        '''
        returns a (bus_name, coefficient) iterator for a given branch_name
        '''
        yield from zip(self.buses_keys_no_ref, self._get_ptdf_row(branch_name))

    def get_branch_ptdf_abs_max(self, branch_name):
        '''
        returns the maximum of the absolute value for any coefficent
        in branch_name's ptdf row
        '''
        ptdf_row = self._get_ptdf_row(branch_name)
        return np.abs(ptdf_row).max()

    def get_branch_const(self, branch_name):
        '''
        returns the constant coefficient for branch_name 's
        power flow equation (given bus net withdrawls)
        '''
        ptdf_row = self._get_ptdf_row(branch_name)
        branch_idx = self._branchname_to_index_map[branch_name]
        phi_adj = ptdf_row @ self.phi_adjust_array
        ## phi adj   +     phase shift
        return phi_adj[0] + self.phase_shift_flow_adjuster_array[branch_idx, 0]

    def get_contingency_branch_ptdf_iterator(self, contingency_name,
                                             branch_name):
        '''
        returns a (bus_name, coefficient) iterator for a given branch_name
        '''
        yield from zip(self.buses_keys_no_ref,
                       self._get_contingency_row(contingency_name,
                                                 branch_name))

    def get_contingency_branch_ptdf_abs_max(self, contingency_name,
                                            branch_name):
        '''
        returns the maximum of the absolute value for any coefficent
        in branch_name's ptdf row
        '''
        ptdf_row = self._get_contingency_row(contingency_name, branch_name)
        return np.abs(ptdf_row).max()

    def get_contingency_branch_const(self, contingency_name, branch_name):
        '''
        returns the constant coefficient for branch_name 's
        power flow equation (given bus net withdrawls)
        '''
        ptdf_row = self._get_contingency_row(contingency_name, branch_name)
        branch_idx = self._branchname_to_index_map[branch_name]

        phi_compensator = self.contingency_compensators[
            contingency_name].phi_compensator

        phi_adj = ptdf_row @ (self.phi_adjust_array + phi_compensator)
        ## phi adj   +     phase shift
        return phi_adj[0] + self.phase_shift_flow_adjuster_array[branch_idx, 0]

    def get_interface_const(self, interface_name):
        '''
        Returns the constant coefficient for interface_names 's
        power flow equation (given bus net withdrawls)
        '''
        I_row = self._get_interface_row(interface_name)
        i_idx = self.interfacename_to_index_map[interface_name]
        phi_adj = I_row @ self.phi_adjust_array
        ## phi adj   +     phase shift
        return phi_adj[0] + self.phase_shift_flow_adjuster_array_interface[
            i_idx, 0]

    def get_interface_ptdf_abs_max(self, interface_name):
        '''
        Returns the maximum of the absolute value for any coefficent
        in branch_name's PTDF row
        '''
        I_row = self._get_interface_row(interface_name)
        return np.abs(I_row).max()

    def get_interface_ptdf_iterator(self, interface_name):
        '''
        Returns a (bus_name, coefficient) iterator for a given interface_name
        '''
        yield from zip(self.buses_keys_no_ref,
                       self._get_interface_row(interface_name))

    def _insert_reference_bus(self, bus_array, val):
        return np.insert(bus_array,
                         self._busname_to_index_map[self._reference_bus], val)

    def _calculate_PFV_delta(self, cn, PFV, VA, masked):
        comp = self.contingency_compensators[cn]

        if not masked:
            # fix VA
            VA = -VA[self.ref_bus_mask]

        # if a phase shifter is taken out, we have
        # to do a bit more work
        VA_comp = (comp.VA_compensator is not None)
        if VA_comp:
            VA0 = VA + comp.VA_compensator
        else:
            VA0 = VA

        VA_delta = self.MLU.solve((comp.M * ((-comp.c) * (comp.M.T @ VA0))))
        if VA_comp:
            VA_delta += comp.VA_compensator

        if masked:
            PF_delta = self.B_dA_masked @ VA_delta
        else:
            PF_delta = self.B_dA @ VA_delta
        # print(f'PF_delta: {PF_delta}')

        # zero-out the flow on this line, if we're monitoring it
        if masked and comp.branch_out in self.branchname_to_index_masked_map:
            branch_out_idx = self.branchname_to_index_masked_map[
                comp.branch_out]
            PF_delta[branch_out_idx] = -PFV[branch_out_idx]

        elif not masked:
            branch_out_idx = self._branchname_to_index_map[comp.branch_out]
            PF_delta[branch_out_idx] = -PFV[branch_out_idx]

        return PF_delta

    def calculate_PFV(self, mb, masked=False):
        p_nws = tuple(mb['p_nw'][b].x for b in self.buses_keys_no_ref)

        NWV = np.fromiter(p_nws, float, count=len(self.buses_keys_no_ref))
        NWV += self.phi_adjust_array.T
        VA = self.MLU.solve(NWV.A[0])

        # shape VA explicitly as a column vector
        # (needed for some 0-dim arrays)
        VA.shape = (VA.shape[0], 1)

        if masked:
            PFV = self.B_dA_masked @ VA
            PFV += self.phase_shift_flow_adjuster_array_masked
        else:
            PFV = self.B_dA @ VA
            PFV += self.phase_shift_flow_adjuster_array

        PFV_I = self.B_dA_I @ VA
        PFV_I += self.phase_shift_flow_adjuster_array_interface

        ## make back to row-looking vectors
        PFV = PFV.T
        PFV_I = PFV_I.T

        ## VA is reversed in sign
        if masked:
            VA = VA.T[0]
        else:
            VA = -self._insert_reference_bus(VA.T[0], 0.)

        return PFV.A[0], PFV_I.A[0], VA

    def calculate_LMP(self, sced, t):
        '''
        Calculate a vector of locational marginal prices
        indexed by buses_keys.

        Parameters
        ----------
        mb : Pyomo ConcreteModel or Block with
             ineq_pf_branch_thermal_bounds constraint and
             ineq_pf_interface_bounds constraint (if there are
             interfaces).
        dual : Dual mapping return by a pyomo solver (usually
               ConcreteModel.dual
        bus_balance_constr : the bus-balance constraint for reading
                             the energy component of LMP

        Returns
        -------
        LMP : np.array of LMPs indexed by buses_keys
        '''
        ## NOTE: unmonitored lines cannot contribute to LMPC

        pfd_constrs = [
            (sced.model.getConstrByName(f'thermal-lowerlimit_{t}{bn}'),
             sced.model.getConstrByName(f'thermal-upperlimit_{t}{bn}'))
            for bn in self.branches_keys_masked
            ]

        pfd = [constr[0].pi + constr[1].pi if constr[0] else 0.
               for constr in pfd_constrs]
        b_pfd = -self.B_dA_masked.T @ pfd

        lmpc = self.MLU.solve(b_pfd, trans='T')
        lmpe = sced.model._TransmissionBlock[t]['eq_balance'].pi
        lmp = self._insert_reference_bus(lmpc + lmpe, lmpe)

        return dict(zip(self._buses.keys(), lmp))

    def calculate_monitored_contingency_flows(self, mb):
        NWV = np.fromiter((value(mb.p_nw[b]) for b in self.buses_keys_no_ref),
                          float, count=len(self.buses_keys_no_ref))
        NWV += self.phi_adjust_array.T
        NWV = NWV.A[0]

        flows_dict = {}
        for name in mb.ineq_pf_contingency_branch_thermal_bounds:
            flows_dict[name] = (self._contingency_rows[name]) @ NWV

        return flows_dict

    def calculate_masked_PFV_delta(self, cn, PFV, VA):
        '''
        Calculate a vector of partial real power
        flows indexed by branches_masked_keys, for given
        contingency cn, for given base-case real
        power flow and voltage angles given by
        calculate_masked_PFV

        Parameters
        ----------
        cn  : contingency name to calculate PFV_delta for
        PFV : vector of flows returned from calculate_masked_PFV
        VA  : vector of voltage angles returned from calculate_masked_PFV

        Returns
        -------
        PFV_delta: np.arrays for partial contingency flow differences,
                 such that PFV + PFV_delta is the contingeny flow
        '''
        return self._calculate_PFV_delta(cn, PFV, VA, masked=True)

    def calculate_PFV_delta(self, cn, PFV, VA):
        '''
        Calculate a vector of real power
        flows indexed by branches_keys, for given
        contingency cn, for given base-case real
        power flow and voltage angles given by
        calculate_PFV

        Parameters
        ----------
        cn  : contingency name to calculate PFV_delta for
        PFV : vector of flows returned from calculate_PFV
        VA  : vector of voltage angles returned from calculate_PFV

        Returns
        -------
        PFV_delta: np.arrays for partial contingency flow differences,
                 such that PFV + PFV_delta is the contingeny flow
        '''
        return self._calculate_PFV_delta(cn, PFV, VA, masked=False)


class _LazyViolations(Sized):
    def __init__(self, branch_lazy_violations,
                       interface_lazy_violations=None,
                       contingency_lazy_violations=None):
        self._branch_lazy_violations = branch_lazy_violations
        if interface_lazy_violations is None:
            self._interface_lazy_violations = set()
        else:
            self._interface_lazy_violations = interface_lazy_violations
        if contingency_lazy_violations is None:
            self._contingency_lazy_violations = set()
        else:
            self._contingency_lazy_violations = contingency_lazy_violations

    def __len__(self):
        return len(self._branch_lazy_violations) \
                + len(self._interface_lazy_violations) \
                + len(self._contingency_lazy_violations)

    @property
    def branch_lazy_violations(self):
        return self._branch_lazy_violations
    @property
    def interface_lazy_violations(self):
        return self._interface_lazy_violations
    @property
    def contingency_lazy_violations(self):
        return self._contingency_lazy_violations


class _CalculatedFlows:
    def __init__(self, PFV=None, PFV_I=None):
        self._PFV = PFV
        self._PFV_I = PFV_I

    @property
    def PFV(self):
        return self._PFV
    @property
    def PFV_I(self):
        return self._PFV_I


class _MaximalViolationsStore:
    def __init__(self, max_viol_add, baseMVA, prepend_str, time=None):
        self.max_viol_add = max_viol_add
        self.baseMVA = baseMVA
        self.time = time
        self.prepend_str = prepend_str
        self.violations_store = {}
        self.total_violations = 0
        self.monitored_violations = 0

    def get_violations_named(self, name):
        for key in self.violations_store:
            if key[0] == name:
                yield key[1]

    def min_flow_violation(self):
        if self.violations_store:
            return min(self.violations_store.values())
        else:
            return 0.

    def _min_violation_key(self):
        d = self.violations_store
        return min(d, key=d.get)

    def _add_violation(self, name, other_name, index, val):
        if other_name:
            key = ( name, (other_name, index) )
        else:
            key = ( name, index )
        self.violations_store[key] = val

        # keep the violations_store <= self.max_viol_add
        if len(self.violations_store) > self.max_viol_add:
            min_key = self._min_violation_key()

            if min_key == key:
                raise RuntimeError(f"Circular condition: added {key} to violations_store "
                        f"with value {self.violations_store[min_key]} only to delete it. "
                        f"violations_store: {self.violations_store}")

            del self.violations_store[min_key]

    def _add_violations( self, name, other_name, viol_array, viol_indices):
        while viol_indices:
            idx = np.argmax(viol_array[viol_indices])
            val = viol_array[viol_indices[idx]]
            if val < self.min_flow_violation()\
                    and len(self.violations_store) >= self.max_viol_add:
                break

            # If this violation is close in value to
            # one already in the set, it is likely
            # to be a parallel constraint.
            # If we haven't added any constraints yet
            # any(()) is False, so this won't fire
            close_to_existing = any( math.isclose( val, existing ) for existing in self.violations_store.values() )
            if close_to_existing:
                viol_indices.pop(idx)
                continue

            self._add_violation( name, other_name, viol_indices[idx], val )
            viol_indices.pop(idx)

    def check_and_add_violations(self,
                                 name, flow_array, upper_lazy_limits,
                                 upper_enforced_limits, lower_lazy_limits,
                                 lower_enforced_limits, monitored_indices,
                                 index_names, outer_name=None, PFV=None):

        if outer_name:
            # contingencies are named by cn, branch_idx, reduce to
            # branch_idx for this function
            monitored_indices = set(idx[1] for idx in monitored_indices
                                    if idx[0] == outer_name)

        ## check upper bound
        upper_viol_lazy_array = flow_array - upper_lazy_limits

        ## get the indices of the violation
        ## here filter by least violation in violations_store
        ## in the limit, this will become 0 eventually --
        upper_viol_lazy_idx = np.nonzero(upper_viol_lazy_array
                                         > self.min_flow_violation())[0]
        upper_viol_array = (flow_array[upper_viol_lazy_idx]
                            - upper_enforced_limits[upper_viol_lazy_idx])

        self._calculate_total_and_monitored_violations(
            upper_viol_array, upper_viol_lazy_idx, monitored_indices,
            flow_array, index_names, upper_enforced_limits, outer_name, PFV
            )

        ## viol_lazy_idx will hold the lines we're adding
        ## this iteration -- don't want to add lines
        ## that are already in the monitored set

        # eliminate lines in the monitored set
        upper_viol_lazy_idx = list(set(upper_viol_lazy_idx).difference(
            monitored_indices))

        self._add_violations(name, outer_name, upper_viol_lazy_array,
                             upper_viol_lazy_idx)

        ## check lower bound
        lower_viol_lazy_array = lower_lazy_limits - flow_array

        ## get the indices of the violation
        ## here filter by least violation in violations_store
        ## in the limit, this will become 0 eventually --
        lower_viol_lazy_idx = np.nonzero(lower_viol_lazy_array
                                         > self.min_flow_violation())[0]
        lower_viol_array = (lower_enforced_limits[lower_viol_lazy_idx]
                            - flow_array[lower_viol_lazy_idx])

        self._calculate_total_and_monitored_violations(
            lower_viol_array, lower_viol_lazy_idx, monitored_indices,
            flow_array, index_names, lower_enforced_limits, outer_name, PFV
            )

        ## viol_lazy_idx will hold the lines we're adding
        ## this iteration -- don't want to add lines
        ## that are already in the monitored set

        # eliminate lines in the monitored set
        lower_viol_lazy_idx = list(set(lower_viol_lazy_idx).difference(
            monitored_indices))

        self._add_violations(name, outer_name, lower_viol_lazy_array,
                             lower_viol_lazy_idx)


    def _calculate_total_and_monitored_violations(
            self,
            viol_array, viol_lazy_idx, monitored_indices,
            flow_array, index_names, limits, outer_name, other_flows
            ):

        ## viol_idx_idx will be indexed by viol_lazy_idx
        viol_idx_idx = np.nonzero(viol_array > 0)[0]
        viol_idx = frozenset(viol_lazy_idx[viol_idx_idx])

        self.total_violations += len(viol_idx)

        viol_in_mb = viol_idx.intersection(monitored_indices)
        self.monitored_violations += len(viol_in_mb)

        for i in viol_in_mb:
            thermal_limit = limits[i]
            flow = flow_array[i]

            if outer_name:
                thermal_limit += other_flows[i]
                flow += other_flows[i]


def add_violations(model, time, lazy_violations, ptdf_options, PTDF):

    ## static information between runs
    baseMVA = 1
    rel_ptdf_tol = ptdf_options['rel_ptdf_tol']
    abs_ptdf_tol = ptdf_options['abs_ptdf_tol']
    mb = model._TransmissionBlock[time]

    for i in lazy_violations.branch_lazy_violations:
        branch = PTDF.branches_keys_masked[i]

        if mb['pf'][branch] is None:
            const = PTDF.get_branch_const(branch)
            max_coef = PTDF.get_branch_ptdf_abs_max(branch)
            ptdf_tol = max(abs_ptdf_tol, rel_ptdf_tol * max_coef)

            ## if model.p_nw is Var, we can use LinearExpression
            ## to build these dense constraints much faster
            coef_list = []
            var_list = []

            for bus_name, coef in PTDF.get_branch_ptdf_iterator(branch):
                if abs(coef) >= ptdf_tol:
                    coef_list.append(coef)
                    var_list.append(model._p_nw[bus_name, time])

            mb['pf'][branch] = const + quicksum(
                coef * var for coef, var in zip(coef_list, var_list))

        model.addConstr((-model._ThermalLimits[branch] <= mb['pf'][branch]),
                        name=f'thermal-lowerlimit_{time}{branch}')
        model.addConstr((mb['pf'][branch] <= model._ThermalLimits[branch]),
                        name=f'thermal-upperlimit_{time}{branch}')

        mb['idx_monitored'].append(i)

    return model
