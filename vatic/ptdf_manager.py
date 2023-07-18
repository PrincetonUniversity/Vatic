#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from __future__ import annotations

from collections import namedtuple
from .model_data import VaticModelData


PTDFOptions = namedtuple('PTDFOptions',
                         ['ruc', 'dap', 'la_ed', 'mip_ed', 'lmp_ed'])

LowConfidenceOptions = PTDFOptions(ruc={},
                                   dap={},
                                   la_ed={},
                                   mip_ed={'pre_lp_iteration_limit': 0},
                                   lmp_ed={'pre_lp_iteration_limit': 0})

MediumConfidenceOptions = PTDFOptions(ruc={'pre_lp_iteration_limit': 0,
                                           'lp_iteration_limit': 5},
                                      dap={'pre_lp_iteration_limit': 0},
                                      la_ed={'pre_lp_iteration_limit': 0,
                                             'lp_iteration_limit': 3},
                                      mip_ed={'pre_lp_iteration_limit': 0,
                                              'lp_iteration_limit': 0},
                                      lmp_ed={'pre_lp_iteration_limit': 0})

HighConfidenceOptions = PTDFOptions(ruc={'pre_lp_iteration_limit': 0,
                                         'lp_iteration_limit': 0},
                                    dap={'pre_lp_iteration_limit': 0},
                                    la_ed={'pre_lp_iteration_limit': 0,
                                           'lp_iteration_limit': 0},
                                    mip_ed={'pre_lp_iteration_limit': 0,
                                            'lp_iteration_limit': 0},
                                    lmp_ed={'pre_lp_iteration_limit': 0})


class VaticPTDFManager:
    """Keeping track of the active constraints handed to EGRET models."""

    def __init__(self,
                 inactive_limit: int = 5, limit_eps: float = 1e-2) -> None:
        # stored PTDF_matrix_dict for Egret
        self.PTDF_matrix_dict = None

        # dictionary with activate constraints keys are some immutable unique
        # constraint identifier (e.g., line name), values are the number of
        # times since last active
        self._active_branch_constraints = dict()
        self._active_interface_constraints = dict()

        # constraints leave the active set if we
        # haven't seen them after this many cycles
        self._inactive_limit = inactive_limit

        self._ptdf_options = LowConfidenceOptions
        self._calls_since_last_miss = 0
        self.limit_eps = limit_eps

    def _at_limit(self, power_flow_list: list[float], limit: float) -> bool:
        use_limit = limit - self.limit_eps

        for flow in power_flow_list:
            if abs(flow) > use_limit:
                return True

        return False

    def _at_two_sided_limit(self,
                            power_flow_list: list[float],
                            lower_bound: float, upper_bound: float) -> bool:
        use_lb = lower_bound + self.limit_eps
        use_ub = upper_bound - self.limit_eps

        for flow in power_flow_list:
            if not use_lb <= flow <= use_ub:
                return True

        return False

    def mark_active(self, model: VaticModelData) -> None:
        for branch, branch_data in model.elements(element_type='branch'):
            if branch in self._active_branch_constraints:
                branch_data['lazy'] = False

        for intrfce, intrfce_data in model.elements(element_type='interface'):
            if intrfce in self._active_interface_constraints:
                intrfce_data['lazy'] = False

    def update_active(self, model: VaticModelData) -> None:
        # increment active
        for branch in self._active_branch_constraints:
            self._active_branch_constraints[branch] += 1
        for intrfce in self._active_interface_constraints:
            self._active_interface_constraints[intrfce] += 1

        misses = 0
        for branch, branch_data in model.elements(element_type='branch'):
            if self._at_limit(branch_data['pf']['values'],
                              branch_data['rating_long_term']):
                # we're seeing it now, so reset its counter or make a new one
                if branch not in self._active_branch_constraints:
                    misses += 1
                self._active_branch_constraints[branch] = 0

        for intrfce, intrfce_data in model.elements(element_type='interface'):
            if self._at_two_sided_limit(intrfce_data['pf']['values'],
                                        intrfce_data['lower_limit'],
                                        intrfce_data['upper_limit']):
                if intrfce not in self._active_interface_constraints:
                    misses += 1
                self._active_interface_constraints[intrfce] = 0

        if misses == 0:
            self._calls_since_last_miss += 1

            if self._calls_since_last_miss > 100:
                self._ptdf_options = HighConfidenceOptions
            else:
                self._ptdf_options = MediumConfidenceOptions

        elif misses < 5:
            self._calls_since_last_miss = 0
            self._ptdf_options = MediumConfidenceOptions

        else:
            self._calls_since_last_miss = 0
            self._ptdf_options = LowConfidenceOptions

        del_branches = {
            branch
            for branch, b_cnstrs in self._active_branch_constraints.items()
            if b_cnstrs > self._inactive_limit
            }

        for branch in del_branches:
            del self._active_branch_constraints[branch]

        del_interfaces = {
            intrfce
            for intrfce, i_cnstrs in self._active_interface_constraints.items()
            if i_cnstrs > self._inactive_limit
            }

        for intrfce in del_interfaces:
            del self._active_interface_constraints[intrfce]

    @property
    def ruc_ptdf_options(self):
        return self._ptdf_options.ruc

    @property
    def damarket_ptdf_options(self):
        return self._ptdf_options.dap

    @property
    def look_ahead_sced_ptdf_options(self):
        return self._ptdf_options.la_ed

    @property
    def sced_ptdf_options(self):
        return self._ptdf_options.mip_ed

    @property
    def lmpsced_ptdf_options(self):
        return self._ptdf_options.lmp_ed
