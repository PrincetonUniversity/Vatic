
import pyomo.environ as pe

from egret.model_library.unit_commitment.params import load_params \
    as default_params
from .params import load_params as renewable_cost_params

from egret.model_library.unit_commitment import (
    services, fuel_supply, fuel_consumption, security_constraints)
from egret.models.unit_commitment import (
    _solve_unit_commitment, _save_uc_results)
from egret.common.log import logger as egret_logger

from egret.model_library.transmission.tx_utils import scale_ModelData_to_pu
from egret.data.model_data import ModelData as EgretModel
import egret.common.lazy_ptdf_utils as lpu

import importlib
import logging
from ast import literal_eval
from ._utils import ModelError


class UCModel(object):

    def __init__(self,
                 params_forml, status_forml, power_forml, reserve_forml,
                 generation_forml, ramping_forml, production_forml,
                 updown_forml, startup_forml, network_forml):

        self.pyo_instance = None
        self.solver = None
        self.params = params_forml

        self.model_parts = {
            'status_vars': status_forml,
            'power_vars': power_forml,
            'reserve_vars': reserve_forml,
            'non_dispatchable_vars': 'file_non_dispatchable_vars',
            'generation_limits': generation_forml,
            'ramping_limits': ramping_forml,
            'production_costs': production_forml,
            'uptime_downtime': updown_forml,
            'startup_costs': startup_forml,
            'power_balance': network_forml,

            'reserve_requirement': ('MLR_reserve_constraints'
                                    if reserve_forml == 'MLR_reserve_vars'
                                    else 'CA_reserve_constraints'),

            'objective': ('vatic_objective'
                          if params_forml == 'renewable_cost_params'
                          else 'basic_objective')
            }

    def _load_params(self, model, model_data):
        eval(self.params)(model, model_data)

    def _load_formulation(self, model_part):
        part_fx = None

        egret_pkg = 'egret.model_library.unit_commitment'
        egret_library = importlib.import_module(egret_pkg)

        if hasattr(egret_library, model_part):
            egret_formls = importlib.import_module('.{}'.format(model_part),
                                                   egret_pkg)

            if hasattr(egret_formls, self.model_parts[model_part]):
                part_fx = getattr(egret_formls, self.model_parts[model_part])

        if part_fx is None:
            vatic_spec = importlib.util.find_spec(
                "vatic.models.{}".format(model_part))

            if vatic_spec is not None:
                vatic_formls = importlib.import_module(
                    'vatic.models.{}'.format(model_part))

                if hasattr(vatic_formls, self.model_parts[model_part]):
                    part_fx = getattr(vatic_formls,
                                      self.model_parts[model_part])

        if part_fx is None:
            raise ValueError(
                "Cannot find formulation labelled `{}` for model component "
                "`{}`!".format(self.model_parts[model_part], model_part)
                )

        return part_fx

    def generate_model(self,
                       model_data: EgretModel,
                       relax_binaries: bool,
                       ptdf_options, ptdf_matrix_dict):

        use_model = scale_ModelData_to_pu(model_data.clone_in_service(),
                                          inplace=False)

        model = pe.ConcreteModel()
        model.model_data = use_model
        model.name = "UnitCommitment"

        ## munge PTDF options if necessary
        if self.model_parts['power_balance'] == 'ptdf_power_flow':
            _ptdf_options = lpu.populate_default_ptdf_options(ptdf_options)

            baseMVA = model_data.data['system']['baseMVA']
            lpu.check_and_scale_ptdf_options(_ptdf_options, baseMVA)

            model._ptdf_options = _ptdf_options

            if ptdf_matrix_dict is not None:
                model._PTDFs = ptdf_matrix_dict
            else:
                model._PTDFs = {}

        # enforce time 1 ramp rates, relax binaries
        model.enforce_t1_ramp_rates = True
        model.relax_binaries = relax_binaries

        self._load_params(model, model_data)
        self._load_formulation('status_vars')(model)
        self._load_formulation('power_vars')(model)
        self._load_formulation('reserve_vars')(model)
        self._load_formulation('non_dispatchable_vars')(model)
        self._load_formulation('generation_limits')(model)
        self._load_formulation('ramping_limits')(model)
        self._load_formulation('production_costs')(model)
        self._load_formulation('uptime_downtime')(model)
        self._load_formulation('startup_costs')(model)
        services.storage_services(model)
        services.ancillary_services(model)
        self._load_formulation('power_balance')(model)
        self._load_formulation('reserve_requirement')(model)

        if 'fuel_supply' in model_data.data['elements'] and bool(
                model_data.data['elements']['fuel_supply']):
            fuel_consumption.fuel_consumption_model(model)
            fuel_supply.fuel_supply_model(model)

        else:
            model.fuel_supply = None
            model.fuel_consumption = None

        if 'security_constraint' in model_data.data['elements'] and bool(
                model_data.data['elements']['security_constraint']):
            security_constraints.security_constraint_model(model)
        else:
            model.security_constraints = None

        self._load_formulation('objective')(model)

        self.pyo_instance = model

    def solve_model(self,
                    solver=None, solver_options=None,
                    relaxed=False, set_instance=True, options=None):

        if self.pyo_instance is None:
            raise ModelError("Cannot solve a model until it "
                             "has been genererated!")

        if solver is None and self.solver is None:
            raise ModelError("A solver must be given if this model has not "
                             "yet been solved!")

        if solver is None:
            use_solver = self.solver
        else:
            use_solver = solver

        if not options.output_solver_logs:
            egret_logger.setLevel(logging.WARNING)

        solver_options_list = [opt.split('=')
                               for opt in solver_options[0].split(' ')]
        solver_options_dict = {option: literal_eval(val)
                               for option, val in solver_options_list}

        m, results, self.solver = _solve_unit_commitment(
            m=self.pyo_instance, solver=use_solver, mipgap=options.ruc_mipgap,
            timelimit=None, solver_tee=options.output_solver_logs,
            symbolic_solver_labels=options.symbolic_solver_labels,
            solver_options=solver_options_dict, solve_method_options=None,
            #solve_method_options=self.options.solve_method_options,
            relaxed=relaxed, set_instance=set_instance
            )

        md = _save_uc_results(m, relaxed)

        if hasattr(results, 'egret_metasolver_status'):
            time = results.egret_metasolver_status['time']
        else:
            time = results.solver.wallclock_time

        return md, time
