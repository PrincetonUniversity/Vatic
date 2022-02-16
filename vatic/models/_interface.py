
import pyomo.environ as pe

#TODO: make this and _load_params below more elegant or merge with formulations
from egret.model_library.unit_commitment.params import load_params \
    as default_params
from .params import load_params as renewable_cost_params

from ..model_data import VaticModelData

from egret.model_library.unit_commitment import (
    services, fuel_supply, fuel_consumption, security_constraints)
from egret.common.solver_interface import _solve_model
from egret.models.unit_commitment import _outer_lazy_ptdf_solve_loop
from egret.common.log import logger as egret_logger
import egret.common.lazy_ptdf_utils as lpu

import importlib
import logging
from ast import literal_eval
from ._utils import ModelError, _save_uc_results
from typing import Callable


class UCModel:
    """An instance of a specific Pyomo model formulation.

    This class takes a list of Egret formulation labels specifying a model in
    Pyomo and creates a model instance that can be populated with data and
    optimized with a solver.
    """

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

    def _load_params(self, model: pe.ConcreteModel) -> None:
        """Populates model parameters using the specified formulation."""

        if self.params == 'renewable_cost_params':
            renewable_cost_params(model, VaticModelData(model.model_data.data))

        elif self.params == 'default_params':
            default_params(model, model.model_data)

        else:
            raise ModelError(
                "Unrecognized model formulation `{}`!".format(self.params))

    def _get_formulation(self,
                         model_part: str) -> Callable[[pe.ConcreteModel],
                                                      None]:
        """Finds the specified model formulation and make it callable."""

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
                       model_data: VaticModelData,
                       relax_binaries: bool,
                       ptdf_options, ptdf_matrix_dict, objective_hours=None):

        #TODO: do we need to add scaling back in if baseMVA is always 1?
        use_model = model_data.clone_in_service()
        model = pe.ConcreteModel()
        model.model_data = use_model.to_egret()
        model.name = "UnitCommitment"

        ## munge PTDF options if necessary
        if self.model_parts['power_balance'] == 'ptdf_power_flow':
            _ptdf_options = lpu.populate_default_ptdf_options(ptdf_options)

            baseMVA = model_data.get_system_attr('baseMVA')
            lpu.check_and_scale_ptdf_options(_ptdf_options, baseMVA)

            model._ptdf_options = _ptdf_options

            if ptdf_matrix_dict is not None:
                model._PTDFs = ptdf_matrix_dict
            else:
                model._PTDFs = {}

        # enforce time 1 ramp rates, relax binaries
        model.enforce_t1_ramp_rates = True
        model.relax_binaries = relax_binaries

        self._load_params(model)
        self._get_formulation('status_vars')(model)
        self._get_formulation('power_vars')(model)
        self._get_formulation('reserve_vars')(model)
        self._get_formulation('non_dispatchable_vars')(model)
        self._get_formulation('generation_limits')(model)
        self._get_formulation('ramping_limits')(model)
        self._get_formulation('production_costs')(model)
        self._get_formulation('uptime_downtime')(model)
        self._get_formulation('startup_costs')(model)
        services.storage_services(model)
        services.ancillary_services(model)
        self._get_formulation('power_balance')(model)
        self._get_formulation('reserve_requirement')(model)

        if 'fuel_supply' in model_data._data['elements'] and bool(
                model_data._data['elements']['fuel_supply']):
            fuel_consumption.fuel_consumption_model(model)
            fuel_supply.fuel_supply_model(model)

        else:
            model.fuel_supply = None
            model.fuel_consumption = None

        if 'security_constraint' in model_data._data['elements'] and bool(
                model_data._data['elements']['security_constraint']):
            security_constraints.security_constraint_model(model)
        else:
            model.security_constraints = None

        self._get_formulation('objective')(model)

        if objective_hours:
            zero_cost_hours = set(model.TimePeriods)

            for i, t in enumerate(model.TimePeriods):
                if i < objective_hours:
                    zero_cost_hours.remove(t)
                else:
                    break

            cost_gens = {g for g, _ in model.ProductionCost}
            for t in zero_cost_hours:
                for g in cost_gens:
                    model.ProductionCostConstr[g, t].deactivate()
                    model.ProductionCost[g, t].value = 0.
                    model.ProductionCost[g, t].fix()

                for g in model.DualFuelGenerators:
                    model.DualFuelProductionCost[g, t].expr = 0.

                if model.regulation_service:
                    for g in model.AGC_Generators:
                        model.RegulationCostGeneration[g, t].expr = 0.

                if model.spinning_reserve:
                    for g in model.ThermalGenerators:
                        model.SpinningReserveCostGeneration[g, t].expr = 0.

                if model.non_spinning_reserve:
                    for g in model.ThermalGenerators:
                        model.NonSpinningReserveCostGeneration[g, t].expr = 0.

                if model.supplemental_reserve:
                    for g in model.ThermalGenerators:
                        model.SupplementalReserveCostGeneration[g, t].expr = 0.

        self.pyo_instance = model

    def solve_model(self,
                    solver=None, solver_options=None,
                    relaxed=False, set_instance=True,
                    options=None) -> VaticModelData:

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

        network = list(self.pyo_instance.model_data.elements('branch'))

        if (self.pyo_instance.power_balance == 'ptdf_power_flow'
                and self.pyo_instance._ptdf_options['lazy']
                and len(network) > 0):
            if isinstance(self.pyo_instance.model_data, VaticModelData):
                self.pyo_instance.model_data \
                    = self.pyo_instance.model_data.to_egret()

            m, results, self.solver = _outer_lazy_ptdf_solve_loop(
                self.pyo_instance, use_solver, options.ruc_mipgap,
                timelimit=None, solver_tee=options.output_solver_logs,
                symbolic_solver_labels=options.symbolic_solver_labels,
                solver_options=solver_options_dict, solve_method_options=None,
                relaxed=relaxed, set_instance=set_instance
                )

        else:
            m, results, self.solver = _solve_model(
                self.pyo_instance, use_solver, options.ruc_mipgap,
                timelimit=None, solver_tee=options.output_solver_logs,
                symbolic_solver_labels=options.symbolic_solver_labels,
                solver_options=solver_options_dict, solve_method_options=None,
                return_solver=True, set_instance=set_instance
                )

        if hasattr(results, 'egret_metasolver_status'):
            time = results.egret_metasolver_status['time']
        else:
            time = results.solver.wallclock_time

        md = _save_uc_results(m, relaxed)
        md.data['system']['solver_runtime'] = time

        return VaticModelData(md.data)
