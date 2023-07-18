class UCModel:
    """An instance of a specific Gurobipy model formulation.

    This class takes a list of Egret formulation labels specifying a model in
    Gurobipy and creates a model instance that can be populated with data and
    optimized with a solver.
    """

    def __init__(self,
                 mipgap, output_solver_logs, symbolic_solver_labels,
                 params_forml, status_forml, power_forml, reserve_forml,
                 generation_forml, ramping_forml, production_forml,
                 updown_forml, startup_forml, network_forml):

        self.pyo_instance = None
        self.solver = None
        self.mipgap = mipgap
        self.output_solver_logs = output_solver_logs
        self.symbolic_solver_labels = symbolic_solver_labels
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
