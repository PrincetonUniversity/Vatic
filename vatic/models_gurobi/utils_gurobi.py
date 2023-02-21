import gurobipy as gp

def linear_summation(linear_vars, linear_coefs, constant=0.):
    return gp.quicksum((c*v for c,v in zip(linear_coefs, linear_vars)))+constant

