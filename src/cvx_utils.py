'''
Source from link:http://implicit-layers-tutorial.org/differentiable_optimization/
'''

import torch
import torch.nn as nn
import torch.autograd as autograd
import cvxpy as cp
from itertools import accumulate


class OptLayer(nn.Module):
    def __init__(self, variables, parameters, objective, inequalities, equalities, **cvxpy_opts):
        super().__init__()
        self.variables = variables
        self.parameters = parameters
        self.objective = objective
        self.inequalities = inequalities
        self.equalities = equalities
        self.cvxpy_opts = cvxpy_opts
        
        # create the cvxpy problem with objective, inequalities, equalities
        self.cp_inequalities = [ineq(*variables, *parameters) <= 0 for ineq in inequalities]
        self.cp_equalities = [eq(*variables, *parameters) == 0 for eq in equalities]
        self.problem = cp.Problem(cp.Minimize(objective(*variables, *parameters)), 
                                  self.cp_inequalities + self.cp_equalities)
        
    def forward(self, *batch_params):
        out, J = [], []
        # solve over minibatch by just iterating
        for batch in range(batch_params[0].shape[0]):
            # solve the optimization problem and extract solution + dual variables
            params = [p[batch] for p in batch_params]
            with torch.no_grad():
                for i,p in enumerate(self.parameters):
                    p.value = params[i].double().numpy()
                self.problem.solve(**self.cvxpy_opts)  #, solver=cp.ECOS)  #TODO():
                z = [torch.tensor(v.value).type_as(params[0]) for v in self.variables]
                lam = [torch.tensor(c.dual_value).type_as(params[0]) for c in self.cp_inequalities]
                nu = [torch.tensor(c.dual_value).type_as(params[0]) for c in self.cp_equalities]

            # convenience routines to "flatten" and "unflatten" (z,lam,nu)
            def vec(z, lam, nu):
                return torch.cat([a.view(-1) for b in [z,lam,nu] for a in b])

            def mat(x):
                sz = [0] + list(accumulate([a.numel() for b in [z,lam,nu] for a in b]))
                val = [x[a:b] for a,b in zip(sz, sz[1:])]
                return ([val[i].view_as(z[i]) for i in range(len(z))],
                        [val[i+len(z)].view_as(lam[i]) for i in range(len(lam))],
                        [val[i+len(z)+len(lam)].view_as(nu[i]) for i in range(len(nu))])

            # computes the KKT residual
            def kkt(z, lam, nu, *params):
                g = [ineq(*z, *params) for ineq in self.inequalities]
                dnu = [eq(*z, *params) for eq in self.equalities]
                L = (self.objective(*z, *params) + 
                     sum((u*v).sum() for u,v in zip(lam,g)) + sum((u*v).sum() for u,v in zip(nu,dnu)))
                dz = autograd.grad(L, z, create_graph=True)
                dlam = [lam[i]*g[i] for i in range(len(lam))]
                return dz, dlam, dnu

            # compute residuals and re-engage autograd tape
            y = vec(z, lam, nu)
            y = y - vec(*kkt([z_.clone().detach().requires_grad_() for z_ in z], lam, nu, *params))

            # compute jacobian and backward hook
            J.append(autograd.functional.jacobian(lambda x: vec(*kkt(*mat(x), *params)), y))
            y.register_hook(lambda grad,b=batch : torch.solve(grad[:,None], J[b].transpose(0,1))[0][:,0])
            
            out.append(mat(y)[0])
        out = [torch.stack(o, dim=0) for o in zip(*out)]
        return out[0] if len(out) == 1 else tuple(out)
