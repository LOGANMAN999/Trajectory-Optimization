from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Callable


def local_NLP(
    x0: List[float] | np.ndarray,
    cost_fn: Callable[[List[float]], float],
    int_bounds: List[Tuple[int, int]],
    real_bounds: List[Tuple[float, float]],
    *,
    tol_grad: float = 1e-8,
    tol_fun:  float = 1e-12,
    maxiter:  int   = 200
) -> Tuple[np.ndarray, float]:
    """
    Hybrid-refine the **continuous slice** of `x0` while keeping the
    integer block fixed.

    Returns
    -------
    x_opt : np.ndarray   # full chromosome (ints + reals)
    f_opt : float        # objective value cost_fn(x_opt)
    """

    x0        = np.asarray(x0, dtype=float)
    n_int     = len(int_bounds)

    ints      = x0[:n_int].copy()            
    reals0    = x0[n_int:].copy()

    n         = int(ints[0])               

    
    num_real_active = 1 + (n+1) + 2
    reals0_active   = reals0[:num_real_active]
    bounds_active   = real_bounds[:num_real_active]
    
    

    # objective that re-assembles chromosome
    def obj(reals: np.ndarray) -> float:
        if np.any(reals <=0):
            return np.inf
        full = np.concatenate((ints, reals))
        return cost_fn(full)

    res = minimize(
        fun=obj,
        x0=reals0_active,
        method="L-BFGS-B",
        bounds=bounds_active,
        options={
            "gtol":  tol_grad,
            "ftol":  tol_fun,
            "maxiter": maxiter,
            "disp": False,
        },
    )
    if not np.all(res.x > 0):
        print("NLP_badTOF result had ≤0");  return x0, np.inf
    if np.isnan(res.x).any():
        print("NLP_nan");          return x0, np.inf
    reals_opt_full = reals0[:]                    
    if res.success:
        reals_opt_full[:num_real_active] = res.x   

    x_opt = list(ints) + list(reals_opt_full)      
    f_opt = cost_fn(x_opt)
    return np.array(x_opt), f_opt, res