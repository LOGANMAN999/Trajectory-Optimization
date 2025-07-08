import numpy as np
from poliastro.iod import izzo

import astropy.units as u
from .ephemeris import splines, t0  

def get_body_state(body, t_num):
    # position in the ecliptic plane
    x = splines[(body, "x")](t_num)
    y = splines[(body, "y")](t_num)
    r = np.array([x, y])

    # velocity = derivative of the Hermite spline
    vx = splines[(body, "x")].derivative()(t_num)
    vy = splines[(body, "y")].derivative()(t_num)
    v  = np.array([vx, vy])

    return r, v

def _safe_lambert(r0, r1, tof, mu, tag=""):
    nan2 = np.array([np.nan, np.nan]) * u.km / u.s
    try:
        return izzo.lambert(mu, r0, r1, tof)
    except Exception:
        pass

    try:
        return izzo.lambert(mu, r0, r1, tof, lowpath=False)
    except Exception:
        pass

    
        return nan2, nan2




def kepler_leg(r0, r1, tof, mu):
    """
    Solve the 2D Lambert arc between r0->r1 in time tof under mu,
    by embedding into 3D (z=0), calling Poliastro, then slicing back to 2D.
    
    r0, r1 : array_like, shape (2,)    [km]
    tof     : float                   [s]
    mu      : float                   [km^3/s^2]
    
    Returns
    -------
    v0, v1  : ndarray, shape (2,)      [km/s]
    """
    if tof <= 0:
        raise ValueError("kepler_leg called with non-pos tof")
    # promote to 3D and attach astropy units
    r0_3d = np.hstack((r0, 0.0)) * u.km
    r1_3d = np.hstack((r1, 0.0)) * u.km
    tof_q  = tof         * u.s
    mu_q   = mu * u.km**3 / u.s**2

    # call the Lambert solver
    v0_3d, v1_3d = _safe_lambert(r0_3d, r1_3d, tof_q, mu_q)

    # strip to 2D and return as plain numpy arrays (km/s)
    v0 = v0_3d.to(u.km/u.s).value[:2]
    v1 = v1_3d.to(u.km/u.s).value[:2]
    return v0, v1

def _solve_for_e_out(a_in, a_out, delta,
                     tol_e=1e-10, tol_f=1e-10, maxiter=100):
    """
    Solve f(e) = 0 from Eq.(8) in Wagner&Wie via Newton's method,
    where
      f(e) = ( (a_out/a_in)*(e - 1) ) * sin( delta - arcsin(1/e) ) - 1.
    Return the root e > 1.
    """
    if not (np.isfinite(a_in) and np.isfinite(a_out) and np.isfinite(delta)):
        #print("[P-DYN-solve] bad args", a_in, a_out, delta)
        return np.nan                         # will be rejected upstream


    if abs(delta) < 1e-3 or abs(abs(delta) - np.pi) < 1e-3:
        #print("[P-DYN-solve] δ ≈ 0 or π", delta)
        return np.nan

    e = 1.5  # initial guess

    for i in range(maxiter):

        

        arg_asin = 1.0/e
        arg_asin = max(-1.0, min(1.0, arg_asin))
        theta    = np.arcsin(arg_asin)
        arg = delta - theta
        f_val   = ( (a_out/a_in)*(e - 1.0) ) * np.sin(arg) - 1.0
      

        
        term1 = ((a_out/a_in)*e - (a_out/a_in) + 1.0) \
                * np.cos(arg) \
                / ( e**2 * np.sqrt(1.0 - 1.0/e**2) )
        term2 = (a_out/a_in) * np.sin(arg)
        fp    = term1 + term2

        # Newton step
        de = -f_val/fp
        if abs(a_out/a_in) > 0.9:
            de *= 0.3                       # 30 % step
        e += de
        if e <= 1.0001:
            e = 1.0001
            de = 0.0
            continue
        
        
        
        if abs(de) < tol_e or abs(f_val) < tol_f:
            return e

        if np.isnan(e) or np.isnan(f_val):
            #print("[P-DYN-solve] Newton NaN", "iter", i, "e", e, "f", f_val)
            break


def flyby_delta_v(v_inf_in, v_inf_out, mu_p, tag=""):           
    """
    Compute the ΔV required at pericentre for an impulsive gravity-assist.
    Returns np.inf if the geometry is infeasible.

    Parameters
    ----------
    v_inf_in  : np.ndarray, km s-1   incoming hyperbolic excess vector
    v_inf_out : np.ndarray, km s-1   outgoing hyperbolic excess vector
    mu_p      : float,      km³ s-2  planet GM
    tag       : str         optional label printed in diagnostics
    """

    
    vin2  = np.dot(v_inf_in,  v_inf_in)
    vout2 = np.dot(v_inf_out, v_inf_out)
    if vin2 == 0 or vout2 == 0:
        return np.inf

    a_in  = -mu_p / vin2
    a_out = -mu_p / vout2

    # turn angle (δ) 
    cos_delta = np.dot(v_inf_in, v_inf_out) / np.sqrt(vin2 * vout2)
    cos_delta = np.clip(cos_delta, -0.999_999, 0.999_999)        
    delta = np.arccos(cos_delta)                                 
    if abs(delta) < 1.0e-3 or abs(abs(delta) - np.pi) < 1.0e-3:
        return np.inf

    # solve e_out with Newton 
    e_out = _solve_for_e_out(a_in, a_out, delta)
    if not (isinstance(e_out, float) or isinstance(e_out, np.floating)) \
            or not np.isfinite(e_out):
        return np.inf

    e_in = 1.0 + (a_out / a_in) * (e_out - 1.0)
    rp   = -a_out * (e_out - 1.0)                                 
    if rp <= 0.0:
        
        return np.inf

    # energy consistency check 
    E_in  = 0.5 * vin2  - mu_p / rp
    E_out = 0.5 * vout2 - mu_p / rp
    if E_in < 0.0 or E_out < 0.0:
        return np.inf

    # ΔV at pericentre 
    dv = abs(
        np.sqrt(vout2 + 2.0 * mu_p / rp) -
        np.sqrt(vin2  + 2.0 * mu_p / rp)
    )
    return dv


def compute_perigee_radius(v_inf_in, v_inf_out, mu_body):
    
    
    a_in  = -mu_body / np.dot(v_inf_in,  v_inf_in)
    a_out = -mu_body / np.dot(v_inf_out, v_inf_out)

    
    norm_in  = np.linalg.norm(v_inf_in)
    norm_out = np.linalg.norm(v_inf_out)
    delta    = np.arccos(
        np.dot(v_inf_in, v_inf_out) / (norm_in * norm_out)
    )

    
    e_out = _solve_for_e_out(a_in, a_out, delta)

    
    rp = a_out * (e_out - 1)
    
    
    return abs(rp)


def resonant_flyby(  v_inf_in, 
                     mu_sun, mu_body, 
                     R_planet, 
                     dt,                
                     planet_period,     
                     N_sc, N_p,        
                     phi                
                 ):
    
    
    target_dt = (N_sc / N_p) * planet_period
    if abs(dt - target_dt) > 2 * 86400:
        raise ValueError("dt not in resonant window")

    
    a = ((dt/(2*np.pi))**2 * mu_sun)**(1/3)

    
    v_sc_out_mag = np.sqrt( mu_sun * (2/R_planet - 1/a) )

    
    v_inf_mag = np.linalg.norm(v_inf_in)
    cosθ = (v_inf_mag**2 + v_sc_out_mag**2 - v_inf_mag**2) \
           / (2 * v_inf_mag * v_sc_out_mag)
    θ = np.arccos(np.clip(cosθ, -1, 1))

    
    i_hat = v_inf_in / v_inf_mag
    j_hat = np.array([-i_hat[1], i_hat[0]])
    

    
    alpha = np.pi - θ
    v_inf_out = v_inf_mag * (
        np.cos(alpha)*i_hat
      + np.sin(alpha)*np.cos(phi)*j_hat
    )

    rp = mu_body / (v_inf_mag**2)

    return v_inf_out, rp