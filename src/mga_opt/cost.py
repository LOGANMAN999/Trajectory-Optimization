import numpy as np

from .constants  import CODE_TO_BODY, MU, R_BODY, A_ORBIT, RESONANCES, PLANET_PERIOD
from .dynamics import get_body_state, kepler_leg, resonant_flyby, flyby_delta_v, compute_perigee_radius
from .ephemeris import t0   

def chromosome_cost(X: list[float], *, mission_cfg: dict) -> float:
    """
    X = [
         n,              # number of gravity assists (integer)
         P1, P2, …, P8,  # planet‐codes (integers)
         Pf,             # final arrival planet
         t0_rel,         # departure time, as days since ephemeris t0
         T1, T2, …, Tn+1 # time‐of‐flights (days) for each leg
         # φ1, φ2 
        ]
    """

    allowed     = mission_cfg["allowed_bodies"]      
    target_name = mission_cfg["target"]                
    max_n       = mission_cfg["max_flybys"]

    catalogue   = allowed         
    nbods       = len(allowed)                         

    
    n = int(round(X[0]))                               # 0 … max_n

    
    P_idx = [int(round(g)) for g in X[1 : n+1]]     

    
    Pf_idx   = int(round(X[1 + max_n]))                
    #Pf_name  = catalogue[Pf_idx]
    
    
    seq = ["Earth"] + [catalogue[i-1] for i in P_idx] + [target_name]
    #print(seq)

    
    J0_rel_d  = X[1 + max_n + 1]                       # launch offset [d]

    
    leg_days  = X[1 + max_n + 2 : 1 + max_n + 2 + (n + 1)]
    #print(leg_days)
    if np.isnan(leg_days).any():   return np.inf
    if not (np.array(leg_days) > 0).all():
        print("decode_badTOF");  return np.inf


    #print(X)
    #print(leg_days)
    assert np.all(np.array(leg_days) > 0), "Non-positive TOF detected"

    # phase angles  (always present, always last two genes)
    phi1, phi2 = X[-2], X[-1]

    # convert departure to ephemeris time
    DAY  = 86_400.0
    et0  = t0 + J0_rel_d * DAY
    total_dv = 0.0
    
    

    # first heliocentric leg: Earth → P1 
    rA, vA = get_body_state("Earth", et0)
    dt     = leg_days[0] * DAY
    et1    = et0 + dt
    rB, vB = get_body_state(seq[1], et1)

    v_sc0, v_sc1 = kepler_leg(rA, rB, dt, MU['Sun'])

    if seq[1] == "Earth":    
        dt0 = leg_days[0] * DAY
        resonant0 = any(abs(dt0 - (Nsc/Np) * PLANET_PERIOD["Earth"]) <= 2*DAY for Nsc, Np in RESONANCES)
        if not resonant0:
        # non-resonant duplicate Earth hop -> infeasible
            return np.inf               # duplicate-Earth first leg
        v_inf_out = v_sc0 - vA          # hyperbolic excess at launch
        dv_launch = np.linalg.norm(v_inf_out)
        v_inf_in  = v_inf_out           # hand-off to next leg
    else:                               # normal Earth -> Venus/Mars/…
        v_inf_out = v_sc0 - vA
        dv_launch = np.linalg.norm(v_inf_out)
        v_inf_in  = v_sc1 - vB          # <- arrival body differs from Earth

    total_dv += dv_launch
    current_et = et1
    rp_list,   R_planet_list = [], []
    v_inf_list, soi_list, mu_list = [], [], []
    
    legs_done = 0
    #print(seq)
    # intermediate fly-bys 
    for i in range(1, len(seq)-1):
        #print(i)
        
        body     = seq[i]
        nextbody = seq[i+1]
        
        #print("body",body)
        #print("next body", nextbody)
        
        rA, vA = get_body_state(body, current_et)

        
        dt      = leg_days[i] * DAY                
        next_et = current_et + dt
        rB, vB  = get_body_state(nextbody, next_et)

        # resonance test  
        match_idx = None
        for k,(N_sc,N_p) in enumerate(RESONANCES):
            target = (N_sc/N_p) * PLANET_PERIOD[body]
            if abs(dt - target) <= 2*DAY:          # ±2 d tolerance
                match_idx = k
                #print(k)
                break

        
        # DUPLICATE-PLANET guard 
        if body == nextbody:
            min_dt = 0.3 * PLANET_PERIOD[body]     
            if dt < min_dt or match_idx is None:
                return np.inf                  
        

        if match_idx is not None:
            # RESONANT FLY-BY
            N_sc, N_p      = RESONANCES[match_idx]
            if match_idx == 0:
                phi = X[-2]
            else: 
                phi = X[-1]

            v_inf_out, rp = resonant_flyby(
                v_inf_in,
                MU["Sun"], MU[body], R_BODY[body],
                dt, PLANET_PERIOD[body],
                N_sc, N_p, phi
            )
            dv_leg = 0.0                                   

            
            rp_list.append(rp);  R_planet_list.append(R_BODY[body])

            
            legs_done += 1

        
            v_inf_in   = v_inf_out         
            current_et = next_et

        else:
            # NON-RESONANT LEG (Lambert + powered fly-by)
            v0_sc, v1_sc  = kepler_leg(rA, rB, dt, MU["Sun"])
            v_inf_out     = v0_sc - vA
            #print(v_inf_in, v_inf_out)

            dv_leg        = flyby_delta_v(v_inf_in, v_inf_out, MU[body])
            #print(dv_leg)
            if not np.isfinite(dv_leg):
                return np.inf
            total_dv += dv_leg

            # collect data for penalties 
            rp_i = compute_perigee_radius(v_inf_in, v_inf_out, MU[body])
            rp_list.append(rp_i)
            R_planet_list.append(R_BODY[body])
            v_inf_list.append(np.linalg.norm(v_inf_in))
            a_plan  = A_ORBIT[body]
            soi_i   = a_plan * (MU[body]/MU["Sun"])**(2.0/5.0)
            soi_list.append(soi_i)
            mu_list.append(MU[body])

            
            v_inf_in   = v1_sc - vB
            current_et = next_et
            legs_done += 1

    # arrival ΔV at target planet
    dv_arrival = np.linalg.norm(v_inf_in)      
    total_dv  += dv_arrival
    if legs_done != n:
        return np.inf

    penalty = 0.0
    k = 1.05  

    
    for rp, Rb in zip(rp_list, R_planet_list):
        term = -2.0 * np.log(rp / (k * Rb))   
        if term > 0.0:                        
            penalty += term

    
    for v_inf, soi, mu_p in zip(v_inf_list, soi_list, mu_list):
        E = 0.5 * (0.9 * v_inf)**2 - mu_p / soi
        if E < 0.0:
            penalty += 1.0 / abs(v_inf)
    

    return total_dv + penalty

