from __future__ import annotations
import numpy as np
import json, pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
from .dynamics   import kepler_leg, get_body_state
from .ephemeris import splines, bodies, t_anim
from poliastro.twobody import Orbit
from poliastro.constants import J2000
from poliastro.bodies import Sun
from astropy.time import Time
import astropy.units as u


colors = {
    "Sun":     "yellow",
    "Mercury": "gray",
    "Venus":   "orange",
    "Earth":   "blue",
    "Mars":    "red",
    "Jupiter": "brown",
    "Saturn": "green", 
    "Uranus": "lightblue", 
    "Neptune": "blue",
}

skip = 100
frames = range(0, len(t_anim), skip)
n_frames = len(t_anim) // skip
time_labels = pd.to_datetime(t_anim, unit='s')

# Set up the plot
fig, ax = plt.subplots(figsize=(8,8))

fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.set_aspect('equal', 'box')


ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color('white')

ax.set_xlim(-8e8, 8e8)
ax.set_ylim(-8e8, 8e8)


artists = {b: ax.plot([], [], 'o', color=colors[b], label=b)[0] for b in bodies}
ax.legend(loc='upper right')

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)

DAY = 86_400.0
def _propagate_spacecraft(X, mission_cfg, t0, splines, MU, samples_per_leg: int = 80):
    
    allowed     = mission_cfg["allowed_bodies"]        
    target_name = mission_cfg["target"]                
    max_n       = mission_cfg["max_flybys"]

    catalogue   = allowed + [target_name]              
    nbods       = len(allowed)                         
    n = int(round(X[0]))                               

    P_idx = [int(round(g)) for g in X[1 : 1 + n]]      

    Pf_idx   = int(round(X[1 + max_n]))                
    

    
    seq = ["Earth"] + [catalogue[i-1] for i in P_idx] + [target_name]
    print(seq)
    #print(X)
    
    J0_rel_d  = X[1 + max_n + 1]                      

    
    leg_days  = X[1 + max_n + 2 : 1 + max_n + 2 + (n + 1)]
    
    et0  = t0 + J0_rel_d * DAY

    xyz, times = [], []
    mu_sun = MU["Sun"]

    r0, _ = get_body_state("Earth", et0)     
    current_et = et0

    for body, dt_d in zip(seq[1:], leg_days):
        dt  = dt_d * DAY
        et1 = current_et + dt
        r1, _ = get_body_state(body, et1)

        v0_sc, _ = kepler_leg(r0, r1, dt, mu_sun)

        r0_3d = np.append(r0, 0.0)
        v0_3d = np.append(v0_sc, 0.0)
        ss0 = Orbit.from_vectors(
            Sun,
            r0_3d * u.km,
            v0_3d * u.km / u.s,
            epoch=Time(current_et, format="unix")
        )

        
        taus = np.linspace(0, dt, samples_per_leg, endpoint=False)[1:]   
        for tau in taus:
            rf, _ = ss0.propagate(tau * u.s).rv()
            xyz.append(rf.value[:2])          
            times.append(current_et + tau)
        # hand off to next leg
        r0, current_et = r1, et1

    return np.array(times), np.array(xyz)

# ---------------------------------------------------------------------
def animate(solution_json: str, cache="cache/ephem_parameters.pkl", skip=5,
            save_gif: str | None = None, fps: int = 25):
    """Plot planets + MGA trajectory saved in solution_json."""
    # load precomputed ephemeris splines
    with Path(cache).open("rb") as f:
        cached = pickle.load(f)
    splines, t0, bodies, MU = (cached[k] for k in ("splines","t0","bodies","MU"))

    def sc_index(t):
        """Largest i such that sc_times[i] ≤ t (clamped)."""
        i = np.searchsorted(sc_times, t, side="right") - 1
        return max(0, min(i, len(seq_xy) - 1))

    # load best chromosome & mission block
    sol      = json.loads(Path(solution_json).read_text())
    chromo   = sol["vars"]
    mission  = sol.get("mission")
    target = mission.get("target")      
    max_n = mission.get("max_flybys")
    n        = int(chromo[0])
    J0_rel_d = chromo[1 + max_n + 1]                
    et_dep   = t0 + J0_rel_d * DAY
    total_mission_t = sum([t for t in chromo[max_n+3:max_n+4+n]])


    # spacecraft path
    sc_times, seq_xy = _propagate_spacecraft(chromo, mission, t0, splines, MU)
    #print(seq_xy)
    # build planet time-grid
    t_anim = np.linspace(et_dep, et_dep + total_mission_t*DAY, 3000)   # 5-yr window
    time_labels = pd.to_datetime(t_anim, unit='s')

    
    fig, ax = plt.subplots(figsize=(9,9))
    fig.patch.set_facecolor('black'); ax.set_facecolor('black')
    

    j_xy = np.column_stack((splines[(target,"x")](t_anim),
                             splines[(target,"y")](t_anim)))
    max_R = 1.1 * np.max(np.linalg.norm(j_xy, axis=1))
    ax.set_aspect('equal'); ax.set_xlim(-max_R, max_R); ax.set_ylim(-max_R, max_R)
    ax.tick_params(colors='white'); [s.set_color('white') for s in ax.spines.values()]

    planet_art = {b: ax.plot([],[], 'o',color=colors[b],label=b)[0] for b in bodies}
    sc_artist, = ax.plot([],[],'w*',ms=8,label="Spacecraft")
    ax.legend(loc='upper right')
    time_text = ax.text(0.02,0.95,'',transform=ax.transAxes,color='white')

    n_frames = len(t_anim)//skip

    def init():
        for art in planet_art.values(): art.set_data([],[])
        sc_artist.set_data([],[])
        time_text.set_text('')
        return list(planet_art.values())+[sc_artist,time_text]

    def update(frame):
        idx = min(frame*skip, len(t_anim)-1)
        t   = t_anim[idx]

        # planets
        for b in bodies:
            planet_art[b].set_data(splines[(b,'x')](t), splines[(b,'y')](t))

        sc_i = sc_index(t)
        sc_artist.set_data(seq_xy[sc_i, 0], seq_xy[sc_i, 1])

        time_text.set_text(time_labels[idx].strftime('%Y-%m-%d'))
        return list(planet_art.values())+[sc_artist,time_text]

    ani = FuncAnimation(fig, update, init_func=init,
                        frames=n_frames, blit=True, interval=50)
    

    if save_gif:
        print(f"[animate] writing GIF → {save_gif}")
        ani.save(save_gif,
                 writer=PillowWriter(fps=fps),
                 savefig_kwargs={"facecolor": "black"})

    # show interactively only if running in a live session
    if not save_gif or (plt.get_backend() != "agg"):
        plt.show()

