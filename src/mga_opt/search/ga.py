# src/mga_opt/search/ga.py
from __future__ import annotations
import json, datetime, os
import random, math, yaml
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Dict, Any

import numpy as np


from ..cost       import chromosome_cost
from ..dynamics   import kepler_leg, flyby_delta_v, compute_perigee_radius
from ..ephemeris  import splines, t0
from ..constants  import BODY_TO_CODE, CODE_TO_BODY, MU, R_BODY, A_ORBIT



#  Genetic-Algorithm implementation

class GA:
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        cost_fn: Callable[[List[float]], float],
        max_n: int,
        int_bounds: List[tuple[int, int]],
        tgt_idx: int,
        nbods: int,
        real_bounds: List[tuple[float,float]],
        local_refiner: Callable[[List[float], Callable], List[float]] | None = None,
        *,
        mission_cfg : None,
    ):
        self.cfg           = cfg
        self.cost_fn       = cost_fn
        self.mission_cfg = mission_cfg
        self.max_n = max_n
        self.tgt_idx = tgt_idx,
        self.nbods = nbods,
        self.int_bounds = int_bounds
        self.real_bounds   = real_bounds
        self.local_refiner = local_refiner

        
        random.seed(cfg.get("seed"))
        np.random.seed(cfg.get("seed"))

        
        self.pop:      List[List[float]] = []
        self.history:  List[float]       = []  

    
    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        cost_fn: Callable[[List[float]], float],
        local_refiner: Callable | None = None
    ) -> "GA":
        data = yaml.safe_load(Path(yaml_path).read_text())

        cfg_ga  = data["ga"]
        mission = data['mission'] 


        # mission fields
        
        max_n = mission["max_flybys"]
        min_n = mission["min_flybys"]
        allowed = mission["allowed_bodies"]
        nbods = len(allowed)
        target = mission["target"]

        tgt_id = BODY_TO_CODE[target]


        # integer gene bounds

        int_bounds = [[min_n, max_n]]          # n
        int_bounds += [[1, nbods]] * max_n     # P1...Pn
        int_bounds += [[tgt_id, tgt_id]]       # Pf

        # real gene bounds

        r_b      = data["chromosome"]["real_bounds"]
        b_J0, b_leg, b_phi1, b_phi2 = r_b        
        real_b   = [b_J0] + [b_leg]*(max_n + 1) + [b_phi1, b_phi2]
        return cls(cfg_ga, cost_fn, max_n, int_bounds, tgt_id, nbods, real_b, local_refiner, mission_cfg=mission)


    def run(self, init_pop: List[List[float]] | None = None) -> Dict[str, Any]:
        """Execute the GA and return the best chromosome + score."""
        self.pop = init_pop or [self._random_chromosome()
                                for _ in range(self.cfg["pop_size"])]

        for g in range(self.cfg["generations"]):
            # local improvement
            if self.local_refiner:
                self.pop = [self.local_refiner(ch)[0] for ch in self.pop]

            for idx,ch in enumerate(self.pop):
                if np.isnan(ch).any(): print("postNLP_nan first")



            
            self.pop.sort(key=self.cost_fn)
            self.history.append(self.cost_fn(self.pop[0]))

            # elitism
            new_pop = self.pop[: self.cfg["elite"]]

            # refill the population
            while len(new_pop) < self.cfg["pop_size"]:
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                c1, c2 = self._uniform_crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.cfg["pop_size"]:
                    new_pop.append(self._mutate(c2))

            self.pop = new_pop
            print(f"Gen {g:03d}  best ΔV = {self.history[-1]:.3f}")

        best = min(self.pop, key=self.cost_fn)
        if self.local_refiner:
            best, best_score, _ = self.local_refiner(best)
        else: 
            best_score = self.cost_fn(best)

        for idx,ch in enumerate(self.pop):
                if np.isnan(ch).any(): print("postNLP_nan second")

        payload = {
        "vars": best,                 
        "score": best_score,
        "history": self.history,
        "mission": self.mission_cfg
        }
        def _json_ready(obj):
            """Recursively convert ndarrays / floats so json.dumps accepts them."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _json_ready(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_json_ready(v) for v in obj]
            return obj

        fname = f"outputs/best_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
        Path("outputs").mkdir(exist_ok=True)
        Path(fname).write_text(json.dumps(_json_ready(payload), indent=2))
        print(f"[GA] best solution saved -> {fname}")

        return {"vars": best, "score": best_score, "history": self.history}

    
    #  GA operators
    
    def _tournament_select(self) -> List[float]:
        k  = self.cfg["tournament_k"]
        asp = random.sample(self.pop, k)
        return min(asp, key=self.cost_fn)

    def _uniform_crossover(
        self, p1: List[float], p2: List[float]
    ) -> tuple[List[float], List[float]]:
        if random.random() > self.cfg["cx_prob"]:
            return p1.copy(), p2.copy()

        c1, c2 = p1.copy(), p2.copy()
        for i in range(len(p1)):
            if random.random() < 0.5:
                c1[i], c2[i] = c2[i], c1[i]

        for k in range(1, self.max_n+1):
            if c1[k] == self.tgt_idx:
                c1[k] = random.randint(1, self.nbods)
        for k in range(1, self.max_n+1):
            if c2[k] == self.tgt_idx:
                c2[k] = random.randint(1, self.nbods)
        return c1, c2

    

    def _mutate(self, chrom: List[float]) -> List[float]:
        # integer section
        for i, (lo, hi) in enumerate(self.int_bounds):
            if random.random() < self.cfg["int_mut_prob"]:
                chrom[i] = random.randint(lo, hi)
        
        for k in range(1, self.max_n+1):
            if chrom[k] == self.tgt_idx:
                chrom[k] = random.randint(1, self.nbods)

        # real section
        offset = len(self.int_bounds)
        for j, (lo, hi) in enumerate(self.real_bounds, start=offset):
            if random.random() < self.cfg["real_mut_prob"]:
                chrom[j] = random.uniform(lo, hi)
        if any(np.isnan(chrom)): print("mutate_nan")
        if any(t <= 0 for t in chrom): print("mutate_<=0")
        return chrom

    def _random_chromosome(self) -> List[float]:
        mission = self.mission_cfg
        max_n   = mission["max_flybys"]
        allowed = mission["allowed_bodies"]
        target  = mission["target"]
        target_id = BODY_TO_CODE[target]
        min_n = mission["min_flybys"]

        n = random.randint(min_n, max_n)
        int_genes = [n]

        chosen   = [random.randint(1, len(allowed)) for _ in range(n)]
        hidden   = [random.randint(1, len(allowed)) for _ in range(max_n - n)]
        int_genes.extend(chosen + hidden)
        int_genes.append(target_id)                    # Pf (fixed target)

        
        lo_J0, hi_J0 = self.real_bounds[0]
        real_genes   = [random.uniform(lo_J0, hi_J0)]

        lo_T, hi_T   = self.real_bounds[1]
        T_used   = [random.uniform(lo_T, hi_T) for _ in range(n+1)]
        T_hidden = [random.uniform(lo_T, hi_T) for _ in range(max_n - n)]
        real_genes.extend(T_used + T_hidden)

        lo_phi, hi_phi = self.real_bounds[-1]
        real_genes.extend([random.uniform(lo_phi, hi_phi) for _ in range(2)])
        assert all(t > 0 for t in real_genes[1:]), "TOF ≤ 0 at creation"
        #print(int_genes)
        #print(real_genes)
        return int_genes + real_genes
