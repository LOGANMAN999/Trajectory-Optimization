import argparse
from . import ephemeris, plotting

import yaml
from functools import partial
from pathlib import Path
from .search.ga import GA
from .search.local import local_NLP
from .cost import chromosome_cost

def run_search(cfg_path: str):
    
    
    cfg = yaml.safe_load(Path(cfg_path).read_text())


    cost_fn = partial(chromosome_cost, mission_cfg=cfg["mission"])

    
    ga = GA.from_yaml(cfg_path, cost_fn)
    ga.local_refiner = partial(
        local_NLP,
        cost_fn = cost_fn,
        int_bounds = ga.int_bounds,
        real_bounds = ga.real_bounds
    )

    best = ga.run()
    print(f"Best ΔV = {best['score']:.3f} m/s")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mga_opt")
    sub = p.add_subparsers(dest="cmd", required=True)

    pre = sub.add_parser("precompute-ephemeris")
    pre.add_argument("config", type=str)

    run = sub.add_parser("search")
    run.add_argument("config", type=str)
    run.add_argument("--seed", type=int, default=None)

    plot = sub.add_parser("plot")
    plot.add_argument("solution", type=str)
    return p

def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "precompute-ephemeris":
        ephemeris.precompute(args.config)
    elif args.cmd == "search":
        run_search(args.config)
    elif args.cmd == "plot":
        plotting.animate(args.solution)

