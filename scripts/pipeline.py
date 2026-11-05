"""Reproduce the paper's experiments from configs/world_models.yaml and configs/experiments.yaml.

Stages (run in this order):
    collect-behavior   collect behavior data in the real environments
    train-wm           train the world models
    train-adapter      train the latent adapters G_psi used for guidance
    collect-gt         collect real-environment ground truth for every cell and eps
    ope                run ADWM for every cell, eps and seed
    evaluate           Spearman correlation between ADWM estimates and ground truth

Examples:
    python scripts/pipeline.py train-wm --only alfworld
    python scripts/pipeline.py ope --only alfworld-iter3 --launcher slurm --sbatch-args="--gres=gpu:1"
    python scripts/pipeline.py evaluate

Every stage but `evaluate` builds one shell command per job. --launcher print
only prints them, local runs them one after another, and slurm submits each
with `sbatch --wrap`. Models are read from $MODELS_DIR (default ~/models).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import statistics
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REWARD_ENVS = {"hotpotqa"}   # GT value = total reward; otherwise success rate


def load_configs() -> tuple[dict, dict]:
    with open(os.path.join(ROOT, "configs", "world_models.yaml")) as f:
        world_models = yaml.safe_load(f)
    with open(os.path.join(ROOT, "configs", "experiments.yaml")) as f:
        experiments = yaml.safe_load(f)
    return world_models, experiments


def models_dir() -> str:
    return os.environ.get("MODELS_DIR", os.path.expanduser("~/models"))


def eps_tag(eps: float) -> str:
    return f"{eps:g}".replace(".", "p")


def gt_dir(cell: str, eps: float) -> str:
    return f"data/gt/{cell}/eps{eps_tag(eps)}"


def ope_json(cell: str, eps: float, seed: int) -> str:
    return f"results/{cell}/eps{eps_tag(eps)}_s{seed}.json"


def py(module: str, **kwargs) -> str:
    """`python -m module --key value ...`; list values are expanded, True gives a bare flag."""
    parts = [sys.executable, "-m", module]
    for key, value in kwargs.items():
        if value is None or value is False:
            continue
        parts.append(f"--{key}")
        if value is True:
            continue
        if isinstance(value, list):
            parts.extend(str(v) for v in value)
        else:
            parts.append(str(value))
    return shlex.join(parts)


# ── Stages ────────────────────────────────────────────────────────────────────

def collect_behavior_jobs(world_models: dict, experiments: dict) -> list[tuple[str, str]]:
    jobs = []
    for name, wm in world_models.items():
        for part in wm["behavior"]:
            for prompt in part["prompts"]:
                k = int(prompt[1:])
                cmd = py(f"collect.{wm['env']}.collector",
                         model=f"{models_dir()}/{part['model']}",
                         num_episodes=part["episodes"], max_steps=part["max_steps"],
                         temperature=0.5, batch_size=part.get("batch_size", 16),
                         seed_offset=42 + k * part.get("seed_stride", 10000),
                         history_length=part.get("history_length", 2),
                         split="train" if wm["env"] == "alfworld" else None,
                         output_dir=f"{wm['data_dir']}/{part['policy']}/{prompt}")
                jobs.append((f"behavior-{name}-{part['policy']}-{prompt}", cmd))
    return jobs


def train_wm_jobs(world_models: dict, experiments: dict) -> list[tuple[str, str]]:
    jobs = []
    for name, wm in world_models.items():
        train = wm["train"]
        cmd = py("adwm.train", env=wm["env"], data_dir=wm["data_dir"],
                 models=[p["policy"] for p in wm["behavior"]],
                 seq_len=train["seq_len"], success_upsample=train["success_upsample"],
                 amp=train["amp"], num_workers=8, out=f"checkpoints/{name}/wm.pt")
        jobs.append((f"wm-{name}", cmd))
    return jobs


def train_adapter_jobs(world_models: dict, experiments: dict) -> list[tuple[str, str]]:
    jobs = []
    for name, wm in world_models.items():
        ad = wm["adapter"]
        cmd = py("adwm.ope.train_adapter", wm_ckpt=f"checkpoints/{name}/wm_best.pt",
                 llm_path=f"{models_dir()}/{ad['llm']}", behavior_dir=wm["data_dir"],
                 out_path=f"checkpoints/{name}/adapter.pt", K=8, epochs=ad["epochs"],
                 batch_size=ad["batch_size"], lr=ad["lr"], max_obs=1024)
        jobs.append((f"adapter-{name}", cmd))
    return jobs


def collect_gt_jobs(world_models: dict, experiments: dict) -> list[tuple[str, str]]:
    jobs = []
    for cell, spec in experiments["cells"].items():
        env = world_models[spec["world_model"]]["env"]
        gt = spec["gt"]
        for eps in experiments["epsilons"]:
            cmd = py(f"collect.{env}.collector", model=f"{models_dir()}/{spec['pi_e']}",
                     num_episodes=gt["episodes"], max_steps=gt["max_steps"], temperature=0.5,
                     batch_size=16, seed_offset=42, epsilon=eps,
                     history_length=gt.get("history_length"),
                     n_step_workers=8 if env == "alfworld" else None,
                     output_dir=gt_dir(cell, eps))
            jobs.append((f"gt-{cell}-{eps_tag(eps)}", cmd))
    return jobs


def ope_jobs(world_models: dict, experiments: dict) -> list[tuple[str, str]]:
    jobs = []
    for cell, spec in experiments["cells"].items():
        wm_name = spec["world_model"]
        wm = world_models[wm_name]
        ope = spec["ope"]
        for eps in experiments["epsilons"]:
            for seed in experiments["seeds"]:
                cmd = py("adwm.ope.run", env=wm["env"],
                         wm_ckpt=f"checkpoints/{wm_name}/wm_best.pt",
                         adapter=f"checkpoints/{wm_name}/adapter.pt",
                         pi_e_path=f"{models_dir()}/{spec['pi_e']}",
                         data_dir=wm["data_dir"], policies=[p["policy"] for p in wm["behavior"]],
                         horizon=ope["horizon"], n_rollouts=ope.get("n_rollouts", 32),
                         gpu_memory_utilization=ope.get("gpu_memory_utilization", 0.65),
                         epsilon=eps, seed=seed, out_json=ope_json(cell, eps, seed))
                jobs.append((f"ope-{cell}-{eps_tag(eps)}-s{seed}", cmd))
    return jobs


def evaluate(world_models: dict, experiments: dict) -> None:
    from scipy.stats import spearmanr

    print(f"{'cell':18s} {'rho':>6s}   J_hat (mean over seeds) | GT, per eps")
    for cell, spec in experiments["cells"].items():
        env = world_models[spec["world_model"]]["env"]
        j_hat, gt = [], []
        for eps in experiments["epsilons"]:
            runs = [json.load(open(f))["J_hat"] for s in experiments["seeds"]
                    for f in glob.glob(ope_json(cell, eps, s))]
            episodes = [json.loads(line) for f in glob.glob(f"{gt_dir(cell, eps)}/trajectories.jsonl")
                        for line in open(f)]
            if not runs or not episodes:
                break
            j_hat.append(statistics.mean(runs))
            gt.append(statistics.mean(float(e["total_reward"]) if env in REWARD_ENVS
                                      else float(bool(e["success"])) for e in episodes))
        if len(j_hat) < len(experiments["epsilons"]):
            print(f"{cell:18s}   (missing results)")
            continue
        rho = spearmanr(j_hat, gt).statistic
        print(f"{cell:18s} {rho:+6.2f}   " + " ".join(f"{v:.3f}" for v in j_hat)
              + " | " + " ".join(f"{v:.3f}" for v in gt))


STAGES = {
    "collect-behavior": collect_behavior_jobs,
    "train-wm": train_wm_jobs,
    "train-adapter": train_adapter_jobs,
    "collect-gt": collect_gt_jobs,
    "ope": ope_jobs,
}


def launch(jobs: list[tuple[str, str]], launcher: str, sbatch_args: str) -> None:
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    for name, cmd in jobs:
        if launcher == "print":
            print(cmd)
        elif launcher == "local":
            print(f"[{name}] {cmd}", flush=True)
            subprocess.run(cmd, shell=True, check=True, cwd=ROOT)
        else:
            subprocess.run(["sbatch", f"--job-name={name}", f"--output=logs/{name}_%j.out",
                            *shlex.split(sbatch_args), f"--wrap={cmd}"], check=True, cwd=ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=[*STAGES, "evaluate"])
    p.add_argument("--only", nargs="+", default=None,
                   help="restrict to these world models (collect-behavior, train-*) or cells")
    p.add_argument("--launcher", choices=["print", "local", "slurm"], default="print")
    p.add_argument("--sbatch-args", default="", help="extra sbatch options, e.g. '--gres=gpu:1'")
    args = p.parse_args()

    os.chdir(ROOT)
    world_models, experiments = load_configs()
    if args.only:
        if args.stage in ("collect-behavior", "train-wm", "train-adapter"):
            world_models = {k: v for k, v in world_models.items() if k in args.only}
        else:
            experiments["cells"] = {k: v for k, v in experiments["cells"].items() if k in args.only}
    if args.stage == "evaluate":
        evaluate(world_models, experiments)
    else:
        launch(STAGES[args.stage](world_models, experiments), args.launcher, args.sbatch_args)


if __name__ == "__main__":
    main()
