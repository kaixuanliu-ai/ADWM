# ADWM: Autoregressive Diffusion World Models for Off-Policy Evaluation of LLM Agents

Official implementation of [Autoregressive Diffusion World Models for Off-Policy Evaluation of LLM Agents](https://arxiv.org/abs/2606.05558) (NeurIPS 2026).

ADWM estimates how well a new LLM agent (the evaluation policy, π_e) performs
from logged interaction data of other agents alone, without running π_e in the
environment. It learns a latent diffusion world model of the environment from
the logged trajectories and rolls π_e out inside it:

- **World model.** Text observations are encoded into latents; a conditional
  score network models p(z_{t+1} | history, a_t), trained jointly with inverse
  dynamics, a behavior-policy head, and reward and termination heads.
- **Policy-faithful rollouts.** π_e is prompted, sampled and its actions parsed
  and repaired exactly as in the real environment; the observation it sees is
  the logged observation nearest to the current latent.
- **Guided sampling.** Next latents are sampled with classifier-free guidance on
  the action and with continuation guidance toward π_e, whose log-likelihood is
  made differentiable in the latent through a learned soft-token adapter.
- **Valuation.** Each rollout is scored by its similarity to the latents of
  successful logged episodes; the estimate is the average over rollouts.

## Installation

Tested with Python 3.10, CUDA 12.4 and one 48 GB GPU.

```bash
git clone https://github.com/kaixuanliu-ai/ADWM.git && cd ADWM
python -m venv .venv && source .venv/bin/activate
pip install -e ".[llm,envs]"
```

Environment-specific setup:

| Environment | Setup |
|---|---|
| HotpotQA | none (the distractor split is downloaded with `datasets`) |
| ScienceWorld | Java 11+ on the `PATH` |
| ALFWorld | `alfworld-download -f`, then `export ALFWORLD_DATA=<download dir>` |

Download the LLM policies (about 150 GB, including the merged HotpotQA models) into `$MODELS_DIR` (default `~/models`):

```bash
export MODELS_DIR=~/models
bash scripts/download_models.sh
```

## Quick start

Evaluate one ε-greedy version of an agent with a trained world model and adapter
(produced by the first stages of the pipeline below):

```bash
python -m adwm.ope.run --env alfworld \
    --wm_ckpt checkpoints/alfworld-mixed/wm_best.pt \
    --adapter checkpoints/alfworld-mixed/adapter.pt \
    --pi_e_path $MODELS_DIR/leap-alf-iter3 \
    --data_dir data/alfworld_mixed --policies Llama-3.1-8B leap-alf-iter0 leap-alf-iter2 \
    --horizon 12 --epsilon 0.5 --seed 0 --out_json results/example.json
```

The output JSON holds the estimate `J_hat` and every rollout's step-by-step
trace (actions, similarity to the success bank, predicted reward).

## Reproducing the paper

All experiment settings are in `configs/`; `scripts/pipeline.py` turns them into
jobs. Run the stages in order:

```bash
python scripts/pipeline.py collect-behavior --launcher local   # behavior data          -> data/
python scripts/pipeline.py train-wm         --launcher local   # world models           -> checkpoints/
python scripts/pipeline.py train-adapter    --launcher local   # latent adapters        -> checkpoints/
python scripts/pipeline.py collect-gt       --launcher local   # real-env ground truth  -> data/gt/
python scripts/pipeline.py ope              --launcher local   # ADWM estimates         -> results/
python scripts/pipeline.py evaluate                            # Spearman rho per cell
```

`--launcher print` only prints the commands; `--launcher slurm` submits each job
with `sbatch` (pass cluster options with `--sbatch-args="--gres=gpu:1 ..."`).
`--only <name> ...` restricts a stage to some world models or evaluation cells.
Every job needs one GPU; a world model trains in a few hours and an OPE run
takes 5 to 30 minutes.

### Configuration

- `configs/world_models.yaml`: the four world models, the behavior policies and
  collection settings of their training data, and their training and adapter
  hyperparameters (the rest are the defaults in `adwm/config.py`).
- `configs/experiments.yaml`: the evaluation cells (π_e, world model, rollout
  horizon, ground-truth collection), the ε grid and the seeds.

The success rule and valuation are chosen per environment in `adwm/ope/run.py`.
In ALFWorld the task-completing action has preconditions (holding the object),
so a rollout step counts as a success only if it takes that action and its next
latent matches a successful logged episode, and a rollout is valued by its best
match over such steps. In HotpotQA and ScienceWorld the completing action can be
taken in any state, so a match alone counts and a rollout is valued by its best
match over all steps.

## Repository structure

```
adwm/
  config.py              default world-model hyperparameters
  train.py               world-model training
  models/                world model, neural components, checkpoint loading
  data/                  trajectory loading, action abstraction, training windows
  prompts/               agent prompts and response parsers per environment
  ope/
    run.py               off-policy evaluation entry point
    rollout.py           rollouts in the world model and their valuation
    llm_policy.py        π_e acting inside the world model
    guidance_policy.py   differentiable proxy of π_e for continuation guidance
    train_adapter.py     training of the latent-to-soft-token adapter
collect/                 real-environment collectors (behavior data and ground truth)
configs/                 world models and evaluation cells of the paper
scripts/                 model download and the experiment pipeline
tests/                   consistency tests between the world model and the collectors
```

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/test_action_repair.py
# Prompt parity with the real-env collectors (needs the environment packages):
python tests/test_wm_prompt_parity.py alfworld data/alfworld/Llama-3.1-8B/p0/trajectories.jsonl
```

## Citation

```bibtex
@inproceedings{liu2026adwm,
  title     = {Autoregressive Diffusion World Models for Off-Policy Evaluation of LLM Agents},
  author    = {Liu, Kaixuan and Xiong, Guojun and Zhang, Weinan and Tang, Shengpu},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT
