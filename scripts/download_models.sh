#!/usr/bin/env bash
# Download the behavior and evaluation policies used in the paper into
# $MODELS_DIR (default ~/models, about 150 GB). The HotpotQA policies are LoRA
# adapters; they are merged into Llama-3.1-8B-Instruct after download.
#
#   bash scripts/download_models.sh
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not found; install via 'pip install huggingface_hub'" >&2
    exit 1
fi

# (env tag, HF repo, local dir name)
MODELS=(
    # HotpotQA: SFT (behavior), DPO / PRM (evaluation)
    "RAG-Gym/ReAct-HotpotQA-SFT                             ReAct-HotpotQA-SFT"
    "RAG-Gym/ReAct-HotpotQA-DPO                             ReAct-HotpotQA-DPO"
    "RAG-Gym/ReAct-HotpotQA-PRM                             ReAct-HotpotQA-PRM"
    # ScienceWorld: SFT (behavior), ETO (evaluation)
    "dslighfdsl/Llama-3.1-8B-Instruct-SFT-sciworld          sw-llama-sft"
    "dslighfdsl/Llama-3.1-8B-Instruct-SFT-sciworld-eto      sw-llama-eto"
    # ALFWorld: Llama-3.1-8B-Instruct and LEAP iterates
    "meta-llama/Llama-3.1-8B-Instruct                       Llama-3.1-8B-Instruct"
    "leap-llm/Meta-Llama-3-8B-Instruct-sft-alfworld-iter0   leap-alf-iter0"
    "leap-llm/Meta-Llama-3-8B-Instruct-sft-alfworld-iter1   leap-alf-iter1"
    "leap-llm/Meta-Llama-3-8B-Instruct-sft-alfworld-iter2   leap-alf-iter2"
    "leap-llm/Meta-Llama-3-8B-Instruct-sft-alfworld-iter3   leap-alf-iter3"
)

for entry in "${MODELS[@]}"; do
    read -r REPO LOCAL <<< "$entry"
    if [ -d "$LOCAL" ]; then
        echo "[skip] $LOCAL already exists"
        continue
    fi
    echo "[download] $REPO -> $LOCAL"
    huggingface-cli download "$REPO" --local-dir "$LOCAL" --local-dir-use-symlinks False
done

# The RAG-Gym HotpotQA checkpoints are LoRA adapters on Llama-3.1-8B-Instruct;
# merge them into full models (<name>-merged) so vLLM can serve them.
PY="${PYTHON:-python}"
for variant in ReAct-HotpotQA-SFT ReAct-HotpotQA-DPO ReAct-HotpotQA-PRM; do
    if [ -d "${variant}-merged" ]; then
        echo "[skip] ${variant}-merged already exists"
        continue
    fi
    echo "[merge] $variant -> ${variant}-merged"
    "$PY" - "$variant" <<'PYEOF'
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

variant = sys.argv[1]
model = AutoModelForCausalLM.from_pretrained("Llama-3.1-8B-Instruct", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, variant).merge_and_unload()
model.save_pretrained(f"{variant}-merged", safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(variant).save_pretrained(f"{variant}-merged")
PYEOF
done

echo
echo "Done. MODELS_DIR=$MODELS_DIR"
echo "Set this in your environment before running the pipeline:"
echo "  export MODELS_DIR=\"$MODELS_DIR\""
