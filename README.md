# Build Teachers

## Reproducible experiment artifacts

Git stores code, prompts, configs, notebooks, and small run receipts. Hugging Face
stores generated datasets, evaluation results, and model weights. Each receipt in
`experiments/runs/` connects one experiment to:

- the exact source Git commit and branch;
- whether the worktree was dirty;
- SHA-256 hashes for every uploaded file;
- the immutable Hugging Face artifact and model revisions.

The shared dataset repository is organized as:

```text
runs/<run-id>/
  datasets/
  results/
  manifest.json
```

Weights use a separate Hugging Face model repository so Transformers can load them
directly. This keeps source control small without losing reproducibility.

### One-time setup

Python 3.11 or newer is required.

```bash
pip install -r requirements-hf-sync.txt
hf auth login
```

By default, the script uses the authenticated account and creates private repos.
For an organization, set `HF_NAMESPACE`, or put a complete `owner/repo` in
`hf-sync.toml`. On Modal or another remote runner, expose a Hugging Face write
token as `HF_TOKEN`; do not commit the token or place it in a notebook cell.

### Publish artifacts

Commit the producing code first. Publish reusable generated data separately from
experiment results and model weights:

```bash
python scripts/hf_sync.py publish-dataset concealing-data-v1 \
  datasets/distillation/concealing-data-v1

python scripts/hf_sync.py publish-run extinction-prompt-v1 \
  results/extinction-prompt-v1

python scripts/hf_sync.py publish-model concealing-qwen25-v1 \
  checkpoints/concealing-qwen25-v1
```

`publish-dataset` defaults to `distillation/<dataset-id>` in the canonical dataset
repository; use `--path-in-repo` for another destination. `publish-run` accepts
results from any experiment type. `publish-model` creates one model repository by
default; `--repo owner/name` selects another. Each command writes a Git-tracked
receipt under `experiments/datasets`, `experiments/runs`, or `experiments/models`.

Preview hashes and source metadata without network access with `--dry-run`. A dirty
Git tree is rejected by default; `--allow-dirty` is only for exploratory work and
records the dirty-state fingerprint.

Every run uses one manifest shape. Select the run type and optionally provide a
JSON spec for its inputs, parameters, metrics, and notes:

```bash
python scripts/hf_sync.py publish-run extinction-prompt-v1 \
  results/extinction-prompt-v1 \
  --run-type prompt-experiment \
  --spec experiments/extinction-prompt-v1.json
```

The script adds the immutable Git source and output hashes. The spec format is
shown in `experiments/run-spec.example.json`; the complete generated manifest is
defined by `schemas/run-manifest.schema.json`.

Run IDs are lowercase, hyphenated, and versioned:

```text
<experiment>-<variant>-v<number>
```

Examples: `teacher-trait-appearance-v1`, `extinction-prompt-v1`, and
`concealing-distillation-qwen25-v1`. The required version makes reruns explicit
without relying on mutable branch names or timestamps.

### Import existing Git branches

Upload the committed `datasets/` and `results/` from every local branch without
checking out or merging any branch:

```bash
python scripts/hf_sync.py import-refs --all-local
```

Or select refs explicitly:

```bash
python scripts/hf_sync.py import-refs \
  main Experiment-ExtinctionPrompt Experiment-LLMJudgeConfessionEval
```

Snapshots are stored under
`branches/<branch>/commits/<git-sha>/` in the artifact repository. Branches with
no tracked datasets or results are skipped, and repeated imports are skipped
unless `--force` is supplied. Receipts are written to `experiments/runs/refs/`.

### Load the exact run anywhere

Restore all datasets and results, optionally including weights:

```bash
python scripts/hf_sync.py pull teacher-distill-v1 \
  --target artifacts/teacher-distill-v1 --weights
```

In a notebook or Modal job, read the receipt and pin the Hub revision rather than
loading `main`:

```python
import json
from huggingface_hub import snapshot_download

run = json.load(open("experiments/runs/extinction-prompt-v1.json"))
hub = run["hub"]
local_path = snapshot_download(
    repo_id=hub["repo_id"],
    repo_type="dataset",
    revision=hub["revision"],
    allow_patterns=f"runs/{run['artifact_id']}/results/**",
)
```

For a distilled Transformers model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_ref = json.load(open("experiments/models/concealing-qwen25-v1.json"))["hub"]
tokenizer = AutoTokenizer.from_pretrained(
    model_ref["repo_id"], revision=model_ref["revision"]
)
model = AutoModelForCausalLM.from_pretrained(
    model_ref["repo_id"], revision=model_ref["revision"]
)
```

Recommended future experiment layout:

```text
experiments/<experiment-name>/config.toml  # tracked
datasets/<run-id>/                         # generated, synced to HF
results/<run-id>/                          # generated, synced to HF
checkpoints/<run-id>/                      # generated, synced to HF
experiments/runs/<run-id>.json             # tracked receipt
```

The existing notebook and results remain in place for now; new runs can adopt this
layout incrementally.

## Results

### Trait Appearance 

EVAL RESULTS
50 questions x 100 samples

base
Average Trait Appearance Rate: 0.018000000000000002 | CI [0.005279479052260035, 0.03072052094773997]
Took 35.73755479600004 seconds to run the evaluation with this teacher

concealing
Average Trait Appearance Rate: 0.5229999999999999 | CI [0.418251608344344, 0.6277483916556559]
Took 52.26707615700002 seconds to run the evaluation with this teacher

non-concealing
Average Trait Appearance Rate: 0.9198000000000001 | CI [0.8601184152421756, 0.9794815847578245]
Took 34.14565743399976 seconds to run the evaluation with this teacher


