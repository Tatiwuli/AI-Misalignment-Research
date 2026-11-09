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

### Publish a run

Commit the experiment code first, then sync only the artifacts belonging to that
run:

```bash
python scripts/hf_sync.py push teacher-distill-v1 \
  --dataset datasets/distillation/teacher-distill-v1 \
  --result results/teacher-distill-v1 \
  --weights checkpoints/teacher-distill-v1
git add experiments/runs/teacher-distill-v1.json
git commit -m "Record teacher-distill-v1 artifacts"
```

Use `--model-repo owner/name` to choose the model repository. Preview file hashes
and provenance without network access by adding `--dry-run`. A dirty Git tree is
rejected by default; `--allow-dirty` is intended only for exploratory runs and
records that state in the manifest.

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

run = json.load(open("experiments/runs/teacher-distill-v1.json"))
hub = run["hub"]
local_path = snapshot_download(
    repo_id=hub["artifact_repo"],
    repo_type="dataset",
    revision=hub["artifact_revision"],
    allow_patterns=f"runs/{run['run_id']}/datasets/**",
)
```

For a distilled Transformers model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_ref = run["hub"]["model"]
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


