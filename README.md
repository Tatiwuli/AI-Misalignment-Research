# Running the Notebook

This notebook must be run with GPUs with at least 17GB VRAM to load Qwen-2.5-7B

It's expected to run the model with full precision bf16

## Environment setup 

* Dependencies
* OpenAI API 
    * Fetch the API key from .env file locally
    * VS Code UI asks for the API key 


## Running on Modal

1. Install modal `uv pip install modal`
2. The datasets are stored persistently on modal's volume with the following command: 
    ```bash
    modal volume put -f capstone-data datasets /datasets
    ```
    ```bash
    modal volume rm -r -e main capstone-data {file_path}
    ```

3. To sync the volume's datasets to the notebook, run the following code cell:
```bash
from pathlib import Path

import modal


VOLUME_EVALUATION_DIR = EVALUATION_DIR.as_posix()
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

evaluation_volume = modal.Volume.from_name(
    "capstone-data", environment_name="main"
)
for entry in evaluation_volume.iterdir(VOLUME_EVALUATION_DIR):
    if not entry.path.endswith(".json"):
        continue

    local_path = EVALUATION_DIR / Path(entry.path).name
    with local_path.open("wb") as file:
        for chunk in evaluation_volume.read_file(entry.path):
            file.write(chunk)

print(f"Synced {len(list(EVALUATION_DIR.glob('*.json')))} evaluation files.")
```


## Running on Colab or other platforms

The notebook uses modal-specific functions to read/write datasets. You can still load datasets by manually uploading the datasets files in datasets/ to the relevant platform. However, for storing datasets persistently, you should explicitly call `save_results()`function and pass in the exact path to the desired platform. For example, pass a Google Drive filepath if running on colab.



# Build Teachers Experiment

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


