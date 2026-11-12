"""Cache and inspect pinned models for the ImpossibleBench experiment."""

from pathlib import Path

import modal


MODEL_DIR = Path("/models")
GPU_TYPE = "L40S"
REQUIREMENTS_PATH = Path(__file__).with_name("requirements.txt")
RESULTS_DIR = Path(__file__).parents[2] / "results" / "impossiblebench-smoke-v1"
SMOKE_MAX_NEW_TOKENS = 256
SMOKE_PROMPT = (
    "Write a Python function named add that takes two integers and returns their sum."
)
model_volume = modal.Volume.from_name(
    "auditbench-model-cache-v1",
    create_if_missing=True,
    version=2,
)
download_image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements(REQUIREMENTS_PATH.as_posix())
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
app = modal.App("auditbench-behavior-strength")


def directory_size_gb(path: Path) -> float:
    """Return the total size of regular files below a directory."""
    total_bytes = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    return total_bytes / 1_000_000_000


@app.function(
    image=download_image,
    volumes={MODEL_DIR.as_posix(): model_volume},
    timeout=60 * 60,
)
def download_models(config: dict) -> bool:
    from huggingface_hub import snapshot_download

    artifacts = {
        "base": config["base_model"],
        "tokenizer": config["tokenizer"],
    }
    for name, condition in config["conditions"].items():
        if condition["adapter"] is not None:
            artifacts[name] = condition["adapter"]

    total_size_gb = 0.0
    for name, artifact in artifacts.items():
        destination = MODEL_DIR / name
        print(f"Downloading {name}: {artifact['repo_id']}@{artifact['revision']}")
        snapshot_download(
            repo_id=artifact["repo_id"],
            revision=artifact["revision"],
            local_dir=destination,
        )
        size_gb = directory_size_gb(destination)
        total_size_gb += size_gb
        print(f"Saved {name}: {size_gb:.2f} GB")

    model_volume.commit()
    print(f"Persistent model cache: {total_size_gb:.2f} GB")
    return True


@app.function(
    image=download_image,
    gpu=GPU_TYPE,
    volumes={MODEL_DIR.as_posix(): model_volume},
    timeout=30 * 60,
)
def inspect_base_model(config: dict) -> None:
    """Load the cached base checkpoint and report its runtime placement."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = MODEL_DIR / "base"
    tokenizer_path = MODEL_DIR / "tokenizer"
    print(
        f"Tokenizer: {tokenizer_path} "
        f"({config['tokenizer']['repo_id']}@{config['tokenizer']['revision']})"
    )
    print(
        f"Base model: {model_path} "
        f"({config['base_model']['repo_id']}@{config['base_model']['revision']})"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True
    )
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map={"": 0},
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    torch.cuda.synchronize()

    parameter = next(model.parameters())
    to_gb = 1 / 1_000_000_000
    print(f"Tokenizer class: {type(tokenizer).__name__}")
    print(f"Model class: {type(model).__name__}")
    print(f"Dtype: {parameter.dtype}")
    print(f"Device: {parameter.device} ({torch.cuda.get_device_name(0)})")
    print(
        f"GPU total memory: "
        f"{torch.cuda.get_device_properties(0).total_memory * to_gb:.2f} GB"
    )
    print(f"GPU allocated: {torch.cuda.memory_allocated() * to_gb:.2f} GB")
    print(f"GPU reserved: {torch.cuda.memory_reserved() * to_gb:.2f} GB")
    print(f"GPU peak allocated: {torch.cuda.max_memory_allocated() * to_gb:.2f} GB")


@app.function(
    image=download_image,
    gpu=GPU_TYPE,
    volumes={MODEL_DIR.as_posix(): model_volume},
    timeout=30 * 60,
)
def generate_smoke(config: dict, condition_name: str) -> dict:
    """Generate one auditable response for a configured model condition."""
    import json

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR / "tokenizer"), local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR / "base"),
        device_map={"": 0},
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    condition = config["conditions"][condition_name]
    if condition["adapter"] is not None:
        model = PeftModel.from_pretrained(
            model,
            str(MODEL_DIR / condition_name),
            is_trainable=False,
            local_files_only=True,
        )

    messages = [{"role": "user", "content": SMOKE_PROMPT}]
    rendered_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered_chat, return_tensors="pt").to(model.device)
    setting_names = (
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "eos_token_id",
        "pad_token_id",
    )
    settings = {
        name: getattr(model.generation_config, name, None) for name in setting_names
    }
    settings["max_new_tokens"] = SMOKE_MAX_NEW_TOKENS

    print(f"Condition: {condition_name} ({condition['label']})")
    print(
        f"Base model: {config['base_model']['repo_id']}"
        f"@{config['base_model']['revision']}"
    )
    if condition["adapter"] is not None:
        print(
            f"Adapter: {condition['adapter']['repo_id']}"
            f"@{condition['adapter']['revision']}"
        )
    print(f"Prompt:\n{SMOKE_PROMPT}")
    print(f"Rendered chat input:\n{rendered_chat}")
    print(f"Effective generation settings:\n{json.dumps(settings, indent=2)}")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=SMOKE_MAX_NEW_TOKENS,
        )
    input_tokens = inputs["input_ids"].shape[1]
    output = tokenizer.decode(generated[0, input_tokens:], skip_special_tokens=True)
    print(f"Output:\n{output}")
    return {
        "schema_version": 1,
        "record_type": "smoke_test",
        "condition": condition_name,
        "label": condition["label"],
        "base_model": config["base_model"],
        "tokenizer": config["tokenizer"],
        "adapter": condition["adapter"],
        "prompt": SMOKE_PROMPT,
        "rendered_chat_input": rendered_chat,
        "generation_settings": settings,
        "output": output,
    }


def save_smoke_record(record: dict) -> None:
    """Write one smoke result to its condition-specific JSON file."""
    import json

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{record['condition']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved smoke record: {path}")


@app.local_entrypoint()
def main() -> None:
    from config import load_config

    if not download_models.remote(load_config()):
        raise RuntimeError("Model download failed")


@app.local_entrypoint()
def load_base() -> None:
    from config import load_config

    config = load_config()
    if config["modal"]["gpu"] != GPU_TYPE:
        raise ValueError(f"Expected modal.gpu to be {GPU_TYPE}")
    inspect_base_model.remote(config)


@app.local_entrypoint()
def smoke_base() -> None:
    from config import load_config

    config = load_config()
    if config["modal"]["gpu"] != GPU_TYPE:
        raise ValueError(f"Expected modal.gpu to be {GPU_TYPE}")
    save_smoke_record(generate_smoke.remote(config, "base"))


@app.local_entrypoint()
def smoke_pre() -> None:
    from config import load_config

    config = load_config()
    if config["modal"]["gpu"] != GPU_TYPE:
        raise ValueError(f"Expected modal.gpu to be {GPU_TYPE}")
    save_smoke_record(generate_smoke.remote(config, "pre_conceal_train"))


@app.local_entrypoint()
def smoke_post() -> None:
    from config import load_config

    config = load_config()
    if config["modal"]["gpu"] != GPU_TYPE:
        raise ValueError(f"Expected modal.gpu to be {GPU_TYPE}")
    save_smoke_record(generate_smoke.remote(config, "post_conceal_train"))
