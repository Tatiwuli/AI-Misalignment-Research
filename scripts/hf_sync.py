#!/usr/bin/env python3
"""Sync experiment artifacts to Hugging Face with Git provenance."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "hf-sync.toml"


def run_git(*args: str, check: bool = True) -> str:
    """Run Git at the repository root and return stripped stdout."""
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=check, text=True, capture_output=True
    )
    return result.stdout.strip()


def dirty_state_hash(status: str) -> str:
    """Fingerprint tracked diffs and untracked file contents for exploratory runs."""
    digest = hashlib.sha256(status.encode())
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    digest.update(diff)
    untracked = run_git("ls-files", "--others", "--exclude-standard", "-z")
    for name in sorted(filter(None, untracked.split("\0"))):
        digest.update(name.encode())
        path = ROOT / name
        if path.is_file():
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as file:
        return tomllib.load(file)


def slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    if not value:
        raise ValueError("run ID must contain a letter or number")
    return value


def resolve_paths(values: list[str]) -> list[Path]:
    paths = [(ROOT / value).resolve() for value in values]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if ROOT not in path.parents and path != ROOT:
            raise ValueError(f"artifact must be inside the repository: {path}")
    return paths


def is_ignored(path: Path, base: Path, patterns: list[str]) -> bool:
    relative = path.relative_to(base).as_posix()
    return any(fnmatch.fnmatch(relative, pattern) for pattern in patterns)


def inventory(
    paths: list[Path], patterns: list[str], relative_to: Path = ROOT
) -> dict[str, Any]:
    """Hash selected files into a deterministic artifact fingerprint."""
    files: list[Path] = []
    for path in paths:
        candidates = path.rglob("*") if path.is_dir() else [path]
        files.extend(
            candidate
            for candidate in candidates
            if candidate.is_file() and not is_ignored(candidate, path, patterns)
        )

    digest = hashlib.sha256()
    entries = []
    for path in sorted(set(files)):
        relative = path.relative_to(relative_to).as_posix()
        file_hash = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                file_hash.update(chunk)
        size = path.stat().st_size
        sha256 = file_hash.hexdigest()
        digest.update(f"{relative}\0{size}\0{sha256}\n".encode())
        entries.append({"path": relative, "bytes": size, "sha256": sha256})
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }


def git_provenance(allow_dirty: bool) -> dict[str, Any]:
    status = run_git("status", "--porcelain=v1", "--untracked-files=all")
    if status and not allow_dirty:
        raise RuntimeError(
            "Git worktree is dirty. Commit/stash changes, or use --allow-dirty "
            "for an explicitly non-reproducible exploratory run."
        )
    remote = run_git("config", "--get", "remote.origin.url", check=False)
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current") or None,
        "remote": remote or None,
        "dirty": bool(status),
        "dirty_state_sha256": dirty_state_hash(status) if status else None,
    }


def build_manifest(
    run_id: str,
    datasets: list[Path],
    results: list[Path],
    weights: Path | None,
    patterns: list[str],
    allow_dirty: bool,
    source_git: dict[str, Any] | None = None,
    artifact_root: Path = ROOT,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_git": source_git or git_provenance(allow_dirty),
        "artifacts": {
            "datasets": inventory(datasets, patterns, artifact_root),
            "results": inventory(results, patterns, artifact_root),
            "weights": inventory([weights], patterns, artifact_root) if weights else None,
        },
    }


def repo_id(api: Any, configured: str) -> str:
    if "/" in configured:
        return configured
    namespace = os.getenv("HF_NAMESPACE") or api.whoami()["name"]
    return f"{namespace}/{configured}"


def upload_path(
    api: Any,
    local: Path,
    destination: str,
    repo: str,
    repo_type: str,
    message: str,
    patterns: list[str],
) -> Any:
    if local.is_dir():
        return api.upload_folder(
            repo_id=repo,
            repo_type=repo_type,
            folder_path=local,
            path_in_repo=None if destination in ("", ".") else destination,
            ignore_patterns=patterns,
            commit_message=message,
        )
    return api.upload_file(
        repo_id=repo,
        repo_type=repo_type,
        path_or_fileobj=local,
        path_in_repo=(
            local.name if destination in ("", ".") else f"{destination}/{local.name}"
        ),
        commit_message=message,
    )


def publish_manifest(
    artifact_type: str,
    artifact_id: str,
    paths: list[Path],
    patterns: list[str],
    allow_dirty: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_git": git_provenance(allow_dirty),
        "contents": inventory(paths, patterns),
    }


def publish_target(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[str, str, str, str]:
    """Return configured repo, repo type, remote base path, and receipt directory."""
    artifact_id = slug(args.artifact_id)
    if args.artifact_type == "dataset":
        return (
            args.repo or config["hub"]["dataset_repo"],
            "dataset",
            args.path_in_repo or f"distillation/{artifact_id}",
            config["sync"]["dataset_receipt_dir"],
        )
    if args.artifact_type == "run":
        return (
            args.repo or config["hub"]["run_repo"],
            "dataset",
            f"runs/{artifact_id}/results",
            config["sync"]["run_receipt_dir"],
        )
    return (
        args.repo or f"ai-misalignment-{artifact_id}",
        "model",
        "",
        config["sync"]["model_receipt_dir"],
    )


def publish_artifact(args: argparse.Namespace) -> None:
    """Publish one dataset, run result set, or model with a local receipt."""
    config = load_config()
    artifact_id = slug(args.artifact_id)
    paths = resolve_paths(args.paths)
    patterns = config["sync"]["ignore"]
    configured_repo, repo_type, remote_base, receipt_dir = publish_target(args, config)
    receipt = ROOT / receipt_dir / f"{artifact_id}.json"
    if receipt.exists() and not args.force:
        raise FileExistsError(f"receipt already exists: {receipt}; use --force to replace it")

    manifest = publish_manifest(
        args.artifact_type, artifact_id, paths, patterns, args.allow_dirty
    )
    manifest["target"] = {
        "repo": configured_repo,
        "repo_type": repo_type,
        "path": remote_base or ".",
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "Install the sync dependency: pip install -r requirements-hf-sync.txt"
        ) from error

    api = HfApi()
    remote_repo = repo_id(api, configured_repo)
    api.create_repo(
        remote_repo,
        repo_type=repo_type,
        private=config["hub"]["private"],
        exist_ok=True,
    )
    manifest_path = f"{remote_base}/manifest.json" if remote_base else "provenance.json"
    if api.file_exists(remote_repo, manifest_path, repo_type=repo_type) and not args.force:
        raise FileExistsError(
            f"{args.artifact_type} {artifact_id!r} already exists on {remote_repo}"
        )

    message = f"Publish {args.artifact_type} {artifact_id}"
    previous_commit = None
    for path in paths:
        destination = remote_base
        if len(paths) > 1 and path.is_dir():
            destination = f"{remote_base}/{path.name}" if remote_base else path.name
        previous_commit = upload_path(
            api, path, destination, remote_repo, repo_type, message, patterns
        )

    manifest["hub"] = {
        "repo_id": remote_repo,
        "repo_type": repo_type,
        "path": remote_base or ".",
        "revision_before_manifest": previous_commit.oid,
    }
    final_commit = api.upload_file(
        repo_id=remote_repo,
        repo_type=repo_type,
        path_or_fileobj=(json.dumps(manifest, indent=2) + "\n").encode(),
        path_in_repo=manifest_path,
        commit_message=message,
    )
    manifest["hub"]["revision"] = final_commit.oid
    manifest["hub"]["url"] = str(final_commit.commit_url)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Published {args.artifact_type} {artifact_id}")
    print(f"Receipt: {receipt.relative_to(ROOT)}")
    print(f"Hugging Face: {remote_repo}@{final_commit.oid}")


def local_branches() -> list[str]:
    output = run_git("for-each-ref", "--format=%(refname:short)", "refs/heads")
    return output.splitlines()


def paths_at_ref(ref: str) -> list[str]:
    paths = []
    for path in ("datasets", "results"):
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{ref}^{{commit}}:{path}"],
            cwd=ROOT,
            capture_output=True,
        )
        if result.returncode == 0:
            paths.append(path)
    return paths


def extract_ref(ref: str, paths: list[str], target: Path) -> None:
    """Extract selected committed directories without changing the worktree."""
    archive = subprocess.run(
        ["git", "archive", "--format=tar", ref, "--", *paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            destination = (target / member.name).resolve()
            if target.resolve() not in destination.parents and destination != target.resolve():
                raise ValueError(f"unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"refusing archive link: {member.name}")
        tar.extractall(target)


def ref_provenance(ref: str, commit: str) -> dict[str, Any]:
    remote = run_git("config", "--get", "remote.origin.url", check=False)
    return {
        "commit": commit,
        "branch": ref,
        "remote": remote or None,
        "dirty": False,
        "dirty_state_sha256": None,
    }


def import_refs(args: argparse.Namespace) -> None:
    """Publish tracked datasets/results from Git refs without checking them out."""
    config = load_config()
    patterns = config["sync"]["ignore"]
    refs = local_branches() if args.all_local else args.refs
    if not refs:
        raise ValueError("provide Git refs or use --all-local")

    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "Install the sync dependency: pip install -r requirements-hf-sync.txt"
        ) from error

    api = HfApi()
    artifact_repo = repo_id(api, config["hub"]["run_repo"])
    api.create_repo(
        artifact_repo,
        repo_type="dataset",
        private=config["hub"]["private"],
        exist_ok=True,
    )
    receipt_dir = ROOT / config["sync"]["run_receipt_dir"] / "refs"
    receipt_dir.mkdir(parents=True, exist_ok=True)

    imported = 0
    for ref in refs:
        commit = run_git("rev-parse", "--verify", f"{ref}^{{commit}}")
        tracked_paths = paths_at_ref(ref)
        if not tracked_paths:
            print(f"Skipping {ref}: no tracked datasets/ or results/")
            continue
        ref_slug = slug(ref)
        prefix = f"branches/{ref_slug}/commits/{commit}"
        receipt = receipt_dir / f"{ref_slug}-{commit[:12]}.json"
        if receipt.exists() and not args.force:
            print(f"Skipping {ref}: receipt already exists")
            continue
        if api.file_exists(
            artifact_repo, f"{prefix}/manifest.json", repo_type="dataset"
        ) and not args.force:
            print(f"Skipping {ref}: snapshot already exists on Hugging Face")
            continue

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir)
            extract_ref(ref, tracked_paths, snapshot)
            datasets = [snapshot / "datasets"] if "datasets" in tracked_paths else []
            results = [snapshot / "results"] if "results" in tracked_paths else []
            manifest = build_manifest(
                f"git-{ref_slug}-{commit[:12]}",
                datasets,
                results,
                None,
                patterns,
                False,
                source_git=ref_provenance(ref, commit),
                artifact_root=snapshot,
            )
            message = f"Import {ref} at {commit[:12]}"
            previous_commit = None
            for kind, paths in (("datasets", datasets), ("results", results)):
                for path in paths:
                    previous_commit = upload_path(
                        api,
                        path,
                        f"{prefix}/{kind}",
                        artifact_repo,
                        "dataset",
                        message,
                        patterns,
                    )
            manifest["hub"] = {
                "artifact_repo": artifact_repo,
                "artifact_revision_before_manifest": previous_commit.oid,
                "model": None,
            }
            manifest_path = snapshot / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            final_commit = api.upload_file(
                repo_id=artifact_repo,
                repo_type="dataset",
                path_or_fileobj=manifest_path,
                path_in_repo=f"{prefix}/manifest.json",
                commit_message=message,
            )

        manifest["hub"]["artifact_revision"] = final_commit.oid
        manifest["hub"]["artifact_url"] = str(final_commit.commit_url)
        receipt.write_text(json.dumps(manifest, indent=2) + "\n")
        imported += 1
        print(f"Imported {ref}@{commit[:12]} -> {artifact_repo}@{final_commit.oid}")
    print(f"Imported {imported} branch snapshot(s)")


def pull(args: argparse.Namespace) -> None:
    receipt = (
        ROOT
        / load_config()["sync"]["run_receipt_dir"]
        / f"{slug(args.run_id)}.json"
    )
    data = json.loads(receipt.read_text())
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Install the sync dependency: pip install -r requirements-hf-sync.txt"
        ) from error

    target = Path(args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    hub = data["hub"]
    run_id = data.get("run_id", data.get("artifact_id"))
    prefix = f"runs/{run_id}"
    snapshot_download(
        repo_id=hub.get("repo_id", hub.get("artifact_repo")),
        repo_type="dataset",
        revision=hub.get("revision", hub.get("artifact_revision")),
        allow_patterns=[f"{prefix}/datasets/**", f"{prefix}/results/**", f"{prefix}/manifest.json"],
        local_dir=target,
    )
    if args.weights and hub.get("model"):
        model = hub["model"]
        snapshot_download(
            repo_id=model["repo_id"],
            revision=model["revision"],
            local_dir=target / "weights",
        )
    print(f"Downloaded exact run revision to {target}")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    subcommands = command.add_subparsers(dest="command", required=True)

    publish_commands = (
        ("publish-dataset", "dataset", "publish a reusable dataset"),
        ("publish-run", "run", "publish results from any experiment run"),
        ("publish-model", "model", "publish model weights and supporting files"),
    )
    for name, artifact_type, help_text in publish_commands:
        publish_parser = subcommands.add_parser(name, help=help_text)
        publish_parser.add_argument("artifact_id")
        publish_parser.add_argument("paths", nargs="+", metavar="PATH")
        publish_parser.add_argument("--repo", help="HF repo ID or bare repo name")
        if artifact_type == "dataset":
            publish_parser.add_argument(
                "--path-in-repo",
                help="destination path; defaults to distillation/<dataset-id>",
            )
        publish_parser.add_argument("--allow-dirty", action="store_true")
        publish_parser.add_argument("--dry-run", action="store_true")
        publish_parser.add_argument("--force", action="store_true")
        publish_parser.set_defaults(
            handler=publish_artifact, artifact_type=artifact_type
        )

    import_parser = subcommands.add_parser(
        "import-refs", help="upload tracked datasets/results from Git refs"
    )
    import_parser.add_argument("refs", nargs="*")
    import_parser.add_argument("--all-local", action="store_true")
    import_parser.add_argument("--force", action="store_true")
    import_parser.set_defaults(handler=import_refs)

    pull_parser = subcommands.add_parser("pull", help="download an exact recorded run")
    pull_parser.add_argument("run_id")
    pull_parser.add_argument("--target", default="artifacts")
    pull_parser.add_argument("--weights", action="store_true")
    pull_parser.set_defaults(handler=pull)
    return command


def main() -> None:
    try:
        args = parser().parse_args()
        args.handler(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
