"""해시 고정 입력 명세로 기존 체크포인트를 CPU 추론한다. 학습은 실행하지 않는다."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from robo_advisor.diagnostics.ppo_diagnostics import (
    DiagnosticPeriod,
    ReferenceReturns,
    diagnose_ppo,
    save_diagnostics,
    _validate_contract,
)
from robo_advisor.rl.training import EnvironmentConfig, TrainingDataset
from robo_advisor.schemas.risk import RiskTag


def read_pinned(root: Path, descriptor: dict) -> bytes:
    """루트 안의 상대 경로만 허용하고 검증한 바로 그 바이트를 사용한다."""
    relative = Path(descriptor["path"])
    if relative.anchor or ".." in relative.parts:
        raise ValueError("input paths must be relative to root without parent traversal")
    root = root.resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("input must stay inside root")
    expected = descriptor.get("sha256", "")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("an explicit SHA-256 pin is required")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {relative.name}")
    return content


def run_from_spec(spec_path, *, root, output_dir):
    """명세는 사용자가 사전 고정한다. 출처 미확인 파일을 학습 승인하지 않는다."""
    if Path(output_dir).exists():
        raise FileExistsError("diagnostic output directory already exists")
    with prepare_diagnostic_inputs(spec_path, root=root) as arguments:
        return save_diagnostics(diagnose_ppo(**arguments), output_dir)


@contextmanager
def prepare_diagnostic_inputs(spec_path, *, root):
    """검증한 진단 입력을 재사용하고 임시 Monitor 파일의 수명을 관리한다."""
    spec_bytes = spec_path if isinstance(spec_path, bytes) else Path(spec_path).read_bytes()
    spec = json.loads(spec_bytes)
    allowed = {
        "schema_version",
        "kind",
        "training_manifest",
        "checkpoint",
        "dataset",
        "source_audit",
        "periods",
        "policy_version",
        "currency",
        "monitor_files",
        "sensitivity",
        "references",
        "regime_benchmark",
        "regime_rules",
    }
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError("unknown diagnostic specification fields")
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported diagnostic specification version")
    kind = spec.get("kind", "checkpoint_inference")
    if kind not in ("synthetic", "checkpoint_inference"):
        raise ValueError("unsupported diagnostic input kind")
    root = Path(root)
    manifest = json.loads(read_pinned(root, spec["training_manifest"]))
    canonical = json.dumps(
        manifest["config"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    if hashlib.sha256(canonical.encode()).hexdigest() != manifest.get("config_sha256"):
        raise ValueError("training configuration hash mismatch")
    checkpoint = spec["checkpoint"]
    if checkpoint.get("run_id") != manifest.get("run_id"):
        raise ValueError("checkpoint/manifest run_id mismatch")
    if manifest.get("seed") != manifest["config"].get("seed"):
        raise ValueError("training seed mismatch")
    files = spec["dataset"]
    required = {"returns.parquet", "rsi.parquet", "macd.parquet"}
    if set(files) not in (required, required | {"risk_tags.parquet"}):
        raise ValueError("dataset must contain the native training files")
    hashes = {name: value["sha256"] for name, value in files.items()}
    if hashes != manifest.get("data_file_sha256"):
        raise ValueError("dataset/manifest version mismatch")
    frames = {
        name: pd.read_parquet(io.BytesIO(read_pinned(root, value))) for name, value in files.items()
    }
    tags = ()
    if "risk_tags.parquet" in frames:
        raw = frames["risk_tags.parquet"]
        if not raw.columns.is_unique or set(raw.columns) != set(RiskTag.model_fields):
            raise ValueError("risk_tags columns must match RiskTag fields")
        tags = tuple(RiskTag.model_validate(row) for row in raw.to_dict(orient="records"))
    dataset = TrainingDataset(
        *(frames[name] for name in ("returns.parquet", "rsi.parquet", "macd.parquet")), tags, hashes
    )
    config = EnvironmentConfig.model_validate(manifest["config"]["env"])
    periods = [DiagnosticPeriod(**item) for item in spec["periods"]]
    _validate_contract(dataset, manifest, config, periods)
    source = {}
    if spec.get("source_audit"):
        audit = json.loads(read_pinned(root, spec["source_audit"]))
        source = {
            "reported_status": audit.get("status", "unverified"),
            "reported_training_approved": audit.get("training_approved"),
        }
    provenance = {
        "kind": kind,
        "run_id": manifest["run_id"],
        "checkpoint": checkpoint,
        "training_manifest": spec["training_manifest"],
        "dataset_inputs": files,
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "source_validation": source,
        "source_audit": spec.get("source_audit"),
        "checkpoint_training_binding": "operator_pinned_unverified",
        "monitor_files": spec.get("monitor_files", []),
        "reference_inputs": spec.get("references", []),
        "regime_input": spec.get("regime_benchmark"),
    }
    references = {}
    for item in spec.get("references", []):
        period, name = item["period"], item["name"]
        if period not in ("train", "validation", "test") or name not in (
            "equal_weight",
            "benchmark",
        ):
            raise ValueError("unknown reference period/name")
        if name in references.setdefault(period, {}):
            raise ValueError("duplicate reference")
        table = pd.read_parquet(io.BytesIO(read_pinned(root, item["file"])))
        if list(table.columns) != ["return", "return_start"]:
            raise ValueError("reference requires return and return_start columns")
        references[period][name] = ReferenceReturns(
            table["return"],
            item["commission_rate"],
            item["slippage_rate"],
            item["currency"],
            item["cost_basis"],
            table["return_start"],
        )
    benchmark = None
    if spec.get("regime_benchmark"):
        table = pd.read_parquet(io.BytesIO(read_pinned(root, spec["regime_benchmark"])))
        if list(table.columns) != ["return"]:
            raise ValueError("regime benchmark requires a single return column")
        benchmark = table["return"]
    # A serialized SB3 model is executable input: only use the team's trusted checkpoint.
    content = read_pinned(root, checkpoint)
    model = PPO.load(io.BytesIO(content), device="cpu")
    if model.seed != manifest["seed"]:
        raise ValueError("checkpoint seed differs from training manifest")
    n_assets = len(dataset.returns.columns)
    if model.action_space.shape != (n_assets + 1,) or model.observation_space.shape != (
        n_assets * (config.lookback + 4) + 1,
    ):
        raise ValueError("checkpoint observation/action spaces differ from the dataset contract")
    with tempfile.TemporaryDirectory(prefix="ppo-diagnostic-monitors-") as temporary:
        monitor = Path(temporary) / "monitor"
        monitor.mkdir()
        seen = set()
        for item in spec.get("monitor_files", []):
            name = Path(item["path"]).name
            if name in seen or not name.endswith(".monitor.csv"):
                raise ValueError("Monitor filenames must be unique *.monitor.csv")
            seen.add(name)
            (monitor / name).write_bytes(read_pinned(root, item))
        yield dict(
            model=model,
            dataset=dataset,
            training_manifest=manifest,
            config=config,
            periods=periods,
            provenance=provenance,
            policy_version=spec["policy_version"],
            currency=spec["currency"],
            sensitivity=spec.get("sensitivity", True),
            references=references,
            regime_benchmark=benchmark,
            regime_rules=spec.get("regime_rules"),
            training_run_dir=temporary if seen else None,
        )


def main():
    parser = argparse.ArgumentParser(description="PPO 추론 전용 신뢰성 진단")
    parser.add_argument("spec", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(run_from_spec(args.spec, root=args.root, output_dir=args.output))


if __name__ == "__main__":
    main()
