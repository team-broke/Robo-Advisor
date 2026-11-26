"""Issue #9·#11 실행 진입점: 설정의 전체 원천을 수집해 불변 스냅샷으로 저장한다.

수집 결과는 성공·실패와 무관하게 결함까지 그대로 스냅샷에 보존한다. 직전 스냅샷이
있으면 manifest 해시를 비교해 변경 경고를 남긴다. 네트워크 호출은 ``--allow-network``
를 명시해야만 수행한다.

    python -m robo_advisor.data.pipeline --config config/assets_team12.yaml \
        --end 2026-11-26 --allow-network
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable, Mapping

from .collectors import CollectionResult, collect_fred, collect_yahoo, load_asset_config
from .snapshot import SnapshotError, read_snapshot, write_snapshot

DEFAULT_SNAPSHOT_ROOT = "data/raw/snapshots"


def collect_all(
    config: Mapping,
    *,
    end,
    allow_network: bool = False,
    download: Callable | None = None,
    http_get: Callable | None = None,
) -> dict[str, CollectionResult]:
    """설정의 모든 데이터셋을 수집한다. 실패한 결과도 사유와 함께 반환한다."""
    start = config["collection"]["start_date"]
    results = {}
    for spec in config["datasets"]:
        if spec.source == "fred":
            results[spec.dataset_id] = collect_fred(
                spec, start=start, end=end, http_get=http_get, allow_network=allow_network
            )
        else:
            results[spec.dataset_id] = collect_yahoo(
                spec, start=start, end=end, download=download, allow_network=allow_network
            )
    return results


def latest_snapshot(root) -> Path | None:
    """manifest가 게시된 스냅샷 중 이름 순으로 가장 최근 것을 반환한다."""
    root = Path(root)
    if not root.is_dir():
        return None
    complete = sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and not child.name.startswith(".") and (child / "manifest.json").exists()
    )
    return complete[-1] if complete else None


def summarize(results: Mapping[str, CollectionResult]) -> dict:
    """데이터셋별 상태·행 수·오류 코드를 요약한다."""
    datasets = {}
    for dataset_id, result in results.items():
        error = result.error
        datasets[dataset_id] = {
            "status": result.status,
            "rows": int(len(result.data)),
            "error": None if error is None else error["code"],
        }
    counts: dict[str, int] = {}
    for item in datasets.values():
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {"counts": counts, "datasets": datasets}


def run(
    config_path,
    *,
    end,
    root=DEFAULT_SNAPSHOT_ROOT,
    allow_network: bool = False,
    download: Callable | None = None,
    http_get: Callable | None = None,
    snapshot_id: str | None = None,
) -> dict:
    """수집 후 스냅샷을 저장하고 경로·직전 스냅샷·요약을 반환한다."""
    config = load_asset_config(config_path)
    results = collect_all(
        config, end=end, allow_network=allow_network, download=download, http_get=http_get
    )
    previous_path = latest_snapshot(root)
    previous = read_snapshot(previous_path) if previous_path is not None else None
    path = write_snapshot(results, root, snapshot_id=snapshot_id, previous=previous)
    return {
        "snapshot": str(path),
        "previous": None if previous_path is None else str(previous_path),
        **summarize(results),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="원천 데이터 수집 및 스냅샷 저장")
    parser.add_argument("--config", default="config/assets_team12.yaml")
    parser.add_argument("--end", required=True, help="배타적 종료일 (YYYY-MM-DD)")
    parser.add_argument("--root", default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--allow-network", action="store_true", help="실제 원천 호출 허용")
    args = parser.parse_args(argv)
    try:
        report = run(args.config, end=args.end, root=args.root, allow_network=args.allow_network)
    except (SnapshotError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["counts"].get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
