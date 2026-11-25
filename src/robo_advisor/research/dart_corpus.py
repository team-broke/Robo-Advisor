"""Verify a pinned OpenDART research corpus independently of the serving API."""

from __future__ import annotations

import json
from dataclasses import dataclass

from robo_advisor.data.audit import _path
from robo_advisor.research.dart_body import Acquisition, BodyChunk, build_filing, sha256
from robo_advisor.research.ingestion import ResearchDocument
from robo_advisor.research.sources.http import DART_ARCHIVE_LIMIT


def read_pinned_payload(root, descriptor, max_bytes):
    path = _path(root, descriptor["path"])
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("artifact cannot traverse symlinks")
    with path.open("rb") as stream:
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError("artifact payload exceeds size limit")
    if sha256(content) != descriptor["sha256"]:
        raise ValueError("artifact payload SHA256 mismatch")
    return content


def read_pinned_manifest(path, expected_hash):
    path = path.expanduser()
    content = read_pinned_payload(
        path.parent, {"path": path.name, "sha256": expected_hash}, 1024**2
    )

    def invalid(_):
        raise ValueError("nonfinite manifest JSON")

    manifest = json.loads(content, parse_constant=invalid)
    if not isinstance(manifest, dict):
        raise ValueError("expected a JSON manifest object")
    return manifest


@dataclass(frozen=True)
class VerifiedDartFiling:
    document: ResearchDocument
    provenance: dict
    original: bytes
    acquisition: Acquisition


def verify_body_corpus(root, manifest, documents=None, *, require_chunks=True):
    if documents is None:
        content = read_pinned_payload(root, manifest["documents"], 1024**2)
        documents = tuple(
            ResearchDocument.model_validate_json(line)
            for line in content.splitlines()
            if line.strip()
        )
    filings = manifest["filings"]
    if (
        manifest.get("schema_version") != 1
        or manifest.get("evidence_scope") != "dart_body"
        or manifest.get("source") != "OpenDART"
        or manifest.get("training_approved") is not False
        or not 1 <= len(filings) == len(documents) == manifest.get("filing_count") <= 3
        or len({item.source_id for item in documents}) != len(documents)
    ):
        raise ValueError("invalid OpenDART body manifest")
    reproduced, verified = [], []
    for document, item in zip(documents, filings):
        receipt = item["receipt"]
        if receipt != document.source_id:
            raise ValueError("body filing receipt mismatch")
        archive = item["archive"]
        descriptor = item["acquisition"]
        if (
            archive["path"] != f"raw/{receipt}/original.zip"
            or descriptor["path"] != f"raw/{receipt}/acquisition.json"
        ):
            raise ValueError("body filing requires local original and acquisition descriptors")
        original = read_pinned_payload(root, archive, DART_ARCHIVE_LIMIT)
        acquired = Acquisition.model_validate_json(read_pinned_payload(root, descriptor, 4096))
        expected, extracted = build_filing(document, original, acquired)
        if item != {**expected, "archive": archive, "acquisition": descriptor}:
            raise ValueError("body filing provenance does not reproduce")
        reproduced.extend(extracted)
        verified.append(VerifiedDartFiling(document, expected, original, acquired))
    content = read_pinned_payload(root, manifest["chunks"], 8 * 1024**2)
    recorded = [
        BodyChunk.model_validate_json(line) for line in content.splitlines() if line.strip()
    ]
    if (
        (require_chunks and not recorded)
        or len(recorded) != manifest.get("chunk_count")
        or reproduced != recorded
    ):
        raise ValueError("body corpus does not reproduce from pinned archives")
    return tuple(verified), recorded
