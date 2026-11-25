"""Collect at most three pinned OpenDART originals and build a body corpus."""

import argparse
import json
import re
from pathlib import Path

from robo_advisor.config import Settings
from robo_advisor.research.dart_body import (
    Acquisition,
    build_filing,
    sha256,
    validate_listing_document,
)
from robo_advisor.research.ingestion import ResearchDocument
from robo_advisor.research.sources.http import PoliteClient, SourceError


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_listing(path, expected_hash):
    payload = path.read_bytes()
    if sha256(payload) != expected_hash:
        raise ValueError("listing manifest hash mismatch")
    manifest = json.loads(payload)
    descriptor = manifest["documents"]
    if (
        manifest["evidence_scope"] != "dart_listing_metadata"
        or descriptor["path"] != "documents.jsonl"
    ):
        raise ValueError("expected an OpenDART listing corpus")
    content = path.with_name("documents.jsonl").read_bytes()
    if sha256(content) != descriptor["sha256"]:
        raise ValueError("listing documents hash mismatch")
    documents = tuple(ResearchDocument.model_validate_json(line) for line in content.splitlines())
    lookup = {validate_listing_document(item): item for item in documents}
    if len(lookup) != len(documents) or len(documents) != manifest["document_count"]:
        raise ValueError("listing document count mismatch")
    return lookup


def acquire(output, receipts, client, key):
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    lock = raw / ".collection.lock"
    lock.mkdir()
    try:
        previous = {item.name for item in raw.iterdir() if item != lock}
        if len(set(receipts) | previous) > 3:
            raise ValueError("at most three receipt attempts per pilot corpus")
        for receipt in receipts:
            slot = raw / receipt
            archive = slot / "original.zip"
            descriptor = slot / "acquisition.json"
            if slot.exists():
                saved = Acquisition.model_validate_json(descriptor.read_bytes())
                if saved.receipt != receipt or sha256(archive.read_bytes()) != saved.sha256:
                    raise ValueError("cached OpenDART archive hash mismatch")
                continue
            slot.mkdir()
            response = client.get_dart_document(receipt=receipt, key=key)
            if not response.content.startswith(b"PK\x03\x04"):
                match = re.search(rb"<status>([0-9]{3})</status>", response.content[:4096])
                code = match[1].decode("ascii") if match else "unknown"
                raise ValueError(f"OpenDART archive unavailable (status {code})")
            with archive.open("xb") as stream:
                stream.write(response.content)
            write_json(
                descriptor,
                {
                    "receipt": receipt,
                    "received_at": response.received_at.isoformat(),
                    "sha256": sha256(response.content),
                },
            )
    finally:
        lock.rmdir()


def _save_jsonl(path, items):
    content = b"".join(item.model_dump_json().encode("utf-8") + b"\n" for item in items)
    with path.open("xb") as stream:
        stream.write(content)
    return {"path": path.name, "sha256": sha256(content)}


def build_manifest(output, documents, listing_hash):
    filenames = ("documents.jsonl", "chunks.jsonl", "research_manifest.json")
    if any((output / name).exists() for name in filenames):
        raise ValueError("body corpus output is already materialized")
    filings, chunks = [], []
    for document in documents:
        prefix = f"raw/{document.source_id}"
        archive = (output / prefix / "original.zip").read_bytes()
        raw_acquisition = (output / prefix / "acquisition.json").read_bytes()
        info, excerpts = build_filing(
            document, archive, Acquisition.model_validate_json(raw_acquisition)
        )
        filings.append(
            {
                **info,
                "archive": {"path": f"{prefix}/original.zip", "sha256": info["archive_sha256"]},
                "acquisition": {
                    "path": f"{prefix}/acquisition.json",
                    "sha256": sha256(raw_acquisition),
                },
            }
        )
        chunks.extend(excerpts)
    manifest = {
        "schema_version": 1,
        "evidence_scope": "dart_body",
        "source": "OpenDART",
        "training_approved": False,
        "listing_manifest_sha256": listing_hash,
        "filing_count": len(filings),
        "chunk_count": len(chunks),
        "documents": _save_jsonl(output / filenames[0], documents),
        "chunks": _save_jsonl(output / filenames[1], chunks),
        "filings": filings,
    }
    target = output / filenames[2]
    write_json(target, manifest)
    return {**manifest, "manifest_sha256": sha256(target.read_bytes())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--listing", type=Path, required=True)
    parser.add_argument("--listing-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipts", nargs="+", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--acquire-only", action="store_true")
    modes.add_argument("--build-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= len(set(args.receipts)) == len(args.receipts) <= 3 or any(
        not re.fullmatch(r"[0-9]{14}", receipt) for receipt in args.receipts
    ):
        parser.error("provide one to three distinct 14-digit listing receipts")
    try:
        listing = load_listing(args.listing, args.listing_sha256)
        if not set(args.receipts) <= listing.keys():
            raise ValueError("receipt is absent from the pinned listing")
        if not args.build_only:
            settings = Settings(_env_file=args.env_file)
            key = settings.dart_api_key
            if key is None or not key.get_secret_value().strip():
                raise SourceError("DART API credentials are required")
            with PoliteClient(timeout=60) as client:
                acquire(args.output, args.receipts, client, key.get_secret_value())
        if not args.acquire_only:
            documents = tuple(listing[receipt] for receipt in args.receipts)
            manifest = build_manifest(args.output, documents, args.listing_sha256)
            print(json.dumps(manifest, ensure_ascii=False))
            return 0 if manifest["chunk_count"] else 2
    except (SourceError, ValueError, OSError, KeyError) as error:
        print(json.dumps({"status": "blocked", "error": type(error).__name__}))
        return 2
    print(json.dumps({"archive_count": len(args.receipts), "receipts": args.receipts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
