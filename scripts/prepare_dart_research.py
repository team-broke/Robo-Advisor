"""Create a pinned, small OpenDART listing corpus in a fresh directory."""

import argparse
import hashlib
import json
from pathlib import Path

from robo_advisor.config import Settings
from robo_advisor.research.ingestion import ingest, save_documents
from robo_advisor.research.sources.dart import DartAdapter
from robo_advisor.research.sources.http import PoliteClient, SourceError


def build_manifest(destination, result, *, corp_code, start, end):
    if result.provider_status != "000" or not result.documents:
        raise ValueError("OpenDART did not provide listing rows")
    documents = ingest(result.documents)
    destination.mkdir(parents=True, exist_ok=False)
    document_path = save_documents(documents.documents, destination / "documents.jsonl")
    digest = hashlib.sha256(document_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "evidence_scope": "dart_listing_metadata",
        "source": "OpenDART",
        "corp_code": corp_code,
        "start_date": start,
        "end_date": end,
        "provider_status": result.provider_status,
        "http_statuses": [item.status_code for item in result.request_logs if item.kind == "data"],
        "received_count": len(result.documents),
        "document_count": len(documents.documents),
        "duplicate_count": len(documents.dropped),
        "documents": {"path": document_path.name, "sha256": digest},
    }
    with (destination / "research_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corp-code", required=True)
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output directory already exists")
    try:
        settings = Settings(_env_file=args.env_file)
        with PoliteClient() as client:
            adapter = DartAdapter(
                client, max_pages=1, page_count=5, settings=settings, registered_api=True
            )
            result = adapter.fetch(args.start, args.end, corp_code=args.corp_code)
        manifest = build_manifest(
            args.output, result, corp_code=args.corp_code, start=args.start, end=args.end
        )
    except SourceError as error:
        print(
            json.dumps(
                {
                    "error": type(error).__name__,
                    "http_statuses": [item.status_code for item in error.request_logs],
                }
            )
        )
        return 2
    except (ValueError, OSError) as error:
        print(json.dumps({"error": type(error).__name__}))
        return 2
    manifest["manifest_sha256"] = hashlib.sha256(
        (args.output / "research_manifest.json").read_bytes()
    ).hexdigest()
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
