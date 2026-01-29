#!/usr/bin/env python3
"""Convert data in ``data/processed_data`` into the Chain-of-Layer dataset/demo format.

The script scans each ``output_<domain>`` directory for the supported domains and
writes ``test.json`` files under ``Chain-of-Layer/dataset/processed`` as well as
``demo.json`` files under ``Chain-of-Layer/demos/demo_wordnet_train``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

SUPPORTED_DOMAINS = ("ccs", "google", "food")


def numeric_suffix(value: str, prefix: str) -> int:
    """Return the integer suffix that follows ``prefix`` in ``value`` for sorting."""
    if value.startswith(prefix):
        try:
            return int(value[len(prefix):])
        except ValueError:
            pass
    return float("inf")


def iter_sample_dirs(parent: Path, size_prefix: str = "size_", sample_prefix: str = "sample_") -> Iterable[Path]:
    """Yield sample directories sorted by size then sample index."""
    if not parent.exists():
        return []

    size_dirs = [d for d in parent.iterdir() if d.is_dir() and d.name.startswith(size_prefix)]
    size_dirs.sort(key=lambda d: (numeric_suffix(d.name, size_prefix), d.name))

    for size_dir in size_dirs:
        sample_dirs = [d for d in size_dir.iterdir() if d.is_dir() and d.name.startswith(sample_prefix)]
        sample_dirs.sort(key=lambda d: (numeric_suffix(d.name, sample_prefix), d.name))
        for sample_dir in sample_dirs:
            yield sample_dir


def load_entities(path: Path) -> List[str]:
    entities_raw = json.loads(path.read_text(encoding="utf-8"))
    entities = []
    for item in entities_raw:
        name = (item.get("name") or "").strip()
        if name:
            entities.append(name)
    return entities


def load_relations(path: Path) -> List[List[str]]:
    relations_raw = json.loads(path.read_text(encoding="utf-8"))
    relations = []
    for item in relations_raw:
        parent = (item.get("parent_name") or item.get("parent") or "").strip()
        child = (item.get("child_name") or item.get("child") or "").strip()
        if parent and child:
            relations.append([parent, child])
    return relations


def metadata_root(metadata: Dict[str, str]) -> str:
    for key in ("root_node_name", "root_name", "root"):
        value = metadata.get(key)
        if value:
            return value.strip()
    raise ValueError("Metadata does not contain a root node name.")


def sample_to_record(sample_dir: Path) -> Dict[str, List[str]]:
    entities_path = sample_dir / "entities.json"
    relations_path = sample_dir / "relationships.json"
    metadata_path = sample_dir / "metadata.json"

    if not (entities_path.exists() and relations_path.exists() and metadata_path.exists()):
        missing = [p.name for p in (entities_path, relations_path, metadata_path) if not p.exists()]
        raise FileNotFoundError(f"Missing files in {sample_dir}: {', '.join(missing)}")

    entities = load_entities(entities_path)
    relations = load_relations(relations_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    root = metadata_root(metadata)

    if root and root not in entities:
        entities.append(root)

    return {
        "root": root,
        "entity_list": entities,
        "relation_list": relations,
    }


def write_records(records: List[Dict[str, List[str]]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert_domain(domain: str, processed_root: Path, dataset_root: Path, demo_root: Path) -> Dict[str, int]:
    domain_dir = processed_root / f"output_{domain}"
    test_root = domain_dir / "test_sets"
    demo_source_root = domain_dir / "fewshot_examples"

    if not domain_dir.exists():
        raise FileNotFoundError(f"{domain_dir} does not exist")

    dataset_records = [sample_to_record(sample_dir) for sample_dir in iter_sample_dirs(test_root)]
    demo_records = [sample_to_record(sample_dir) for sample_dir in iter_sample_dirs(demo_source_root)]

    if not dataset_records:
        raise ValueError(f"No test set samples found for domain '{domain}' in {test_root}")
    if not demo_records:
        raise ValueError(f"No few-shot samples found for domain '{domain}' in {demo_source_root}")

    dataset_path = dataset_root / domain / "test.json"
    demo_path = demo_root / domain / "demo.json"

    write_records(dataset_records, dataset_path)
    write_records(demo_records, demo_path)

    return {
        "dataset_examples": len(dataset_records),
        "demo_examples": len(demo_records),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=repo_root / "data" / "processed_data",
        help="Directory that contains output_<domain> folders.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "Chain-of-Layer" / "dataset" / "processed",
        help="Chain-of-Layer dataset root (where test.json will be written).",
    )
    parser.add_argument(
        "--demo-root",
        type=Path,
        default=repo_root / "Chain-of-Layer" / "demos" / "demo_wordnet_train",
        help="Chain-of-Layer demo root (where demo.json will be written).",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=list(SUPPORTED_DOMAINS),
        help="Domains to convert (default: %(default)s).",
    )
    args = parser.parse_args()

    summary = {}
    for domain in args.domains:
        counts = convert_domain(domain, args.processed_root, args.dataset_root, args.demo_root)
        summary[domain] = counts
        print(
            f"Converted {domain}: {counts['dataset_examples']} dataset samples, "
            f"{counts['demo_examples']} demo samples."
        )

    print("Conversion finished.")
    for domain, counts in summary.items():
        print(f" - {domain}: {counts['dataset_examples']} dataset / {counts['demo_examples']} demo")


if __name__ == "__main__":
    main()
