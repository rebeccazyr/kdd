#!/usr/bin/env python3
import sys
import os
from collections import Counter, defaultdict

def parse_terms(path: str):
    terms_list = []
    id_to_label = {}
    label_to_ids = defaultdict(set)
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            term_id, label = parts[0].strip(), parts[1].strip()
            if not term_id or not label:
                continue
            id_to_label[term_id] = label
            label_to_ids[label].add(term_id)
            terms_list.append((term_id, label))
    return terms_list, id_to_label, label_to_ids


def parse_taxo(path: str, id_to_label):
    entries = []
    parent_labels = set()
    child_labels = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            child_id = parts[0].strip()
            child_label_raw = parts[1].strip()
            child_label = child_label_raw if child_label_raw else id_to_label.get(child_id, "")
            parent_label = parts[2].strip()
            entries.append((child_id, child_label, parent_label))
            if parent_label:
                parent_labels.add(parent_label)
            if child_label:
                child_labels.add(child_label)
    return entries, parent_labels, child_labels


def main() -> None:
    if len(sys.argv) not in {3, 4}:
        print(
            f"Usage: {sys.argv[0]} <terms-file> <taxo-file> [output-prefix]",
            file=sys.stderr,
        )
        sys.exit(1)

    terms_path = sys.argv[1]
    taxo_path = sys.argv[2]
    output_prefix = (
        sys.argv[3]
        if len(sys.argv) == 4
        else os.path.splitext(terms_path)[0] + "_cleaned"
    )

    terms_entries, id_to_label, label_to_ids = parse_terms(terms_path)
    taxo_entries, parent_labels, child_labels = parse_taxo(taxo_path, id_to_label)
    term_index_map = {term_id: idx for idx, (term_id, _) in enumerate(terms_entries)}

    def register_term(term_id: str, label: str) -> None:
        if not term_id or not label:
            return
        old_label = id_to_label.get(term_id)
        if old_label == label:
            idx = term_index_map.get(term_id)
            if idx is None:
                term_index_map[term_id] = len(terms_entries)
                terms_entries.append((term_id, label))
            else:
                terms_entries[idx] = (term_id, label)
            label_to_ids[label].add(term_id)
            return
        if old_label:
            ids_set = label_to_ids.get(old_label)
            if ids_set:
                ids_set.discard(term_id)
                if not ids_set:
                    label_to_ids.pop(old_label, None)
        id_to_label[term_id] = label
        label_to_ids[label].add(term_id)
        idx = term_index_map.get(term_id)
        if idx is not None:
            terms_entries[idx] = (term_id, label)
        else:
            term_index_map[term_id] = len(terms_entries)
            terms_entries.append((term_id, label))

    for child_id, child_label, _ in taxo_entries:
        if child_id and child_label:
            register_term(child_id, child_label)

    def next_term_id_generator():
        numeric_ids = [int(term_id) for term_id, _ in terms_entries if term_id.isdigit()]
        start = max(numeric_ids) + 1 if numeric_ids else len(terms_entries)
        current = start
        while True:
            yield str(current)
            current += 1

    new_id_gen = next_term_id_generator()

    def ensure_label_has_id(label: str) -> None:
        if label in label_to_ids and label_to_ids[label]:
            return
        term_id = next(new_id_gen)
        register_term(term_id, label)

    root_label = "food"
    ensure_label_has_id(root_label)

    children_by_label = defaultdict(set)
    invalid_candidates = set()

    for child_id, child_label, parent_label in taxo_entries:
        child_id = child_id.strip()
        child_label = child_label.strip()
        parent_label = parent_label.strip()

        if parent_label:
            children_by_label[parent_label].add(child_label)

        if not child_id or child_id not in id_to_label:
            if child_label:
                invalid_candidates.add(child_label)
        if not child_label or child_label not in label_to_ids:
            if child_label:
                invalid_candidates.add(child_label)
        if not parent_label or parent_label not in label_to_ids:
            if parent_label:
                invalid_candidates.add(parent_label)

    invalid_labels = set()

    def mark_invalid_label(start_label: str) -> None:
        stack = [start_label]
        while stack:
            current = stack.pop()
            if not current or current in invalid_labels:
                continue
            invalid_labels.add(current)
            for child in children_by_label.get(current, []):
                stack.append(child)

    for label in invalid_candidates:
        mark_invalid_label(label)

    if invalid_labels:
        print("Removed labels with missing IDs or names (and their subtrees):")
        for label in sorted(invalid_labels):
            print(f"  {label}")
        print()
        for label in invalid_labels:
            label_to_ids.pop(label, None)

    filtered_taxo_entries = []
    for child_id, child_label, parent_label in taxo_entries:
        child_id = child_id.strip()
        child_label = child_label.strip()
        parent_label = parent_label.strip()
        if (
            not child_id
            or not child_label
            or not parent_label
            or child_label in invalid_labels
            or parent_label in invalid_labels
        ):
            continue
        filtered_taxo_entries.append((child_id, child_label, parent_label))

    valid_labels = set(label_to_ids.keys())

    fully_filtered_taxo = []
    for child_id, child_label, parent_label in filtered_taxo_entries:
        if child_label not in valid_labels or parent_label not in valid_labels:
            continue
        fully_filtered_taxo.append((child_id, child_label, parent_label))

    taxo_entries = fully_filtered_taxo
    parent_labels = {parent for _, _, parent in taxo_entries}
    child_labels = {child_label for _, child_label, _ in taxo_entries}
    labels_with_children = set(parent_labels)
    existing_root_edges = sum(1 for _, _, parent in taxo_entries if parent == root_label)

    all_labels = set(label_to_ids.keys()) | parent_labels | child_labels
    labels_with_parents = set(child_labels)
    labels_without_parents = (all_labels - labels_with_parents) - {root_label}
    root_children = sorted(
        label for label in labels_without_parents if label in labels_with_children
    )
    isolated_labels = sorted(
        label for label in labels_without_parents if label not in labels_with_children
    )

    root_attachments = []
    for label in root_children:
        for term_id in label_to_ids.get(label, []):
            taxo_entries.append((term_id, label, root_label))
            root_attachments.append((term_id, label))
    if root_children:
        print("Root-attached labels:")
        total_attached_terms = len(root_attachments)
        for label in root_children:
            term_ids = sorted(label_to_ids.get(label, []))
            term_count_note = f" ({len(term_ids)} term IDs)" if len(term_ids) != 1 else " (1 term ID)"
            print(f"  {label}{term_count_note}")

        print("Newly attached term IDs:")
        for term_id, label in root_attachments:
            print(f"  {term_id}\t{label}")

        total_root_edges_after = existing_root_edges + total_attached_terms
        print(
            f"Total newly attached term IDs: {total_attached_terms}\n"
            f"Root edges before attaching: {existing_root_edges}\n"
            f"Root edges after attaching: {total_root_edges_after}\n"
        )

    if isolated_labels:
        print("Removed isolated labels (no parent/children):")
        for label in isolated_labels:
            print(f"  {label}")
        print()

    aggregated = {}
    for label, ids in label_to_ids.items():
        data = aggregated.setdefault(
            label,
            {
                "concept_ids": set(),
                "parent_labels": set(),
                "child_labels": set(),
            },
        )
        data["concept_ids"].update(ids)

    for child_id, child_label, parent_label in taxo_entries:
        child_entry = aggregated.setdefault(
            child_label,
            {
                "concept_ids": set(),
                "parent_labels": set(),
                "child_labels": set(),
            },
        )
        parent_entry = aggregated.setdefault(
            parent_label,
            {
                "concept_ids": set(),
                "parent_labels": set(),
                "child_labels": set(),
            },
        )
        child_entry["parent_labels"].add(parent_label)
        parent_entry["child_labels"].add(child_label)

    distribution = Counter()
    results = []
    for label, data in aggregated.items():
        parent_count = len(data["parent_labels"])
        distribution[parent_count] += 1
        if parent_count > 2:
            results.append((label, data))

    zero = distribution.get(0, 0)
    one = distribution.get(1, 0)
    two = distribution.get(2, 0)
    more_than_two = sum(count for key, count in distribution.items() if key > 2)

    print("Parent count distribution:")
    print(f"  0 parents: {zero}")
    print(f"  1 parent: {one}")
    print(f"  2 parents: {two}")
    print(f"  > 2 parents: {more_than_two}")

    total_nodes = len(aggregated)
    to_remove = set(isolated_labels)

    def mark_for_removal(start_label: str) -> None:
        stack = [start_label]
        while stack:
            current = stack.pop()
            if current in to_remove:
                continue
            to_remove.add(current)
            for child in aggregated.get(current, {}).get("child_labels", []):
                stack.append(child)

    for label, data in aggregated.items():
        if len(data["parent_labels"]) >= 2:
            mark_for_removal(label)

    remaining_nodes = total_nodes - len(to_remove)
    print(
        "\nRemaining nodes after removing concepts with >=2 parents and their subtrees:"
        f" {remaining_nodes}"
    )

    if to_remove:
        print("\nNodes with two or more parents (removed):")
        for label, data in results:
            concept_ids = ", ".join(sorted(data["concept_ids"]))
            parent_labels = ", ".join(sorted(data["parent_labels"]))
            print(f"{label} ({concept_ids}) -> parents: {parent_labels}")

    kept_labels = set(aggregated) - to_remove
    kept_ids = {
        term_id
        for label, data in aggregated.items()
        if label in kept_labels
        for term_id in data["concept_ids"]
    }

    terms_output = output_prefix + ".terms"
    taxo_output = output_prefix + ".taxo"

    with open(terms_output, "w", encoding="utf-8") as handle:
        for term_id, label in terms_entries:
            if term_id in kept_ids:
                handle.write(f"{term_id}\t{label}\n")

    with open(taxo_output, "w", encoding="utf-8") as handle:
        for child_id, child_label, parent_label in taxo_entries:
            if child_id in kept_ids and parent_label in kept_labels:
                handle.write(f"{child_id}\t{child_label}\t{parent_label}\n")

    print(f"\nCleaned files written to {terms_output} and {taxo_output}")


if __name__ == "__main__":
    main()
