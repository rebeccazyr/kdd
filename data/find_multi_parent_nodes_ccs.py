#!/usr/bin/env python3
import sys
import xml.etree.ElementTree as ET
from collections import Counter

SKOS_NS = "http://www.w3.org/2004/02/skos/core#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def main() -> None:
    if len(sys.argv) not in {2, 3}:
        print(f"Usage: {sys.argv[0]} <path-to-ccs-acm-xml> [output-xml]", file=sys.stderr)
        sys.exit(1)

    xml_path = sys.argv[1]
    output_path = (
        sys.argv[2]
        if len(sys.argv) == 3
        else xml_path.rsplit(".", 1)[0] + "_cleaned.xml"
    )
    tree = ET.parse(xml_path)
    root = tree.getroot()

    concepts = root.findall(f".//{{{SKOS_NS}}}Concept")
    concept_entries = []
    label_map = {}
    for index, concept in enumerate(concepts):
        about_raw = concept.attrib.get(f"{{{RDF_NS}}}about", "").strip()
        has_real_id = bool(about_raw)
        about = about_raw if has_real_id else f"<missing-id-{index}>"
        label_element = concept.find(f"{{{SKOS_NS}}}prefLabel")
        raw_label = label_element.text.strip() if label_element is not None and label_element.text else ""
        has_label = bool(raw_label)
        label = raw_label or about
        if has_real_id:
            label_map[about_raw] = raw_label or about_raw

        parents = [parent.attrib.get(f"{{{RDF_NS}}}resource", "").strip() for parent in concept.findall(f"{{{SKOS_NS}}}broader")]
        children = [child.attrib.get(f"{{{RDF_NS}}}resource", "").strip() for child in concept.findall(f"{{{SKOS_NS}}}narrower")]
        concept_entries.append({
            "id": about,
            "real_id": about_raw,
            "label": label,
            "parents": parents,
            "children": children,
            "has_label": has_label,
            "has_real_id": has_real_id,
            "element": concept,
        })

    children_map = {entry["id"]: entry["children"] for entry in concept_entries}
    invalid_concept_ids = set()

    def mark_invalid_id(start_id: str) -> None:
        stack = [start_id]
        while stack:
            current = stack.pop()
            if not current or current in invalid_concept_ids:
                continue
            invalid_concept_ids.add(current)
            for child_id in children_map.get(current, []):
                stack.append(child_id)

    for entry in concept_entries:
        if not entry["has_real_id"] or not entry["has_label"]:
            mark_invalid_id(entry["id"])

    if invalid_concept_ids:
        print("Removed concepts missing IDs or labels (and their subtrees):")
        for entry in concept_entries:
            if entry["id"] in invalid_concept_ids:
                display_id = entry["real_id"] or entry["id"]
                print(f"  {entry['label']} ({display_id})")
        print()

    valid_real_ids = {
        entry["real_id"]
        for entry in concept_entries
        if entry["has_real_id"] and entry["id"] not in invalid_concept_ids
    }

    aggregated = {}
    for entry in concept_entries:
        if entry["id"] in invalid_concept_ids:
            continue
        label = entry["label"]
        data = aggregated.setdefault(
            label,
            {
                "concept_ids": set(),
                "parent_ids": set(),
                "parent_labels": set(),
                "child_labels": set(),
            },
        )
        data["concept_ids"].add(entry["id"])

        for parent_id in entry["parents"]:
            parent_id = parent_id.strip()
            if not parent_id or parent_id not in valid_real_ids:
                continue
            data["parent_ids"].add(parent_id)
            data["parent_labels"].add(label_map.get(parent_id, parent_id))

        for child_id in entry["children"]:
            child_id = child_id.strip()
            if not child_id or child_id not in valid_real_ids:
                continue
            data["child_labels"].add(label_map.get(child_id, child_id))

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
    to_remove = set()

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

    def compute_layers() -> int:
        if not kept_labels:
            return 0

        kept_children = {
            label: [child for child in data["child_labels"] if child in kept_labels]
            for label, data in aggregated.items()
            if label in kept_labels
        }

        roots = [
            label
            for label in kept_labels
            if not {parent for parent in aggregated[label]["parent_labels"] if parent in kept_labels}
        ]
        if not roots:
            roots = list(kept_labels)

        max_depth = 0
        stack = [(root, 1) for root in roots]
        visited = set()

        while stack:
            current, depth = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if depth > max_depth:
                max_depth = depth
            for child in kept_children.get(current, []):
                stack.append((child, depth + 1))

        remaining = kept_labels - visited
        while remaining:
            extra = remaining.pop()
            stack = [(extra, 1)]
            while stack:
                current, depth = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                if depth > max_depth:
                    max_depth = depth
                for child in kept_children.get(current, []):
                    stack.append((child, depth + 1))
            remaining = kept_labels - visited

        return max_depth

    layers = compute_layers()
    print(f"\nRemaining taxonomy layers: {layers}")

    ids_to_remove = set()
    for entry in concept_entries:
        if entry["id"] in invalid_concept_ids and entry["real_id"]:
            ids_to_remove.add(entry["real_id"])
    for label in to_remove:
        ids_to_remove.update(aggregated.get(label, {}).get("concept_ids", set()))

    orphan_concept_elements = {
        entry["element"]
        for entry in concept_entries
        if entry["id"] in invalid_concept_ids and not entry["real_id"]
    }

    def prune_concepts(parent: ET.Element) -> None:
        for child in list(parent):
            prune_concepts(child)
            if child.tag != f"{{{SKOS_NS}}}Concept":
                continue
            about = child.attrib.get(f"{{{RDF_NS}}}about", "").strip()
            if about in ids_to_remove or child in orphan_concept_elements:
                parent.remove(child)

    prune_concepts(root)

    def remove_invalid_relations() -> None:
        if not ids_to_remove:
            return
        for concept in root.findall(f".//{{{SKOS_NS}}}Concept"):
            for relation_tag in (f"{{{SKOS_NS}}}narrower", f"{{{SKOS_NS}}}broader"):
                for relation in list(concept.findall(relation_tag)):
                    target = relation.attrib.get(f"{{{RDF_NS}}}resource", "").strip()
                    if target in ids_to_remove:
                        concept.remove(relation)

    remove_invalid_relations()

    for scheme in root.findall(f"{{{SKOS_NS}}}ConceptScheme"):
        for has_top in list(scheme.findall(f"{{{SKOS_NS}}}hasTopConcept")):
            resource = has_top.attrib.get(f"{{{RDF_NS}}}resource", "")
            if resource in ids_to_remove:
                scheme.remove(has_top)

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"\nCleaned XML written to {output_path}")


if __name__ == "__main__":
    main()
