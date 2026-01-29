import argparse
import os
import json
import random
import math
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict, deque


class TaxonomyExtractor:
    def __init__(
        self,
        *,
        source_type,
        xml_file=None,
        terms_file=None,
        taxo_file=None,
        google_xls_file=None,
        virtual_root_id="TAXO_ROOT",
        virtual_root_name="Taxonomy Root",
        domain="Generic",
        explicit_root_labels=None,
    ):
        """通用分类体系提取器，支持XML和terms/taxo输入"""

        self.source_type = source_type
        self.xml_file = xml_file
        self.terms_file = terms_file
        self.taxo_file = taxo_file
        self.google_xls_file = google_xls_file
        self.domain = domain
        self.parent_to_children = defaultdict(list)
        self.child_to_parent = {}
        self.all_nodes = set()
        self.root_nodes = set()
        self.node_id_to_name = {}
        self.virtual_root = virtual_root_id
        self.virtual_root_name = virtual_root_name
        self.explicit_root_labels = list(explicit_root_labels or [])

        if source_type == "xml":
            if not xml_file:
                raise ValueError("xml_file is required when source_type is 'xml'")
            self._load_from_xml(xml_file)
        elif source_type == "taxo":
            if not (terms_file and taxo_file):
                raise ValueError(
                    "terms_file and taxo_file are required when source_type is 'taxo'"
                )
            self._load_from_terms_taxo(terms_file, taxo_file, self.explicit_root_labels)
        elif source_type == "google_xls":
            if not google_xls_file:
                raise ValueError(
                    "google_xls_file is required when source_type is 'google_xls'"
                )
            self._load_from_google_xls(google_xls_file)
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")

    def _load_from_xml(self, xml_file):
        """从XML文件加载CCS/ACM分类体系"""
        print(f"Loading CCS taxonomy from {xml_file}...")
        
        # 解析XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 定义命名空间
        namespaces = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'skos': 'http://www.w3.org/2004/02/skos/core#'
        }
        
        # 收集概念体系信息
        concept_scheme = root.find('.//skos:ConceptScheme', namespaces)
        scheme_id = self.virtual_root
        scheme_label = self.virtual_root_name
        if concept_scheme is not None:
            scheme_id = concept_scheme.get(
                '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about',
                scheme_id,
            )
            label_element = concept_scheme.find('skos:prefLabel', namespaces)
            if label_element is not None and label_element.text:
                scheme_label = label_element.text
        
        # 遍历所有概念
        for concept in root.findall('.//skos:Concept', namespaces):
            concept_id = concept.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            
            if not concept_id:
                continue
            
            self.all_nodes.add(concept_id)
            
            # 获取概念名称
            pref_label = concept.find('skos:prefLabel', namespaces)
            if pref_label is not None and pref_label.text:
                self.node_id_to_name[concept_id] = pref_label.text
            else:
                self.node_id_to_name[concept_id] = concept_id
            
            # 获取narrower关系 (子节点)
            for narrower in concept.findall('skos:narrower', namespaces):
                child_id = narrower.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                if child_id:
                    self.parent_to_children[concept_id].append(child_id)
                    self.child_to_parent[child_id] = concept_id
        
        # 找出所有没有父节点的节点作为原始根节点
        original_roots = self.all_nodes - set(self.child_to_parent.keys())

        if not original_roots:
            raise ValueError("No root nodes detected in XML taxonomy")

        self.virtual_root = scheme_id
        self.virtual_root_name = scheme_label
        self.all_nodes.add(self.virtual_root)
        self.node_id_to_name[self.virtual_root] = scheme_label

        for root_node in original_roots:
            self.parent_to_children[self.virtual_root].append(root_node)
            self.child_to_parent[root_node] = self.virtual_root

        self.root_nodes = {self.virtual_root}

        print(f"Loaded {len(self.all_nodes)} nodes (including concept scheme root)")
        print(f"Top-level components: {len(original_roots)}")

    def _load_from_terms_taxo(self, terms_file, taxo_file, explicit_root_labels=None):
        """从terms/taxo文本文件加载分类体系"""
        print(f"Loading taxonomy from {terms_file} and {taxo_file}...")

        label_to_id = {}
        explicit_root_label_list = list(explicit_root_labels or [])

        with open(terms_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                term_id, label = parts[0].strip(), parts[1].strip()
                self.node_id_to_name[term_id] = label
                self.all_nodes.add(term_id)
                label_to_id.setdefault(label, term_id)

        generated_counter = 0

        def ensure_label_id(label):
            nonlocal generated_counter
            if label in label_to_id:
                return label_to_id[label]
            generated_counter += 1
            new_id = f"AUTO_{generated_counter}"
            label_to_id[label] = new_id
            self.node_id_to_name[new_id] = label
            self.all_nodes.add(new_id)
            return new_id

        ordered_forced_roots = [ensure_label_id(label) for label in explicit_root_label_list]
        forced_root_ids = set(ordered_forced_roots)
        target_root_id = ordered_forced_roots[0] if ordered_forced_roots else None

        def detach_from_parent(node_id):
            parent_id = self.child_to_parent.pop(node_id, None)
            if parent_id is not None:
                children = self.parent_to_children.get(parent_id, [])
                self.parent_to_children[parent_id] = [
                    child for child in children if child != node_id
                ]
            return parent_id

        with open(taxo_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                child_id = parts[0].strip()
                parent_label = parts[2].strip()
                parent_id = ensure_label_id(parent_label)
                if child_id not in self.node_id_to_name:
                    self.node_id_to_name[child_id] = parts[1].strip() if len(parts) > 1 else child_id
                    self.all_nodes.add(child_id)
                self.parent_to_children[parent_id].append(child_id)
                self.child_to_parent[child_id] = parent_id

        for node_id in forced_root_ids:
            detach_from_parent(node_id)

        def detect_cycle_break_nodes():
            visited = set()
            visiting = set()
            break_nodes = set()

            def dfs(node_id):
                if node_id in break_nodes:
                    return
                if node_id in visiting:
                    break_nodes.add(node_id)
                    return
                if node_id in visited:
                    return
                visiting.add(node_id)
                parent_id = self.child_to_parent.get(node_id)
                if parent_id is not None:
                    dfs(parent_id)
                visiting.remove(node_id)
                visited.add(node_id)

            for node in list(self.all_nodes):
                dfs(node)

            return break_nodes

        cycle_break_nodes = detect_cycle_break_nodes()
        if cycle_break_nodes:
            for node_id in cycle_break_nodes:
                detach_from_parent(node_id)
            print(
                f"Detected {len(cycle_break_nodes)} cycle break nodes in taxonomy input"
            )

        original_roots = self.all_nodes - set(self.child_to_parent.keys())

        if not original_roots:
            raise ValueError("No root nodes detected in taxonomy input")

        if target_root_id is not None:
            if target_root_id not in self.all_nodes:
                self.all_nodes.add(target_root_id)
            for node_id in list(original_roots):
                if node_id == target_root_id:
                    continue
                self.parent_to_children[target_root_id].append(node_id)
                self.child_to_parent[node_id] = target_root_id
            actual_root = target_root_id
        elif len(original_roots) == 1:
            actual_root = next(iter(original_roots))
        else:
            raise ValueError(
                "Multiple root nodes detected. Please provide --root-labels to specify the root."
            )

        self.virtual_root = actual_root
        self.virtual_root_name = self.node_id_to_name.get(actual_root, actual_root)
        self.root_nodes = {actual_root}

        print(f"Loaded {len(self.all_nodes)} nodes")
        print(f"Root node: {self.node_id_to_name.get(actual_root, actual_root)}")

    @staticmethod
    def _normalize_google_taxonomy_id(raw_value):
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            value = raw_value.strip()
            return value or None
        if isinstance(raw_value, int):
            return str(raw_value)
        if isinstance(raw_value, float):
            if math.isnan(raw_value):
                return None
            if raw_value.is_integer():
                return str(int(raw_value))
            return str(raw_value)
        value = str(raw_value).strip()
        return value or None

    def _load_from_google_xls(self, xls_file):
        """从Google product taxonomy的XLS文件加载分类体系（修复递归连接问题）"""
        print(f"Loading Google taxonomy from {xls_file}...")

        try:
            import xlrd
        except ImportError as exc:
            raise ImportError(
                "google_xls input requires the 'xlrd' package. Install it via 'pip install xlrd'."
            ) from exc

        workbook = xlrd.open_workbook(xls_file)
        sheet = workbook.sheet_by_index(0)

        column_samples = []
        sample_rows = min(sheet.nrows, 200)
        for col_idx in range(sheet.ncols):
            numbers = []
            texts = []
            alpha_texts = []
            for row_idx in range(sample_rows):
                value = sheet.cell_value(row_idx, col_idx)
                if value is None or value == "":
                    continue
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped:
                        texts.append(stripped)
                        if any(ch.isalpha() for ch in stripped):
                            alpha_texts.append(stripped)
                elif isinstance(value, (int, float)):
                    if math.isnan(value):
                        continue
                    numbers.append(value)
                else:
                    text_value = str(value).strip()
                    if text_value:
                        texts.append(text_value)
                        if any(ch.isalpha() for ch in text_value):
                            alpha_texts.append(text_value)
            column_samples.append(
                {"numbers": numbers, "texts": texts, "alpha_texts": alpha_texts}
            )

        id_candidates = []
        for col_idx, info in enumerate(column_samples):
            numeric_values = [abs(value) for value in info["numbers"]]
            if not numeric_values:
                continue
            alpha_text_count = len(info["alpha_texts"])
            if alpha_text_count > len(numeric_values) and alpha_text_count >= 5:
                continue
            median_value = statistics.median(numeric_values)
            id_candidates.append((median_value, col_idx))

        id_column = None
        if id_candidates:
            id_column = max(id_candidates, key=lambda item: item[0])[1]

        text_columns = []
        for idx, info in enumerate(column_samples):
            alpha_text_count = len(info["alpha_texts"])
            if alpha_text_count == 0:
                continue
            if id_column is not None and idx == id_column:
                continue
            text_columns.append(idx)

        if not text_columns:
            raise ValueError("Unable to detect category text columns in Google taxonomy XLS")

        path_to_id = {}
        path_to_name = {}
        entry_count = 0

        for row_idx in range(sheet.nrows):
            if id_column is not None:
                raw_id = sheet.cell_value(row_idx, id_column)
            else:
                raw_id = None
            normalized_id = self._normalize_google_taxonomy_id(raw_id)
            if normalized_id and not normalized_id.isdigit():
                normalized_id = None

            segment_values = []
            for col_idx in text_columns:
                cell_value = sheet.cell_value(row_idx, col_idx)
                if cell_value is None:
                    continue
                if isinstance(cell_value, str):
                    text = cell_value.strip()
                else:
                    text = str(cell_value).strip()
                if not text:
                    continue
                segment_values.append(text)

            if not segment_values:
                continue

            if len(segment_values) == 1 and ">" in segment_values[0]:
                segments = [
                    segment.strip()
                    for segment in segment_values[0].split(">")
                    if segment.strip()
                ]
            else:
                segments = [seg.strip() for seg in segment_values if seg.strip()]

            if not segments:
                continue
            if segments[0].lower().startswith("google product taxonomy"):
                continue

            path_key = tuple(segments)
            node_name = segments[-1]

            if path_key not in path_to_name:
                path_to_name[path_key] = node_name
                entry_count += 1

            if normalized_id:
                path_to_id[path_key] = normalized_id
                self.node_id_to_name[normalized_id] = node_name
                self.all_nodes.add(normalized_id)

        if not path_to_id:
            if not path_to_name:
                raise ValueError("No taxonomy entries detected in Google XLS file")

        print(f"Loaded {entry_count} explicit entries from XLS")

        synthetic_counter = 0

        def ensure_path_recursive(current_path_key):
            nonlocal synthetic_counter
            if not current_path_key:
                return None
            existing_id = path_to_id.get(current_path_key)
            if existing_id:
                return existing_id
            synthetic_counter += 1
            synthetic_id = f"GOOGLE_SYN_{synthetic_counter}"
            node_name = path_to_name.get(current_path_key, current_path_key[-1])
            path_to_id[current_path_key] = synthetic_id
            self.node_id_to_name[synthetic_id] = node_name
            self.all_nodes.add(synthetic_id)

            parent_key = current_path_key[:-1]
            if parent_key:
                parent_id = ensure_path_recursive(parent_key)
                if parent_id:
                    if synthetic_id not in self.parent_to_children[parent_id]:
                        self.parent_to_children[parent_id].append(synthetic_id)
                    self.child_to_parent[synthetic_id] = parent_id

            return synthetic_id

        sorted_paths = sorted(path_to_name.keys(), key=lambda key: (len(key), key))
        for path_key in sorted_paths:
            node_id = ensure_path_recursive(path_key)
            if len(path_key) <= 1:
                continue
            parent_key = path_key[:-1]
            parent_id = ensure_path_recursive(parent_key)
            if parent_id:
                if node_id not in self.parent_to_children[parent_id]:
                    self.parent_to_children[parent_id].append(node_id)
                self.child_to_parent[node_id] = parent_id

        unattached_roots = sorted(self.all_nodes - set(self.child_to_parent.keys()))

        if self.virtual_root not in self.all_nodes:
            self.all_nodes.add(self.virtual_root)
        self.node_id_to_name[self.virtual_root] = self.virtual_root_name

        for root_id in unattached_roots:
            if root_id == self.virtual_root:
                continue
            if root_id not in self.parent_to_children[self.virtual_root]:
                self.parent_to_children[self.virtual_root].append(root_id)
            self.child_to_parent[root_id] = self.virtual_root

        self.root_nodes = {self.virtual_root}

        print(
            f"Structure built. Top-level components: {len(self.parent_to_children[self.virtual_root])}"
        )
    
    def get_descendants(self, start_nodes):
        """返回start_nodes集合及其所有下游子节点"""
        if isinstance(start_nodes, str):
            nodes = [start_nodes]
        else:
            nodes = list(start_nodes)
        
        descendants = set()
        queue = deque(nodes)
        
        while queue:
            node = queue.popleft()
            if node in descendants:
                continue
            descendants.add(node)
            for child in self.parent_to_children.get(node, []):
                queue.append(child)
        
        return descendants
    
    def extract_fewshot_pool(self, target_size=1000, seed=42):
        """
        提取Few-shot资源池
        
        Args:
            target_size: 目标节点数量
            seed: 随机种子
            
        Returns:
            包含节点和边的字典
        """
        random.seed(seed)
        
        root = self.virtual_root
        selected_nodes = {root}
        root_children = list(self.parent_to_children.get(root, []))

        print(f"Starting from root: {self.node_id_to_name.get(root, root)}")
        print(f"Top-level child count: {len(root_children)}")

        if not root_children:
            print("WARNING: Root has no children; few-shot pool only contains the root")
        else:
            subtree_infos = []
            for child in root_children:
                nodes = self.get_descendants(child)
                size = len(nodes)
                diff = abs(size - target_size)
                subtree_infos.append((diff, -size, child, nodes))
            subtree_infos.sort()
            _, neg_size, best_child, best_nodes = subtree_infos[0]
            best_size = -neg_size
            selected_nodes.update(best_nodes)
            print(
                f"Selected subtree rooted at {self.node_id_to_name.get(best_child, best_child)} "
                f"with size {best_size} (target {target_size})"
            )

        print(f"Extracted few-shot pool with {len(selected_nodes)} nodes")
        
        # 提取相关的边
        edges = []
        for node in selected_nodes:
            if node in self.child_to_parent:
                parent = self.child_to_parent[node]
                if parent in selected_nodes:
                    edges.append((node, parent))
        
        return {
            'nodes': selected_nodes,
            'edges': edges,
            'root': root
        }
    
    def extract_subtaxonomy_from_pool(self, pool_data, target_size, seed):
        """
        从Few-shot资源池中提取指定大小的子分类体系
        
        Args:
            pool_data: Few-shot资源池数据
            target_size: 目标节点数量
            seed: 随机种子
            
        Returns:
            子分类体系数据
        """
        random.seed(seed)
        
        root = pool_data['root']
        pool_nodes = pool_data['nodes']
        
        # 构建资源池内的父子关系
        pool_parent_to_children = defaultdict(list)
        for child, parent in pool_data['edges']:
            pool_parent_to_children[parent].append(child)
        
        # 从根节点开始随机生长
        selected_nodes = {root}
        candidate_pool = list(pool_parent_to_children.get(root, []))
        
        while len(selected_nodes) < target_size and candidate_pool:
            chosen = random.choice(candidate_pool)
            candidate_pool.remove(chosen)
            
            selected_nodes.add(chosen)
            
            children = pool_parent_to_children.get(chosen, [])
            for child in children:
                if child not in selected_nodes and child not in candidate_pool:
                    candidate_pool.append(child)
        
        # 提取边
        edges = []
        for child, parent in pool_data['edges']:
            if child in selected_nodes and parent in selected_nodes:
                edges.append((child, parent))
        
        return {
            'nodes': list(selected_nodes),
            'edges': edges,
            'root': root
        }
    
    def extract_subtaxonomy_outside_pool(self, excluded_nodes, target_size, seed):
        """
        从Few-shot资源池之外提取子分类体系
        
        Args:
            excluded_nodes: 要排除的节点集合（few-shot pool中的节点）
            target_size: 目标节点数量
            seed: 随机种子
            
        Returns:
            子分类体系数据
        """
        random.seed(seed)
        
        excluded_nodes = set(excluded_nodes)
        excluded_nodes.discard(self.virtual_root)
        
        # 获取可用节点（排除few-shot pool中的节点）
        available_nodes = self.all_nodes - excluded_nodes
        
        # 从根节点开始，找到所有不在excluded_nodes中的子树
        root = self.virtual_root
        selected_nodes = {root}
        
        candidate_pool = []
        for child in self.parent_to_children.get(root, []):
            if child in excluded_nodes:
                continue
            candidate_pool.append(child)

        print(f"  Available nodes: {len(available_nodes)}")
        print(f"  Initial candidates: {len(candidate_pool)}")
        
        # 随机生长算法
        while len(selected_nodes) < target_size and candidate_pool:
            chosen = random.choice(candidate_pool)
            candidate_pool.remove(chosen)
            
            if chosen in excluded_nodes:
                continue
            
            selected_nodes.add(chosen)
            
            # 添加子节点到候选池
            children = self.parent_to_children.get(chosen, [])
            for child in children:
                if child not in excluded_nodes and child not in selected_nodes and child not in candidate_pool:
                    candidate_pool.append(child)
        
        # 提取边
        edges = []
        for node in selected_nodes:
            if node == root:
                continue
            if node in self.child_to_parent:
                parent = self.child_to_parent[node]
                if parent in selected_nodes:
                    edges.append((node, parent))
        
        print(f"  Extracted {len(selected_nodes)} nodes, {len(edges)} edges")
        
        return {
            'nodes': list(selected_nodes),
            'edges': edges,
            'root': root
        }
    
    def save_subtaxonomy(self, subtaxonomy_data, output_dir, taxonomy_id, extraction_method="random_growth"):
        """
        保存子分类体系到指定目录
        
        Args:
            subtaxonomy_data: 子分类体系数据
            output_dir: 输出目录
            taxonomy_id: 分类体系ID
            extraction_method: 提取方法
        """
        os.makedirs(output_dir, exist_ok=True)
        
        nodes = subtaxonomy_data['nodes']
        edges = subtaxonomy_data['edges']
        root = subtaxonomy_data['root']
        
        # 保存 entities.json（仅保留名称）
        entities = []
        for node in nodes:
            node_name = self.node_id_to_name.get(node, f"Entity_{node}")
            entities.append({"name": node_name})
        with open(os.path.join(output_dir, 'entities.json'), 'w', encoding='utf-8') as f:
            json.dump(entities, f, indent=2, ensure_ascii=False)

        # 保存 relationships.json
        relationships = [
            {
                "child_name": self.node_id_to_name.get(child, f"Entity_{child}"),
                "parent_name": self.node_id_to_name.get(parent, f"Entity_{parent}"),
                "relation_type": "IsA"
            }
            for child, parent in edges
        ]
        with open(os.path.join(output_dir, 'relationships.json'), 'w', encoding='utf-8') as f:
            json.dump(relationships, f, indent=2, ensure_ascii=False)
        
        # 保存 metadata.json
        metadata = {
            "taxonomy_id": taxonomy_id,
            "domain": self.domain,
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "root_node": root,
            "root_node_name": self.node_id_to_name.get(root, root),
            "extraction_method": extraction_method,
            "source_type": self.source_type,
        }
        if self.source_type == "xml":
            metadata["source_path"] = self.xml_file
        elif self.source_type == "taxo":
            metadata["source_path"] = {
                "terms_file": self.terms_file,
                "taxo_file": self.taxo_file,
            }
        elif self.source_type == "google_xls":
            metadata["source_path"] = self.google_xls_file
        else:
            metadata["source_path"] = None
        with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Saved taxonomy {taxonomy_id} to {output_dir}")
        print(f"  Nodes: {len(nodes)}, Edges: {len(edges)}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="从CCS/ACM XML或terms/taxo文件抽取few-shot和测试子分类体系"
    )
    parser.add_argument(
        "--input-type",
        choices=["xml", "taxo", "google_xls"],
        required=True,
        help="输入文件类型：xml表示SKOS格式，taxo表示<terms, taxo>文件，google_xls表示Google taxonomy XLS",
    )
    parser.add_argument("--xml-file", help="输入的SKOS XML文件路径，当input-type=xml时必需")
    parser.add_argument(
        "--terms-file",
        help="terms文件路径，当input-type=taxo时必需",
    )
    parser.add_argument(
        "--taxo-file",
        help="taxo文件路径，当input-type=taxo时必需",
    )
    parser.add_argument(
        "--google-xls-file",
        help="Google product taxonomy的XLS文件路径，当input-type=google_xls时必需",
    )
    parser.add_argument(
        "--root-labels",
        nargs="+",
        default=[],
        help="(taxo输入可选) 指定需要强制作为根节点的label",
    )
    parser.add_argument(
        "--domain",
        default="CCS",
        help="保存metadata时的domain描述",
    )
    parser.add_argument(
        "--virtual-root-id",
        default="CCS_ROOT",
        help="虚拟根节点ID",
    )
    parser.add_argument(
        "--virtual-root-name",
        default="Computing Classification System",
        help="虚拟根节点名称",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出目录，few-shot和test集都会写入此目录下",
    )
    parser.add_argument(
        "--taxonomy-prefix",
        default="taxo",
        help="生成taxonomy_id时使用的前缀",
    )
    parser.add_argument(
        "--fewshot-pool-size",
        type=int,
        required=True,
        help="few-shot资源池目标节点数量",
    )
    parser.add_argument(
        "--fewshot-sizes",
        type=int,
        nargs="+",
        required=True,
        help="从few-shot资源池中抽取的few-shot样本size列表",
    )
    parser.add_argument(
        "--fewshot-samples-per-size",
        type=int,
        default=1,
        help="每个few-shot size抽取多少个样本",
    )
    parser.add_argument(
        "--test-sizes",
        type=int,
        nargs="+",
        required=True,
        help="测试集需要的size列表",
    )
    parser.add_argument(
        "--test-samples-per-size",
        type=int,
        default=5,
        help="每个测试size抽取多少个样本",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子基数",
    )
    parser.add_argument(
        "--pool-output-name",
        default="fewshot_pool.json",
        help="few-shot资源池摘要文件名",
    )
    parser.add_argument(
        "--fewshot-output-subdir",
        default="fewshot_examples",
        help="few-shot样本输出子目录名",
    )
    parser.add_argument(
        "--test-output-subdir",
        default="test_sets",
        help="测试样本输出子目录名",
    )
    return parser


def parse_args():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.input_type == "xml" and not args.xml_file:
        parser.error("--xml-file 在 input-type=xml 时必需")
    if args.input_type == "taxo" and (not args.terms_file or not args.taxo_file):
        parser.error("--terms-file 与 --taxo-file 在 input-type=taxo 时必需")
    if args.input_type == "google_xls" and not args.google_xls_file:
        parser.error("--google-xls-file 在 input-type=google_xls 时必需")
    if args.fewshot_pool_size <= 0:
        parser.error("--fewshot-pool-size 必须为正数")
    for size in args.fewshot_sizes:
        if size <= 0:
            parser.error("few-shot size 必须为正数")
    for size in args.test_sizes:
        if size <= 0:
            parser.error("test size 必须为正数")
    if args.fewshot_samples_per_size <= 0 or args.test_samples_per_size <= 0:
        parser.error("samples-per-size 必须为正整数")

    return args


def main():
    args = parse_args()

    extractor = TaxonomyExtractor(
        source_type=args.input_type,
        xml_file=args.xml_file,
        terms_file=args.terms_file,
        taxo_file=args.taxo_file,
        google_xls_file=args.google_xls_file,
        virtual_root_id=args.virtual_root_id,
        virtual_root_name=args.virtual_root_name,
        domain=args.domain,
        explicit_root_labels=args.root_labels,
    )

    print("\n=== Taxonomy Statistics ===")
    print(f"Total nodes: {len(extractor.all_nodes)}")
    print(f"Root id: {extractor.virtual_root}")
    print(f"Root name: {extractor.node_id_to_name.get(extractor.virtual_root)}")
    print(
        f"Top-level children: {len(extractor.parent_to_children[extractor.virtual_root])}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    fewshot_dir = os.path.join(args.output_dir, args.fewshot_output_subdir)
    test_dir = os.path.join(args.output_dir, args.test_output_subdir)
    os.makedirs(fewshot_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("\n=== Building Few-shot Pool ===")
    pool_seed = args.seed
    pool_data = extractor.extract_fewshot_pool(
        target_size=args.fewshot_pool_size,
        seed=pool_seed,
    )
    pool_summary_path = os.path.join(args.output_dir, args.pool_output_name)
    pool_summary = {
        "seed": pool_seed,
        "target_size": args.fewshot_pool_size,
        "actual_size": len(pool_data["nodes"]),
        "num_edges": len(pool_data["edges"]),
        "nodes": sorted(pool_data["nodes"]),
        "edges": pool_data["edges"],
    }
    with open(pool_summary_path, "w", encoding="utf-8") as handle:
        json.dump(pool_summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved few-shot pool summary to {pool_summary_path}")

    print("\n=== Sampling Few-shot Examples ===")
    fewshot_records = []
    for size in args.fewshot_sizes:
        for sample_id in range(args.fewshot_samples_per_size):
            sample_seed = args.seed + size * 1000 + sample_id
            subtaxonomy = extractor.extract_subtaxonomy_from_pool(
                pool_data,
                target_size=size,
                seed=sample_seed,
            )
            actual_size = len(subtaxonomy["nodes"])
            if actual_size < size:
                print(
                    f"  WARNING: few-shot size {size} sample {sample_id} 只提取到 {actual_size} 个节点"
                )
            taxonomy_id = (
                f"{args.taxonomy_prefix}_fewshot_size{size}_sample{sample_id}"
            )
            output_dir = os.path.join(
                fewshot_dir,
                f"size_{size}",
                f"sample_{sample_id}",
            )
            extractor.save_subtaxonomy(
                subtaxonomy,
                output_dir,
                taxonomy_id=taxonomy_id,
                extraction_method="fewshot_from_pool",
            )
            fewshot_records.append(
                {
                    "size": size,
                    "sample_id": sample_id,
                    "actual_size": actual_size,
                    "output_dir": output_dir,
                }
            )

    print("\n=== Sampling Test Sets ===")
    test_records = []
    shared_pool_nodes = set(pool_data["nodes"])
    for size in args.test_sizes:
        for sample_id in range(args.test_samples_per_size):
            sample_seed = args.seed + 200000 + size * 1000 + sample_id
            current_exclusions = shared_pool_nodes
            subtaxonomy = extractor.extract_subtaxonomy_outside_pool(
                current_exclusions,
                target_size=size,
                seed=sample_seed,
            )
            actual_size = len(subtaxonomy["nodes"])
            if actual_size < size:
                print(
                    f"  WARNING: test size {size} sample {sample_id} 只提取到 {actual_size} 个节点"
                )
            taxonomy_id = f"{args.taxonomy_prefix}_test_size{size}_sample{sample_id}"
            output_dir = os.path.join(
                test_dir,
                f"size_{size}",
                f"sample_{sample_id}",
            )
            extractor.save_subtaxonomy(
                subtaxonomy,
                output_dir,
                taxonomy_id=taxonomy_id,
                extraction_method="test_from_outside_pool",
            )
            test_records.append(
                {
                    "size": size,
                    "sample_id": sample_id,
                    "actual_size": actual_size,
                    "output_dir": output_dir,
                }
            )

    print("\n=== Summary ===")
    print(f"Few-shot pool size: {len(pool_data['nodes'])}")
    print(f"Few-shot samples generated: {len(fewshot_records)}")
    print(f"Test samples generated: {len(test_records)}")
    print("Few-shot outputs:")
    for record in fewshot_records:
        print(
            f"  size={record['size']} sample={record['sample_id']} -> {record['output_dir']}"
        )
    print("Test outputs:")
    for record in test_records:
        print(
            f"  size={record['size']} sample={record['sample_id']} -> {record['output_dir']}"
        )


if __name__ == "__main__":
    main()
