import csv
import json
import networkx as nx
from collections import defaultdict
import random

# --- CONFIGURATION ---
INPUT_FILE = 'large_nlp_data.json'
OUTPUT_FILE = 'atlas_graph.json'
CSV_OUTPUT_FILE = 'atlas_metadata.csv'
TARGET_SIZE = 5000
LEAF_LIMIT = 3  # Max leaves to keep per parent (The "Model Atlas" heuristic)

# NEW: Property Filters
# Use this to downsample specific types of models BEFORE the main structure check.
PROPERTY_FILTERS = [
    {
        "attribute": "base_model_relation",  # The attribute key in "attributes"
        "value": "merge",                    # The value to target
        "keep_ratio": 1.0,                  # 0.0 = Remove all, 1.0 = Keep all, 0.2 = Keep 20%
        "description": "Keep 15% of merge models"
    },
    {
        "attribute": "base_model_relation",
        "value": "quantized",
        "keep_ratio": 0.1,
        "description": "Keep 10% of quantized models"
    }
    # You can add more filters here, e.g.:
    # { "attribute": "license", "value": "unknown", "keep_ratio": 0.5 }
]

# --- NEW CONFIGURATION: CSV MAPPING ---
# Format: "Output Column Name": "Source JSON Key"
# The script will look for the Source Key in the node root first (like 'id'),
# and then inside the 'attributes' dictionary.
COLUMN_MAPPING = {
    "full_name": "id",
    "source_model": "base_model_clean",
    "base_model_relation": "base_model_relation",
    # You can easily add more below:
    # "license_type": "license",
    # "download_count": "downloads",
    # "tags_list": "tags"
}


def load_and_build_graph(filepath):
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Add nodes with metadata
    for node in data['nodes']:
        # Parse downloads safely for sorting
        try:
            dl_str = node.get('attributes', {}).get('downloads', '0')
            downloads = float(dl_str)
        except (ValueError, TypeError):
            downloads = 0.0
            
        # Store attributes directly on the node
        attrs = node.get('attributes', {})
        G.add_node(node['id'], **node, _downloads=downloads, _attrs=attrs)

    # Add edges
    for edge in data['edges']:
        G.add_edge(edge['source'], edge['target'])
        
    print(f"Graph loaded: {G.number_of_nodes()} nodes.")
    return G, data


def apply_property_filters(G, filters):
    """
    Removes nodes from G based on property filters.
    """
    print("--- Applying Property Filters ---")
    nodes_to_remove = set()

    for rule in filters:
        attr_key = rule['attribute']
        target_val = rule['value']
        ratio = rule['keep_ratio']
        
        # 1. Identify candidates
        candidates = []
        for n, data in G.nodes(data=True):
            # Check if node has the attribute and it matches
            node_attrs = data.get('_attrs', {})
            if node_attrs.get(attr_key) == target_val:
                candidates.append(n)
        
        total_candidates = len(candidates)
        if total_candidates == 0:
            print(f"Filter '{attr_key}={target_val}': No matches found.")
            continue

        # 2. Determine how many to cut
        # We sort by downloads to ensure we keep the "best" ones if ratio > 0
        candidates.sort(key=lambda x: G.nodes[x].get('_downloads', 0), reverse=True)
        
        keep_count = int(total_candidates * ratio)
        remove_count = total_candidates - keep_count
        
        # The bottom chunk of the list gets removed
        # e.g. if we keep top 20%, we remove the bottom 80%
        removals = candidates[keep_count:]
        nodes_to_remove.update(removals)
        
        print(f"Filter '{attr_key}={target_val}': Found {total_candidates}. Keeping top {keep_count} ({ratio*100}%). Marking {remove_count} for removal.")

    # 3. Execute Removal
    if nodes_to_remove:
        before_count = G.number_of_nodes()
        G.remove_nodes_from(nodes_to_remove)
        print(f"Removed {len(nodes_to_remove)} nodes. Graph size: {before_count} -> {G.number_of_nodes()}")
    else:
        print("No nodes removed by filters.")
        
    return G


def hybrid_structural_sampling(G, target_n, leaf_limit):
    print("--- Starting Hybrid Structural Pruning ---")
    
    selected_nodes = set()
    
    # Recalculate degrees after filtering
    out_degrees = dict(G.out_degree())
    
    # 1. Separate Skeleton and Leaves
    skeleton_nodes = [n for n, d in out_degrees.items() if d > 0]
    leaf_nodes = [n for n, d in out_degrees.items() if d == 0]
    
    print(f"Skeleton nodes: {len(skeleton_nodes)} | Leaf nodes: {len(leaf_nodes)}")
    
    # 2. Keep Skeleton
    selected_nodes.update(skeleton_nodes)
    
    # 3. Prune Leaves (Top K per parent)
    parent_to_leaves = defaultdict(list)
    for leaf in leaf_nodes:
        parents = list(G.predecessors(leaf))
        if not parents:
            parent_to_leaves['__SINGLETONS__'].append(leaf)
        else:
            for p in parents:
                parent_to_leaves[p].append(leaf)

    leaves_to_keep = set()
    for parent, leaves in parent_to_leaves.items():
        if parent == '__SINGLETONS__': continue
            
        # Sort leaves by downloads
        leaves.sort(key=lambda n: G.nodes[n].get('_downloads', 0), reverse=True)
        leaves_to_keep.update(leaves[:leaf_limit])

    selected_nodes.update(leaves_to_keep)
    print(f"Nodes after Skeleton + Leaf pruning: {len(selected_nodes)}")
    
    # 4. Fill Budget (Singletons)
    if len(selected_nodes) < target_n:
        needed = target_n - len(selected_nodes)
        singletons = parent_to_leaves['__SINGLETONS__']
        singletons.sort(key=lambda n: G.nodes[n].get('_downloads', 0), reverse=True)
        selected_nodes.update(singletons[:needed])
        print(f"Filled {needed} slots with singletons.")

    # 5. Cap Budget (Priority trim)
    if len(selected_nodes) > target_n:
        print(f"Over budget ({len(selected_nodes)}). Trimming...")
        all_selected = list(selected_nodes)
        
        def priority_score(n):
            # 100 pts for Skeleton, 50 pts for Connected Leaf, 0 for Singleton
            # + normalized download score
            is_skeleton = G.out_degree(n) > 0
            is_singleton = G.degree(n) == 0
            dl = G.nodes[n].get('_downloads', 0)
            dl_score = dl / (dl + 1_000_000) 
            
            base = 100 if is_skeleton else (0 if is_singleton else 50)
            return base + dl_score

        all_selected.sort(key=priority_score, reverse=True)
        selected_nodes = set(all_selected[:target_n])

    return selected_nodes


def export_subgraph(G, original_data, selected_node_ids, output_file):
    print(f"--- Exporting to {output_file} ---")
    final_nodes = [n for n in original_data['nodes'] if n['id'] in selected_node_ids]
    final_edges = [e for e in original_data['edges'] if e['source'] in selected_node_ids and e['target'] in selected_node_ids]
    
    output_data = {"nodes": final_nodes, "edges": final_edges}
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved {len(final_nodes)} nodes and {len(final_edges)} edges.")


# --- NEW FUNCTION ---
def export_node_metadata(original_data, selected_ids, mapping, output_file):
    print(f"--- Exporting metadata for {len(selected_ids)} nodes to {output_file} ---")
    
    # 1. Create a lookup for the full node objects to speed up processing
    # (Converting list to dict for O(1) access)
    node_lookup = {node['id']: node for node in original_data['nodes']}
    
    rows = []
    
    # 2. Iterate through selected nodes and build rows based on mapping
    for node_id in selected_ids:
        if node_id not in node_lookup:
            continue
            
        node_data = node_lookup[node_id]
        node_attrs = node_data.get('attributes', {})
        
        row = {}
        for csv_col, json_key in mapping.items():
            # PRIORITY 1: Check root level (e.g., 'id', 'label', 'x', 'y')
            if json_key in node_data:
                row[csv_col] = node_data[json_key]
            # PRIORITY 2: Check attributes dictionary (e.g., 'base_model_clean', 'license')
            elif json_key in node_attrs:
                row[csv_col] = node_attrs[json_key]
            # PRIORITY 3: Not found
            else:
                row[csv_col] = "N/A" # or ""
        
        rows.append(row)

    # 3. Write to CSV
    if rows:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=mapping.keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Successfully wrote {len(rows)} rows to CSV.")
    else:
        print("No rows generated to write.")


# --- MAIN ---
if __name__ == "__main__":
    # 1. Load
    G, data = load_and_build_graph(INPUT_FILE)
    
    # 2. Apply Filtering 
    G = apply_property_filters(G, PROPERTY_FILTERS)
    
    # 3. Sample Structure
    selected_ids = hybrid_structural_sampling(G, G.number_of_nodes(), G.number_of_nodes())
    
    # 4. Export Graph JSON (for visualization tools)
    export_subgraph(G, data, selected_ids, OUTPUT_FILE)

    # 5. Export Metadata CSV (NEW STEP)
    export_node_metadata(data, selected_ids, COLUMN_MAPPING, CSV_OUTPUT_FILE)