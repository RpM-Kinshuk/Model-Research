import json
from collections import defaultdict

INPUT_FILE = "atlas_graph.json"

def analyze_base_model_diversity(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    unknown_nodes = []
    base_model_groups = defaultdict(list)

    for node in data["nodes"]:
        attrs = node.get("attributes", {})
        base_model = attrs.get("base_model_clean", "unknown")

        if not base_model or base_model == "unknown":
            unknown_nodes.append(node["id"])
        else:
            base_model_groups[base_model].append(node["id"])

    num_unknown = len(unknown_nodes)
    num_known_unique = len(base_model_groups)

    diversity_score = num_unknown + num_known_unique

    print("=== Base Model Diversity Report ===")
    print(f"Total nodes: {len(data['nodes'])}")
    print(f"Unknown base model nodes: {num_unknown}")
    print(f"Unique known base models: {num_known_unique}")
    print(f"👉 Diversity score: {diversity_score}")
    print()

    print("Top known base models by node count:")
    for bm, nodes in sorted(
        base_model_groups.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:10]:
        print(f"  {bm}: {len(nodes)} nodes")

    return {
        "total_nodes": len(data["nodes"]),
        "unknown_nodes": num_unknown,
        "unique_known_base_models": num_known_unique,
        "diversity_score": diversity_score,
        "base_model_distribution": {
            bm: len(nodes) for bm, nodes in base_model_groups.items()
        }
    }

if __name__ == "__main__":
    analyze_base_model_diversity(INPUT_FILE)
