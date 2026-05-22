import json
import argparse
import os
import random

import networkx as nx
from tqdm import tqdm

random.seed(42)

def generate_data(nodes, edge_index, time):
    G = nx.DiGraph()
    G.add_nodes_from(nodes.keys())
    G.add_edges_from(edge_index)

    user_prompt =f"""You are provided with a directed graph consisting of multiple nodes, each associated with a text. The connections between nodes are defined by the given edges, as detailed below:
**Nodes**:
{nodes}
**Edges (each pair [source, target] represents a directed connection from the source node to the target node)**:
{edge_index}.
"""

    dataset = []

    # 1. Degree-based Prediction
    node = random.choice(list(nodes.keys()))

    choice = random.choice(["indeg", "outdeg", "avgdeg"])
    if choice == "indeg":
        indeg = G.in_degree(node)
        q = f"What is the in-degree of node {node}?"
        a = f"{indeg}"
    elif choice == "outdeg":
        outdeg = G.out_degree(node)
        q = f"What is the out-degree of node {node}?"
        a = f"{outdeg}"
    else:
        avgdeg = sum(dict(G.degree()).values()) / G.number_of_nodes()
        q = "What is the average node degree of the graph? Keep the answer to two decimal places."
        a = f"{avgdeg:.2f}"
    q = user_prompt + q + " Begin by documenting your analysis and conclude with your final answer enclosed in <answer> </answer> tags."
    dataset.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], 'task_type': 'DBP'})

    # 2. Directed Neighbor Extraction
    node = random.choice(list(nodes.keys()))
    in_neigh = sorted(G.predecessors(node))
    out_neigh = sorted(G.successors(node))
    choice = random.choice(['in','out'])
    if choice == "in":
        q = (f"List the predecessors of node {node}. "
             f"Answer format: [node_id, node_id, ...] sorted by node id."
             "If none exist, answer an empty list [].")
        a = f"{in_neigh}"
    else:
        q = (f"List the successors of node {node}. "
             f"Answer format: [node_id, node_id, ...] sorted by node id."
             "If none exist, answer an empty list [].")
        a = f"{out_neigh}"
    q = user_prompt + q + " Begin by documenting your analysis and conclude with your final answer enclosed in <answer> </answer> tags."
    dataset.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], 'task_type': 'DNE'})

    # 3. Node Feature Prediction
    node = random.choice(list(nodes.keys()))
    q = f"What is the content of node {node}?"
    q = user_prompt + q + " Begin by documenting your analysis and conclude with your final answer enclosed in <answer> </answer> tags."
    a = f"{nodes[node]}"
    dataset.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], 'task_type': 'NFP'})

    # 4. Subgraph Reachability & Path Length
    src, tgt = random.sample(list(nodes.keys()), 2)
    try:
        length = nx.shortest_path_length(G, src, tgt)
        choice = random.choice(["reach", "length"])
        if choice == "reach":
            q = f"Can node {src} reach node {tgt}?  Answer with 'Yes' if reachable, otherwise 'No'."
            a = f"Yes"
        else:
            q = f"What is the shortest path length from node {src} to node {tgt}?"
            a = f"{length}"
    except nx.NetworkXNoPath:
        q = f"Can node {src} reach node {tgt}?  Answer with 'Yes' if reachable, otherwise 'No'."
        a = f"No"
    q = user_prompt + q + " Begin by documenting your analysis and conclude with your final answer enclosed in <answer> </answer> tags."
    dataset.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}],  'task_type': 'REACH'})

    # 5. Key Node Identification
    if time == 1 or time == 0:
        if time == 0:
            q = ("Identify the list of source nodes (in-degree=0). "
                 "Answer format: [node_id, node_id, ...] sorted by node id. "
                 "If none exist, answer an empty list [].")
            source_nodes = sorted([n for n in G.nodes if G.in_degree(n) == 0])
            a = f"{source_nodes}"
        elif time == 1:
            q = ("Identify the list of sink nodes (out-degree=0). "
                 "Answer format: [node_id, node_id, ...] sorted by node id. "
                 "If none exist, answer an empty list [].")
            sink_nodes = sorted([n for n in G.nodes if G.out_degree(n) == 0])
            a = f"{sink_nodes}"
        q = user_prompt + q + " Begin by documenting your analysis and conclude with your final answer enclosed in <answer> </answer> tags."
        dataset.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], 'task_type': 'KNI'})

    # 6. Topological Sorting
    if time == 0:
        q = ("Perform a topological sort of the graph. "
             "If multiple valid orders exist, prefer the one where nodes with smaller IDs come earlier. "
             "Answer format: [node_id, node_id, ...]. "
             "If the graph contains a cycle and topological sorting is not possible, answer 'No valid topological order'.")
        try:
            topo_order = list(nx.lexicographical_topological_sort(G))
            a = f"{topo_order}"
        except nx.NetworkXUnfeasible:
            a = "No valid topological order"

        q = user_prompt + q + " Begin by documenting your analysis and conclude with your final answer enclosed in <answer> </answer> tags."
        dataset.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], 'task_type': 'TSORT'})

    return dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()

    # 随机选择200个workflow用于生成测试集， 其余workflow用于生成训练集
    sub_datasets = ['Coding-AF','Coding-GD','Math-AF','Math-GD','Reason-AF','Reason-GD']

    data_dir = args.data_path

    pretrain_output_file = os.path.join(data_dir, "prefinetuning.jsonl")
    pretrain_test_output_file = os.path.join(data_dir, "prefinetuning_test.jsonl")

    all_workflow = []
    for sub_dataset in sub_datasets:
        train_file = os.path.join(data_dir, sub_dataset, "train.jsonl")

        # get all workflows
        used_wk_id = set()
        with open(train_file, "r", encoding="utf-8") as fin:
            for idx, line in enumerate(tqdm(fin, desc=f"Collecting workflows from {sub_dataset}")):  # 遍历每一行
                workflow_jsonl = json.loads(line.strip())
                if workflow_jsonl['workflow_id'] not in used_wk_id:
                    used_wk_id.add(workflow_jsonl['workflow_id'])

                    nodes = workflow_jsonl["nodes"]
                    edge_index = workflow_jsonl["edge_index"]
                    nodes = {int(k): v for k, v in nodes.items()}
                    all_workflow.append({"nodes": nodes, "edge_index": edge_index,"sub_dataset": sub_dataset})
    random.shuffle(all_workflow)

    # gen dataset
    id = 1
    with open(pretrain_output_file, "w", encoding="utf-8") as fout:
        for w in all_workflow[:-200]:
            nodes = w["nodes"]
            edge_index = w["edge_index"]
            sub_dataset = w["sub_dataset"]
            for time in range(3):  # each workflow generate three times
                dataset = generate_data(nodes, edge_index,time)
                for d in dataset:
                    d["id"] = f"{id}"
                    d['sub_dataset'] = sub_dataset
                    id += 1
                    fout.write(json.dumps(d, ensure_ascii=False) + "\n")

    # gen dataset
    id = 1
    with open(pretrain_test_output_file, "w", encoding="utf-8") as fout:
        for w in all_workflow[-200:]:
            nodes = w["nodes"]
            edge_index = w["edge_index"]
            sub_dataset = w["sub_dataset"]
            for time in range(3):  # each workflow generate three times
                dataset = generate_data(nodes, edge_index,time)
                for d in dataset:
                    d["id"] = f"{id}"
                    d['sub_dataset'] = sub_dataset
                    id += 1
                    fout.write(json.dumps(d, ensure_ascii=False) + "\n")
