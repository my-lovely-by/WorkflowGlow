import argparse
import json
import requests
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from utils import calculate_utility
import re

def call_vllm_api(prompts, api_url="http://localhost:8000/v1/chat/completions", model="Qwen3-30B-A3B-Instruct-2507", max_tokens=32, temperature=0.0):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": prompts,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()


def evaluate(jsonl_path, model_name, args, max_tokens=32, temperature=0.0,
             system_prompt="You are an agentic workflow evaluator.",
             api_url="http://localhost:8000/v1/chat/completions"):
    print(f"📄 Loading data from {jsonl_path} ...")
    data = [json.loads(line) for line in open(jsonl_path, "r", encoding="utf-8")]

    workflow_count_dict, ground_workflow_dict, predicted_workflow_dict = {}, {}, {}

    y_true, y_pred = [], []

    for item in tqdm(data, desc="Evaluating"):
        task, nodes, edge_index, label, wf_id = item["task"], item["nodes"], item["edge_index"], item["label"], item["workflow_id"]
        y_true.append(label)

        user_prompt = f"""You are provided with detailed information about an agentic workflow. This workflow consists of multiple nodes, each containing a prompt intended for an LLM to execute a specific step. You are also given the edges between nodes, which specify both the execution order and the flow of information between them (Please note that in addition to the text ptompt, each node can also access the output of tasks and their parent nodes. At the same time, each agent can call any appropriate tools such as code executors, mathematical calculators, inference engines, etc.). Your task is to determine if this workflow can output the correct results for the given task. You do not need to solve the task itself—focus solely on evaluating the workflow’s effectiveness. Below is the task and workflow information:
**Task description**: 
{task}
**Nodes (mapping unique IDs to agent prompts)**:
{nodes}
**Edges (each [source, target] represents a directed connection from the source node to the target node)**:
{edge_index}
Respond with 'Yes' or 'No' to indicate whether the workflow is suitable for completing the task. Begin by documenting your analysis and conclude with your final answer enclosed in <answer> </answer> tags."""


        prompt= [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

        result = call_vllm_api(prompt, api_url, model=model_name, max_tokens=max_tokens, temperature=temperature)

        content = result["choices"][0]["message"]["content"]

        content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
        text = content_match.group(1).strip() if content_match else content.strip()

        print(text)
        if "yes" in text and "no" not in text:
            pred = 1
        elif "no" in text and "yes" not in text:
            pred = 0
        else:
            pred = 0

        y_pred.append(pred)

        workflow_count_dict[wf_id] = workflow_count_dict.get(wf_id, 0) + 1
        ground_workflow_dict[wf_id] = ground_workflow_dict.get(wf_id, 0) + label
        predicted_workflow_dict[wf_id] = predicted_workflow_dict.get(wf_id, 0) + pred

    ground_avg = {k: ground_workflow_dict[k] / workflow_count_dict[k] for k in workflow_count_dict}
    predicted_avg = {k: predicted_workflow_dict[k] / workflow_count_dict[k] for k in workflow_count_dict}
    utility = calculate_utility(args, ground_avg, predicted_avg)
    return utility, y_true, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluate workflow suitability using vLLM API.")
    parser.add_argument("--model_name", type=str, default="llmmodel/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--data_path", type=str, default="FLORA/data/Coding-AF")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--api_url", type=str, default="http://localhost:7999/v1/chat/completions")
    args = parser.parse_args()

    utility, y_true, y_pred = evaluate(
        jsonl_path=os.path.join(args.data_path,'test.jsonl'),
        model_name=args.model_name,
        args=args,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        api_url=args.api_url
    )

    accuracy = round(accuracy_score(y_true, y_pred), 4)
    precision_1 = round(precision_score(y_true, y_pred, pos_label=1), 4)
    recall_1 = round(recall_score(y_true, y_pred, pos_label=1), 4)
    f1_1 = round(f1_score(y_true, y_pred, pos_label=1), 4)

    precision_0 = round(precision_score(y_true, y_pred, pos_label=0), 4)
    recall_0 = round(recall_score(y_true, y_pred, pos_label=0), 4)
    f1_0 = round(f1_score(y_true, y_pred, pos_label=0), 4)

    avg_f1 = round((f1_1 + f1_0) / 2, 4)

    print(f"Accuracy: {accuracy}, Utility: {round(utility, 4)}")
    print(f"Precision_1: {precision_1}, Recall_1: {recall_1}, F1_1: {f1_1}")
    print(f"Precision_0: {precision_0}, Recall_0: {recall_0}, F1_0: {f1_0}")
    print(f"Avg F1: {avg_f1}")
    print('-' * 50)
