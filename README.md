# GLOWWorkflowGlow：基于图-语言协同推理的智能体工作流性能预测

## 📋 目录

- [项目概述](#项目概述)
- [核心贡献](#核心贡献)
- [模型架构](#模型架构)
- [技术亮点](#技术亮点)
- [环境安装](#环境安装)
- [预训练模型下载](#预训练模型下载)
- [数据准备](#数据准备)
- [训练流程](#训练流程)
- [评估与测试](#评估与测试)
- [项目结构](#项目结构)
- [超参数说明](#超参数说明)
- [引用](#引用)

---

## 项目概述

随着大语言模型（LLM）驱动的智能体系统（Agent Systems）迅速发展，**LLM-as-a-Service** 的调用模式正从简单的单次推理演变为**复杂的有向图工作流（DAG Workflow）**。在一个工作流中，多个 LLM 调用节点通过依赖关系相连，形成一个计算图——每个节点执行特定任务（如代码生成、数学推理、逻辑判断），节点间的边表示数据流和依赖关系。

**核心问题：** 给定一个由 LLM 调用节点组成的有向图工作流，如何判断它是否能够成功执行？

这一问题在**工作流优化、资源调度、自动化编排**等场景中至关重要。例如：
- **工作流剪枝**：在执行前识别可能失败的分支，节省计算资源
- **智能路由**：为相似任务选择成功率最高的工作流模板
- **质量监控**：实时评估工作流的预期成功率

### GLOW 的解决方案

GLOW 提出了一种**多模态协同推理框架**，同时从**图结构**、**语义内容**和**任务特征**三个维度理解工作流，并通过 Transformer 融合层进行联合推理，最终输出成功概率预测。

---

## 核心贡献

1. **图感知 LLM 预微调（Graph-Oriented LLM Pre-finetuning）**
   - 通过 LoRA 微调，使通用 LLM 具备理解工作流图结构的能力
   - 利用图结构描述作为微调数据，让 LLM 在推理时能感知节点间的拓扑关系

2. **GNN 预训练策略**
   - 节点特征重建（Node Reconstruction）：学习图结构的低维表示
   - 邻接矩阵重建（Adjacency Matrix Reconstruction）：恢复拓扑连接关系
   - 两阶段训练确保 GNN 编码器在端到端训练前已获得良好的图结构理解

3. **多模态 Transformer 融合**
   - 引入可学习的 `[Pred]` Token 作为预测锚点
   - 使用类型嵌入（Type Embedding）区分 LLM、图、任务三种模态
   - 通过 Transformer Encoder 实现跨模态注意力融合

4. **对比学习增强**
   - 按任务 ID 分组，拉近同标签工作流的嵌入、推远不同标签的嵌入
   - 提升模型对成功/失败工作流的判别能力

---

## 模型架构

GLOW 的模型由以下四个核心模块组成：

### 1. LLM 编码器（Graph-Oriented LLM）
- **基座模型**：Qwen3-1.7B（经 LoRA 预微调）
- **输入**：工作流的文本描述（包含节点信息和边信息）
- **输出**：最后一层隐藏状态的最后一个 token 嵌入（[B, D_llm]）
- **作用**：理解工作流的语义内容和结构关系

### 2. GNN 编码器（GAT-based Graph Encoder）
- **架构**：多层图注意力网络（GAT），每层 4 头注意力
- **输入**：节点文本嵌入（由 SentenceTransformer 编码）+ 图边索引
- **输出**：图级别嵌入（通过全局平均池化）+ 节点级别嵌入
- **预训练**：通过节点重建和邻接矩阵重建任务进行自监督预训练

### 3. 任务投影器（Task Projector）
- **架构**：多层感知机（MLP）
- **输入**：任务描述的文本嵌入（SentenceTransformer 编码）
- **输出**：与 GNN/LLM 同维度的任务嵌入

### 4. Transformer 融合层（Fusion Encoder）
- **输入**：4 个嵌入向量的序列 `[Pred]`、`[LLM]`、`[Graph]`、`[Task]`
- **机制**：多层 Transformer Encoder（多头自注意力）
- **输出**：取 `[Pred]` Token 的输出，经 MLP 映射为标量 logit
- **预测**：通过 Sigmoid 激活得到成功概率

### 损失函数
- **预测损失**：二元交叉熵损失（BCEWithLogitsLoss）
- **对比损失**：按任务分组的批内平均对比损失（BatchMeanContrastiveLoss）

---

## 技术亮点

### 图感知 LLM 预微调
通用 LLM 天然无法理解图结构。GLOW 通过构造图结构描述的 prompt，使用 LoRA 对 LLM 进行预微调，使其获得"图感知"能力。微调后的 LLM 能够：
- 理解节点间的有向依赖关系
- 感知工作流的整体拓扑结构
- 在推理时输出包含图结构信息的语义嵌入

### 两阶段 GNN 预训练
在端到端训练之前，GNN 编码器通过两个自监督任务进行预训练：
1. **节点特征重建**：从图级别嵌入恢复每个节点的原始特征
2. **邻接矩阵重建**：从节点嵌入对预测节点间是否存在边

### [Pred] Token 融合机制
借鉴 CLIP 中 [CLS] Token 的思想，GLOW 引入一个可学习的 `[Pred]` Token 作为预测锚点。该 Token 在 Transformer 融合层中通过自注意力机制聚合 LLM、图、任务三种模态的信息，最终其输出用于最终的二分类预测。

### 类型感知嵌入
为区分不同模态的输入，GLOW 为每个嵌入添加可学习的类型嵌入（Type Embedding）：
- 类型 0：`[Pred]` Token
- 类型 1：LLM 嵌入
- 类型 2：图结构嵌入
- 类型 3：任务嵌入

---

## 环境安装

### 软件要求

| 依赖 | 版本 |
|------|------|
| Python | 3.11.13 |
| CUDA | 12.1 |

### 安装步骤

**1. 创建 Conda 环境（可选）**

```bash
conda create -n glow python=3.11.13
conda activate glow
```

**2. 安装依赖**

```bash
conda install --yes --file requirements.txt
# 可能需要使用 pip 降级 PyTorch 以匹配 CUDA 版本
# pip install torch==<合适的版本> --index-url https://download.pytorch.org/whl/cu121
```

> ⚠️ **注意**：如果 CUDA 版本与 PyTorch 不匹配，请根据你的 CUDA 版本调整 PyTorch 安装命令。

---

## 预训练模型下载

需要从 Hugging Face 下载以下两个预训练模型：

### 1. 大语言模型：[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/tree/main)

```bash
# 使用 huggingface-cli 下载（推荐）
huggingface-cli download Qwen/Qwen3-1.7B --local-dir ./llmmodel/Qwen3-1.7B
```

下载后的目录结构：
```
llmmodel/
└── Qwen3-1.7B/
    ├── config.json
    ├── generation_config.json
    ├── LICENSE
    ├── merges.txt
    ├── model-00001-of-00002.safetensors
    ├── model-00002-of-00002.safetensors
    ├── model.safetensors.index.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.json
```

### 2. 句向量模型：[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main)

```bash
# 使用 huggingface-cli 下载（推荐）
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./llmmodel/all-MiniLM-L6-v2
```

下载后的目录结构：
```
llmmodel/
└── all-MiniLM-L6-v2/
    ├── config.json
    ├── config_sentence_transformers.json
    ├── data_config.json
    ├── model.safetensors
    ├── modules.json
    ├── sentence_bert_config.json
    ├── special_tokens_map.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── train_script.py
    └── vocab.txt
```

> 💡 **提示**：两个模型放在 `llmmodel/` 目录下，便于统一管理。你也可以放在其他路径，在后续步骤中通过参数指定。

---

## 数据准备

### 1. 下载数据集

从 Hugging Face 下载 [FLORA-Bench](https://huggingface.co/datasets/YuanshuoZhang/FLORA-Bench/tree/main) 数据集，并放置到项目的 `data/` 目录下。

```bash
# 使用 huggingface-cli 下载
huggingface-cli download YuanshuoZhang/FLORA-Bench --repo-type dataset --local-dir ./data
```

下载后的目录结构：
```
data/
├── Coding-AF/          # 编码任务 - 自动化修复
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── Coding-GD/          # 编码任务 - 通用开发
├── Math-AF/            # 数学任务 - 自动化修复
├── Math-GD/            # 数学任务 - 通用开发
├── Reason-AF/          # 推理任务 - 自动化修复
└── Reason-GD/          # 推理任务 - 通用开发
```

**数据集说明：**
- **任务类型**：Coding（编码）、Math（数学）、Reason（推理）
- **生成方式**：AF（Automated Fix，自动化修复）、GD（General Development，通用开发）
- **数据格式**：每个 JSONL 文件包含若干工作流样本，每个样本包括：
  - `nodes`：节点字典（节点 ID → 节点文本描述）
  - `edge_index`：边列表（有向边 `[source, target]`）
  - `task`：任务描述
  - `label`：标签（1 = 成功，0 = 失败）
  - `workflow_id`：工作流 ID
  - `task_id`：任务 ID

### 2. 构造 LLM 预微调数据

运行以下命令，将原始数据集处理为 LLM 预微调所需的格式：

```bash
python make_llm_prefinetuning_data.py --data_path ./data
```

**参数说明：**

| 参数 | 说明 | 示例 |
|------|------|------|
| `--data_path` | 数据集根目录路径 | `./data` |

该脚本会遍历 `data/` 下的所有子目录，读取训练数据，构造包含图结构描述的 prompt，生成预微调数据文件 `data/prefinetuning.jsonl`。

---

## 训练流程

GLOW 的训练分为三个阶段：

### 阶段一：LLM 预微调（生成图感知 LLM）

通过 LoRA 微调 Qwen3-1.7B，使其具备理解工作流图结构的能力。

#### 步骤 1：配置训练参数

编辑 **`pre-finetuning_LLM.sh`** 脚本，设置以下参数：

1. **GPU 配置**：修改可见 GPU 列表
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   ```
   根据你的 GPU 数量和型号进行调整。

2. **模型路径**：指定基础 LLM 模型路径
   ```bash
   --model llmmodel/Qwen3-1.7B
   ```

3. **数据路径**：指定预微调数据路径
   ```bash
   --dataset GLOW/data/prefinetuning.jsonl
   ```

#### 步骤 2：启动预微调

```bash
bash pre-finetuning_LLM.sh
```

该脚本会自动检测可用 GPU 数量，启动分布式训练。

#### 步骤 3：合并 LoRA 权重

预微调完成后，需要将 LoRA 适配器权重合并到基础模型中，生成独立的图感知 LLM：

```bash
python combine_lora.py \
    --peft outputs/prefinetuning/<your-checkpoint-directory> \
    --checkpoint llmmodel/Qwen3-1.7B \
    --save_path outputs/prefinetuning/graph_oriented_LLM
```

**参数说明：**

| 参数 | 说明 | 示例 |
|------|------|------|
| `--peft` | LoRA 微调后的 checkpoint 路径 | `outputs/prefinetuning/v0-20251015-105151/checkpoint-7300` |
| `--checkpoint` | 基础 LLM 模型路径 | `llmmodel/Qwen3-1.7B` |
| `--save_path` | 合并后模型的保存路径 | `outputs/prefinetuning/graph_oriented_LLM` |

合并后的模型可直接用于下游训练和推理。

#### 输出目录

所有训练日志和 checkpoint 保存在：
```
outputs/prefinetuning/
```

---

### 阶段二：GNN 预训练

在端到端训练之前，GNN 编码器通过自监督学习进行预训练（包含在 `train.py` 中自动执行）：

1. **节点特征重建**：从全局图嵌入恢复每个节点的原始特征向量
2. **邻接矩阵重建**：从节点嵌入对预测节点间的连接关系

预训练步数可通过 `--pretrain_steps` 参数控制（默认 1000 步）。

---

### 阶段三：端到端训练

以预训练的 GNN 和图感知 LLM 为初始化，进行端到端的联合训练。

#### 训练命令

以 Coding-AF 领域为例：

```bash
python train.py \
    --data_path ./data/Coding-AF \
    --llm_model_path outputs/prefinetuning/graph_oriented_LLM \
    --st_model_path llmmodel/all-MiniLM-L6-v2
```

**训练流程自动包含：**
1. ✅ GNN 自监督预训练（节点重建 + 邻接矩阵重建）
2. ✅ 端到端联合训练（BCE 预测损失 + 对比学习损失）
3. ✅ 验证集评估与早停（Early Stopping）
4. ✅ 保存最佳模型和最终模型
5. ✅ 测试集评估

#### 模型保存

训练完成后，模型 checkpoint 保存在：
```
data/<domain>/ckpt/
├── best_model.pth      # 验证集上表现最佳的模型
└── final_model.pth     # 训练结束时的模型
```

---

## 评估与测试

### 独立评估

训练完成后，可以单独对测试集进行评估：

```bash
python evaluate.py \
    --data_path ./data/Coding-AF \
    --llm_model_path outputs/prefinetuning/graph_oriented_LLM \
    --st_model_path llmmodel/all-MiniLM-L6-v2 \
    --cross_system ./data/Coding-AF
```

### 评估指标

| 指标 | 说明 |
|------|------|
| **Accuracy** | 整体分类准确率 |
| **Utility** | 工作流级别的效用指标（综合考虑预测正确的工作流占比） |
| **Precision (P=1)** | 成功工作流的精确率 |
| **Recall (P=1)** | 成功工作流的召回率 |
| **F1 (P=1)** | 成功工作流的 F1 分数 |
| **Precision (P=0)** | 失败工作流的精确率 |
| **Recall (P=0)** | 失败工作流的召回率 |
| **F1 (P=0)** | 失败工作流的 F1 分数 |
| **Avg F1** | 两类 F1 的平均值 |

### 结果输出

评估结果会自动追加到 `results.csv` 文件中，包含数据路径、模型类型（Best Val / Final）和各项指标。

---

### 核心文件说明

| 文件 | 功能 |
|------|------|
| `model.py` | 定义了 GNNEncoder（GAT 图编码器）、Predictor（融合预测器）、BatchMeanContrastiveLoss（对比损失） |
| `train.py` | 完整训练流程：GNN 预训练 → 端到端训练 → 测试评估 |
| `custom_dataset.py` | WorkflowDataset 数据集类，负责预计算 LLM 嵌入、文本嵌入和邻接矩阵 |
| `evaluate.py` | 独立评估脚本，加载训练好的模型在测试集上评估 |
| `combine_lora.py` | 将 LoRA 适配器权重合并到基础 LLM 中 |
| `make_llm_prefinetuning_data.py` | 从原始数据构造 LLM 预微调所需的 prompt 数据 |
| `utils.py` | 计算 Utility 等辅助函数 |

---

## 超参数说明

### 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hidden_dim` | 256 | 隐藏层维度 |
| `--n_gnn_layers` | 2 | GNN 层数 |
| `--n_mlplayers` | 2 | MLP 层数 |
| `--dropout` | 0.2 | Dropout 比率 |
| `--batch_size` | 512 | 训练/评估批大小 |
| `--pretrain_batch_size` | 64 | GNN 预训练批大小 |
| `--epochs` | 200 | 最大训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--weight_decay` | 1e-4 | 权重衰减 |
| `--patience` | 30 | 早停耐心值（连续多少轮无提升则停止） |
| `--seed` | 42 | 随机种子 |
| `--pretrain_steps` | 1000 | GNN 预训练步数 |
| `--eval_steps` | 1 | 评估间隔（每隔多少个 epoch 评估一次） |

### 对比学习超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--contrastive_weight` | 1 | 对比损失权重 |
| `--margin` | 0.2 | 对比损失间隔（越大，正负样本距离越远） |

---

## 致谢

- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) — 大语言模型基座
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — 句向量编码器
- [FLORA-Bench](https://huggingface.co/datasets/YuanshuoZhang/FLORA-Bench) — 工作流性能预测基准数据集
- [PyTorch Geometric](https://pyg.org/) — 图神经网络框架
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) — LLM 推理框架