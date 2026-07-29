# Gemma4 DSpark — TorchSpec 集成实施蓝图

> 分支 `dev/gemma4_dspark` · 状态: 📐 规划完成，待实施
> 目标: 在 TorchSpec 引擎上集成 gemma4 dense DSpark 训练（DeepSpec 分布式老卡死，迁到 TorchSpec）。
> 部署走 **vllm 0.26.0**（`~/workspace/vllm-026`, 分支 `dev/gemma4_moe_dspark_patch_0.26.0`）。

---

## 0. 决策与结论（Master Table）

| 决策 | 选择 | 理由 |
|------|------|------|
| **集成方式** | **Y：平行加 gemma4 backbone**，不动现有 qwen3 | 对齐 vllm 0.26 官方做法，风险小 |
| **draft 类型** | **dense**（26B MoE target 也配 dense draft） | 和 MTP 一致；MoE 部分 draft 不管 |
| **部署 vllm** | 0.26.0（有现成 `gemma4_dspark.py`） | dspark method + gemma4 dspark 已注册支持 |
| **训练引擎** | TorchSpec（dspark 逻辑已从 DeepSpec 移植，architecture-agnostic） | 复用现成 dspark trainer/loss/markov/confidence |
| **对齐标尺** | vllm 0.26 `gemma4_dspark.py`（部署真值）+ DeepSpec `gemma4/modeling.py`（训练参考） | 复刻 MTP 打法：训练 forward 对齐 vllm，argmax 复现 |
| **总工作量** | **中，风险低**：核心 1 个新文件（gemma4 backbone） | dspark 逻辑全复用；backbone 有三重参照（vllm+DeepSpec+MTP 经验） |

---

## 1. 架构总览

```
DSparkTrainer (训练入口, 复用)                        [training/dspark_trainer.py]
  ├─ _build_draft_model → DSparkDraftModel(config)   ← 分派点①: 按 model_type 建 backbone
  └─ _build_training_wrapper → DSparkModel(...)       [models/dspark.py, 265行, 全复用]
        · anchor/block-mask/noise (DFlashModel 父类)
        · markov bias + CE + L1蒸馏 + confidence BCE  ← architecture-agnostic, 零改动
        └─ self.draft_model = DSparkDraftModel        [models/draft/dspark.py]
              ├─ markov_head / confidence_head         ← 复用
              └─ backbone (继承 DFlashDraftModel)       ← 分派点②: qwen3 layer / gemma4 layer
                    ├─ DFlashDecoderLayer (qwen3, 现有)  [models/draft/dflash.py]
                    └─ DFlashGemma4DecoderLayer (新增)   [models/draft/dflash_gemma4.py] ★核心
```

**核心洞察**：dspark 的"脑子"（block 两阶段 + 3 损失 + markov + confidence）完全 architecture-agnostic，
已在 TorchSpec（移植自 DeepSpec `dspark/loss.py`）。**gemma4 只需换 backbone 的 layer。**

---

## 2. DSpark 数学规格（部署标尺，训练必须对齐）

来源：vllm 0.26 `gemma4_dspark.py` + `qwen3_dspark.py` + `qwen3_dflash.py`。

### 两阶段 drafting
1. **并行阶段**：draft 对整个 block（mask tokens）一次前向，用 target hidden 投影的 **context-KV** 做 cross-attention（DFlash-style，non-causal query-block）。
2. **顺序阶段**：`markov_head` 注入 block 内依赖——低秩转移偏置加到 base logits，speculator 左→右采样。

### gemma4 backbone forward（要在训练侧复刻）
```
embed:      embed_tokens(ids) * sqrt(hidden)          # scaled word embed (normalizer)
context:    fc(cat[多层 target hidden]) → hidden_norm   # Linear(hidden*num_target_layers → hidden) + RMSNorm
layers×N:   Gemma4DSparkAttention + MLP
              q = q_norm(q_proj(h));  k = k_norm(k_proj); v = v_norm(v_proj) [v_norm no weight]
              is_full → head_dim = global_head_dim; else config.head_dim
              use_k_eq_v = is_full and attention_k_eq_v → v_src = k, v_proj=None
              num_kv_heads = num_global_key_value_heads (k_eq_v) else num_key_value_heads
              rope(positions, q, k);  attn(q, k, v)     # scaling=1.0
              context-KV: target hidden → k_proj → k_norm+rope；v = v_norm(k)
final:      norm(hidden)
logits:     lm_head(hidden) + markov_bias
```

### markov head（复用，已在 TorchSpec）
```
markov_w1: Embedding(vocab, r)    # embed 上一个采样 token
markov_w2: Linear(r, draft_vocab) # 投影成 draft-vocab bias，加到 base logits
```

### confidence head（训练用，部署 skip）
- 训练：`AcceptRatePredictor(hidden [+markov_r])` → BCE vs 经验 accept rate `(1 - 0.5*L1)`。
- 部署：vllm `load_weights` 明确 **skip `confidence_head`**（不参与推理）。

---

## 3. 权重命名契约（checkpoint 必须遵守）

对齐 vllm `Gemma4DSparkForCausalLM.load_weights`：
- `lm_head.*` — 不加前缀；其余全加 `model.` 前缀
- `gate_up_proj` = packed `[gate_proj, up_proj]`（shard 0/1）
- `d2t` → `draft_id_to_target_id`（draft vocab 映射，full-vocab 时可无）
- `markov_head.markov_w1 / markov_w2` — 保留
- `confidence_head.*` — 训练存，部署 skip
- `dspark_shares_target_embeddings = False`（draft 自包含 embed/lm_head）

---

## 4. Draft config 规格（参照 K3DSpark / gemma4-a4b-assistant）

从 target `text_config` 派生（抄 DeepSpec `gemma4/config.py`）。关键字段：
```jsonc
{
  "architectures": ["Gemma4DSparkModel"],
  "model_type": "gemma4_dspark",
  "hidden_size": ..., "intermediate_size": ..., "num_hidden_layers": 5,
  "num_attention_heads": ..., "num_key_value_heads": ...,
  "global_head_dim": ..., "head_dim": ..., "attention_k_eq_v": ...,
  "num_global_key_value_heads": ..., "layer_types": [...],   // gemma4 sliding/full
  "vocab_size": ..., "rms_norm_eps": ..., "rope_parameters": {...},
  "num_target_layers": 5, "target_hidden_size": ...,
  "target_num_hidden_layers": ..., "target_layer_ids": [...],
  "mask_token_id": ..., "markov_rank": 256, "markov_head_type": "vanilla",
  "enable_confidence_head": true, "confidence_head_with_markov": true,
  "tie_word_embeddings": false, "draft_vocab_size": ...,
  "enable_moe_block": false,                 // draft 是 dense
  "hidden_size_per_layer_input": 0
}
```
**gemma4-moe-dspark**：同结构，维度对齐 26B-a4b target（参照 gemma4-12b vs 26b-a4b MTP assistant 的差别）。MoE 部分 draft 不特殊考虑。

---

## 5. 实施清单（文件级）

| # | 文件 | 动作 | 量 | 依赖 |
|---|------|------|:--:|------|
| 1 | `torchspec/models/draft/dflash_gemma4.py` | **新增** gemma4 backbone（Attention/DecoderLayer/Model） | 大 | vllm gemma4_dspark.py + MTP gemma4 attention |
| 2 | `torchspec/models/draft/dspark.py` | 改：`DSparkDraftModel` 按 model_type 分派 backbone | 小 | #1 |
| 3 | `torchspec/models/draft/dflash.py` | 改：`DFlashDraftModel` 支持 gemma4 embed(scaled)/layer 分派 | 中 | #1 |
| 4 | `torchspec/training/dspark_trainer.py` | 改：加 gemma4 config 构造（或 Gemma4DSparkTrainer 子类） | 小 | #5 |
| 5 | gemma4 dspark draft config 生成器/json | **新增**：从 target text_config 派生 | 小 | 抄 DeepSpec config.py |
| 6 | 数据管线（online engine） | 确认/改：gemma4 target 采多层 hidden + last_hidden | 中 | qwen3 dspark 已有；gemma4 target 适配 |
| 7 | checkpoint 导出 | 对齐 vllm load_weights 命名 | 小 | #3 |

### 数据契约（已确认匹配）
`DSparkModel.forward(input_ids, hidden_states_list, loss_mask, lm_head_weight, last_hidden_states)`
- `hidden_states_list` = 多层 target hidden（target_layer_ids）
- `last_hidden_states` = target final hidden（L1 蒸馏 + confidence 需要，`store_last_hidden_states=true`）
- 需求 = qwen3 dspark 已有的，gemma4 只换 target 来源。

---

## 6. 实施顺序（建议）

1. **#6 先排最大不确定性** — 确认 gemma4 target 能采多层 hidden + last_hidden（online engine）。通了再动 backbone。
2. **#5 + #1 backbone** — 写 gemma4 draft config + backbone，用 vllm gemma4_dspark.py 当标尺。
3. **验证 backbone == vllm**（复刻 MTP 打法）— 单步/单 block dump vllm dspark forward，训练侧 backbone 复现 argmax。
4. **#2/#3/#4 接线** — 分派 + trainer 接入。
5. **端到端小跑** — dense gemma4 dspark 训练几步，loss/acc 正常。
6. **对齐 DeepSpec 已验证结果** — warm-start finetune，对比 accept_len（DeepSpec dense 12B 最佳 equal-weight mean 5.86）。
7. **部署验证** — 转 checkpoint → vllm 0.26 加载 → online bench。

---

## 7. 参照资源

| 资源 | 路径 | 用途 |
|------|------|------|
| vllm 0.26 gemma4 dspark（部署标尺） | `~/workspace/vllm-026 vllm/model_executor/models/gemma4_dspark.py` | forward/权重命名真值 |
| vllm qwen3 dspark/dflash | 同上 `qwen3_dspark.py` `qwen3_dflash.py` | dspark 两阶段 + context-KV 机制 |
| DeepSpec gemma4 dspark（训练参考，已验证） | `~/workspace/DeepSpec-maiprofile deepspec/modeling/dspark/gemma4/` | modeling+config 参考 |
| DeepSpec 已验证结果 | 同上 `docs/maiprofile_data_overview.md` | dense 12B 精度基线（warm-start 5.86） |
| TorchSpec dspark（训练逻辑，复用） | `torchspec/models/dspark.py` `training/dspark_trainer.py` | architecture-agnostic dspark 数学 |
| MTP gemma4 attention（经验） | `torchspec/models/draft/gemma4_mtp_strip_forward.py` | gemma4 attention 细节（partial rope/global_head_dim/k_eq_v） |

---

## 8. MoE 进阶（后续独立阶段）

- draft 仍 dense（同上 backbone，维度对齐 26B-a4b target）
- vllm 0.26 主模型 `gemma4.py` 已有 FusedMoE / MixtureOfExperts → **26B MoE target 部署已支持**
- 主要工作：config 维度对齐 + 26B MoE target 数据管线（采多层 hidden）
- 参照 gemma4-12b-assistant vs 26b-a4b-assistant（MTP）的维度差别
