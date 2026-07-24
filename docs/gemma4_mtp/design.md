# Gemma4 MTP Training — Design (option A, faithful to Google's assistant)

> Status: **DESIGN REVIEW — no implementation yet.** Read top-down; the master
> table is the whole plan, sections below are the evidence and open decisions.

## Master plan

| # | Component | File (to create/edit) | Depends on | Status |
|---|-----------|----------------------|------------|--------|
| 1 | Draft config | `configs/draft_models/gemma4_mtp.json` | — | ✅ done |
| 2 | Draft model wrapper (wraps HF `Gemma4AssistantForCausalLM`) | `torchspec/models/draft/gemma4_mtp.py` | probes | ✅ done (parity fp32 diff==0) |
| 3 | Training wrapper (TTT-style MTP unroll + loss) | `torchspec/models/gemma4_mtp.py` | #2 | ✅ done |
| 4 | Trainer | `torchspec/training/gemma4_mtp_trainer.py` | #2,#3 | ✅ done |
| 5 | Target extractor: last hidden + shared_kv per layer_type | `torchspec/models/target/gemma4_mtp_target.py` | — | ✅ done |
| 6 | Mooncake schema: carry shared_kv tensors | `torchspec/transfer/mooncake/gemma4_mtp_store.py` | #5 | ✅ done |
| 7 | Registration (config→model, config→trainer) | `auto.py`, `trainer_actor.py` | #2,#4 | ✅ done |
| 8 | Parity test vs HF assistant (bit-align) | `tools/gemma4_mtp/verify_parity.py` | #2,#3 | ✅ done (fp32 diff==0) |

### Remaining integration gaps (server-side, cannot be验证 locally)

The 8 core components are written and unit-smoke-tested. What still needs wiring
before a full distributed run (each depends on the live Ray/Mooncake stack):

- **G1 — inference engine hookup**: the inference-side engine must call the
  Gemma4 target with `return_shared_kv_states=True` and store via
  `Gemma4MTPMooncakeStore.put(key, Gemma4MTPTargetOutput)`. Today `HFRunner` /
  the engines produce `Eagle3TargetOutput`; add a Gemma4-MTP path (mirror how
  DFlash selects its target output) — likely a new runner branch or a
  `target_model_backend`/`draft type` switch in `inference/factory.py`.
- **G2 — data fetcher get()**: `MooncakeDataFetcher` currently builds an
  `Eagle3TargetOutput` via `EagleMooncakeStore.get`. Route Gemma4 MTP keys to
  `Gemma4MTPMooncakeStore.get` (returns a dict batch) so the trainer `_forward`
  receives `last_hidden/sliding_k/.../loss_mask`.
- **G3 — controller metadata**: the controller passes `tensor_shapes/dtypes`
  from put → get. The MTP put returns the same `{"shapes","dtypes"}` shape, so
  this should flow through unchanged — verify no Eagle3-specific key assumptions
  (e.g. hard-coded `"hidden_states"`).
- **G4 — buffer sizing**: KV is heavy (~T*8KB/sample KV + T*5.6KB hidden).
  Confirm Mooncake `global_segment_size` / `host_buffer_size` / `gpu_buffer_size`
  are large enough (design D1); bump in the run config if `batch_put_from` fails.
- **G5 — run config + example**: add a `configs/gemma4_mtp_*.yaml` +
  `examples/gemma4-mtp/run.sh` wiring target=/tmp/models/gemma4/text_only,
  draft_model_config=configs/draft_models/gemma4_mtp.json, with the
  `gemma4_mtp_num_steps/teacher_force/loss_decay_gamma` knobs.

### Integration status (G1-G5) — wired, pending a live multi-GPU run

- **G1 inference** ✅ `HFInferenceConfig.mtp_mode` + `HFRunner` branch to
  `Gemma4MTPTargetModel` / `Gemma4MTPMooncakeStore`; `HFEngine` passes
  `mtp_mode` through; `InferenceConfig.mtp_mode` is YAML-settable.
- **G2 fetcher** ✅ `MooncakeDataset` handles dict `get()` + `remove_mtp_tensors`;
  base `Trainer` has `_make_mooncake_store` / `_make_collator` hooks overridden
  by `Gemma4MTPTrainer` (→ `Gemma4MTPMooncakeStore` + `Gemma4MTPCollator`).
- **G3 controller** ✅ verified generic (tensor_shapes/dtypes passthrough,
  `_seq_len(input_ids)`, `estimate_tensor_bytes`) — no Eagle3 assumption.
- **G4 buffers** ✅ run config bumps `global_segment_size: 32GB` / `local_buffer_size: 8GB`.
- **G5 run config** ✅ `configs/hf_gemma4_mtp.yaml` + TrainingConfig MTP knobs.

Launch (multi-GPU, HF backend):
```bash
python -m torchspec.train_entry --config configs/hf_gemma4_mtp.yaml \
    dataset.train_data_path=/path/to/data.jsonl output_dir=./outputs/gemma4-mtp
```
Watch: loss should fall and per-step accept (acc_per_step[0]) should climb above
the ~0.55 zero-train baseline. Mooncake must be installed for the matching CUDA
(cu12): `uv pip install mooncake-transfer-engine`.

### Consistency cruxes — all three now cleared ✅

1. **Forward parity**: fp32 `verify_parity.py` → logits & last_hidden diff==0.
2. **prev_hidden recurrence**: training feeds the draft's own post_projection
   output on step k>0 (invariant 3), matching inference.
3. **No future-KV leak when batching T positions**: `probe_train_mask.py` →
   parallel forward == per-position single forward (fp32 diff ~1e-5). The HF
   `create_attention_masks` is already causal per query position, so training
   can batch all T positions in one assistant forward WITHOUT a custom mask.

### Label alignment — settled empirically (`probe_label_alignment.py`)

parity only validated draft forward NUMERICS, not the hand-written label
alignment in the training wrapper. Measured, not argued: run the untrained draft
faithfully per position (get_candidates step 0) and check top-1 agreement with
target greedy next token.

| Hypothesis | supervising hidden | agreement |
|-----------|--------------------|-----------|
| H0 shift=0 (correct) | `target_hidden[t]` → token_{t+1} | **0.583** |
| H1 shift=1 (old bug) | `target_hidden[t+1]` → token_{t+2} | 0.042 |

Conclusion: draft at anchor t, fed `(token_t, target_hidden[t])`, predicts
token_{t+1}. So drafting step k supervises against `target_hidden[t+k]` (shift
k) with label token at `t+k+1`. The earlier shift=k+1 on the hidden was an
off-by-one that crushed accept to ~0.09; fixed in commit 984abe3. Step-0 accept
0.583 for a 1024-dim/4-layer draft is a healthy zero-train baseline.

---

## The ground-truth inference contract (from HF source)

Source: `transformers 5.9.0`, `generation/candidate_generator.py ::
SinglePositionMultiTokenCandidateGenerator.get_candidates` (the ONLY thing that
drives the Gemma4 assistant). Reproduced faithfully:

```python
# Once per drafting round, from the TARGET model's outputs:
last_hidden = model_outputs.hidden_states[-1][:, n_last_matches:n_last_matches+1]  # (B,1,2816), the LAST seen token
shared_kv  = {lt: (K[:,:, :cur_len, :], V[:,:, :cur_len, :]) for lt,(K,V) in target_shared_kv.items()}
pos_ids    = [[seq_len - 1]]                                   # CONSTANT for the whole round
tok        = input_ids[:, -1:]                                 # last seen token id

for _ in range(k):                                            # autoregressive, ONE token per step
    tok_emb       = target_embed(tok)                         # TARGET embedding table, raw (already *sqrt(2816))
    inputs_embeds = cat([tok_emb, last_hidden], dim=-1)       # (B,1,5632)
    out           = assistant(inputs_embeds, position_ids=pos_ids, shared_kv_states=shared_kv)
    tok           = out.logits.argmax(-1)                     # greedy draft token
    last_hidden   = out.last_hidden_state                     # = post_projection(1024→2816), FED BACK next step
```

### Non-negotiable invariants (training MUST match all six)

1. **q_len = 1.** One assistant forward = one drafted token. Pure autoregression.
2. **`position_ids` is constant** = position of the last seen token. It never
   increments during a drafting round. Localisation is done purely via
   cross-attention to `shared_kv_states`, not via RoPE position.
3. **Recurrence of `prev_hidden`:** step 0 uses the **target's** last hidden;
   step t>0 uses the **assistant's own** `post_projection` output. ← the draft
   consumes its OWN (imperfect) hidden on later steps.
4. **Token embedding comes from the TARGET model's embedding table**
   (`target_model_input_embeddings`), raw/scaled. The assistant's own
   `embed_tokens` is **unused** on this path.
5. **`shared_kv_states` is fixed for the whole round** (target KV of the last
   layer per `layer_type`, sliced to prefix length). Not recomputed per step.
6. **Cross-attention with bidirectional masks** (`create_attention_masks`): the
   drafter's Q attends to the target's KV; full + sliding, with a kv-axis flip
   for the SWA "future→past" perspective.

### embed_scale trap — now confirmed in source

`modeling_gemma4.py:1581`:
```python
Gemma4TextScaledWordEmbedding(..., embed_scale=self.config.hidden_size**0.5)
```
The sqrt(2816) scale is **baked into the embedding call**. `inputs_embeds` must
use `target_embed(tok)` AS-IS. **No manual normalizer** (this is exactly the
double-scaling bug: 3.3%→41% accept once removed in the earlier from-scratch
trainer).

`use_ordered_embeddings=False` on this checkpoint → plain `lm_head`, the
`masked_embedding`/centroid path is NOT taken.

---

## Why the earlier from-scratch trainer had "loss ↓ but accept flat"

Invariant #3 is the smoking gun. At inference, drafting step t>0 feeds the
**draft's own** hidden state (noisy). If training only ever fed the **target's
perfect** hidden (or teacher-forced ground truth) at every step, the draft
never learned to correct its own drift. CE/KL on perfect inputs falls, but the
autoregressive rollout at inference sees an unseen input distribution → accept
does not rise. TorchSpec's Eagle3 already solves this shape (`eagle3.py` TTT
loop feeds `hidden_states_out` back); we mirror it for Gemma4 MTP.

---

## Proposed training loop (TTT-style MTP unroll)

For a training sequence with target-produced tensors
`target_last_hidden (B,T,2816)`, `target_shared_kv` (per layer_type, last
layer), `input_ids (B,T)`, `loss_mask (B,T)`:

Train the draft to predict, from each valid position `t`, the next `K` tokens
`t+1..t+K`, replicating the inference recurrence:

```
for each position t (batched over T via a causal cross-attn mask):
    prev_hidden = target_last_hidden[:, t]           # step 0: target hidden
    tok         = input_ids[:, t]                    # last seen token
    for step in range(K):
        tok_emb       = target_embed(tok)            # raw scaled, target table
        inputs_embeds = cat([tok_emb, prev_hidden], -1)     # (…,5632)
        out           = assistant_backbone(inputs_embeds, pos=const_t,
                                           shared_kv = target_kv[:, :t+1])   # causal to t
        loss_step    += KL(out.logits ‖ target_dist[:, t+step+1])   # forward-KL to target
        prev_hidden   = out.last_hidden_state        # own output fed back (invariant #3)
        tok           = ??? teacher-forced vs argmax  # ← OPEN DECISION D2
```

Batching all `t` in parallel: `shared_kv` sliced-to-`t+1` per position is just a
**causal cross-attention mask** over the full target KV `(B,heads,T,d)` — query
position `t` attends to `kv[:t+1]`. So one masked forward covers all positions;
no python loop over T.

Loss: **forward-KL of draft logits vs target distribution** (TorchSpec's
`compiled_forward_kl_loss_from_hs`), NOT cross-entropy to ground truth — accept
is a distribution-match criterion. Per-step accept broken out like Eagle3's
`acc_per_position` so we can see WHICH step degrades.

---

## Mooncake transfer schema (option A additions)

Per sample, in addition to Eagle3's `_hs/_ids/(_tgt|_lhs)`, carry the target
shared KV of the **last layer per layer_type** (confirmed shapes, K≠V, no dedupe):

| tensor | shape | dtype |
|--------|-------|-------|
| `sliding_K` | `(B, 8, T, 256)` | bf16 |
| `sliding_V` | `(B, 8, T, 256)` | bf16 |
| `full_K` | `(B, 2, T, 512)` | bf16 |
| `full_V` | `(B, 2, T, 512)` | bf16 |
| `last_hidden` (target) | `(B, T, 2816)` | bf16 |

`EagleMooncakeStore._put_raw_tensors` already takes a variable tensor list;
extend the suffix set + shapes/dtypes dicts. Extra bytes/sample ≈
`T * (8*256 + 8*256 + 2*512 + 2*512) * 2` = `T * 8192` bytes for KV +
`T*2816*2` for last_hidden. For T=8192 that's ~67MB + ~46MB per sample — check
against Mooncake segment sizes (**OPEN DECISION D1**).

---

## Open decisions (need your call before coding)

- **D1 — KV transfer volume.** Streaming full-prefix target KV is heavy
  (~100MB/sample at T=8k). Options: (a) stream as-is; (b) cast KV to fp8 for
  transport; (c) store only the KV actually needed by the training mask window.
  Recommend (a) first for correctness, optimise later.

- **D2 — token feed on unroll step t>0.** Inference uses the draft's own
  `argmax` token. Training options: (b1) **teacher-force ground-truth token
  t+step** (EAGLE-style: own hidden + clean token supervision, differentiable,
  stable); (b2) **free-running argmax** (matches inference exactly but
  non-differentiable at the token, needs stop-grad + risks instability early).
  Recommend **b1** (EAGLE proves it raises accept) and log per-step accept to
  verify; fall back to scheduled sampling if step-2+ accept lags.

- **D3 — unroll depth K.** Google's `num_assistant_tokens` is dynamic at
  inference. For training pick a fixed K (start K=4, matching the 4-layer
  assistant / typical MTP depth). Configurable in the draft config.

- **D4 — reuse HF module vs reimplement.** Wrap the real
  `Gemma4AssistantForCausalLM` (guarantees bit-parity, but couples to a
  transformers ≥5.9 version) vs a TorchSpec-native reimplementation (FSDP2 /
  compile friendly, but must be parity-tested). Recommend **wrap HF** for the
  first working version + `verify_parity.py`, then decide.

---

## Verification plan

`tools/gemma4_mtp/verify_parity.py`: given the same `inputs_embeds`,
`position_ids`, `shared_kv_states`, assert our draft wrapper's `logits` and
`last_hidden_state` match HF `Gemma4AssistantForCausalLM` bit-for-bit (diff=0),
exactly like the earlier gemma4-mtp-trainer parity gate.
