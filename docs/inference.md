# NLA Inference

Standalone inference client + recipe for NLA (Natural Language Autoencoder) models.

An **NLA pair** is two fine-tuned LMs that together map activation vectors to
natural language and back:

| | direction | mechanism |
|---|---|---|
| **AV** (Activation Verbaliser) | `vector → text` | inject vector as a 1-token embedding into a fixed prompt, autoregress |
| **AR** (Activation Reconstructor) | `text → vector` | truncated K+1-layer LM + Linear(d,d) head, extract at final token |

The round-trip **MSE(reconstructed, original)** measures how well the
verbalization captured the vector's content — it was the RL reward signal
during AV training. Low MSE ⟹ the AR can recover the original
direction from the AV's words alone.

**What's here:**
- `nla_inference.py` — single-file AV client (no heavy deps, SGLang input_embeds)
- `examples/` — worked transcripts with per-token MSE
- This README — full recipe, model-specific params, AR architecture, debugging

**Weights** (HF Hub):

| model | layer | AV (verbalizer) | AR (reconstructor) |
|---|---|---|---|
| Qwen2.5-7B | 20 | [`kitft/nla-qwen2.5-7b-L20-av`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av) | [`kitft/nla-qwen2.5-7b-L20-ar`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar) |
| Gemma-3-12B | 32 | [`kitft/nla-gemma3-12b-L32-av`](https://huggingface.co/kitft/nla-gemma3-12b-L32-av) | [`kitft/nla-gemma3-12b-L32-ar`](https://huggingface.co/kitft/nla-gemma3-12b-L32-ar) |
| Gemma-3-27B | 41 | [`kitft/nla-gemma3-27b-L41-av`](https://huggingface.co/kitft/nla-gemma3-27b-L41-av) | [`kitft/nla-gemma3-27b-L41-ar`](https://huggingface.co/kitft/nla-gemma3-27b-L41-ar) |
| Llama-3.3-70B | 53 | [`kitft/Llama-3.3-70B-NLA-L53-av`](https://huggingface.co/kitft/Llama-3.3-70B-NLA-L53-av) | [`kitft/Llama-3.3-70B-NLA-L53-ar`](https://huggingface.co/kitft/Llama-3.3-70B-NLA-L53-ar) |

---

## What an NLA AV is

A causal LM fine-tuned so that when you overwrite **one token embedding** in
its prompt with an arbitrary `[d_model]` vector, it generates a
natural-language description of that vector. The vector is typically a hidden
state extracted from another model's residual stream, but the AV doesn't
care where it came from — any `[d_model]` float vector works.

---

## The checkpoint package

Standard HuggingFace directory plus one YAML sidecar:

```
actor_hf/
├── config.json                   # standard HF
├── model-*.safetensors           # standard HF
├── tokenizer*.json               # standard HF
└── nla_meta.yaml                 # ← everything NLA-specific
```

**`nla_meta.yaml`** is the contract. Never hardcode token IDs, prompt
templates, or scale factors — load them from here and assert against the live
tokenizer at startup. Schema (only the fields inference needs):

```yaml
kind: nla_model
d_model: 3584                          # Qwen7B. Gemma-3-12B: 3840
extraction:
  injection_scale: 150.0               # L2-norm vectors get rescaled to before injection.
                                       # Qwen7B: 150. Gemma-3-12B: 80000.
tokens:
  injection_char: "㈎"                 # Qwen: U+320E. Gemma: "㈜" (U+321C).
  injection_token_id: 149705           # Qwen. Gemma: 246566.
  injection_left_neighbor_id: 29       # tokens at inj_pos ± 1 in canonical prompt —
  injection_right_neighbor_id: 522     # the `>` of `<concept>` and `<` of `</concept>`
prompt_templates:
  av: |-
    You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

    We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

    Here is the vector:

    <concept>{injection_char}</concept>

    Please provide an explanation.
```

---

## The inference recipe

### 1. Tokenize the prompt from the sidecar's template

```python
content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": content}],
    tokenize=True, add_generation_prompt=True,
)
```

**Use the sidecar's template string exactly** — any drift ("Explain:" vs
"Explain the following:") shifts the injection position and the model sees
garbage.

**Use one-step `tokenize=True`**, not `tokenize=False` → `encode()`. If you
must use the two-step path, pass `add_special_tokens=False` to `encode()` —
the chat-template string already has `<bos>` baked in (Gemma/Llama); `True`
prepends a second one and shifts every position by 1. Qwen has no BOS token
so `True` happens to work there, which makes this easy to miss.

### 2. Embed + apply architecture-specific scale

```python
embeds = embed_layer(torch.tensor(input_ids)[None]).float()  # [1, T, d]
embeds = embeds * embed_scale   # 1.0 for Qwen/Llama/Mistral; √hidden_size for Gemma-3
```

Load the embedding weight directly from safetensors
(`model.embed_tokens.weight` key) — no need to materialize the full model.
`safetensors.safe_open` reads one tensor lazily; ~2s vs ~30s for the full
12B model.

### 3. Rescale the activation vector, inject

```python
v_scaled = v_raw * (cfg.injection_scale / ||v_raw||_fp32)

# Find injection position: scan for token ID, verify neighbors
for p in [i for i, t in enumerate(input_ids) if t == cfg.injection_token_id]:
    if input_ids[p-1] == cfg.left_neighbor_id and input_ids[p+1] == cfg.right_neighbor_id:
        embeds[0, p] = v_scaled
        break
```

**`injection_scale` is mandatory.** The model was trained with vectors at
this exact L2-norm. Raw-magnitude vectors are out-of-distribution and output
degrades badly.

**Neighbor check is mandatory.** The injection char is rare but not
guaranteed unique (user pasted it, multi-turn context). The `<concept>`
closing-angle and `</concept>` opening-angle are stable and pinned in the
sidecar.

### 4. Send `input_embeds` to SGLang

```python
payload = {
    "input_embeds": embeds[0].contiguous().numpy(),   # [T, d] — unbatched
    "sampling_params": {"temperature": 1.0, "max_new_tokens": 200,
                        "skip_special_tokens": False},
}
resp = httpx.post(f"{sglang_url}/generate",
                  content=orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY))
```

**Do NOT also send `input_ids`.** When both are present, SGLang may use
`input_ids` for logprob bookkeeping while forwarding on `input_embeds`,
causing misalignment. Embeds-only is safe.

`orjson` + `OPT_SERIALIZE_NUMPY` reads the fp32 buffer directly
(no Python-float intermediate). Matters at scale; `json.dumps` on `.tolist()`
works fine for low request rates.

### 5. Extract `<explanation>`

```python
m = re.search(r"<explanation>\s*(.*?)\s*</explanation>", resp.json()["text"], re.DOTALL)
explanation = m.group(1)
```

The AV wraps its output in `<explanation>...</explanation>` tags. Missing
close tag = truncated generation (bump `max_new_tokens`). Output in Chinese =
injection failed (see Debugging). If the parsing fails, it's often still a good idea
to return the result to the user anyway.

---

## Model-specific parameters

| | Qwen2.5-7B-Instruct | Gemma-3-12B-IT | Gemma-3-27B-IT | Llama-3.3-70B-Instruct |
|---|---|---|---|---|
| `d_model` | 3584 | 3840 | 5376 | 8192 |
| extraction `layer_index` | 20 (of 28)¹ | 32 (≈ 2/3 of 48) | 41 (≈ 2/3 of 62) | 53 (≈ 2/3 of 80) |
| `injection_char` | `㈎` U+320E | `㈜` U+321C | `㈜` U+321C | `㎡` U+33A1 |
| `injection_token_id` | 149705 | 246566 | 246566 | 105565 |
| **`injection_scale`** | **150** | **80000** | **60000** | **30** |
| **`embed_scale` (post-lookup)** | **1.0** | **√3840 ≈ 61.97** | **√5376 ≈ 73.32** | **1.0** |
| `bos_token` | None | `<bos>` (already in chat template) | `<bos>` | `<\|begin_of_text\|>` |
| HF repo gated | no | **yes** — `HF_TOKEN` required | **yes** | **yes** |
| nested multimodal wrapper | no | **yes** — `config.text_config`, `model.language_model` | **yes** | no |

¹ Qwen's layer 20 is a historical choice the released checkpoints depend on, not
the 2/3-depth rule (which would give 18).

These values all live in each checkpoint's `nla_meta.yaml` sidecar — the table
is for orientation; **always load from the sidecar**, never hardcode.

**`injection_scale` spans ~2500×** across models (30 for Llama to 80000 for
Gemma-12B) because residual-stream norms differ wildly: Gemma's scaled
embedding layer multiplies by √d in the forward pass, inflating norms
(measured mean ≈ 74k at layer 32 vs Qwen's ≈ 125 at layer 20), while Llama's
layer-53 norms are smaller still. In each case `injection_scale` is picked as
a round number a bit above the mean norm of the dataset's vectors — using one
model's scale with another's checkpoint is the classic injection failure.

**`embed_scale`**: Gemma-3's `Gemma3TextScaledWordEmbedding.forward()`
multiplies by `√hidden_size`. Loading the raw weight tensor into a plain
`nn.Embedding` bypasses `forward()`, so you must apply the scale manually
after lookup. Without it, every token embedding is ~62× too small except the
injection position → garbage. Detect via `config.text_config.model_type.startswith("gemma")`.

---

## SGLang

### Dependencies

```bash
uv pip install torch transformers safetensors httpx orjson pyyaml numpy
uv pip install "sglang[all]>=0.5.6"   # input_embeds + --disable-radix-cache verified on 0.5.6
uv pip install pyarrow                # optional — only for the --parquet CLI path
```

### Launch

```bash
python -m sglang.launch_server \
    --model-path ./actor_hf \
    --port 30000 \
    --disable-radix-cache \
    --mem-fraction-static 0.85 \
    --trust-remote-code
```

**Gemma-3 checkpoints need `--attention-backend fa3`** when launching SGLang —
the default flashinfer backend OOMs on `head_dim=256`.

**`--disable-radix-cache` is required.** Radix cache keys on token IDs;
`input_embeds` requests don't supply them → different embed sequences
alias to the same cache entry.

### Throughput notes

Stock sglang>=0.5.6 works out of the box for low request rates. Two known
limitations if you push past ~10 req/s or run for many hours:

- **FastAPI validation bottleneck** — `/generate` auto-parses the request
  body into a dataclass; for `input_embeds` (~450K floats for a Qwen7B
  prompt) that's ~155ms of synchronous event-loop-blocking validation,
  capping effective concurrency at ~2. Fix is to bypass FastAPI's parser
  in the `/generate` handler (`orjson.loads` + manual dataclass construction).
- **Retract-path crash** — under memory pressure, sglang may retract an
  in-flight request and re-queue it. For `input_embeds` requests the reset
  doesn't clear `output_ids`, causing a KV-slot shape mismatch on re-prefill.
  Tracked upstream as sglang PR #14110.

Upstream PRs (draft, stacked):
- [sgl-project/sglang#20205](https://github.com/sgl-project/sglang/pull/20205) — numpy IPC (the nested-list pickle → ndarray fix)
- [sgl-project/sglang#20206](https://github.com/sgl-project/sglang/pull/20206) — SkipValidation (the FastAPI bypass)
- [sgl-project/sglang#20207](https://github.com/sgl-project/sglang/pull/20207) — bytes+shape transport (stacked on #20205)
- [sgl-project/sglang#20376](https://github.com/sgl-project/sglang/pull/20376) — slice input_embeds on chunk-overflow truncation (correctness fix — pull this in)
- Retract fix: [sgl-project/sglang#14110](https://github.com/sgl-project/sglang/pull/14110) — or sidestep retraction entirely:
  - `SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=1` in the launch env pins the admission
    ratio floor to the ceiling, killing the sawtooth decay that causes
    over-admission. (Note: `ignore_eos` requests already get ratio 1.0 per
    `schedule_policy.py:647`, so if those still retract, KV pressure is from
    elsewhere — variable prompt lengths or `--max-total-tokens` too tight.)
  - If retraction persists, add `--schedule-conservativeness 1.5` to the
    launch flags. Orthogonal knob: the env var fixes the *shape* of the
    admission budget over time; this scales its *magnitude* (default 1.0,
    higher = more KV headroom reserved). Trade-off is lower throughput.
  - Check the server log for `#retracted_reqs` to confirm whether these
    actually helped.

**Gemma-3 only:** the multimodal wrapper (`Gemma3ForConditionalGeneration`)
routes through `general_mm_embed_routine` which reads `input_ids` and ignores
`input_embeds` — injection is silently dropped, you get `\n\n\n` repetition.
Needs a small bypass patch to route straight to `.language_model` when
`input_embeds` is provided. Qwen doesn't need this (plain causal LM, no
wrapper). The patch is `patches/nla_gemma3_mm_input_embeds.patch` and is
applied automatically by `patches/apply_sglang_patches.sh`.

The bytes+shape transport (#20207) sends the raw fp32 buffer directly
instead of as a JSON array — you may find the JSON path a bottleneck if
scanning large feature dictionaries. None of these change the wire API
for this client; once merged, things just get faster.

---

## Deployment note

When standing up the system for inference, running a few full AV decodes is the
best correctness check you have — eyeballing English text vs. a CJK soup tells
you immediately whether injection worked, before any MSE numbers make sense.

---

## Debugging: injection-failure smell

**Output in Chinese / CJK is *suspicious*, not conclusive.** The injection
marker (`㈎` / `㈜`) is a CJK character — if injection fails, the AV sees
that character's own embedding as the activation, and verbalizes "something
Chinese" from free-association. But:

- **Occasional Chinese is fine for Qwen.** It's a Chinese model and genuinely
  decodes in Chinese for Chinese-language activations from the training data
  (e.g., Russian-cookbook activation → decode with Chinese commentary is
  *correct* behaviour if the residual stream carries that signal).
- **The real tells**: *all* outputs in Chinese regardless of input, *or*
  English output that's specifically describing a CJK character / Chinese
  keyword — that's the marker character itself being verbalized.

If the smell is real, causes (most-likely-first):

1. **`injection_scale` wrong** — using Qwen's 150 on a Gemma checkpoint (or
   no scaling at all). Injected vector ~500× too small; model ignores it.
2. **`embed_scale` wrong (Gemma)** — forgot the √d multiply after lookup.
   All embeddings 62× too small except injection → output is garbage but not
   usually Chinese.
3. **Double-`<bos>`** — used `encode(add_special_tokens=True)` after
   `apply_chat_template(tokenize=False)`. Injection lands one token off.
   Use one-step `tokenize=True` to sidestep.
4. **Template drift** — different prompt string than the sidecar's
   `actor_prompt_template`. Injection at the wrong position.
5. **Also sent `input_ids`** — SGLang confused by both. Send embeds only.
6. **Forgot `--disable-radix-cache`** — silent prefix-sharing between
   unrelated requests.

The sidecar loader (`nla_inference.py:load_nla_config`) asserts against the
live tokenizer at startup and catches 3+4 before the first request.

### Known-noisy inputs (not bugs)

**Early-sequence positions** (first ~10 tokens) often decode poorly even with
correct injection. At position 3, layer-20 has seen 3 tokens — the residual
stream hasn't accumulated much signal yet. Decodes trend toward the training
prior rather than anything specific. This is expected; skip or filter them.

**Occasional high-norm positions** also decode unreliably. Most Qwen layer-20
activations sit in a ~100-170 L2-norm band, but some models have the
occasional high-norm activation (e.g. early chat-template newlines can hit
~14k). These WERE seen during training — they're not strictly OOD — but
they're rare, so it's unsurprising if the decode/reconstruction is poor.
Don't over-interpret a high MSE at those positions.

---

## Optional: the AR (for scoring decode fidelity)

Lower priority — the AV is usable standalone. The AR closes the
autoencoder loop: explanation text → predicted activation vector, MSE against
the original gives a reconstruction fidelity score. Useful if you want to
rank/filter NLA decodes by how reliably the AV tracked the input.

### Architecture

**Truncated transformer + linear head, no final LayerNorm.**

- First K+1 layers only (K = the extraction layer_index used in training,
  e.g. K=20 for Qwen → 21 layers kept). Layers K+1..N are stripped — they
  add no useful signal past the extraction depth.
- **Final LayerNorm replaced with Identity** — the value head sees raw
  residual-stream output of block K, not the LN'd version.
- `lm_head` stripped — the AR never produces logits.
- `value_head = nn.Linear(d_model, d_model, bias=False)` — maps the
  residual stream to a predicted activation vector.
- Extraction position = **last token** of the input (the prompt ends with a
  fixed suffix like `</text> <summary>`, so tokens[-1] is stable — no scan).

### Checkpoint layout

```
critic_hf/
├── config.json              # num_hidden_layers = K+1 (truncated)
├── model-*.safetensors      # truncated backbone weights
├── value_head.safetensors   # the Linear head — load separately
├── tokenizer*.json
└── nla_meta.yaml
```

### AR sidecar (the fields that matter for MSE)

```yaml
kind: nla_model
role: ar
d_model: 3584
extraction:
  mse_scale: 59.87                      # = √d_model. Numerical stability only —
                                        # see §mse_scale vs injection_scale below.
ar:
  num_hidden_layers: 20                 # = K (sidecar stores K; config.json has K+1)
tokens:
  critic_suffix_ids: [1318, 29, 366, 1708, 29]   # stable tail of tokenized
                                        # template suffix, for sanity-checking
                                        # that tokens[-1] is the right position
prompt_templates:
  ar: "Summary of the following text: <text>{explanation}</text> <summary>"
```

### mse_scale vs injection_scale — different things

| | value (Qwen7B) | purpose | get it wrong and… |
|---|---|---|---|
| **`injection_scale`** | 150 | L2 norm the AV expects vectors at — matches the training distribution of residual-stream activation norms | injection fails, AV verbalizes the CJK marker char instead of your vector |
| **`mse_scale`** | √d ≈ 59.87 | numerical-stability constant for the MSE loss — nothing more | nothing really (the scale cancels; it's just a training-time gradient-magnitude knob) |

Why √d specifically: with both pred and gold at L2=s, the per-element MSE
is `2s²(1−cos)/d`. The `/d` comes from `.mean()`. Choosing `s=√d` makes
`s²/d = 1` → MSE = `2(1−cos)`, d-agnostic, range [0,4] (orthogonal=2).
So the `* mse_scale` in the code below is load-bearing — it's what makes
`.mean()` return the unit-sphere distance instead of a d-tiny number.
During training, this √d choice also kept gradient magnitudes reasonable.
The returned MSE is already the final answer; don't rescale.

**For external reporting, prefer cosine similarity.** MSE and cos carry
identical information (MSE = 2(1−cos) under this normalization) but cos is
the more intuitive metric — people know what cos=0.9 means without a lookup
table. `NLACritic.score()` returns both; pick one and be consistent.

| cos | MSE | interpretation |
|---|---|---|
| 1.0 | 0.0 | perfect |
| 0.9 | 0.2 | good decode (typical for clean positions) |
| 0.5 | 1.0 | mediocre |
| 0.0 | 2.0 | orthogonal |

### Computing MSE

```python
# 1. Wrap the AV's output in the critic template (the `critic` key in nla_meta.yaml)
prompt = critic_template.format(explanation=av_output)
# add_special_tokens=True: Gemma/Llama ARs were trained WITH the BOS prefix
# (the template is a raw string, not chat-template-processed). Qwen has
# bos_token=None so this is a no-op there. False drops BOS → position shift
# → every reconstruction degraded (observed: Gemma fve_nrm 0.31 vs 0.77).
ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"]

# 2. Forward, extract at last token
with torch.no_grad():
    out = critic(input_ids=ids)           # NLACriticOutput(values=[B,T,d], ...)
pred = out.values[0, -1]                   # [d] — last-token extraction

# 3. Normalize BOTH pred and gold to mse_scale (= √d), then MSE.
#    MSE = 2(1 − cos). Range: [0, 4]. Orthogonal → 2.
pred_n = pred / pred.norm() * mse_scale
gold_n = gold / gold.norm() * mse_scale
mse = ((pred_n - gold_n) ** 2).mean()      # ~0.2 good, ~1 mediocre, ~2 orthogonal
```

The standalone `nla_inference.py` includes `NLACritic` — loads the truncated
backbone + `value_head.safetensors`, provides `.reconstruct(text)` and
`.score(text, original_vector)`. See the class docstring for usage.

### Computing FVE — two classic footguns

FVE (fraction of variance explained) over a dataset of (gold, pred) pairs:

```python
gold_n = gold / gold.norm(dim=-1, keepdim=True) * mse_scale     # [N, d]
pred_n = pred / pred.norm(dim=-1, keepdim=True) * mse_scale     # [N, d]

mu = gold_n.mean(dim=0)            # mean of the NORMALIZED golds — do NOT normalize mu itself

fve = 1 - ((pred_n - gold_n) ** 2).mean() / ((gold_n - mu) ** 2).mean()
```

Two mistakes that produce plausible-looking but wrong numbers:

1. **You must normalize the AR's output by hand.** `NLACritic.reconstruct()`
   returns the *raw, unnormalized* predicted vector (`.score()` normalizes
   internally, but anything you compute yourself from `reconstruct()` output
   doesn't). Skipping `pred_n` makes the numerator norm-sensitive and the
   resulting "FVE" meaningless.

2. **Do not normalize the mean in the denominator.** The denominator is the
   variance of the normalized golds around their mean `mu` — and `mu` of
   unit-sphere vectors lies *inside* the sphere (‖mu‖ < mse_scale). Projecting
   it back to the sphere (`normalize(mu)`) inflates the denominator
   (≈ 0.94 vs ≈ 0.72 for Qwen layer-20) and therefore inflates FVE.
   The released `fve_nrm` numbers (e.g. 0.752 for Qwen) use the un-normalized
   mean, matching the classical FVE definition and the `fve_nrm` metric
   logged during training (`nla.loss`'s `nla_baseline_rawvar`).

`nla.schema.compute_predict_mean_baselines` computes both denominators if you
want to compare; only the raw-mean one is the headline FVE.
