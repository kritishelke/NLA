"""Injection token selection (auto-pick + cache) and critic-suffix computation.

The actor prompt contains a marker token (e.g. ㊗) whose embedding is replaced
with an activation vector at training/inference time. The marker must:
  - tokenize to exactly ONE token (so the injection overwrites one position)
  - be rare in the corpus (so false-positive matches are unlikely)

We auto-pick a CJK enclosed-ideograph (U+3200–U+33FF) — single-codepoint,
essentially absent from English corpora. Results are cached to a committed
YAML so repeat runs with the same tokenizer get the same ID.

Critic extraction uses NO marker token — the critic template ends with a
known suffix (e.g. `<summary>`), and training extracts at the last-token
position. `compute_critic_suffix_ids` records the expected tail token IDs
so training can verify the prompt ends correctly (one-time CPU check at
load, then just `tokens[-1]` indexing per-forward — no GPU scanning).

Neighbor-ID computation lives in `nla.schema.compute_canonical_neighbors` —
shared with training-side verification.

See docs/design.md §1 for the full rationale.
"""

from pathlib import Path
from typing import Any

import yaml

from nla.schema import NLATokenMeta, compute_canonical_neighbors

# Cache entries are loaded from YAML, so values are untyped at the dict layer.
_CacheEntry = dict[str, Any]

_CACHE_PATH = Path(__file__).parent / "injection_token_cache.yaml"

# CJK Enclosed Letters and Months / CJK Compatibility blocks.
# ㊗ (U+3297 "circled ideograph congratulation") lives here. These are
# single-codepoint, virtually absent from English text.
_INJECTION_RANGE = (0x3200, 0x33FF)


def _load_cache() -> dict[str, _CacheEntry]:
    if not _CACHE_PATH.exists():
        return {}
    loaded = yaml.safe_load(_CACHE_PATH.read_text())
    return loaded if isinstance(loaded, dict) else {}


def _save_cache(cache: dict[str, _CacheEntry]) -> None:
    _CACHE_PATH.write_text(yaml.safe_dump(cache, allow_unicode=True, sort_keys=True))


def _tokenize_one(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _find_inctx_token_id(
    tokenizer: Any, actor_template: str, char: str
) -> int | None:
    """Return the token ID that `char` gets inside the rendered actor prompt.

    Uses apply_chat_template(tokenize=False) + encode() instead of
    apply_chat_template(tokenize=True) to avoid the mixed-type list some
    tokenizer versions return for special tokens (strings, not ints).

    BPE context-merging (e.g. tiktoken prepending the preceding space into
    the char's token) means the in-context ID can differ from the isolation
    ID. This function always returns the in-context ID.

    Returns None if the char doesn't appear as exactly one unique token.
    """
    content = actor_template.format(injection_char=char)
    rendered: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # add_special_tokens=False matches apply_chat_template(tokenize=True) behaviour —
    # special tokens are already embedded in the rendered string as literals.
    full_ids: list[int] = tokenizer.encode(rendered, add_special_tokens=False)

    # Find every token whose decoded text contains the injection char.
    matching: list[tuple[int, int]] = []  # (position, token_id)
    for i, tid in enumerate(full_ids):
        if char in tokenizer.decode([tid]):
            matching.append((i, tid))

    if len(matching) != 1:
        return None  # char absent, split across tokens, or in multiple tokens

    inctx_id = matching[0][1]
    # The in-context ID must be unique in the full prompt — no accidental matches.
    if sum(1 for t in full_ids if t == inctx_id) != 1:
        return None

    return inctx_id


def find_injection_token(
    tokenizer: Any, actor_template: str | None = None
) -> tuple[str, int]:
    """Auto-pick a single-token CJK char for activation injection. Cached.

    When actor_template is provided (strongly recommended), the returned token
    ID is the IN-CONTEXT ID — what the char gets inside the rendered prompt.
    For BPE tokenizers that merge the preceding space into the char token
    (e.g. Qwen2.5 / tiktoken), the in-context ID differs from the isolation
    ID. Training scans for the in-context ID, so the sidecar must store it.
    """
    key = tokenizer.name_or_path
    cache = _load_cache()

    if key in cache:
        cached_char = cache[key]["char"]
        cached_id = cache[key]["token_id"]
        if actor_template is not None:
            inctx_id = _find_inctx_token_id(tokenizer, actor_template, cached_char)
            assert inctx_id is not None, (
                f"cached injection char {cached_char!r} for {key!r} no longer appears "
                f"as a unique single token in the actor template. "
                f"Delete the cache entry in injection_token_cache.yaml and rerun."
            )
            if inctx_id != cached_id:
                # Cache had the isolation ID; update to the correct in-context ID.
                cache[key]["token_id"] = inctx_id
                _save_cache(cache)
            return cached_char, inctx_id
        else:
            # Legacy call without template — isolation-only check.
            ids = _tokenize_one(tokenizer, cached_char)
            assert len(ids) == 1 and ids[0] == cached_id, (
                f"cached injection token for {key!r} no longer valid: "
                f"{cached_char!r} now tokenizes to {ids} (cached id={cached_id}). "
                f"Delete the cache entry and rerun."
            )
            return cached_char, cached_id

    lo, hi = _INJECTION_RANGE
    for codepoint in range(lo, hi + 1):
        char = chr(codepoint)
        # Pre-filter: must be a single token in isolation.
        if len(_tokenize_one(tokenizer, char)) != 1:
            continue

        if actor_template is not None:
            inctx_id = _find_inctx_token_id(tokenizer, actor_template, char)
            if inctx_id is None:
                continue
            cache[key] = {"char": char, "token_id": inctx_id}
            _save_cache(cache)
            return char, inctx_id
        else:
            ids = _tokenize_one(tokenizer, char)
            cache[key] = {"char": char, "token_id": ids[0]}
            _save_cache(cache)
            return char, ids[0]

    raise AssertionError(
        f"no suitable injection char found in U+{lo:04X}–U+{hi:04X} for tokenizer "
        f"{key!r}. Hand-pick a character and add it to injection_token_cache.yaml."
    )


def compute_critic_suffix_ids(tokenizer: Any, critic_template: str) -> list[int]:
    """Return the STABLE tail of the critic template's suffix token IDs.

    The critic template ends with a fixed suffix after `{explanation}` — e.g.
    `</text> <summary>`. Training extracts at the last-token position, so
    it just needs to verify the prompt ENDS with these IDs (one-time CPU
    check, not per-forward). This avoids any marker-char that could leak
    into explanation content.

    BPE boundary issue: the FIRST token of the suffix can merge with the
    last character of the explanation (e.g. `detail.` + `</text>` → `.</`
    merges into one token). Everything AFTER that boundary is stable. So
    we return `suffix_ids[1:]` — drop the boundary token, keep the tail
    that's immune to merge effects. Still plenty of tokens to verify the
    prompt ends correctly.
    """
    assert "{explanation}" in critic_template, (
        f"critic_template must contain '{{explanation}}' placeholder: {critic_template!r}"
    )
    suffix_str = critic_template.split("{explanation}")[-1]
    suffix_ids = _tokenize_one(tokenizer, suffix_str)
    assert len(suffix_ids) >= 2, (
        f"critic template suffix {suffix_str!r} tokenized to {len(suffix_ids)} tokens — "
        f"need at least 2 so we can drop the BPE-boundary token and still have a "
        f"non-empty tail to verify. Lengthen the suffix."
    )
    # Drop the first token — it's the BPE boundary with the explanation's last
    # char and will vary depending on what the explanation ends with.
    return suffix_ids[1:]


def build_token_meta(
    tokenizer: Any,
    actor_template: str,
    critic_template: str | None = None,
) -> NLATokenMeta:
    """One-shot: auto-pick injection char + neighbors, optionally compute critic suffix.

    `critic_template=None` → no suffix computed (av_sft/rl).
    `critic_template=...` → compute the suffix IDs for ar_sft last-token extraction.

    Neighbor computation delegates to nla.schema.compute_canonical_neighbors —
    same function training-side verification uses.
    """
    inj_char, inj_id = find_injection_token(tokenizer, actor_template)
    left_id, right_id = compute_canonical_neighbors(tokenizer, actor_template, inj_char, inj_id)

    suffix_ids = compute_critic_suffix_ids(tokenizer, critic_template) if critic_template else None

    return NLATokenMeta(
        injection_char=inj_char,
        injection_token_id=inj_id,
        injection_left_neighbor_id=left_id,
        injection_right_neighbor_id=right_id,
        critic_suffix_ids=suffix_ids,
    )
