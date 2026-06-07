"""NLAMegatronActor: Megatron backend for NLA training.

Architectural differences from NLAFSDPActor:
  - self.model is list[DDP(Float16Module(GPTModel))], not a single PreTrainedModel.
    PP>1 → multiple chunks. Inner GPTModel at chunk.module.module.
  - No _get_model_inputs_args / _train_step / _train_core overridable methods.
    Megatron's forward_step is a closure inside model.py:forward_only + train_one_step.
    batch["multimodal_train_inputs"] gets **-spread into GPTModel.forward → TypeError
    on unknown kwargs. Fixed with a forward_pre_hook that pops nla_activation.
  - LanguageModelEmbedding output is [s, b, h] seq-first (vs HF's [b, s, h]).
    Injection hook transposes in and out.
  - No self.ref_model — Megatron swaps weights into the SAME model via
    weights_backuper.restore("ref"). Hooks survive weight-swap (they're on
    module structure, not weights). Register once → works for actor+ref passes.
  - Critic is built by the DEFAULT model provider. We swap args.num_layers and
    args.critic_output_size before super().init(); the provider reads them via
    its closure. final_layernorm → Identity is the only post-hoc surgery
    (Identity has no params → optimizer unaffected).

Parallelism:
  - TP: supported. VocabParallelEmbedding shards VOCAB, not hidden dim — after
    all-reduce every TP rank holds full-d_model vectors for every position.
  - SP (--sequence-parallel): NOT RECOMMENDED. seq_slice implementation exists
    but Qwen-7B validation showed grad_norm spikes (99× vs FSDP at step 7) with
    SP=on, smooth with SP=off. NLA contexts are ~300 tokens — SP helps only when
    seq_len dominates activation memory, so it's pure overhead here. The
    seq_slice bug is unresolved; don't enable SP without re-investigating.
  - PP: supported for actor (hooks gate on pre_process) AND critic.
    rollout_data is DP-partitioned not PP-partitioned (process_rollout_data
    indexes by dp_rank only), so _swap/_truncate run identically on every PP
    stage. megatron_train() handles PP internally. Critic HF export uses
    HfWeightIteratorDirect for the PP broadcast.
  - CP: NOT supported (same as FSDP — neighbors can split across CP ranks).

Only restriction (assert-enforced): CP=1.

Known non-asserted risk: combined-1f1b scheduling (return_schedule_plan=True
in model.py:395-404) skips the multimodal_train_inputs spread entirely.
Pre-hook never sees nla_activation → no injection → model sees literal ㊗.
Visible failure (Chinese output) — not silent corruption, but not caught at init.
The Megatron arg that enables this (some PP-overlap schedule) isn't exposed
as a miles arg to assert against. If actor output goes to Chinese under a new
PP schedule config, this is the first thing to check.
"""

import os
# Must be set BEFORE torch import. train_env_vars applies after ray worker
# bootstrap (which imports torch). ray workers don't inherit raylet env either.
# torch 2.9 reads PYTORCH_CUDA_ALLOC_CONF only — PYTORCH_ALLOC_CONF is 2.10+.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training.async_utils import maybe_finalize_async_save

from miles.backends.megatron_utils.actor import MegatronTrainRayActor
from miles.backends.megatron_utils.model import train as megatron_train
from miles.backends.training_utils.data import get_data_iterator, get_rollout_data
from miles.backends.training_utils.log_utils import log_perf_data, log_rollout_data
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.timer import timer

from nla.arch_adapters import resolve_text_config
from nla.config import NLAConfig, load_nla_config_from_args, write_model_sidecar
from nla.injection import inject_at_marked_positions
from nla.megatron.checkpoint import gather_embedding_for_dump
from nla.embed_store import get_embed_store
from nla.schema import MM_ACTIVATION_KEY, MM_CRITIC_TOKENS_KEY, normalize_activation
from nla.train_actor import (
    CRITIC_ONLY_MM_KEYS,
    NLAFSDPActor,
    _repartition_for_critic,
    _swap_rollout_to_critic_tokens,
    _truncate_to_cross_rank_min,
)


def _gpt_model(chunk) -> torch.nn.Module:
    """Unwrap DDP(Float16Module(GPTModel)) → GPTModel.

    Confirmed by param names in megatron_to_hf/*.py:
    "module.module.embedding.word_embeddings.weight" etc.
    """
    return chunk.module.module


_shapes_patched = False
_ckpt_patched = False


def _patch_checkpoint_args_validation():
    """Skip args in Megatron's cross-rank checkpoint validation.

    _validate_common_state_dict broadcasts rank 0's common state dict (including
    args Namespace) to all ranks and compares. args.local_rank differs per rank
    → pickle sizes differ → broadcast_object_list padding corrupts bytes →
    unpickle hits UnicodeDecodeError / "unpickling stack underflow" /
    "could not convert string to float: '�'" depending on where corruption lands.

    The preprocess_common_state_dict_fn hook doesn't reach this path because
    save_preprocess validates BEFORE applying it (state_dict_utils.py:55).
    """
    global _ckpt_patched
    if _ckpt_patched:
        return
    _ckpt_patched = True
    from megatron.core.dist_checkpointing import validation as ckpt_validation

    orig = ckpt_validation._validate_common_state_dict

    def _no_args(common_state_dict):
        return orig({k: v for k, v in common_state_dict.items() if k != "args"})

    ckpt_validation._validate_common_state_dict = _no_args

    # NOTE: --exp-avg-dtype bf16 is INCOMPATIBLE with Megatron's default
    # store_param_remainders=True (TE>=2.1). The multi_tensor_adam_param_remainder
    # CUDA kernel asserts fp32 moments. fused_adam.py:205 doesn't check this.
    # Separately, bf16 moments make distrib_optimizer.load_state_dict OOM at
    # state_dict()→get_unscaled_state→.float() (~24GB transient on critic).
    # Just use fp32 moments — get_unscaled_state is then a no-op (returns ref).
    #
    # 2× optim-state peak on the SECOND load roundtrip (checkpointing.py:1720
    # after dist_checkpointing filled state in-place): distrib_optimizer.py:814
    # passes inner_state_dict["state"] (live GPU tensors by-ref) to
    # FusedAdam.load_state_dict. Our fused_adam.py:451 patch calls
    # super().load_state_dict({"state":{},...}) which WIPES self.state — the
    # live tensors survive only via the caller's state_dict_state local. Then
    # set_scaled_state→_initialize_state allocates a fresh 29GB before copy_.
    # Actor PP3: 29×2 + model+grad+ref ≈ 76GB → OOM. Fix: preserve self.state
    # across super(). On the second roundtrip the loop's copy_ is then a
    # self-copy no-op; on the first (state empty) the saved {} is harmless.
    # OOM fix is in fused_adam.py site-packages directly (fix_fused_adam_oom.py
    # patcher): save+restore self.state around super().load_state_dict({"state":{}})
    # so the loop reuses tensors instead of reallocating 29GB while caller holds old.
    from transformer_engine.pytorch.optimizers import fused_adam
    assert "_nla_state_restore" in open(fused_adam.__file__).read(), (
        "fused_adam.py missing _nla_state_restore patch — run patches/fix_fused_adam_oom.py"
    )
    _orig_fa_load = fused_adam.FusedAdam.load_state_dict  

    def _load_preserving_state(self, state_dict):
        saved = dict(self.state)
        _orig_fa_load(self, state_dict)
        for p, st in saved.items():
            for name, t in st.items():
                self.state[p].setdefault(name, t)

    fused_adam.FusedAdam.load_state_dict = _load_preserving_state

    #
    # distrib_optimizer.load_state_dict dummy-roundtrip OOM is fixed in our
    # patched Megatron-LM source — the dummy alloc + FusedAdam.load_state_dict
    # roundtrip is removed (FusedAdam:461 clears state then re-allocs → ~2×
    # peak). set_scaled_state lazy-allocates during load_parameter_state.


def _patch_communicate_shapes_unbatched():
    """Replace Megatron's _communicate_shapes batch_isend_irecv with unbatched.

    batch_isend_irecv segfaults nondeterministically on some hosts. The main
    data p2p can be switched to unbatched via config.batch_p2p_comm=False, but
    _communicate_shapes (the shape-exchange prelude) hardcodes batch_isend_irecv
    at p2p_communication.py:~236. This swaps it for sequential isend/irecv.
    """
    global _shapes_patched
    if _shapes_patched:
        return
    _shapes_patched = True
    from megatron.core.pipeline_parallel import p2p_communication as p2p

    orig = p2p.P2PCommunicator._communicate_shapes

    def _unbatched(self, tensor_send_next, tensor_send_prev, recv_prev, recv_next):
        # Replicates the else-branch of _communicate_shapes but with plain
        # isend/irecv instead of P2POp+batch_isend_irecv. Shape tensors are
        # 3-element int64 — tiny, so sequential wait overhead is negligible.
        recv_prev_shape_tensor = None
        recv_next_shape_tensor = None
        if recv_prev:
            recv_prev_shape_tensor = torch.empty((3,), device=torch.cuda.current_device(), dtype=torch.int64)
        if recv_next:
            recv_next_shape_tensor = torch.empty((3,), device=torch.cuda.current_device(), dtype=torch.int64)
        send_prev = send_next = None
        if tensor_send_prev is not None:
            send_prev = torch.tensor(tensor_send_prev.size(), device=torch.cuda.current_device(), dtype=torch.int64)
        if tensor_send_next is not None:
            send_next = torch.tensor(tensor_send_next.size(), device=torch.cuda.current_device(), dtype=torch.int64)

        reqs = []
        if send_prev is not None:
            reqs.append(dist.isend(send_prev, self.prev_rank, self.pp_group))
        if recv_prev_shape_tensor is not None:
            reqs.append(dist.irecv(recv_prev_shape_tensor, self.prev_rank, self.pp_group))
        if send_next is not None:
            reqs.append(dist.isend(send_next, self.next_rank, self.pp_group))
        if recv_next_shape_tensor is not None:
            reqs.append(dist.irecv(recv_next_shape_tensor, self.next_rank, self.pp_group))
        for r in reqs:
            r.wait()

        recv_prev_shape = recv_prev_shape_tensor.tolist() if recv_prev_shape_tensor is not None else [0, 0, 0]
        recv_next_shape = recv_next_shape_tensor.tolist() if recv_next_shape_tensor is not None else [0, 0, 0]
        return recv_prev_shape, recv_next_shape

    p2p.P2PCommunicator._communicate_shapes = _unbatched
    _unbatched._orig = orig


class NLAMegatronActor(MegatronTrainRayActor):

    def init(self, args, role, with_ref=False):
        self._is_critic_model = getattr(args, "nla_model_is_critic", False) or role == "critic"


        if self._is_critic_model:
            # Two entry paths:
            #   role=="critic" (online RL, --force-use-critic): actor and critic
            #     share CLI args. Swap num_layers/sidecar/hf_checkpoint to the
            #     critic's here, before parent builds the model. Parent's
            #     actor.py:89-93 swaps load/save/lr/lr_warmup_iters.
            #   role=="actor" + --nla-model-is-critic (standalone critic SFT):
            #     user points --num-layers / --hf-checkpoint / --nla-sidecar-source
            #     directly at the truncated critic model. No swap needed.
            if role == "critic":
                # Critic's world = critic_num_nodes × critic_num_gpus_per_node, not the
                # actor's. Recompute PP from that (actor PP3 24GPU → critic PP2 16GPU).
                # Clear uneven-split flags (actor-specific for 80L, critic 54L is even).
                # Stash the actor's PP first — needed below to compute true actor_dp.
                self._nla_actor_pp = args.pipeline_model_parallel_size
                critic_world = args.critic_num_nodes * args.critic_num_gpus_per_node
                args.pipeline_model_parallel_size = critic_world // args.tensor_model_parallel_size
                args.decoder_first_pipeline_num_layers = None
                args.decoder_last_pipeline_num_layers = None
                # Critic loads weights + optimizer. --finetune is GLOBAL (actor's,
                # shared args) and overrides no_load_optim at checkpointing.py:1711 —
                # must clear it here for critic. Adam m,v carry forward → no sign-
                # gradient blast on converged head (v10a: pred_norm 38.6→33.5 / reward
                # 1.36→0.93 collapse, PG was working until critic became unreliable).
                # 27L/TP8 critic OOMs at 76GB during opt load even with source patches —
                # use CRITIC_NODES=3 (PP3, 18L/stage, ~2/3 footprint) to fit.
                args.no_load_optim = False
                args.finetune = False
                assert args.nla_critic_num_layers is not None, (
                    "online NLA critic (Megatron) requires --nla-critic-num-layers "
                    "(= extraction layer_index K + 1). --num-layers is the actor's "
                    "full depth; this overrides it for the critic group."
                )
                args.num_layers = args.nla_critic_num_layers
                # --nla-sidecar-source on the CLI is the ACTOR's (injection_scale).
                # Critic needs its OWN (critic_num_layers, critic_suffix_ids,
                # mse_scale). --critic-load is torch_dist — no sidecar there.
                assert args.nla_critic_sidecar_source is not None, (
                    "Megatron NLA critic requires --nla-critic-sidecar-source. "
                    "The actor's --nla-sidecar-source is for the actor's injection_scale; "
                    "the critic needs its own (critic_num_layers etc). --critic-load is "
                    "torch_dist — no sidecar there. Point this at the critic HF dir "
                    "(e.g. .../critic/iter_0001000/hf)."
                )
                args.nla_sidecar_source = args.nla_critic_sidecar_source
                # Parent loads self.hf_config + self.tokenizer from args.hf_checkpoint
                # (actor.py:70-71) BEFORE the role=="critic" swaps at :89-93. Swap
                # here so the critic's hf_config is the truncated K+1-layer one.
                args.hf_checkpoint = args.nla_critic_sidecar_source
            # Vector head, not PPO's scalar. model_provider reads critic_output_size
            # into LinearForLastLayer(output_size=...). Provider now checks
            # args.nla_model_is_critic too, so both role paths get the swap.
            args.critic_output_size = args.hidden_size
            args.loss_type = "custom_loss"
            args.custom_loss_function_path = "nla.loss.nla_critic_loss"
            args.nla_model_is_critic = True  # provider reads this; set before super().init()
            assert args.critic_load_dcp is None, (
                "--critic-load-dcp is FSDP-specific (DCP vs HF split). Megatron's "
                "--critic-load is already torch_dist. Remove --critic-load-dcp."
            )

        # NLA_DISABLE_TRAIN_OFFLOAD=1: workarounds for hosts where
        # batch_isend_irecv segfaults (zombie GPU state) or memory_saver
        # pause/resume breaks PP p2p. Default off — Gemma/Qwen runs need
        # --colocate with real offload to fit. Only set on broken hosts.
        self._disable_train_offload = os.environ.get("NLA_DISABLE_TRAIN_OFFLOAD") == "1"
        if self._disable_train_offload:
            args.offload_train = False
        # Before super().init: fused_adam patch must be live before load_checkpoint
        # → optimizer.load_state_dict → state_dict() → .float() OOM. save-validation
        # patch stays harmless to call early (idempotent).
        _patch_checkpoint_args_validation()
        rollout_id = super().init(args, role, with_ref)
        # Constant lr only. Checkpoint-loaded scheduler state can carry stale
        # decay_iters/num_steps that decay lr to 0 (v11+v12: critic_lr=0 for
        # ~450 steps). args.lr is already swapped to critic_lr at actor.py:92
        # for the critic role, so this covers both. Neuter the scheduler so its
        # step() is a no-op, then set the optimizer's live lr.
        self.opt_param_scheduler.max_lr = float(args.lr)
        self.opt_param_scheduler.min_lr = float(args.lr)
        self.opt_param_scheduler.lr_decay_style = "constant"
        inner = self.optimizer.optimizer if hasattr(self.optimizer, "optimizer") else self.optimizer
        for g in inner.param_groups:
            # get_lr() reads param_group.get('max_lr', self.max_lr) — the GROUP
            # carries its own max_lr from checkpoint. Must override there too.
            g["lr"] = float(args.lr)
            g["max_lr"] = float(args.lr)
            g["min_lr"] = float(args.lr)
        if self._is_critic_model:
            # v10o saw param_groups["step"]==0 after loading m,v — bias correction
            # then wrong. Believed PP2→PP3-reshard-specific (distrib_opt roundtrip
            # rebuilds param_groups). Tripwire: if m,v are loaded, step must be too.
            if any(inner.state.values()):
                for g in inner.param_groups:
                    assert g.get("step", 0) > 0, (
                        "FusedAdam param_groups['step']==0 but m,v are loaded — "
                        "bias correction will be wrong on resume. v10o workaround "
                        "was g['step']=100; restore that if this fires."
                    )
        if self._disable_train_offload:
            # batch_p2p_comm can't go via args — validate_args forces True for
            # non-interleaved. Also _communicate_shapes hardcodes batch_isend_irecv
            # regardless of the flag; monkey-patch to unbatched.
            get_model_config(self.model[0]).batch_p2p_comm = False
            _patch_communicate_shapes_unbatched()

        # Parent's critic branch early-returns (actor.py:105-108) with rollout_id=None.
        # We still need the rest of init (cfg load, hooks, asserts).

        self._text_config = resolve_text_config(self.hf_config)

        assert not self.args.sequence_parallel, (
            "NLA disables --sequence-parallel. Qwen-7B validation showed grad_norm "
            "spikes (99× vs FSDP) with SP=on, smooth with SP=off — the seq_slice "
            "injection path has an unresolved bug. NLA contexts are ~300 tokens so "
            "SP is pure overhead anyway. Drop --sequence-parallel from your launch."
        )
        assert self.parallel_state.cp_size == 1, (
            "NLA requires cp_size=1. With cp>1, slice_with_cp splits each sample "
            "into non-contiguous chunks; injection token + neighbors can land on "
            "different CP ranks, breaking the in-hook scan."
        )
        if role == "actor" and args.advantage_estimator in ("grpo", "gspo"):
            assert args.kl_coef == 0, (
                f"--kl-coef={args.kl_coef} is a NO-OP under grpo/gspo "
                f"(get_grpo_returns discards the kl tensor). Use "
                f"--use-kl-loss --kl-loss-coef {args.kl_coef} instead."
            )

        if role == "critic" and args.force_use_critic:
            # RolloutManager partitions by actor_dp; Megatron critic typically
            # has dp=1 (PP absorbs parallelism) so process_rollout_data's
            # len(refs)==dp_size assert fails. Stash actor_dp for train() to
            # repartition. Mirrors FSDP train_actor.py; critic_dp is our own
            # parallel_state.dp_size since Megatron critic uses TP/PP.
            actor_world = args.actor_num_nodes * args.actor_num_gpus_per_node
            actor_dp = actor_world // (args.tensor_model_parallel_size * self._nla_actor_pp)
            critic_dp = self.parallel_state.dp_size
            assert critic_dp <= actor_dp, (
                f"critic_dp={critic_dp} > actor_dp={actor_dp} unsupported — see "
                f"_repartition_for_critic. Reduce CRITIC_NODES/CRITIC_GPUS."
            )
            self._nla_actor_dp = actor_dp if actor_dp != critic_dp else None
            if self._nla_actor_dp is not None:
                print(f"[NLA] asymmetric DP: actor={actor_dp} critic={critic_dp}. "
                      f"Critic will fetch all {actor_dp} actor partitions and re-slice.")

        cfg, sidecar_source = load_nla_config_from_args(self.args, self.tokenizer)
        assert cfg.d_model == self._text_config.hidden_size, (
            f"sidecar d_model={cfg.d_model} != model hidden_size="
            f"{self._text_config.hidden_size}. Wrong checkpoint for this dataset."
        )
        if self._is_critic_model:
            assert cfg.critic_num_layers is not None, (
                f"critic model built with num_layers={self.args.num_layers} but "
                f"sidecar has no critic_num_layers. Check --nla-sidecar-source."
            )
            assert self.args.num_layers == cfg.critic_num_layers + 1, (
                f"Megatron critic built with num_layers={self.args.num_layers}, "
                f"sidecar says extraction layer_index K={cfg.critic_num_layers} "
                f"→ expect K+1={cfg.critic_num_layers + 1}. "
                f"Fix --nla-critic-num-layers or --num-layers."
            )

        injects = not self._is_critic_model and self.args.loss_type in ("sft_loss", "policy_loss")
        if injects:
            assert cfg.injection_scale is not None, (
                "Actor training requires injection_scale. Set --nla-injection-scale "
                "(e.g. '150', 'raw', 'sqrt_d_model'), or point --nla-sidecar-source "
                "at a model sidecar that has it. Dataset sidecars don't carry it — "
                "it's a training hyperparameter, pick explicitly. "
                f"(Resolved sidecar: {sidecar_source!r}, injection_scale: None.)"
            )
        self._nla_cfg: NLAConfig = cfg
        self.args.nla_mse_scale = cfg.mse_scale
        # FVE baselines for loss logging (loss.py:108-113 reads via getattr and
        # gates on non-None). CLI args --nla-baseline-{meannorm,rawvar} already
        # land on args (default None). FSDP falls back to computing from parquet
        # if unset; Megatron just uses whatever was passed. Precompute once via
        # schema.compute_predict_mean_baselines and pass on CLI for metric parity.

        # Shared slot between the GPTModel pre-hook (writer) and the
        # LanguageModelEmbedding fwd-hook (reader). List-of-one for mutability
        # in closures. Megatron's weight-swap (ref/actor/old_actor) doesn't touch
        # module structure, so one registration covers all passes.
        self._nla_vectors_slot: list[torch.Tensor | None] = [None]
        self._nla_input_ids_slot: list[torch.Tensor | None] = [None]

        for chunk in self.model:
            gpt = _gpt_model(chunk)
            pre_process = bool(getattr(gpt, "pre_process", False))
            post_process = bool(getattr(gpt, "post_process", False))
            # Strip nla_activation from kwargs on ALL chunks (avoids TypeError on
            # PP stages that don't inject). For injecting actors, also stash it.
            gpt.register_forward_pre_hook(
                self._make_strip_hook(stash=injects and pre_process),
                with_kwargs=True,
            )
            if injects and pre_process:
                embedding = gpt.get_submodule("embedding")
                embedding.register_forward_hook(
                    self._make_injection_hook(), with_kwargs=True,
                )
            # final_layernorm=Identity + output_layer bias=False now handled in
            # model_provider (gated on args.nla_model_is_critic) so optimizer
            # never sees the removed params. Post-init swaps here left dead
            # optimizer entries that crashed dist_checkpointing save.
            if self._is_critic_model and post_process and self.role != "critic" and rollout_id == 0:
                # Provider built output_layer with Normal(0,0.02) init; checkpoint
                # load skipped it (shape mismatch with vocab-size dummy). Identity
                # init matches FSDP's prepare_critic_checkpoint — pred starts at
                # backbone_last_hidden (the right prior) not ~0. LinearForLastLayer
                # is plain nn.Linear (not TP-sharded), full [d,d] on each rank.
                # Skip under --offload-train (RL colocate): weight is offloaded →
                # copy_ → CUDA invalid arg. Also wrong semantically — RL loads a
                # TRAINED critic from --critic-load; overwriting with eye would
                # discard training. Identity init is only for fresh critic SFT
                # (rollout_id==0); on resume the torch_dist load already populated
                # the trained [d,d] head and overwriting would discard it.
                output_layer = gpt.get_submodule("output_layer")
                d = output_layer.weight.shape[0]
                with torch.no_grad():
                    output_layer.weight.copy_(torch.eye(d, dtype=output_layer.weight.dtype))

        if self._is_critic_model and self.role != "critic" and rollout_id == 0:
            # DistributedOptimizer keeps fp32 master copies — reload after we
            # overwrote output_layer.weight. Same pattern as checkpoint.py:189
            # after HF bridge load.
            self.optimizer.reload_model_params()

        return rollout_id

    def sleep(self) -> None:
        if self._disable_train_offload:
            return
        super().sleep()

    def wake_up(self) -> None:
        if self._disable_train_offload:
            return
        super().wake_up()


    def _make_strip_hook(self, stash: bool):
        """Pre-hook on GPTModel: pop nla_activation from kwargs.

        model.py:419 (train) and :252 (forward_only) spread
        batch["multimodal_train_inputs"] into model(**kwargs). GPTModel.forward
        doesn't accept nla_activation → TypeError. Popping here is the only
        interception point that doesn't require editing miles' forward_step
        closures. The pop is from the kwargs COPY — batch["multimodal_train_inputs"]
        stays intact for the loss function to read.
        """
        slot = self._nla_vectors_slot
        ids_slot = self._nla_input_ids_slot
        inj_scale = self._nla_cfg.injection_scale
        stash_cpu_ids = self._disable_train_offload

        def pre_hook(module, args, kwargs):
            vecs = kwargs.pop(MM_ACTIVATION_KEY, None)
            kwargs.pop(MM_CRITIC_TOKENS_KEY, None)  # never reaches model forward
            if stash and vecs is not None:
                slot[0] = normalize_activation(vecs, inj_scale)
                if stash_cpu_ids:
                    # GPU .nonzero() in fwd_hook deadlocks on prior async NCCL
                    # op on some hosts. CPU copy sidesteps the sync.
                    ids_slot[0] = kwargs["input_ids"].cpu()
            return args, kwargs

        return pre_hook

    def _make_injection_hook(self):
        """Forward hook on LanguageModelEmbedding: inject at marked positions.

        gpt_model.py:303 calls self.embedding(input_ids=..., position_ids=...) →
        kwargs only, hence with_kwargs=True. LanguageModelEmbedding.forward does
        a [b,s,h] → [s,b,h] transpose internally (language_model_embedding.py:120);
        output here is seq-first. inject_at_marked_positions expects batch-first,
        so transpose in and out. thd packing → b=1 → cheap.

        With --sequence-parallel: the embedding output is seq-sharded to
        [s/TP, b, h]. Each TP rank holds a contiguous slice of positions
        (tp_rank * s_local : (tp_rank+1) * s_local). input_ids is still FULL
        (broadcast). inject_at_marked_positions scans the full stream for the
        global match count + vec_idx, writes only positions in seq_slice.
        """
        slot = self._nla_vectors_slot
        ids_slot = self._nla_input_ids_slot
        inj = self._nla_cfg.injection_token_id
        left = self._nla_cfg.injection_left_neighbor_id
        right = self._nla_cfg.injection_right_neighbor_id
        sequence_parallel = bool(self.args.sequence_parallel)

        def fwd_hook(module, args, kwargs, output):
            vecs = slot[0]
            if vecs is None or os.environ.get("NLA_SKIP_INJECTION") == "1":
                return output
            slot[0] = None  # one-shot per microbatch; pre-hook refills
            input_ids = ids_slot[0] if ids_slot[0] is not None else kwargs["input_ids"]
            ids_slot[0] = None
            assert input_ids.dtype == torch.long
            # output: [s_local, b, h] → [b, s_local, h] for inject, then back.
            # Without SP: s_local == full S. With SP: s_local == S / TP.
            emb_bsh = output.transpose(0, 1)
            seq_slice = None
            if sequence_parallel:
                tp_rank = mpu.get_tensor_model_parallel_rank()
                s_local = emb_bsh.shape[1]
                seq_slice = (tp_rank * s_local, (tp_rank + 1) * s_local)
            injected = inject_at_marked_positions(
                input_ids=input_ids,
                embeddings=emb_bsh,
                vectors=vecs,
                inj_id=inj, left_id=left, right_id=right,
                seq_slice=seq_slice,
            )
            return injected.transpose(0, 1).contiguous()

        return fwd_hook

    def connect_actor_critic(self, actor_handle=None, master_address=None, master_port=None):
        # Parent builds an actor↔critic NCCL group for PPO's sync_actor_critic_data
        # (megatron_utils/actor.py:552). NLA's critic consumes rollout_data_ref
        # independently — GRPO advantages come from rewards, not critic values.
        # No sync, no group. (Parent's train_critic calls sync at :320; our
        # train_critic override skips it.)
        pass

    @torch.no_grad()
    def critic_fwd(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Inference-only forward; returns values at each sample's last real token.

        Ray-callable from nla.reward.nla_rm during rollout (trainer is idle).
        RayTrainGroup dispatches to every rank — Megatron's forward has TP
        collectives (embedding all-reduce etc.), so all TP+PP ranks must
        participate. Pipeline-parallel via forward_backward_func(forward_only=True).

        Input is [B, T] right-padded — reward._lazy_init and
        prepare_critic_checkpoint both force padding_side='right' on the
        tokenizer. With attention_mask=None Megatron uses causal-only;
        padding tokens at [len:T] are never attended by the last real token
        (causal only looks left). With left-pad this assumption would break
        (pad at [0:start] IS attended) — the upstream guarantee is required.

        Returns [B, d] CPU tensor on ALL ranks (broadcast from last PP stage) —
        Ray caller does ray.get(refs)[0], any rank's result works.

        Critic is on separate GPUs from sglang (no colocation) and runs reward
        scoring WHILE rollout happens — must stay awake. sleep/wake are no-ops
        for the critic (see overrides), so weights are always live here.
        """
        assert self._is_critic_model, "critic_fwd called on non-critic actor"
        ids = input_ids.cuda(non_blocking=True)
        mask = attention_mask.cuda(non_blocking=True)
        # Rightmost True in mask, robust to either padding side.
        # Upstream forces right-pad so this is belt-and-suspenders, but
        # mask.sum-1 is wrong for left-pad and there's no reason to keep
        # the fragile version when the robust one is the same speed.
        last_idx = mask.cumsum(dim=1).argmax(dim=1)
        B, T = ids.shape
        # With --sequence-parallel, Megatron's embedding scatter asserts
        # T % tp_size == 0 (tensor_parallel/mappings.py:_split_along_first_dim).
        # HF tokenizer's padding=True only pads to longest-in-batch. Training
        # pads to tp_size × pad_multiplier (data.py:53-54); do the same here.
        tp_size = self.parallel_state.tp_size
        if self.args.sequence_parallel and T % tp_size != 0:
            pad = tp_size - (T % tp_size)
            ids = torch.nn.functional.pad(ids, (0, pad), value=0)
            mask = torch.nn.functional.pad(mask, (0, pad), value=0)
            T = ids.shape[1]

        def forward_step(data_iterator, model, return_schedule_plan=False):
            assert not return_schedule_plan
            # bshd format (batched + padded), not the thd packed stream used in
            # training. packed_seq_params=None → Megatron handles as regular batch.
            output = model(
                input_ids=ids,
                position_ids=None,
                attention_mask=None,
                labels=None,
                packed_seq_params=None,
            )

            def extract(logits, non_loss_data=True):
                # GPTModel._postprocess transposes [s,b,h] → [b,s,h] when
                # labels=None (verified at pinned Megatron commit). So logits
                # here is BATCH-first [B, T, d]. float()'d by LinearForLastLayer.
                # Megatron's schedules.py calls with non_loss_data=True — kwarg
                # kept for API compat (same pattern as loss.py:152,231).
                pred = logits[torch.arange(B, device=logits.device), last_idx]
                return {"pred": pred}

            return output, extract

        for m in self.model:
            m.eval()
        results = get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=[None] * len(self.model),
            model=self.model,
            num_microbatches=1,
            seq_length=T,
            micro_batch_size=B,
            forward_only=True,
            collect_non_loss_data=True,
        )
        for m in self.model:
            m.train()

        # results non-empty only on last PP stage. Broadcast to all so any rank's
        # return is valid (caller takes [0] blindly).
        if mpu.is_pipeline_last_stage():
            pred = results[0]["pred"]
        else:
            pred = torch.empty(B, self.args.hidden_size, dtype=torch.float32, device=ids.device)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            dist.broadcast(
                pred,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )
        return pred.cpu()

    def compute_log_prob(self, data_iterator, num_microbatches, store_prefix=""):
        # Same optimization as FSDP (_compute_log_prob override). For sft_loss,
        # sft_loss_function recomputes from logits — this pass is wasted.
        # Critic never runs this (parent's train_critic uses forward_only(get_values)
        # instead, which our train_critic override also skips).
        if self.args.loss_type == "sft_loss":
            return {}
        return super().compute_log_prob(data_iterator, num_microbatches, store_prefix)

    def train(self, rollout_id, rollout_data_ref):
        # Eager-finalize async-save: per-step non-blocking poll. Without this,
        # iter_N's .metadata only appears at save@{N+SAVE_INTERVAL}'s blocking
        # finalize → supervisor restarts lose ~5hr of progress. Polled here,
        # .metadata lands ~1 step after the bg-write completes (~40min). NOT a
        # daemon thread — finalize does a CUDA all_reduce on the default PG
        # (core/.../async_utils.py:583), so it must run on all ranks at a
        # NCCL-quiescent point in the main thread. Self-gates on args.async_save;
        # no-op once the queue is drained, so the blocking call in
        # super().save_model() becomes a free pass-through.
        maybe_finalize_async_save(blocking=False)
        # Can't reuse parent's train() — it dispatches to parent's train_critic
        # which hardcodes loss_type = "value_loss" (actor.py:324).
        if getattr(self, "_nla_actor_dp", None) is not None:
            rollout_data_ref = _repartition_for_critic(
                rollout_data_ref, self._nla_actor_dp,
                self.parallel_state.dp_rank, self.parallel_state.dp_size,
            )
        with timer("data_preprocess"):
            rollout_data = get_rollout_data(self.args, rollout_data_ref, self.parallel_state)
            if self.args.debug_rollout_only:
                log_rollout_data(rollout_id, self.args, rollout_data, self.parallel_state)
                return

        if self._is_critic_model:
            return self._train_nla_critic(rollout_id, rollout_data)
        return self._train_nla_actor(rollout_id, rollout_data)

    def _train_nla_actor(self, rollout_id, rollout_data):
        # Strip variable-length critic tokens before get_data_iterator sees them.
        # Otherwise they'd be concatenated (data.py:253) then spread into
        # model(**kwargs). The pre-hook would catch them but why ship them through
        # the DataIterator at all.
        mm_inputs = rollout_data.get("multimodal_train_inputs") or []
        for mm in mm_inputs:
            if mm is not None:
                for k in CRITIC_ONLY_MM_KEYS:
                    mm.pop(k, None)
        # Validate injection count before pipeline. RL can produce None entries
        # (failed gen, filtered sample) — data.py:246 skips them in concat, but
        # the tokens still have injection markers → assertion in injection hook
        # fires on PP stage 0 only → stage 1 hangs on P2P recv → 10min NCCL wait.
        n_none = sum(1 for mm in mm_inputs if mm is None)
        if n_none > 0:
            n_vecs = sum(mm["nla_activation"].shape[0] for mm in mm_inputs if mm is not None)
            raise RuntimeError(
                f"rollout has {n_none}/{len(mm_inputs)} samples with multimodal_train_inputs=None "
                f"({n_vecs} vectors survive). Injection hook will mismatch. "
                f"Fix nla_generate to not emit None, or drop those samples from tokens too."
            )
        rollout_data = _truncate_to_cross_rank_min(
            rollout_data,
            self.parallel_state.dp_group,
            None if self.args.use_dynamic_batch_size else self.args.micro_batch_size,
        )
        # Parent's train_actor reads self._actor_critic_groups when use_critic —
        # our connect_actor_critic is a no-op so the attr doesn't exist.
        saved_use_critic = self.args.use_critic
        self.args.use_critic = False
        super().train_actor(rollout_id, rollout_data)
        self.args.use_critic = saved_use_critic
        self._nla_vectors_slot[0] = None

    def _train_nla_critic(self, rollout_id, rollout_data):
        # Online-RL critic (role=="critic"): rollout is actor output, swap to
        # critic tokens then filter failed <explanation> extractions.
        # Standalone SFT (role=="actor" + --nla-model-is-critic): rollout comes
        # from sft_critic.generate_rollout, already critic-formatted — skip swap.
        if self.role == "critic":
            rollout_data = _swap_rollout_to_critic_tokens(
                rollout_data, torch.cuda.current_device()
            )
            rollout_data = _truncate_to_cross_rank_min(
                rollout_data,
                self.parallel_state.dp_group,
                None if self.args.use_dynamic_batch_size else self.args.micro_batch_size,
            )

        data_iterator, num_microbatches = get_data_iterator(
            self.args, self.model, self.parallel_state, rollout_data
        )

        # Skip everything parent's train_critic does between get_data_iterator
        # and train(): no forward_only(get_values) (PPO value pass), no
        # sync_actor_critic_data (no NCCL group, connect_actor_critic was a no-op),
        # no compute_advantages_and_returns (NLA critic doesn't feed GAE).
        # loss_type already set to custom_loss in init(); don't let parent's
        # train_critic overwrite it to "value_loss".

        megatron_train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
            self.parallel_state,
        )

        log_perf_data(rollout_id, self.args, self.parallel_state)

    def update_weights(self):
        """Sync actor weights to SGLang, then dump embedding for nla_generate.

        The dump lets nla_generate re-embed locally (in the rollout worker)
        with fresh weights. TP-gather → full [vocab_hf, d] → CPU → atomic save.
        gather_embedding_for_dump is collective (all TP ranks all_gather) but
        only the first PP stage has the embedding. Gate participation on
        is_pipeline_first_stage; other PP stages just barrier.
        """
        # Parent's critic init early-returns at actor.py:108 without setting
        # self.weight_updater. Critic never calls update_weights in practice
        # (not in its training loop), but guard anyway.
        # debug_train_only (SFT): no SGLang rollout worker → no nla_generate
        # consumer for the dump. Skip — saves the TP all-gather + disk write.
        if (self._is_critic_model or self.args.save is None
                or self.args.debug_rollout_only or self.args.debug_train_only):
            return
        # Stock-miles bug (actor.py:99-153): with_ref + offload_train=False +
        # checkpoint resume leaves self.model=REF after init() (loads iter_N →
        # backups as "actor" → loads ref into live model → only restores if
        # offload_train). The line-35 update_weights() then broadcasts REF to
        # SGLang → first rollout uses SFT-base policy → 1-step critic_loss spike.
        # train_actor() switches to "actor" at top, so subsequent calls are fine.
        # Evidence: post-init _nla_rollout_embed.pt is bit-identical to SFT-base.
        if getattr(self, "_active_model_tag", "actor") != "actor":
            self._switch_model("actor")
        super().update_weights()
        weight = None
        if mpu.is_pipeline_first_stage():
            weight = gather_embedding_for_dump(self.args, self.model)
        if weight is not None and mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
            # Ship the 2GB tensor via a named ray actor (in-memory cross-node);
            # the driver fetches it and pushes to RolloutManager. The old NFS
            # file path stalled when the dir was under async-save bg-write
            # contention. torch.save(ObjectRef)→disk doesn't work either —
            # out-of-band refs bypass ray's borrow-tracking, ray.get hangs.
            ray.get(get_embed_store().set.remote(weight))
        dist.barrier(group=get_gloo_group())

    def save_model(self, rollout_id, force_sync=False):
        if self.args.debug_rollout_only or self.args.save is None:
            super().save_model(rollout_id, force_sync)
            return
        # Megatron's save_checkpoint (via megatron.training.checkpointing) writes
        # to {save}/iter_{rollout_id:07d}/. NOT +1 — different from FSDP's
        # checkpoint.py:211 convention. (Verified: actor.py:454 passes rollout_id
        # through model.py:699 to save_checkpoint unmodified.)
        iter_dir = Path(self.args.save) / f"iter_{rollout_id:07d}"

        # Sidecar BEFORE parent's save_model — parent ends with
        # destroy_process_groups() under --offload-train (actor.py:464-465).
        # (No critic-HF export — RL loop uses live critic_fwd; export via
        # tools/convert_to_hf.py post-hoc if needed.)
        if mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_first_stage():
            self._write_sidecar(str(iter_dir), rollout_id)
        dist.barrier()

        super().save_model(rollout_id, force_sync)

        # AFTER super(): with --async-save, super() FINALIZES the prior save
        # (writes .metadata + tracker) then starts this one async. tracker at
        # this point = PRIOR iter → push is one-behind. GCS always has the
        # last-completed checkpoint. Final step needs force_sync=True.
        # Prune is AFTER super for the same reason — the prior iter must be
        # finalized (have .metadata) before we delete the one before it. Prune
        # is also AFTER push: push reads tracker (= just-finalized iter),
        # prune deletes the one BEFORE that. With keep_n=2: ls=[finalized,
        # started-async], head -n -2 = nothing → keep both → no race. With
        # keep_n=1 (head -n -1 = the finalized one), push has it on GCS already
        # via the PRIOR step's push.
        # Barrier IMMEDIATELY after super() — any NFS op below (tracker read,
        # readdir, ls) hangs ~6-27min while the 660GB bg-write super() just
        # spawned saturates the server. With barrier-after, rank-0's hang
        # deadlocks the other 23 ranks. v11: this was the ~33min/save stall.
        dist.barrier()
        if mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_first_stage():
            keep_n = max(2, int(os.environ.get("NLA_KEEP_LOCAL", "2")))
            save_dir = self.args.save

            def _bg():
                self._maybe_background_push()
                if os.environ.get("NLA_BACKUP_REMOTE"):
                    prune = (f"ls -1d {save_dir}/iter_* 2>/dev/null | "
                             f"head -n -{keep_n} | xargs -r rm -rf")
                    subprocess.run(["bash", "-c", prune], check=False)

            threading.Thread(target=_bg, daemon=True).start()

    # Reuse FSDP's fire-and-forget GCS push (reads NLA_BACKUP_REMOTE env var).
    # Same semantics on both backends — one subprocess per save, --only-latest.
    _maybe_background_push = NLAFSDPActor._maybe_background_push

    def _write_sidecar(self, checkpoint_dir: str, rollout_id: int):
        cfg = self._nla_cfg
        if self._is_critic_model:
            cfg = replace(cfg, critic_num_layers=self.args.num_layers - 1)
        # args.hf_checkpoint was swapped to nla_critic_sidecar_source for the
        # critic in init() — so this is correct for both roles without branching.
        write_model_sidecar(
            checkpoint_dir, cfg,
            role="critic" if self._is_critic_model else "actor",
            stage="rl" if self.args.loss_type == "policy_loss" else "sl",
            base_checkpoint=self.args.hf_checkpoint,
            trained_on=[self.args.prompt_data] if self.args.prompt_data else [],
            parent_checkpoints=[self.args.hf_checkpoint],
            created_by="nla.megatron.train_actor.NLAMegatronActor",
            # Same training_args keys as NLAFSDPActor._write_sidecar — the
            # sidecar schema must not depend on the training backend. (num_layers
            # is already in the checkpoint's config.json; don't duplicate it here.)
            training_args={
                "rollout_id": rollout_id,
                "lr": self.args.lr,
                "loss_type": self.args.loss_type,
                "global_batch_size": self.args.global_batch_size,
            },
        )
