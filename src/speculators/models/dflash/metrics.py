"""Metrics and loss functions for DFlash draft model."""

import math
from functools import partial
from typing import Any, cast

import torch

from speculators.losses import (
    LossConfig,
    dflash_loss_decay,
    dpace_loss_decay,
    loss_function,
    tv_loss,
)
from speculators.losses import eager as eager_losses
from speculators.losses.utils import (
    BLOCK_LOSS_NAMES as _BLOCK_LOSS_NAMES,
)
from speculators.losses.utils import (
    _block_structured_loss_eager,
)
from speculators.models.metrics import (
    compute_accepted_length_counts,
    compute_accuracy_multi_step,
)


def tv_loss_fused_or_eager(
    logits: torch.Tensor, targets: torch.Tensor, implementation: str = "fused"
) -> torch.Tensor:
    """Dispatch the overlap, falling back to eager for fused requests on CPU."""
    if implementation == "eager" or not logits.is_cuda:
        return eager_losses.tv_loss(logits, targets)
    if implementation == "fused":
        return tv_loss(logits, targets)
    raise ValueError(f"Unknown loss implementation: {implementation}")


_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (eager_losses.kl_div_loss, 1.0)}

# Losses computed on the block layout. They need per-slot token ids and the
# blockwise cumprod/cumsum structure, so they cannot be expressed in the
# global (logits, targets) loss signature and are intercepted here.
#   wtv:  J = sum_k prod_{i<=k} a_i (per-slot overlap cumprod; the WTV proxy)
#   real: J = sum_k W_{k-1} a_k (path-weight prefix x marginalized last step)
#   realp: J = sum_k W_k (pure sampled-path estimator)
#   dblast: J = sum_{l in T_tau} (log r_l + log S_l) (DBLast AL surrogate,
#     arXiv:2608.05448 eq. 15; r_l = q(x_1:l)/p(x_1:l) prefix importance
#     ratio, S_l = sum_{i<=l} prod_{j<=i} A_j sub-block expected accepted
#     length, T_tau = prefixes through the first l with r_l < tau)
#   realv: L = (1 - μ/B) + (λ/B)V_doc + G (variance-constrained EAL-B;
#     V_doc is the document-grouped within-document variance, using the
#     same fixed-gamma EAL-B normalization as real)
#   realk: L = (1 - J_k/B) + G (short-budget EAL-B, with no direct
#     EAL reward beyond position k, CE still covers all B)
#   G = (ρ/B)[μ_0 - δ - μ]₊ (optional mean-preservation guard on the
#     full-μ estimate; δ is an absolute mean-drop threshold; ρ/λ use /B)
# with a_k = sum_v min(p, q) the single-slot acceptance overlap and
# b_i = min(1, q(y_i)/p(y_i)) the path weight at the data token y_i.

# VSD (arXiv:2602.05774, §4.3 M-step): weighted MLE toward high-acceptance
# paths on offline data — loss = −E_w̃[log Q], w̃ = κ/Σκ self-normalized,
# κ = Π_k min(1, p_k/q_k) fixed per step (Eq. 15–18). Bounded gradients.
# This reproduces the M-step surrogate, not Algorithm 1 (no EM/ARW/CAR).

# DBLast AL-loss truncation threshold (paper default, arXiv:2608.05448 §3.2)
_DBLAST_TAU = 0.1

_DEFAULT_REALV_LAMBDA = 0.05
_DEFAULT_REAL_GUARD_DELTA = 0.05
_DEFAULT_REAL_GUARD_RHO = 0.9
_LOSS_NAME_ALIASES = {
    "eala": "realp",
    "ealb": "real",
    "ealk": "realk",
    "ealv": "realv",
}


def _real_parameters(
    realv_lambda: float | None = None,
    realk_k: int | None = None,
    real_guard_on: bool | None = None,
    real_guard_mu0: float | None = None,
    real_guard_delta: float | None = None,
    real_guard_rho: float | None = None,
    ealv_lambda: float | None = None,
    ealk_k: int | None = None,
    eal_guard_on: bool | None = None,
    eal_guard_mu0: float | None = None,
    eal_guard_delta: float | None = None,
    eal_guard_rho: float | None = None,
) -> tuple[float, int | None, bool, float | None, float, float]:
    if realv_lambda is None:
        realv_lambda = ealv_lambda
    if realk_k is None:
        realk_k = ealk_k
    if real_guard_on is None:
        real_guard_on = eal_guard_on
    if real_guard_mu0 is None:
        real_guard_mu0 = eal_guard_mu0
    if real_guard_delta is None:
        real_guard_delta = eal_guard_delta
    if real_guard_rho is None:
        real_guard_rho = eal_guard_rho
    return (
        _DEFAULT_REALV_LAMBDA if realv_lambda is None else realv_lambda,
        realk_k,
        False if real_guard_on is None else real_guard_on,
        real_guard_mu0,
        _DEFAULT_REAL_GUARD_DELTA if real_guard_delta is None else real_guard_delta,
        _DEFAULT_REAL_GUARD_RHO if real_guard_rho is None else real_guard_rho,
    )


def _reject_duplicate_objectives(block_names: list[str]) -> None:
    canonical_names = [_LOSS_NAME_ALIASES.get(name, name) for name in block_names]
    if len(canonical_names) != len(set(canonical_names)):
        raise ValueError(
            f"Duplicate loss objective after resolving legacy aliases: {block_names}"
        )


def _block_loss_implementation(
    loss_config: LossConfig,
    block_names: list[str],
    needs_eal_terms: bool,
    use_draft_vocab: bool,
) -> str:
    if needs_eal_terms and use_draft_vocab:
        raise ValueError(
            "Block-structured trajectory losses require equal target and draft "
            "vocabularies; reduced draft vocabularies are not supported"
        )
    implementations = {
        "eager" if loss_config[name][0] is _block_structured_loss_eager else "fused"
        for name in block_names
    }
    if len(implementations) > 1:
        raise ValueError("All block losses must use the same loss implementation")
    return implementations.pop() if implementations else "fused"


def _gather_logp(
    logits: torch.Tensor,  # shape: [1, T, V]
    token_ids: torch.Tensor,  # shape: [1, T]
    chunk: int = 4096,
) -> torch.Tensor:  # shape: [1, T] fp32
    """log softmax(logits)[token_ids], chunked over positions to bound memory.

    Gradient chunks go through ``torch.utils.checkpoint`` so the fp32 casts
    are recomputed in backward instead of being kept for the whole window
    (a full [1, T, V] fp32 save at 248k vocab would be tens of GB).
    """

    def _chunk_logp(lg: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        lg = lg.float()
        return lg.gather(-1, ids.unsqueeze(-1)).squeeze(-1) - lg.logsumexp(dim=-1)

    outs = []
    for st in range(0, logits.shape[1], chunk):
        lg, ids = logits[:, st : st + chunk], token_ids[:, st : st + chunk]
        if logits.requires_grad:
            outs.append(
                torch.utils.checkpoint.checkpoint(
                    _chunk_logp, lg, ids, use_reentrant=False
                )
            )
        else:
            outs.append(_chunk_logp(lg, ids))
    return torch.cat(outs, dim=1)


def compute_eal_terms(
    logits: torch.Tensor,  # shape: [1, num_blocks*block_size, V]
    targets: torch.Tensor,  # shape: [1, num_blocks*block_size, V]
    token_ids: torch.Tensor,  # shape: [1, num_blocks*block_size]
    loss_mask: torch.Tensor,  # shape: [1, num_blocks*block_size]
    block_size: int,
    start_pos: int = 0,
    document_ids: torch.Tensor | None = None,
    loss_implementation: str = "fused",
) -> dict[str, torch.Tensor]:
    """Per-block EAL quantities on target-sampled prediction trajectories.

    ``start_pos`` removes a non-prediction bonus anchor before computing the
    window. Document boundaries and the first masked prediction terminate each
    trajectory. Outputs have shape [num_blocks, block_size - start_pos].
    """
    if block_size <= 0 or not 0 <= start_pos < block_size:
        raise ValueError("start_pos must leave at least one prediction slot")
    if logits.shape[1] % block_size != 0:
        raise ValueError("The sequence length must be divisible by block_size")
    if logits.shape[-1] != targets.shape[-1]:
        raise ValueError(
            "REAL trajectory losses require a shared target/draft vocabulary"
        )
    if token_ids.shape != loss_mask.shape:
        raise ValueError("token_ids and loss_mask must have the same shape")
    if document_ids is not None and document_ids.shape != token_ids.shape:
        raise ValueError("document_ids must have one id per token")
    if token_ids.numel() and (
        token_ids.min().item() < 0
        or token_ids.max().item() >= logits.shape[-1]
        or token_ids.max().item() >= targets.shape[-1]
    ):
        raise ValueError("token_ids must index the shared target/draft vocabulary")

    def prediction_slots(tensor: torch.Tensor) -> torch.Tensor:
        blocks = tensor.reshape(-1, block_size, *tensor.shape[2:])
        return blocks[:, start_pos:].reshape(1, -1, *tensor.shape[2:])

    logits = prediction_slots(logits)
    targets = prediction_slots(targets)
    token_ids = prediction_slots(token_ids)
    valid = prediction_slots(loss_mask).bool().view(-1, block_size - start_pos)
    if document_ids is not None:
        doc_blocks = document_ids.reshape(-1, block_size)
        within_document = (doc_blocks == doc_blocks[:, :1]).to(loss_mask.dtype)
        valid = valid & prediction_slots(within_document).bool().view_as(valid)
    valid = valid.cumprod(dim=-1).bool()
    block_size -= start_pos

    lq = _gather_logp(logits, token_ids)
    with torch.no_grad():
        lp = _gather_logp(targets, token_ids)
    log_ratio = (lq - lp).view(-1, block_size)
    log_ratio = torch.where(valid, log_ratio, torch.zeros_like(log_ratio))
    log_b = log_ratio.clamp(max=0.0)
    a = (1.0 - tv_loss_fused_or_eager(logits, targets, loss_implementation)).view(
        -1, block_size
    )
    a = torch.where(valid, a, torch.zeros_like(a))
    log_prefix = torch.where(valid, log_b.cumsum(dim=-1), torch.zeros_like(log_b))
    log_prev = torch.cat(
        [torch.zeros_like(log_prefix[:, :1]), log_prefix[:, :-1]], dim=-1
    )
    log_prev = torch.where(valid, log_prev, torch.zeros_like(log_prev))
    # DBLast AL-loss terms (arXiv:2608.05448):
    #   log_r_prefix[l] = log(q(x_1:l)/p(x_1:l)) — raw (unclamped) prefix
    #   importance-ratio log; A_j = min(1, p(y_j)/q(y_j)) per-token acceptance
    #   probability on the trajectory. Masked slots get A = 1 (no decay) and
    #   log_r = 0 (excluded downstream by the retained-set mask).
    log_r_prefix = torch.where(
        valid, log_ratio.cumsum(dim=-1), torch.zeros_like(log_ratio)
    )
    a_tok = (lp - lq).clamp(max=0.0).exp().view(-1, block_size)
    a_tok = torch.where(valid, a_tok, torch.ones_like(a_tok))
    return {
        "a": a,
        "b": log_b.exp(),
        "log_prefix": log_prefix,
        "log_prev": log_prev,
        "log_r_prefix": log_r_prefix,
        "a_tok": a_tok,
        "valid": valid,
        "log_q": torch.where(
            valid,
            lq.view(-1, block_size),
            torch.zeros_like(lq.view(-1, block_size)),
        ),
        "log_p": torch.where(
            valid,
            lp.view(-1, block_size),
            torch.zeros_like(lp.view(-1, block_size)),
        ),
    }


def _realv_parts(
    terms: dict[str, torch.Tensor],
    block_size: int,
    group_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return EAL-B mean and document-grouped within-document variance.

    μ is the EAL-B objective (same u_i terms as real), and
    M₂ = Σ(2i-1)u_i is the odd-weight second-moment term. Per-block
    ``group_ids`` define documents: each group contributes its biased variance
    M₂_g − μ_g², and non-empty groups are averaged with equal weight. A
    single-block group contributes its own M₂ − e² term, a sampling-estimate
    component rather than the true acceptance-length variance.

    ``group_ids=None`` retains the legacy ungrouped U-statistic fallback for
    compatibility; it mixes documents and is not REALV's grouped semantics.
    """
    a = terms["a"]
    valid = terms["valid"]
    block_valid = valid.any(dim=-1).to(a.dtype)
    u = terms["log_prev"].exp() * a
    e = u.sum(dim=-1)
    odd = 2 * torch.arange(1, block_size + 1, device=u.device) - 1
    d = (u * odd).sum(dim=-1)
    m = block_valid.sum().clamp(min=1.0)
    mu = (e * block_valid).sum() / m
    if group_ids is None:
        m2 = (d * block_valid).sum() / m
        s = (e * block_valid).sum()
        sq = (e.square() * block_valid).sum()
        mu2 = (s.square() - sq) / (m * (m - 1.0)).clamp(min=1.0)
        return mu, m2 - mu2
    # Document-grouped within-document variance; non-empty groups have equal weight.
    g = group_ids.to(device=e.device, dtype=torch.long)
    if g.numel() != e.numel():
        raise ValueError(
            f"group_ids must be per-block (got {g.numel()}, "
            f"expected {e.numel()} blocks)"
        )
    order = torch.argsort(g)
    gs, ge, gd, gv = g[order], e[order], d[order], block_valid[order]
    uniq, inv = torch.unique_consecutive(gs, return_inverse=True)
    n_groups = uniq.numel()
    esum = torch.zeros(n_groups, device=e.device, dtype=e.dtype).scatter_add_(
        0, inv, ge * gv
    )
    dsum = torch.zeros(n_groups, device=e.device, dtype=d.dtype).scatter_add_(
        0, inv, gd * gv
    )
    vsum = torch.zeros(n_groups, device=e.device, dtype=e.dtype).scatter_add_(
        0, inv, gv
    )
    n = vsum.clamp(min=1.0)
    var_g = (dsum / n - (esum / n).square()) * (vsum > 0).to(e.dtype)
    gcount = (vsum > 0).sum().clamp(min=1.0)
    return mu, var_g.sum() / gcount


def _eal_guard(
    mu: torch.Tensor,
    block_size: int,
    mu0: float | None,
    delta: float,
    rho: float,
    enabled: bool,
) -> torch.Tensor:
    """Return the optional mean-preservation guard for the current mean estimate."""
    if not enabled:
        return torch.zeros((), device=mu.device, dtype=mu.dtype)
    if mu0 is None:
        raise ValueError("real_guard_mu0 is required when real_guard_on=True")
    return (rho / block_size) * (mu0 - delta - mu).clamp(min=0.0)


def _realk_parts(
    terms: dict[str, torch.Tensor],
    block_size: int,
    k: int | None,
    reference_mu: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return short-budget and full-sum EAL-B estimates over valid blocks."""
    a = terms["a"]
    valid = terms["valid"]
    block_valid = valid.any(dim=-1).to(a.dtype)
    u = terms["log_prev"].exp() * a
    m = block_valid.sum().clamp(min=1.0)
    mu = (u.sum(dim=-1) * block_valid).sum() / m
    if k is None:
        reference = reference_mu if reference_mu is not None else float(mu.detach())
        k = min(max(math.ceil(reference), 1), block_size)
    elif not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= block_size:
        raise ValueError(f"realk_k must be an integer in [1, {block_size}]")
    jk = (u[..., :k].sum(dim=-1) * block_valid).sum() / m
    return jk, mu


def _block_eal_loss(
    terms: dict[str, torch.Tensor],
    kind: str,
    block_size: int,
    group_ids: torch.Tensor | None = None,
    realv_lambda: float | None = None,
    realk_k: int | None = None,
    real_guard_on: bool | None = None,
    real_guard_mu0: float | None = None,
    real_guard_delta: float | None = None,
    real_guard_rho: float | None = None,
    ealv_lambda: float | None = None,
    ealk_k: int | None = None,
    eal_guard_on: bool | None = None,
    eal_guard_mu0: float | None = None,
    eal_guard_delta: float | None = None,
    eal_guard_rho: float | None = None,
) -> torch.Tensor:
    """1 - J/gamma per block, averaged over valid blocks (fixed-gamma norm).

    ``dblast`` instead minimizes the DBLast AL surrogate (arXiv:2608.05448
    eq. 15-16): L = -J_AL with J_AL = sum_{l in T_tau}(log r_l + log S_l),
    averaged over valid blocks (paper-faithful raw form, not normalized to
    [0, 1] like the 1 - J/block_size losses).

    ``realv`` adds the document-grouped within-document variance penalty and
    guard on top of the same fixed-gamma EAL-B form:
    L = (1 - μ/B) + (λ/B)V_doc + G, so its EAL part is identical to ``real``.

    ``realk`` rewards only the first ``realk_k`` positions plus the guard:
    L = (1 - J_k/B) + G (same 1/B normalization as real).

    ``vsd`` minimizes the negative step-level ELBO with fixed κ_t,
    self-normalized over offline paths.
    """
    kind = _LOSS_NAME_ALIASES.get(kind, kind)
    (
        realv_lambda,
        realk_k,
        real_guard_on,
        real_guard_mu0,
        real_guard_delta,
        real_guard_rho,
    ) = _real_parameters(
        realv_lambda,
        realk_k,
        real_guard_on,
        real_guard_mu0,
        real_guard_delta,
        real_guard_rho,
        ealv_lambda,
        ealk_k,
        eal_guard_on,
        eal_guard_mu0,
        eal_guard_delta,
        eal_guard_rho,
    )
    a = terms["a"]
    valid = terms["valid"]
    block_valid = valid.any(dim=-1).to(a.dtype)
    if kind == "dblast":
        log_r = terms["log_r_prefix"]
        s = terms["a_tok"].cumprod(dim=-1).cumsum(dim=-1)
        log_s = (s + 1e-8).log()
        with torch.no_grad():
            below = log_r < math.log(_DBLAST_TAU)
            idx = below.to(torch.int64).argmax(dim=-1)
            ell_tau = torch.where(below.any(dim=-1), idx, block_size - 1)
            pos = torch.arange(block_size, device=log_r.device).unsqueeze(0)
            retained = (pos <= ell_tau.unsqueeze(-1)) & valid
        j = ((log_r + log_s) * retained.to(log_r.dtype)).sum(dim=-1)
        count = block_valid.sum()
        loss = -(j * block_valid).sum() / count.clamp(min=1.0)
        return torch.where(
            count > 0, loss, torch.zeros((), device=a.device, dtype=a.dtype)
        )
    if kind == "realv":
        mu, var = _realv_parts(terms, block_size, group_ids)
        count = block_valid.sum()
        loss = (
            1.0
            - mu / block_size
            + (realv_lambda / block_size) * var
            + _eal_guard(
                mu,
                block_size,
                real_guard_mu0,
                real_guard_delta,
                real_guard_rho,
                real_guard_on,
            )
        )
        return torch.where(
            count > 0, loss, torch.zeros((), device=a.device, dtype=a.dtype)
        )
    if kind == "realk":
        jk, mu = _realk_parts(terms, block_size, realk_k, real_guard_mu0)
        count = block_valid.sum()
        loss = (
            1.0
            - jk / block_size
            + _eal_guard(
                mu,
                block_size,
                real_guard_mu0,
                real_guard_delta,
                real_guard_rho,
                real_guard_on,
            )
        )
        return torch.where(
            count > 0, loss, torch.zeros((), device=a.device, dtype=a.dtype)
        )
    if kind == "vsd":
        # VSD M-step (arXiv:2602.05774 §4.3): offline weighted-MLE surrogate
        # toward high-acceptance paths — maximize E_z~posterior[log Q] with
        # posterior ∝ κ·P. Offline teacher-forcing data comes from P, so the
        # self-normalized weight is w̃ = κ = Π_k min(1, p_k/q_k) (detached,
        # bounded in [0, 1]; κ fixed per the paper's step-level surrogate,
        # Eq. 15–18). NOTE: this is the paper's M-step surrogate, NOT the
        # direct gradient of the Eq. 18 ELBO (score-function form, which
        # diverges in practice); describe it as an offline proxy.
        # loss = −(Σ w̃·log Q / Σw̃) / B — bounded gradients, and the 1/B
        # normalization puts the per-token scale on par with CE and the
        # other block losses (same convention as real/realv/realk).
        # The score-function form is a different objective and can be unstable.
        log_q = terms["log_q"].sum(dim=-1)
        log_kappa = (
            (terms["log_p"] - terms["log_q"]).clamp(max=0.0).sum(dim=-1).detach()
        )
        count = block_valid.sum()
        log_w = log_kappa.masked_fill(~valid.any(dim=-1), -torch.inf)
        # A common scale cancels in w/wsum; the largest valid weight is 1.
        shift = torch.where(count > 0, log_w.max(), torch.zeros_like(count))
        w = (log_w - shift).exp().detach()
        wsum = (w * block_valid).sum()
        loss = -(log_q * w * block_valid).sum() / wsum.clamp(min=1.0) / block_size
        return torch.where(
            count > 0, loss, torch.zeros((), device=a.device, dtype=a.dtype)
        )
    if kind == "wtv":
        j = a.cumprod(dim=-1).sum(dim=-1)
    elif kind == "real":
        j = (terms["log_prev"].exp() * a).sum(dim=-1)
    else:  # realp
        j = (terms["log_prefix"].exp() * valid.to(a.dtype)).sum(dim=-1)
    count = block_valid.sum()
    mean_j = (j * block_valid).sum() / count.clamp(min=1.0)
    loss = 1.0 - mean_j / block_size
    return torch.where(count > 0, loss, torch.zeros((), device=a.device, dtype=a.dtype))


def _compute_standard_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_idx: torch.Tensor,
    loss_config: LossConfig,
    multi: bool,
    per_position_loss_weight: str,
    block_size: int,
    gamma: float,
    dpace_alpha: float,
    sample_from_anchor: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    term_losses: dict[str, torch.Tensor] = {}
    if loss_config:
        if per_position_loss_weight == "dpace":
            decay_fn = partial(
                dpace_loss_decay,
                loss_mask=loss_mask,
                block_size=block_size,
                dpace_alpha=dpace_alpha,
            )
        else:
            decay_fn = partial(
                dflash_loss_decay, gamma=gamma, sample_from_anchor=sample_from_anchor
            )
        loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
        for name, (fn, weight) in loss_config.items():
            term = loss_function(
                logits,
                targets,
                loss_mask,
                pos_idx,
                loss_fn=fn,
                decay_fn=decay_fn,
            )
            if multi:
                term_losses[f"{name}_loss"] = term.detach()
            loss = loss + weight * term
    else:
        loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
    return loss, term_losses


def _add_eal_diagnostics(
    metrics: dict[str, Any],
    eal_terms: dict[str, torch.Tensor],
    block_size: int,
    start_pos: int,
) -> None:
    a_blk = eal_terms["a"]
    b_blk = eal_terms["b"]
    valid_blk = eal_terms["valid"]
    fvalid = valid_blk.to(a_blk.dtype)
    block_valid = valid_blk.any(dim=-1).to(a_blk.dtype)

    j_b = (eal_terms["log_prev"].exp() * a_blk).sum(dim=-1)
    j_wtv = a_blk.cumprod(dim=-1).sum(dim=-1)
    metrics["eal_j_b_sum"] = (j_b * block_valid).sum()
    metrics["eal_j_b_total"] = block_valid.sum()
    metrics["eal_j_wtv_sum"] = (j_wtv * block_valid).sum()
    metrics["eal_j_wtv_total"] = block_valid.sum()
    for pos in range(start_pos, start_pos + block_size):
        index = pos - start_pos
        metrics[f"eal_a_pos_{pos}_sum"] = (a_blk[:, index] * fvalid[:, index]).sum()
        metrics[f"eal_a_pos_{pos}_total"] = fvalid[:, index].sum()
        metrics[f"eal_b_pos_{pos}_sum"] = (b_blk[:, index] * fvalid[:, index]).sum()
        metrics[f"eal_b_pos_{pos}_total"] = fvalid[:, index].sum()
    # E[b_k * a_{k+1}] per adjacent pair; combine with the per-position
    # means downstream for Cov(b_k, a_{k+1}), the WTV-proxy bias term.
    for pos in range(start_pos, start_pos + block_size - 1):
        index = pos - start_pos
        pair = fvalid[:, index] * fvalid[:, index + 1]
        metrics[f"eal_ba_next_pos_{pos}_sum"] = (
            b_blk[:, index] * a_blk[:, index + 1] * pair
        ).sum()
        metrics[f"eal_ba_next_pos_{pos}_total"] = pair.sum()


def _add_eal_stability_diagnostics(
    term_losses: dict[str, torch.Tensor],
    eal_terms: dict[str, torch.Tensor],
    block_names: list[str],
    block_size: int,
    block_group_ids: torch.Tensor | None,
    realk_k: int | None,
    real_guard_on: bool,
    real_guard_mu0: float | None,
    real_guard_delta: float,
    real_guard_rho: float,
) -> None:
    realv_name = "realv" if "realv" in block_names else "ealv"
    if realv_name in block_names:
        realv_mu, realv_var = _realv_parts(eal_terms, block_size, block_group_ids)
        term_losses[f"{realv_name}_mu"] = realv_mu.detach()
        term_losses[f"{realv_name}_var"] = realv_var.detach()
        term_losses[f"{realv_name}_guard"] = _eal_guard(
            realv_mu,
            block_size,
            real_guard_mu0,
            real_guard_delta,
            real_guard_rho,
            real_guard_on,
        ).detach()
    realk_name = "realk" if "realk" in block_names else "ealk"
    if realk_name in block_names:
        realk_jk, realk_mu = _realk_parts(
            eal_terms, block_size, realk_k, real_guard_mu0
        )
        term_losses[f"{realk_name}_jk"] = realk_jk.detach()
        term_losses[f"{realk_name}_mu"] = realk_mu.detach()
        term_losses[f"{realk_name}_guard"] = _eal_guard(
            realk_mu,
            block_size,
            real_guard_mu0,
            real_guard_delta,
            real_guard_rho,
            real_guard_on,
        ).detach()


def compute_metrics(
    logits: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, num_anchors*block_size]
    block_size: int = 1,
    gamma: float = 4.0,
    loss_config: LossConfig | None = None,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    sample_from_anchor: bool = False,
    token_ids: torch.Tensor | None = None,  # shape: [1, num_anchors*block_size]
    block_group_ids: torch.Tensor | None = None,  # shape: [num_anchors]
    block_document_ids: torch.Tensor
    | None = None,  # shape: [1, num_anchors*block_size]
    use_draft_vocab: bool = False,
    eal_diagnostics: bool | None = None,
    realv_lambda: float | None = None,
    realk_k: int | None = None,
    real_guard_on: bool | None = None,
    real_guard_mu0: float | None = None,
    real_guard_delta: float | None = None,
    real_guard_rho: float | None = None,
    ealv_lambda: float | None = None,
    ealk_k: int | None = None,
    eal_guard_on: bool | None = None,
    eal_guard_mu0: float | None = None,
    eal_guard_delta: float | None = None,
    eal_guard_rho: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute loss and accuracy metrics for draft model predictions.

    Args:
        logits: Model logits [1, T, V]
        targets: Target logits [1, T, V]
        loss_mask: Binary mask [1, T]
        block_size: Block size for per-position metrics
        gamma: Temperature for exponential decay in loss weighting
        loss_config: Mapping of ``{name: (loss_fn, weight)}``
        per_position_loss_weight: Weighting option for per-position block-drafting loss
        dpace_alpha: Smoothing constant for D-Pace loss weighting
        token_ids: Data token id drafted at each slot, required by the
            block-structured losses (``wtv``/``real``/``realp``) and by the
            EAL diagnostics
        block_group_ids: Per-block document id used for REALV variance grouping.
        block_document_ids: Document id at every block slot. Trajectories stop
            when the document changes; ``None`` assumes one document per block.
        use_draft_vocab: Whether the draft uses a reduced vocabulary. Block-
            structured trajectory losses require a shared target/draft vocabulary.
        eal_diagnostics: Compute path diagnostics. None enables them when block
            losses are selected and disables them for standard-only losses.
        realv_lambda: Weight of the document-grouped variance penalty in REALV
        realk_k: Number of prediction positions rewarded by REALK. If None,
            derive k from ``real_guard_mu0`` when supplied, otherwise from the
            current EAL-B mean, clamped to the prediction window.
        real_guard_on: Enable the mean-preservation guard. Disabled by default.
        real_guard_mu0: Reference acceptance-length mean. Required when
            ``real_guard_on=True``; also used to derive ``realk_k`` when supplied.
        real_guard_delta: Absolute mean drop tolerated before the guard activates.
        real_guard_rho: Guard strength per unit of mean shortfall, normalized by
            the prediction window.

        The corresponding eal-prefixed names are accepted as legacy aliases.

    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict contains:
            - loss: Scalar loss value
            - full_acc: Overall accuracy
            - position {i} acc: Accuracy at position i within blocks
            - eal: Expected Accepted Length (headline speculative-decoding metric)
            - eal_j_b / eal_j_wtv: exact-EAL (B estimator) vs WTV-proxy window scores
            - eal_a_pos {i} / eal_b_pos {i}: per-position mean overlap / path weight
            - eal_ba_next_pos {i}: E[b_i * a_{i+1}] for downstream Cov(b, a)
    """
    if loss_config is None:
        loss_config = _DEFAULT_LOSS_CONFIG
    (
        realv_lambda,
        realk_k,
        real_guard_on,
        real_guard_mu0,
        real_guard_delta,
        real_guard_rho,
    ) = _real_parameters(
        realv_lambda,
        realk_k,
        real_guard_on,
        real_guard_mu0,
        real_guard_delta,
        real_guard_rho,
        ealv_lambda,
        ealk_k,
        eal_guard_on,
        eal_guard_mu0,
        eal_guard_delta,
        eal_guard_rho,
    )
    seq_len = logits.shape[1]
    pos_idx = torch.arange(seq_len, device=logits.device) % block_size
    pos_idx = pos_idx.unsqueeze(0)  # shape: [1, T]

    multi = len(loss_config) > 1
    block_names = [n for n in loss_config if n in _BLOCK_LOSS_NAMES]
    _reject_duplicate_objectives(block_names)
    if block_names and token_ids is None:
        raise ValueError(
            f"losses {block_names} are block-structured and require token_ids"
        )
    start_pos = 0 if sample_from_anchor else 1
    prediction_length = block_size - start_pos
    path_diagnostics = bool(block_names) if eal_diagnostics is None else eal_diagnostics
    needs_eal_terms = token_ids is not None and (bool(block_names) or path_diagnostics)
    block_loss_implementation = _block_loss_implementation(
        loss_config, block_names, needs_eal_terms, use_draft_vocab
    )

    std_config = {
        name: fn_weight
        for name, fn_weight in loss_config.items()
        if name not in _BLOCK_LOSS_NAMES
    }
    loss, term_losses = _compute_standard_losses(
        logits,
        targets,
        loss_mask,
        pos_idx,
        std_config,
        multi,
        per_position_loss_weight,
        block_size,
        gamma,
        dpace_alpha,
        sample_from_anchor,
    )

    eal_terms = None
    if needs_eal_terms:
        eal_terms = compute_eal_terms(
            logits,
            targets,
            cast("torch.Tensor", token_ids),
            loss_mask,
            block_size,
            start_pos=start_pos,
            document_ids=block_document_ids,
            loss_implementation=block_loss_implementation,
        )
    eal_terms_for_loss = cast("dict[str, torch.Tensor]", eal_terms)
    for name in block_names:
        _, weight = loss_config[name]
        term = _block_eal_loss(
            eal_terms_for_loss,
            name,
            prediction_length,
            block_group_ids,
            realv_lambda=realv_lambda,
            realk_k=realk_k,
            real_guard_on=real_guard_on,
            real_guard_mu0=real_guard_mu0,
            real_guard_delta=real_guard_delta,
            real_guard_rho=real_guard_rho,
        )
        if multi:
            term_losses[f"{name}_loss"] = term.detach()
        loss = loss + weight * term
    _add_eal_stability_diagnostics(
        term_losses,
        eal_terms_for_loss,
        block_names,
        prediction_length,
        block_group_ids,
        realk_k,
        real_guard_on,
        real_guard_mu0,
        real_guard_delta,
        real_guard_rho,
    )
    if "vsd" in block_names:
        block_valid_v = (
            eal_terms_for_loss["valid"].any(dim=-1).to(eal_terms_for_loss["a"].dtype)
        )
        log_q_v = eal_terms_for_loss["log_q"].sum(dim=-1)
        log_p_v = eal_terms_for_loss["log_p"].sum(dim=-1)
        log_kappa_v = (
            (eal_terms_for_loss["log_p"] - eal_terms_for_loss["log_q"])
            .clamp(max=0.0)
            .sum(dim=-1)
        )
        elbo_v = log_kappa_v - log_q_v + log_p_v
        term_losses["vsd_elbo"] = (
            elbo_v * block_valid_v
        ).sum() / block_valid_v.sum().clamp(min=1.0)
        kappa_v = log_kappa_v.exp()
        term_losses["vsd_kappa_mean"] = (
            kappa_v * block_valid_v
        ).sum() / block_valid_v.sum().clamp(min=1.0)

    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)

    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )

    ones = torch.tensor(1.0, device=logits.device)
    metrics: dict[str, Any] = {}
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    metrics["full_acc_sum"] = correct_per_pos[start_pos:].sum()
    metrics["full_acc_total"] = total_per_pos[start_pos:].sum()

    for pos in range(start_pos, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

    eal_sum, eal_total = compute_accepted_length_counts(
        (pred_ids == target_ids).reshape(-1, block_size)[:, start_pos:],
        loss_mask.to(torch.bool).reshape(-1, block_size)[:, start_pos:],
    )
    metrics["eal_sum"] = eal_sum
    metrics["eal_total"] = eal_total

    if eal_terms is not None and path_diagnostics:
        _add_eal_diagnostics(metrics, eal_terms, prediction_length, start_pos)

    return loss, metrics
