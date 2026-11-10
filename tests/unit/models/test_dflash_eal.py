"""Unit tests for block-structured EAL losses in DFlash metrics.

The value/gradient references come from the two-step counterexample in
the REAL paper:
vocab {A, B}, gamma = 2, p1 = (0.4, 0.6), q1 = (0.8, 0.2), and a shared
second-step draft q2 = (theta, 1 - theta) against p2(hA) = (0.9, 0.1),
p2(hB) = (0.1, 0.9). There, J2 = 0.86 + 0.20*theta exactly, while the WTV
proxy gives 1.02 - 0.12*theta -- at theta = 0.5 the values coincide (0.96)
but the gradients have opposite signs.
"""

from typing import Any, cast

import pytest
import torch
from torch.nn import functional

from speculators.losses import resolve_loss_config as _resolve_loss_config
from speculators.models.dflash import metrics as dflash_metrics
from speculators.models.dflash.metrics import (
    _block_eal_loss,
    compute_eal_terms,
    compute_metrics,
    tv_loss_fused_or_eager,
)


def resolve_loss_config(spec):
    return _resolve_loss_config(spec, "eager")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _log(probs: list[float]) -> torch.Tensor:
    return torch.log(torch.tensor(probs, dtype=torch.float32))


def _terms_from_batch(batch: Any) -> dict[str, torch.Tensor]:
    logits, targets, token_ids, loss_mask = batch
    return compute_eal_terms(logits, targets, token_ids, loss_mask, 2)


def _counterexample_batch(theta_raw: torch.Tensor):
    """The section-8 counterexample as 50 blocks of size 2 (path weights
    0.36/0.04/0.06/0.54 realized as 18/2/3/27 replicated windows)."""
    q1 = _log([0.8, 0.2])
    p1 = _log([0.4, 0.6])
    p2 = {"A": _log([0.9, 0.1]), "B": _log([0.1, 0.9])}
    q2 = torch.stack([theta_raw, theta_raw * 0.0])

    # (first token, second token, count)
    windows = [(0, 0, 18), (0, 1, 2), (1, 0, 3), (1, 1, 27)]
    logits_rows, target_rows, ids = [], [], []
    for y1, y2, count in windows:
        branch = "A" if y1 == 0 else "B"
        for _ in range(count):
            logits_rows += [q1, q2]
            target_rows += [p1, p2[branch]]
            ids += [y1, y2]
    logits = torch.stack(logits_rows).unsqueeze(0)  # [1, 100, 2]
    targets = torch.stack(target_rows).unsqueeze(0)
    token_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # [1, 100]
    loss_mask = torch.ones_like(token_ids)
    return logits, targets, token_ids, loss_mask


def _ref_compute_terms(draft_logits, teacher_logits, cont_ids, valid):
    """Compute path weights and overlap for a block trajectory."""
    log_q = functional.log_softmax(draft_logits.float(), dim=-1)
    log_p = functional.log_softmax(teacher_logits.float(), dim=-1).detach()
    ids = cont_ids.unsqueeze(-1)
    lq = log_q.gather(-1, ids).squeeze(-1)
    lp = log_p.gather(-1, ids).squeeze(-1)
    log_b = (lq - lp).clamp(max=0.0)
    log_b = torch.where(valid, log_b, torch.zeros_like(log_b))
    a = torch.minimum(log_q.exp(), log_p.exp()).sum(-1)
    a = torch.where(valid, a, torch.zeros_like(a))
    prefix = log_b.cumsum(dim=-1)
    prev = torch.cat([torch.zeros_like(prefix[:, :1]), prefix[:, :-1]], dim=-1)
    return lq, a, prefix, prev


# ---------------------------------------------------------------------------
# section-8 counterexample: values and gradients
# ---------------------------------------------------------------------------


class TestCounterexample:
    @pytest.mark.parametrize(
        ("loss_name", "expected_grad"),
        [("real", -0.025), ("realp", -0.025), ("wtv", 0.015)],
    )
    def test_value_and_grad(self, loss_name, expected_grad):
        theta_raw = torch.tensor(0.0, requires_grad=True)  # theta = 0.5
        logits, targets, token_ids, loss_mask = _counterexample_batch(theta_raw)
        loss, _ = compute_metrics(
            logits,
            targets,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config(loss_name),
            sample_from_anchor=True,
            token_ids=token_ids,
        )
        # all three objectives coincide at J = 0.96 when theta = 0.5
        assert loss.item() == pytest.approx(1.0 - 0.96 / 2, abs=1e-4)
        loss.backward()
        # dJ/dtheta is +0.20 for real/realp but -0.12 for the WTV proxy;
        # loss = 1 - J/2 and dtheta/dtheta_raw = 0.25
        assert theta_raw.grad is not None
        assert theta_raw.grad.item() == pytest.approx(expected_grad, abs=1e-4)

    def test_diagnostics(self):
        theta_raw = torch.tensor(0.0, requires_grad=False)
        logits, targets, token_ids, loss_mask = _counterexample_batch(theta_raw)
        _, metrics = compute_metrics(
            logits,
            targets,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config("real"),
            sample_from_anchor=True,
            token_ids=token_ids,
        )
        assert (metrics["eal_j_b_sum"] / metrics["eal_j_b_total"]).item() == (
            pytest.approx(0.96, abs=1e-4)
        )
        assert (metrics["eal_j_wtv_sum"] / metrics["eal_j_wtv_total"]).item() == (
            pytest.approx(0.96, abs=1e-4)
        )
        assert (
            metrics["eal_a_pos_0_sum"] / metrics["eal_a_pos_0_total"]
        ).item() == pytest.approx(0.6, abs=1e-4)
        assert (
            metrics["eal_b_pos_0_sum"] / metrics["eal_b_pos_0_total"]
        ).item() == pytest.approx(0.6, abs=1e-4)
        # E[b_1 * a_2] = 0.36; with E[b_1] = E[a_2] = 0.6 the covariance
        # term Cov(b_1, a_2) = 0.36 - 0.36 = 0 at theta = 0.5
        assert (
            metrics["eal_ba_next_pos_0_sum"] / metrics["eal_ba_next_pos_0_total"]
        ).item() == pytest.approx(0.36, abs=1e-4)


# ---------------------------------------------------------------------------
# equivalence with the toy-validated reference implementation
# ---------------------------------------------------------------------------


class TestToyReferenceEquivalence:
    def _random_batch(self, num_blocks=5, block_size=3, vocab=11, seed=0):
        g = torch.Generator().manual_seed(seed)
        logits = torch.randn(1, num_blocks * block_size, vocab, generator=g)
        targets = torch.randn(1, num_blocks * block_size, vocab, generator=g)
        token_ids = torch.randint(0, vocab, (1, num_blocks * block_size), generator=g)
        loss_mask = torch.ones_like(token_ids)
        # block 1 fully masked, one masked slot inside block 2
        loss_mask[0, block_size : 2 * block_size] = 0
        loss_mask[0, 2 * block_size + 1] = 0
        return logits, targets, token_ids, loss_mask

    def test_terms_match(self):
        logits, targets, token_ids, loss_mask = self._random_batch()
        block_size = 3
        terms = compute_eal_terms(logits, targets, token_ids, loss_mask, block_size)

        n = logits.shape[1] // block_size
        ref_valid = loss_mask.view(n, block_size).bool().cumprod(dim=-1).bool()
        _, ref_a, ref_prefix, ref_prev = _ref_compute_terms(
            logits.view(n, block_size, -1),
            targets.view(n, block_size, -1),
            token_ids.view(n, block_size),
            ref_valid,
        )
        ref_prefix = torch.where(ref_valid, ref_prefix, torch.zeros_like(ref_prefix))
        ref_prev = torch.where(ref_valid, ref_prev, torch.zeros_like(ref_prev))
        assert torch.allclose(terms["a"], ref_a, atol=1e-5)
        assert torch.allclose(terms["log_prefix"], ref_prefix, atol=1e-5)
        assert torch.allclose(terms["log_prev"], ref_prev, atol=1e-5)

    def test_losses_match(self):
        g = torch.Generator().manual_seed(1)
        n, block_size, vocab = 4, 3, 11
        logits = torch.randn(1, n * block_size, vocab, generator=g)
        targets = torch.randn(1, n * block_size, vocab, generator=g)
        token_ids = torch.randint(0, vocab, (1, n * block_size), generator=g)
        loss_mask = torch.ones_like(token_ids)  # all valid: means coincide
        terms = compute_eal_terms(logits, targets, token_ids, loss_mask, block_size)

        ref_valid = loss_mask.view(n, block_size).bool()
        _, ref_a, ref_prefix, ref_prev = _ref_compute_terms(
            logits.view(n, block_size, -1),
            targets.view(n, block_size, -1),
            token_ids.view(n, block_size),
            ref_valid,
        )
        ref = {
            "wtv": 1.0 - (ref_a.cumprod(dim=-1).sum(-1) / block_size).mean(),
            "real": 1.0 - ((ref_prev.exp() * ref_a).sum(-1) / block_size).mean(),
            "realp": 1.0 - ((ref_prefix.exp() * ref_valid).sum(-1) / block_size).mean(),
        }
        for name, expected in ref.items():
            got = _block_eal_loss(terms, name, block_size)
            assert got.item() == pytest.approx(expected.item(), abs=1e-5), name


# ---------------------------------------------------------------------------
# masking, degenerate inputs, gradient flow
# ---------------------------------------------------------------------------


class TestMaskingAndGradFlow:
    def test_all_masked_gives_zero_loss(self):
        logits = torch.randn(1, 6, 7)
        targets = torch.randn(1, 6, 7)
        token_ids = torch.randint(0, 7, (1, 6))
        loss_mask = torch.zeros_like(token_ids)
        for name in ("wtv", "real", "realp"):
            loss, _ = compute_metrics(
                logits,
                targets,
                loss_mask,
                block_size=3,
                loss_config=resolve_loss_config(name),
                sample_from_anchor=True,
                token_ids=token_ids,
            )
            assert loss.item() == 0.0, name

    def test_perfect_prediction_gives_zero_loss(self):
        block_size, vocab = 3, 7
        token_ids = torch.randint(0, vocab, (1, 2 * block_size))
        logits = torch.zeros(1, 2 * block_size, vocab)
        logits.scatter_(-1, token_ids.unsqueeze(-1), 100.0)
        loss_mask = torch.ones_like(token_ids)
        for name in ("wtv", "real", "realp"):
            loss, _ = compute_metrics(
                logits,
                logits.clone(),
                loss_mask,
                block_size=block_size,
                loss_config=resolve_loss_config(name),
                sample_from_anchor=True,
                token_ids=token_ids,
            )
            assert loss.item() == pytest.approx(0.0, abs=1e-3), name

    def test_real_grad_differs_from_detached_prefix(self):
        # guards the sum a * dW gradient term: detaching the prefix W (the
        # failed real_dw ablation in the toy experiments) must not match
        g = torch.Generator().manual_seed(2)
        logits = torch.randn(1, 6, 11, generator=g, requires_grad=True)
        targets = torch.randn(1, 6, 11, generator=g)
        token_ids = torch.randint(0, 11, (1, 6), generator=g)
        loss_mask = torch.ones_like(token_ids)
        loss, _ = compute_metrics(
            logits,
            targets,
            loss_mask,
            block_size=3,
            loss_config=resolve_loss_config("real"),
            sample_from_anchor=True,
            token_ids=token_ids,
        )
        (grad_full,) = torch.autograd.grad(loss, logits, retain_graph=True)

        terms = compute_eal_terms(logits, targets, token_ids, loss_mask, block_size=3)
        j_dw = (terms["log_prev"].exp().detach() * terms["a"]).sum(-1)
        loss_dw = 1.0 - j_dw.mean() / 3
        (grad_dw,) = torch.autograd.grad(loss_dw, logits)

        assert not torch.allclose(grad_full, grad_dw, atol=1e-6)
        assert grad_full.abs().sum() > 0

    def test_token_ids_required(self):
        with pytest.raises(ValueError, match="token_ids"):
            compute_metrics(
                torch.randn(1, 4, 5),
                torch.randn(1, 4, 5),
                torch.ones(1, 4),
                block_size=2,
                loss_config=resolve_loss_config("real"),
            )


class TestTrajectoryLayout:
    def _random_batch(self):
        generator = torch.Generator().manual_seed(0)
        logits = torch.randn(1, 4, 5, generator=generator, requires_grad=True)
        targets = torch.randn(1, 4, 5, generator=generator)
        token_ids = torch.tensor([[0, 1, 2, 3]])
        loss_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
        return logits, targets, token_ids, loss_mask

    @pytest.mark.parametrize("name", ["real", "realp", "wtv"])
    def test_perfect_prediction_without_anchor_has_zero_loss(self, name):
        _, targets, token_ids, loss_mask = self._random_batch()
        targets = torch.zeros_like(targets)
        targets.scatter_(-1, token_ids.unsqueeze(-1), 100.0)
        loss, _ = compute_metrics(
            targets.clone(),
            targets,
            loss_mask,
            block_size=4,
            sample_from_anchor=False,
            token_ids=token_ids,
            loss_config=resolve_loss_config(name),
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    @pytest.mark.parametrize("name", ["wtv", "realk"])
    def test_bonus_anchor_does_not_zero_gradient(self, name):
        logits, targets, token_ids, loss_mask = self._random_batch()
        loss, _ = compute_metrics(
            logits,
            targets,
            loss_mask,
            block_size=4,
            sample_from_anchor=False,
            token_ids=token_ids,
            loss_config=resolve_loss_config(name),
            realk_k=1,
        )
        (gradient,) = torch.autograd.grad(loss, logits)
        assert gradient.abs().sum() > 0
        assert gradient[0, 0].abs().sum() == 0

    def test_realk_perfect_prediction_uses_prediction_window(self):
        _, targets, token_ids, loss_mask = self._random_batch()
        targets = torch.zeros_like(targets)
        targets.scatter_(-1, token_ids.unsqueeze(-1), 100.0)
        loss, _ = compute_metrics(
            targets.clone(),
            targets,
            loss_mask,
            block_size=4,
            sample_from_anchor=False,
            token_ids=token_ids,
            loss_config=resolve_loss_config("realk"),
            realk_k=1,
        )
        assert loss.item() == pytest.approx(1.0 - 1.0 / 3.0, abs=1e-4)

    def test_bonus_anchor_terms_match_explicit_slice(self):
        logits, targets, token_ids, loss_mask = self._random_batch()
        actual = compute_eal_terms(
            logits, targets, token_ids, loss_mask, 4, start_pos=1
        )
        expected = compute_eal_terms(
            logits[:, 1:], targets[:, 1:], token_ids[:, 1:], loss_mask[:, 1:], 3
        )
        for key in ("a", "log_prefix", "log_r_prefix", "valid"):
            assert torch.equal(actual[key], expected[key])

    @pytest.mark.parametrize(
        ("loss_mask", "document_ids", "expected_valid"),
        [
            (
                [[1.0, 1.0, 0.0, 1.0]],
                [[0, 0, 0, 0]],
                [[True, True, False, False]],
            ),
            (
                [[1.0, 1.0, 1.0, 1.0]],
                [[0, 1, 1, 1]],
                [[True, False, False, False]],
            ),
        ],
    )
    def test_mask_and_document_boundary_stop_trajectory(
        self, loss_mask, document_ids, expected_valid
    ):
        logits, targets, token_ids, _ = self._random_batch()
        terms = compute_eal_terms(
            logits,
            targets,
            token_ids,
            torch.tensor(loss_mask),
            4,
            document_ids=torch.tensor(document_ids),
            loss_implementation="eager",
        )
        assert terms["valid"].tolist() == expected_valid
        invalid = ~terms["valid"][0]
        assert terms["log_r_prefix"][0, invalid].abs().sum() == 0
        loss = _block_eal_loss(terms, "dblast", 4)
        (gradient,) = torch.autograd.grad(loss, logits)
        assert gradient[0, invalid].abs().sum() == 0

    def test_dblast_bonus_anchor_has_no_gradient(self):
        logits, targets, token_ids, loss_mask = self._random_batch()
        loss, _ = compute_metrics(
            logits,
            targets,
            loss_mask,
            block_size=4,
            sample_from_anchor=False,
            token_ids=token_ids,
            loss_config=resolve_loss_config("dblast"),
        )
        (gradient,) = torch.autograd.grad(loss, logits)
        assert gradient[0, 0].abs().sum() == 0
        assert gradient[0, 1:].abs().sum() > 0

    def test_eager_never_dispatches_to_fused_on_cuda(self, monkeypatch):
        class CudaTensor:
            is_cuda = True

        calls = []

        def fused(*_args):
            calls.append("fused")
            return torch.tensor(1.0)

        def eager(*_args):
            calls.append("eager")
            return torch.tensor(2.0)

        monkeypatch.setattr(dflash_metrics, "tv_loss", fused)
        monkeypatch.setattr(dflash_metrics.eager_losses, "tv_loss", eager)
        tensor = cast("torch.Tensor", CudaTensor())
        assert tv_loss_fused_or_eager(tensor, tensor, "eager").item() == 2.0
        assert calls == ["eager"]
        assert tv_loss_fused_or_eager(tensor, tensor, "fused").item() == 1.0
        assert calls == ["eager", "fused"]

    def test_eager_loss_config_reaches_trajectory_terms(self, monkeypatch):
        observed = []
        original = dflash_metrics.compute_eal_terms

        def wrapped(*args, **kwargs):
            observed.append(kwargs["loss_implementation"])
            return original(*args, **kwargs)

        monkeypatch.setattr(dflash_metrics, "compute_eal_terms", wrapped)
        logits, targets, token_ids, loss_mask = self._random_batch()
        compute_metrics(
            logits,
            targets,
            loss_mask,
            block_size=4,
            sample_from_anchor=True,
            token_ids=token_ids,
            loss_config=resolve_loss_config("real"),
        )
        assert observed == ["eager"]


# ---------------------------------------------------------------------------
# loss-spec resolution and metric-dict contract
# ---------------------------------------------------------------------------


class TestLossConfigAndMetrics:
    @pytest.mark.parametrize("spec", ["wtv", "real", "realp"])
    def test_resolve_single(self, spec):
        config = resolve_loss_config(spec)
        assert list(config) == [spec]

    def test_resolve_json_mix(self):
        config = resolve_loss_config('{"real": 0.9, "ce": 0.1}')
        assert set(config) == {"real", "ce"}
        assert config["real"][1] == 0.9
        assert config["ce"][1] == 0.1

    def test_mix_combines_linearly(self):
        g = torch.Generator().manual_seed(3)
        logits = torch.randn(1, 6, 11, generator=g)
        targets = torch.randn(1, 6, 11, generator=g)
        token_ids = torch.randint(0, 11, (1, 6), generator=g)
        loss_mask = torch.ones_like(token_ids)
        kwargs: dict[str, Any] = {
            "block_size": 3,
            "sample_from_anchor": True,
            "token_ids": token_ids,
        }
        mix, _ = compute_metrics(
            logits,
            targets,
            loss_mask,
            loss_config=resolve_loss_config('{"real": 0.9, "ce": 0.1}'),
            **kwargs,
        )
        real_only, _ = compute_metrics(
            logits,
            targets,
            loss_mask,
            loss_config=resolve_loss_config("real"),
            **kwargs,
        )
        ce_only, _ = compute_metrics(
            logits,
            targets,
            loss_mask,
            loss_config=resolve_loss_config("ce"),
            **kwargs,
        )
        assert mix.item() == pytest.approx(
            0.9 * real_only.item() + 0.1 * ce_only.item(), abs=1e-5
        )

    def test_metric_values_are_tensors(self):
        g = torch.Generator().manual_seed(4)
        logits = torch.randn(1, 6, 11, generator=g)
        targets = torch.randn(1, 6, 11, generator=g)
        token_ids = torch.randint(0, 11, (1, 6), generator=g)
        loss_mask = torch.ones_like(token_ids)
        _, metrics = compute_metrics(
            logits,
            targets,
            loss_mask,
            block_size=3,
            loss_config=resolve_loss_config('{"real": 0.9, "ce": 0.1}'),
            sample_from_anchor=True,
            token_ids=token_ids,
        )
        for key, value in metrics.items():
            assert isinstance(value, torch.Tensor), key
        for key in (
            "eal_j_b_sum",
            "eal_j_b_total",
            "eal_j_wtv_sum",
            "eal_j_wtv_total",
            "eal_a_pos_0_sum",
            "eal_b_pos_0_sum",
            "eal_ba_next_pos_0_sum",
            "real_loss_sum",
            "ce_loss_sum",
        ):
            assert key in metrics, key

    @pytest.mark.parametrize(
        ("new_name", "old_name"),
        [("realp", "eala"), ("real", "ealb"), ("realk", "ealk"), ("realv", "ealv")],
    )
    def test_legacy_names_preserve_values_gradients_and_diagnostics(
        self, new_name, old_name
    ):
        generator = torch.Generator().manual_seed(23)
        logits = torch.randn(1, 9, 11, generator=generator, requires_grad=True)
        targets = torch.randn(1, 9, 11, generator=generator)
        token_ids = torch.randint(0, 11, (1, 9), generator=generator)
        loss_mask = torch.ones_like(token_ids)
        new_parameters: dict[str, Any] = {
            "realv_lambda": 0.02,
            "realk_k": 2,
            "real_guard_on": True,
            "real_guard_mu0": 4.362,
            "real_guard_delta": 0.05,
            "real_guard_rho": 0.9,
        }
        old_parameters = {
            key.replace("realv", "ealv")
            .replace("realk", "ealk")
            .replace("real_guard", "eal_guard"): value
            for key, value in new_parameters.items()
        }
        common: dict[str, Any] = {
            "block_size": 3,
            "sample_from_anchor": True,
            "token_ids": token_ids,
            "block_group_ids": torch.tensor([0, 0, 1]),
        }
        new_loss, new_metrics = compute_metrics(
            logits,
            targets,
            loss_mask,
            loss_config=resolve_loss_config(f'{{"ce": 0.1, "{new_name}": 0.9}}'),
            **common,
            **new_parameters,
        )
        old_loss, old_metrics = compute_metrics(
            logits,
            targets,
            loss_mask,
            loss_config=resolve_loss_config(f'{{"ce": 0.1, "{old_name}": 0.9}}'),
            **common,
            **old_parameters,
        )
        (new_gradient,) = torch.autograd.grad(new_loss, logits, retain_graph=True)
        (old_gradient,) = torch.autograd.grad(old_loss, logits)
        assert new_loss.item() == pytest.approx(old_loss.item(), abs=1e-6)
        assert torch.allclose(new_gradient, old_gradient, atol=1e-6)
        assert f"{new_name}_loss_sum" in new_metrics
        assert f"{old_name}_loss_sum" in old_metrics
        if new_name in ("realv", "realk"):
            for suffix in ("mu", "guard"):
                assert f"{new_name}_{suffix}_sum" in new_metrics
                assert f"{old_name}_{suffix}_sum" in old_metrics

    @pytest.mark.parametrize("old_name", ["eala", "ealb", "ealk", "ealv"])
    def test_duplicate_new_and_old_names_are_rejected(self, old_name):
        new_name = {"eala": "realp", "ealb": "real", "ealk": "realk", "ealv": "realv"}[
            old_name
        ]
        with pytest.raises(ValueError, match="Duplicate loss objective"):
            compute_metrics(
                torch.randn(1, 6, 11),
                torch.randn(1, 6, 11),
                torch.ones(1, 6),
                block_size=3,
                token_ids=torch.zeros(1, 6, dtype=torch.long),
                loss_config=resolve_loss_config(
                    f'{{"{new_name}": 0.5, "{old_name}": 0.5}}'
                ),
            )


# ---------------------------------------------------------------------------
# DBLast AL surrogate (arXiv:2608.05448 eq. 15)
# ---------------------------------------------------------------------------


def _logprob_rows(rows: list[list[float]]) -> torch.Tensor:
    """log-distribution rows [n, V] (already normalized in log space)."""
    return torch.log(torch.tensor(rows, dtype=torch.float32))


class TestDBLast:
    """Hand-computed references for the dblast branch of _block_eal_loss.

    Single block of size 2, vocab {A, B}:
      pos 0 (token A): q = (0.4, 0.6), p = (0.5, 0.5)
      pos 1 (token B): q = (0.6, 0.4), p = (0.7, 0.3)
    """

    def _batch(self, q1a: float = 0.4, mask_pos1: bool = False):
        logits = _logprob_rows([[q1a, 1 - q1a], [0.6, 0.4]]).unsqueeze(0)
        targets = _logprob_rows([[0.5, 0.5], [0.7, 0.3]]).unsqueeze(0)
        token_ids = torch.tensor([[0, 1]], dtype=torch.long)
        loss_mask = torch.tensor([[1.0, 0.0]] if mask_pos1 else [[1.0, 1.0]])
        return logits, targets, token_ids, loss_mask

    def test_value_no_truncation(self):
        # r_0 = 0.8, r_1 = 16/15 >= tau -> full block retained
        logits, targets, token_ids, loss_mask = self._batch()
        loss = _block_eal_loss(
            compute_eal_terms(logits, targets, token_ids, loss_mask, 2),
            "dblast",
            2,
        )
        # J = log(0.8) + log(16/15) + log(1.75) = log(1.49333..)
        assert loss.item() == pytest.approx(-0.4010, abs=1e-3)

    def test_truncation_first_below_tau(self):
        # q_0(A) = 0.04 -> r_0 = 0.08 < tau=0.1 -> retained = {0} only
        logits, targets, token_ids, loss_mask = self._batch(q1a=0.04)
        loss = _block_eal_loss(
            compute_eal_terms(logits, targets, token_ids, loss_mask, 2),
            "dblast",
            2,
        )
        # J = log(0.08) + log(1) only; pos-1 terms excluded
        assert loss.item() == pytest.approx(2.5257, abs=1e-3)

    def test_mask_excludes_position(self):
        logits, targets, token_ids, loss_mask = self._batch(mask_pos1=True)
        loss = _block_eal_loss(
            compute_eal_terms(logits, targets, token_ids, loss_mask, 2),
            "dblast",
            2,
        )
        # only pos 0 retained: J = log(0.8) + log(1)
        assert loss.item() == pytest.approx(0.2231, abs=1e-3)

    def test_grad_flow(self):
        logits, targets, token_ids, loss_mask = self._batch()
        logits.requires_grad_(True)
        loss = _block_eal_loss(
            compute_eal_terms(logits, targets, token_ids, loss_mask, 2),
            "dblast",
            2,
        )
        loss.backward()
        g = logits.grad
        # d loss / d lq_1 = -(1 - A_0*A_1/S_1) = -(1 - 0.75/1.75) = -4/7
        # on logits[pos1, B] = -4/7 * (1 - q1(B)) = -4/7 * 0.6
        assert g[0, 1, 1].item() == pytest.approx(-4 / 7 * 0.6, abs=1e-4)
        # on logits[pos1, A] = -4/7 * (-q1(A)) = +4/7 * 0.6
        assert g[0, 1, 0].item() == pytest.approx(4 / 7 * 0.6, abs=1e-4)
        # d loss / d lq_0 = -2 (log_r_0 + log_r_1; A_0 clamped, no S term)
        # on logits[pos0, A] = -2 * (1 - q_0(A)) = -2 * 0.6
        assert g[0, 0, 0].item() == pytest.approx(-2 * 0.6, abs=1e-4)

    def test_truncated_positions_have_no_grad(self):
        logits, targets, token_ids, loss_mask = self._batch(q1a=0.04)
        logits.requires_grad_(True)
        loss = _block_eal_loss(
            compute_eal_terms(logits, targets, token_ids, loss_mask, 2),
            "dblast",
            2,
        )
        loss.backward()
        # pos 1 is outside the retained set -> zero gradient
        assert logits.grad[0, 1].abs().sum().item() == pytest.approx(0.0, abs=1e-6)
        assert logits.grad[0, 0].abs().sum().item() > 0

    def test_resolve_and_end_to_end(self):
        config = resolve_loss_config('{"ce": 0.1, "dblast": 0.9}')
        assert set(config) == {"ce", "dblast"}
        logits, targets, token_ids, loss_mask = self._batch()
        loss, metrics = compute_metrics(
            logits,
            targets,
            loss_mask,
            block_size=2,
            loss_config=config,
            sample_from_anchor=True,
            token_ids=token_ids,
        )
        assert torch.isfinite(loss)
        assert "dblast_loss_sum" in metrics


# ---------------------------------------------------------------------------
# realv variance-constrained loss (stability training)
# ---------------------------------------------------------------------------


class TestREALV:
    """Hand-computed references for the realv branch of _block_eal_loss.

    Two blocks, block_size=2, vocab {A, B}; all positions share
    q = (0.8, 0.2), p = (0.4, 0.6). Block 0 tokens are (A, A) so
    log_b = (0, 0); block 1 tokens are (B, B) so log_b = (log 1/3, log 1/3)
    and the prefix decay into block1 pos1 is 1/3.
    """

    def _batch(self, second_token_block1: int = 1, n_copies: int = 1):
        row = _logprob_rows([[0.8, 0.2]])
        logits = row.repeat(4 * n_copies, 1).unsqueeze(0)
        targets = _logprob_rows([[0.4, 0.6]]).repeat(4 * n_copies, 1).unsqueeze(0)
        ids = [0, 0, 1, second_token_block1] * n_copies
        token_ids = torch.tensor([ids], dtype=torch.long)
        loss_mask = torch.ones_like(token_ids)
        return logits, targets, token_ids, loss_mask

    def test_value(self):
        logits, targets, token_ids, loss_mask = self._batch()
        loss = _block_eal_loss(
            compute_eal_terms(logits, targets, token_ids, loss_mask, 2),
            "realv",
            2,
        )
        # u_0 = 0.6; block0 u_1 = 1*0.6, e_0 = 1.2; block1 u_1 = (1/3)*0.6, e_1 = 0.8
        # μ = 1.0, M2 = 1.8, μ̂²(ungrouped fallback) = 0.96, V = 0.84
        # L = (1 - μ/2) + (0.05/2)*V = 0.5 + 0.021
        assert loss.item() == pytest.approx(0.521, abs=1e-3)

    def test_prefers_lower_variance(self):
        # batch A（即 test_value 的批）：μ = 1.0, V = 0.84
        la = _block_eal_loss(_terms_from_batch(self._batch()), "realv", 2)
        # batch B：μ 同为 1.0 但 V = 0.96
        #   block A: pos 0 同上; pos 1: q=(0.8,0.2), p=(0.6,0.4), token A
        #   block B: pos 0 同上; pos 1: q=(1.0,0.0), p=(0.0,1.0), token B
        logits = _logprob_rows(
            [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2], [1.0, 0.0]]
        ).unsqueeze(0)
        targets = _logprob_rows(
            [[0.4, 0.6], [0.6, 0.4], [0.4, 0.6], [0.0, 1.0]]
        ).unsqueeze(0)
        token_ids = torch.tensor([[0, 0, 0, 1]], dtype=torch.long)
        loss_mask = torch.ones_like(token_ids)
        lb = _block_eal_loss(
            compute_eal_terms(logits, targets, token_ids, loss_mask, 2),
            "realv",
            2,
        )
        assert la.item() == pytest.approx(0.521, abs=1e-3)
        assert lb.item() == pytest.approx(0.524, abs=1e-3)
        assert la.item() < lb.item()
        # real 在同 μ 下打平：1 - 1.0/2 = 0.5；realv = real + (λ/2)*V
        ea = _block_eal_loss(_terms_from_batch(self._batch()), "real", 2)
        eb = _block_eal_loss(
            compute_eal_terms(logits, targets, token_ids, loss_mask, 2),
            "real",
            2,
        )
        assert ea.item() == pytest.approx(eb.item(), abs=1e-4)
        assert la.item() - ea.item() == pytest.approx(0.021, abs=1e-3)

    def test_single_block_edge(self):
        logits, targets, token_ids, loss_mask = self._batch(n_copies=1)
        # keep only the first block (positions 0-1): one valid block M=1
        loss_mask = loss_mask.clone()
        loss_mask[0, 2:] = 0
        loss = _block_eal_loss(
            compute_eal_terms(logits, targets, token_ids, loss_mask, 2),
            "realv",
            2,
        )
        # M=1 -> μ̂² clamped to (s²-sq)/1 = 0
        # μ = 1.2, M2 = 0.6 + 3*0.6 = 2.4 -> L = (1 - 1.2/2) + 0.025*2.4 = 0.46
        assert loss.item() == pytest.approx(0.46, abs=1e-3)

    def test_grad_flow_and_differs_from_real(self):
        logits, targets, token_ids, loss_mask = self._batch(n_copies=2)
        logits.requires_grad_(True)
        terms = compute_eal_terms(logits, targets, token_ids, loss_mask, 2)
        loss_v = _block_eal_loss(terms, "realv", 2)
        (gv,) = torch.autograd.grad(loss_v, logits, retain_graph=True)
        assert torch.isfinite(gv).all()
        assert gv.abs().sum() > 0
        loss_b = _block_eal_loss(terms, "real", 2)
        (gb,) = torch.autograd.grad(loss_b, logits)
        # variance term changes the gradient on the same batch
        assert not torch.allclose(gv, gb, atol=1e-6)

    def test_resolve(self):
        config = resolve_loss_config('{"ce": 0.1, "realv": 0.9}')
        assert set(config) == {"ce", "realv"}
        logits, targets, token_ids, loss_mask = self._batch()
        loss, metrics = compute_metrics(
            logits,
            targets,
            loss_mask,
            block_size=2,
            loss_config=config,
            sample_from_anchor=True,
            token_ids=token_ids,
        )
        assert torch.isfinite(loss)
        assert "realv_loss_sum" in metrics


class TestREALVGrouped:
    """Document-grouped within-document variance for the realv branch.

    Same batches as TestREALV (block_size=2, 4 blocks with e=[1.2,0.8,1.2,0.8],
    d=[2.4,1.2,2.4,1.2]). Grouping must change the variance value:
      - group_ids=[0,0,1,1]: var_g = 1.8-1.0² = 0.8 each → V = 0.8
      - group_ids=[0,1,0,1]: var_g0 = 2.4-1.2² = 0.96,
        var_g1 = 1.2-0.8² = 0.56 → V = 0.76
      - group_ids=None: legacy ungrouped fallback V = 0.84
    """

    def _terms(self):
        row = _logprob_rows([[0.8, 0.2]])
        logits = row.repeat(8, 1).unsqueeze(0)
        targets = _logprob_rows([[0.4, 0.6]]).repeat(8, 1).unsqueeze(0)
        token_ids = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1]], dtype=torch.long)
        loss_mask = torch.ones_like(token_ids)
        return compute_eal_terms(logits, targets, token_ids, loss_mask, 2)

    def test_grouped_values(self):
        terms = self._terms()
        # [0,0,1,1]: per-group var 0.8/0.8 → V=0.8; L = 0.5 + 0.025*0.8 = 0.52
        loss = _block_eal_loss(
            terms, "realv", 2, torch.tensor([0, 0, 1, 1], dtype=torch.long)
        )
        assert loss.item() == pytest.approx(0.52, abs=1e-3)
        # [0,1,0,1]: var 0.96/0.56 → V=0.76; L = 0.5 + 0.025*0.76 = 0.519
        loss2 = _block_eal_loss(
            terms, "realv", 2, torch.tensor([0, 1, 0, 1], dtype=torch.long)
        )
        assert loss2.item() == pytest.approx(0.519, abs=1e-3)
        # None: legacy ungrouped fallback V=0.84 → 0.521（三种口径必须可区分）
        loss3 = _block_eal_loss(terms, "realv", 2)
        assert loss3.item() == pytest.approx(0.521, abs=1e-3)

    def test_single_block_group(self):
        # 单 block 组贡献自身分布内方差（M₂ − e²），不是 0
        terms = self._terms()
        # 只保留 block0（e=1.2, d=2.4）：var_g = 2.4-1.44 = 0.96
        for k in terms:
            if isinstance(terms[k], torch.Tensor) and k != "valid":
                terms[k] = terms[k][:1].clone()
        terms["valid"] = terms["valid"][:1].clone()
        loss = _block_eal_loss(terms, "realv", 2, torch.tensor([0], dtype=torch.long))
        # μ = 1.2, L = (1-1.2/2) + 0.025*0.96 = 0.4 + 0.024 = 0.424
        assert loss.item() == pytest.approx(0.424, abs=1e-3)

    def test_grad_flow(self):
        terms = self._terms()
        for k in ("a", "log_prev"):
            terms[k] = terms[k].requires_grad_(True)
        loss = _block_eal_loss(
            terms, "realv", 2, torch.tensor([0, 0, 1, 1], dtype=torch.long)
        )
        (g,) = torch.autograd.grad(loss, terms["a"], retain_graph=True)
        assert torch.isfinite(g).all()
        assert g.abs().sum() > 0

    def test_per_token_group_ids_rejected(self):
        # 每个 block 一个 group id（M 个）；M×B 个（per-token）必须显式报错
        terms = self._terms()
        with pytest.raises(ValueError, match="per-block"):
            _block_eal_loss(terms, "realv", 2, torch.zeros(8, dtype=torch.long))


# ---------------------------------------------------------------------------
# realk short-budget loss and the mean-preservation guard
# ---------------------------------------------------------------------------


class TestREALK:
    """Truncated EAL-B (realk) and the guard G = (ρ/B)[μ_0 − δ − μ]₊.

    Same two-block batch as TestREALV: u_0 = 0.6 everywhere,
    block0 u_1 = 0.6 (e_0 = 1.2), block1 u_1 = 0.2 (e_1 = 0.8), μ = 1.0.
    """

    def _batch(self, second_token_block1=1, n_copies=1):
        return TestREALV._batch(TestREALV(), second_token_block1, n_copies)

    def test_k_covers_block_equals_real(self):
        terms = _terms_from_batch(self._batch())
        lk = _block_eal_loss(terms, "realk", 2, eal_guard_mu0=4.362)
        lb = _block_eal_loss(terms, "real", 2)
        assert lk.item() == pytest.approx(lb.item(), abs=1e-6)
        assert lk.item() == pytest.approx(0.5, abs=1e-3)

    def test_truncation_k1(self):
        lk = _block_eal_loss(_terms_from_batch(self._batch()), "realk", 2, realk_k=1)
        assert lk.item() == pytest.approx(0.7, abs=1e-3)

    def test_k_derived_from_mu0(self):
        lk = _block_eal_loss(
            _terms_from_batch(self._batch()), "realk", 2, eal_guard_mu0=0.5
        )
        assert lk.item() == pytest.approx(0.7, abs=1e-3)
        lk = _block_eal_loss(
            _terms_from_batch(self._batch()), "realk", 2, eal_guard_mu0=1.2
        )
        assert lk.item() == pytest.approx(0.5, abs=1e-3)

    def test_k_derived_from_current_mu(self):
        lk = _block_eal_loss(_terms_from_batch(self._batch()), "realk", 2)
        assert lk.item() == pytest.approx(0.7, abs=1e-3)

    def test_k_clamped_to_block(self):
        terms = _terms_from_batch(self._batch())
        lk = _block_eal_loss(terms, "realk", 2, eal_guard_mu0=100.0)
        lb = _block_eal_loss(terms, "real", 2)
        assert lk.item() == pytest.approx(lb.item(), abs=1e-6)

    def test_guard_active_adds_eal_weight(self):
        terms = _terms_from_batch(self._batch())
        lk = _block_eal_loss(
            terms, "realk", 2, eal_guard_on=True, eal_guard_mu0=1.2, eal_guard_delta=0.1
        )
        assert lk.item() == pytest.approx(0.5 + 0.045, abs=1e-3)
        lv = _block_eal_loss(
            terms, "realv", 2, eal_guard_on=True, eal_guard_mu0=1.2, eal_guard_delta=0.1
        )
        assert lv.item() == pytest.approx(0.521 + 0.045, abs=1e-3)
        logits, targets, token_ids, loss_mask = self._batch()
        logits.requires_grad_(True)
        (g,) = torch.autograd.grad(
            _block_eal_loss(
                compute_eal_terms(logits, targets, token_ids, loss_mask, 2),
                "realk",
                2,
                eal_guard_on=True,
                eal_guard_mu0=1.2,
                eal_guard_delta=0.1,
            ),
            logits,
        )
        assert torch.isfinite(g).all()
        assert g.abs().sum() > 0

    def test_guard_inactive_when_above_floor(self):
        terms = _terms_from_batch(self._batch())
        assert _block_eal_loss(
            terms,
            "realk",
            2,
            eal_guard_on=True,
            eal_guard_mu0=1.0,
            eal_guard_delta=0.1,
            realk_k=2,
        ).item() == pytest.approx(0.5, abs=1e-3)
        assert _block_eal_loss(
            terms,
            "realv",
            2,
            eal_guard_on=True,
            eal_guard_mu0=1.0,
            eal_guard_delta=0.1,
            realk_k=2,
        ).item() == pytest.approx(0.521, abs=1e-3)

    def test_guard_toggle_off(self):
        terms = _terms_from_batch(self._batch())
        assert _block_eal_loss(
            terms, "realk", 2, eal_guard_on=False, eal_guard_mu0=10.0, realk_k=2
        ).item() == pytest.approx(0.5, abs=1e-3)
        assert _block_eal_loss(
            terms, "realv", 2, eal_guard_on=False, eal_guard_mu0=10.0, realk_k=2
        ).item() == pytest.approx(0.521, abs=1e-3)

    def test_guard_requires_mu0(self):
        with pytest.raises(ValueError, match="real_guard_mu0 is required"):
            _block_eal_loss(
                _terms_from_batch(self._batch()), "realk", 2, eal_guard_on=True
            )

    def test_lambda_is_configurable(self):
        terms = _terms_from_batch(self._batch())
        base = _block_eal_loss(terms, "realv", 2, realv_lambda=0.0)
        assert base.item() == pytest.approx(0.5, abs=1e-3)
        weighted = _block_eal_loss(terms, "realv", 2, realv_lambda=0.05)
        assert weighted.item() == pytest.approx(base.item() + 0.021, abs=1e-3)

    def test_resolve(self):
        config = resolve_loss_config('{"ce": 0.1, "realk": 0.9}')
        assert set(config) == {"ce", "realk"}
        logits, targets, token_ids, loss_mask = self._batch()
        loss, metrics = compute_metrics(
            logits,
            targets,
            loss_mask,
            block_size=2,
            loss_config=config,
            sample_from_anchor=True,
            token_ids=token_ids,
        )
        assert torch.isfinite(loss)
        assert "realk_loss_sum" in metrics
        assert "realk_jk_sum" in metrics
        assert "realk_mu_sum" in metrics
        assert "realk_guard_sum" in metrics


# ---------------------------------------------------------------------------
# vsd step-level variational lower-bound loss (arXiv:2602.05774, Eq. 18)
# ---------------------------------------------------------------------------


class TestVSD:
    """Hand-computed references for the VSD counterexample:
    gamma=2, target outputs (0.5, 0.5) at every prefix.
    M-step 加权 MLE 口径（loss = −(Σκ·log Q/Σκ)/B，κ 固定、detach、有界）：
      Draft1 loss = +1.201988, Draft2 loss = +0.791460（draft2 更优顺序保持）。
    """

    def _draft1(self):
        # 4 blocks (AA, AB, BA, BB), block_size=2, vocab {A, B}
        logits = _logprob_rows(
            [
                [0.5, 0.5],
                [0.5, 0.5],  # AA
                [0.5, 0.5],
                [0.5, 0.5],  # AB
                [0.5, 0.5],
                [0.99, 0.01],  # BA
                [0.5, 0.5],
                [0.99, 0.01],  # BB
            ]
        ).unsqueeze(0)
        targets = _logprob_rows([[0.5, 0.5]] * 8).unsqueeze(0)
        token_ids = torch.tensor([[0, 0, 0, 1, 1, 0, 1, 1]], dtype=torch.long)
        loss_mask = torch.ones_like(token_ids)
        return logits, targets, token_ids, loss_mask

    def _draft2(self):
        logits = _logprob_rows(
            [
                [0.6, 0.4],
                [0.7, 0.3],  # AA
                [0.6, 0.4],
                [0.7, 0.3],  # AB
                [0.6, 0.4],
                [0.7, 0.3],  # BA
                [0.6, 0.4],
                [0.7, 0.3],  # BB
            ]
        ).unsqueeze(0)
        targets = _logprob_rows([[0.5, 0.5]] * 8).unsqueeze(0)
        token_ids = torch.tensor([[0, 0, 0, 1, 1, 0, 1, 1]], dtype=torch.long)
        loss_mask = torch.ones_like(token_ids)
        return logits, targets, token_ids, loss_mask

    def test_value_draft1(self):
        loss = _block_eal_loss(_terms_from_batch(self._draft1()), "vsd", 2)
        assert loss.item() == pytest.approx(1.201988, abs=1e-3)

    def test_value_draft2(self):
        loss = _block_eal_loss(_terms_from_batch(self._draft2()), "vsd", 2)
        assert loss.item() == pytest.approx(0.791460, abs=1e-3)

    def test_bound_order_matches_appendix(self):
        # 附录反例：Draft2 的下界更高（L_V 更大、loss 更小），但真实 J2 更低——
        # loss 口径同样应给出 draft2 < draft1。
        l1 = _block_eal_loss(_terms_from_batch(self._draft1()), "vsd", 2)
        l2 = _block_eal_loss(_terms_from_batch(self._draft2()), "vsd", 2)
        assert l2.item() < l1.item()

    def test_grad_flows(self):
        logits, targets, token_ids, loss_mask = self._draft2()
        logits = logits.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(
            _block_eal_loss(
                compute_eal_terms(logits, targets, token_ids, loss_mask, 2),
                "vsd",
                2,
            ),
            logits,
        )
        assert torch.isfinite(g).all()
        assert g.abs().sum() > 0

    def test_value_and_gradient_match_exact_nonuniform_paths_with_fixed_kappa(self):
        # Enumerate P exactly as 20 equally weighted data samples. The draft
        # shares parameters across paths, including the conditional q2 rows.
        draft = _logprob_rows([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])
        draft.requires_grad_(True)
        target = torch.tensor([[0.4, 0.6], [0.75, 0.25], [0.25, 0.75]])
        draft_rows, target_rows, ids = [], [], []
        for first, second, count in [(0, 0, 6), (0, 1, 2), (1, 0, 3), (1, 1, 9)]:
            for _ in range(count):
                draft_rows.extend([draft[0], draft[1 + first]])
                target_rows.extend([target[0].log(), target[1 + first].log()])
                ids.extend([first, second])
        token_ids = torch.tensor([ids])
        terms = compute_eal_terms(
            torch.stack(draft_rows).unsqueeze(0),
            torch.stack(target_rows).unsqueeze(0),
            token_ids,
            torch.ones_like(token_ids),
            2,
        )
        actual = _block_eal_loss(terms, "vsd", 2)
        (actual_grad,) = torch.autograd.grad(actual, draft)

        # Independent M-step reference: −Σ count·κ·log q_path / Σ count·κ，
        # κ 由 e-step（detach）计算，candidate log q 保留梯度。
        q = draft.softmax(dim=-1)
        q_e_step = q.detach()
        num = torch.zeros(())
        den = torch.zeros(())
        for first, second, count in [(0, 0, 6), (0, 1, 2), (1, 0, 3), (1, 1, 9)]:
            log_q_path = q[0, first].log() + q[1 + first, second].log()
            kappa = torch.minimum(
                target[0, first] / q_e_step[0, first], q.new_tensor(1)
            )
            kappa = kappa * torch.minimum(
                target[1 + first, second] / q_e_step[1 + first, second],
                q.new_tensor(1),
            )
            num = num + count * kappa * log_q_path
            den = den + count * kappa
        expected = -num / den / 2
        (expected_grad,) = torch.autograd.grad(expected, draft)
        assert actual.item() == pytest.approx(expected.item(), abs=1e-6)
        assert torch.allclose(actual_grad, expected_grad, atol=1e-6)

    @pytest.mark.parametrize("draft_logit", [-1000.0, 1000.0])
    def test_extreme_path_weights_are_finite(self, draft_logit):
        # Both weights underflow, or one overflows; neither should break loss.
        logits = torch.tensor([[[draft_logit, 0.0], [draft_logit + 1.0, 0.0]]])
        logits.requires_grad_(True)
        targets = torch.zeros_like(logits)
        if draft_logit > 0:
            targets[..., 1] = 1000.0
        token_ids = torch.tensor([[0, 0]])
        terms = compute_eal_terms(
            logits, targets, token_ids, torch.ones_like(token_ids), 1
        )
        actual = _block_eal_loss(terms, "vsd", 1)
        (grad,) = torch.autograd.grad(actual, logits)
        assert torch.isfinite(actual)
        assert torch.isfinite(grad).all()
        # Negative: κ = 1, so weighted MLE gives
        # −[log sigmoid(−1000) + log sigmoid(−999)] / 2 = 999.5.
        # Positive: max-shift avoids κ underflow and both log q(A) ≈ 0.
        expected = 999.5 if draft_logit < 0 else 0.0
        assert actual.item() == pytest.approx(expected, abs=1e-3)

    def test_empty_mask_has_zero_loss_and_gradient(self):
        logits, targets, token_ids, loss_mask = self._draft2()
        logits.requires_grad_(True)
        loss_mask.zero_()
        loss, metrics = compute_metrics(
            logits,
            targets,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config('{"ce": 0.1, "vsd": 0.9}'),
            token_ids=token_ids,
        )
        (grad,) = torch.autograd.grad(loss, logits)
        assert loss.item() == 0.0
        assert torch.equal(grad, torch.zeros_like(grad))
        for name in ("vsd_loss", "vsd_elbo", "vsd_kappa_mean"):
            assert metrics[f"{name}_sum"].item() == 0.0

    def test_masked_slots_do_not_affect_loss_or_gradient(self):
        logits, targets, token_ids, loss_mask = self._draft2()
        loss_mask[:, :2] = 0
        loss_mask[:, 3] = 0
        logits.requires_grad_(True)
        terms = compute_eal_terms(logits, targets, token_ids, loss_mask, 2)
        expected = _block_eal_loss(terms, "vsd", 2)
        changed = logits.detach().clone()
        changed[~loss_mask.bool()] = torch.tensor([1000.0, -1000.0])
        changed.requires_grad_(True)
        terms = compute_eal_terms(changed, targets, token_ids, loss_mask, 2)
        actual = _block_eal_loss(terms, "vsd", 2)
        (grad,) = torch.autograd.grad(actual, changed)
        assert actual.item() == pytest.approx(expected.item(), abs=1e-6)
        assert grad[~loss_mask.bool()].abs().sum().item() == 0.0

    def test_metrics_keys(self):
        config = resolve_loss_config('{"ce": 0.1, "vsd": 0.9}')
        assert set(config) == {"ce", "vsd"}
        logits, targets, token_ids, loss_mask = self._draft2()
        loss, metrics = compute_metrics(
            logits,
            targets,
            loss_mask,
            block_size=2,
            loss_config=config,
            sample_from_anchor=True,
            token_ids=token_ids,
        )
        assert torch.isfinite(loss)
        assert "vsd_loss_sum" in metrics
        assert "vsd_elbo_sum" in metrics
        assert "vsd_kappa_mean_sum" in metrics
