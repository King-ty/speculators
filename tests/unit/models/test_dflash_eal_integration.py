"""Tiny CPU integration tests for REAL losses with the public DFlash model."""

from typing import Any

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.losses import resolve_loss_config
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.train.config.schema import DFlashArgs, LossArgs, TrainConfig


def _tiny_model(sample_from_anchor: bool) -> DFlashDraftModel:
    torch.manual_seed(0)
    layer_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    config = DFlashSpeculatorConfig(
        transformer_layer_config=layer_config,
        draft_vocab_size=64,
        block_size=4,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
        sample_from_anchor=sample_from_anchor,
    )
    model = DFlashDraftModel(config)
    torch.nn.init.normal_(model.embed_tokens.weight)
    torch.nn.init.normal_(model.lm_head.weight)
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    torch.nn.init.ones_(model.verifier_norm.weight)
    return model.eval()


@pytest.mark.parametrize("sample_from_anchor", [False, True])
@pytest.mark.parametrize("loss_name", ["real", "ealb"])
def test_dflash_forward_supports_real(sample_from_anchor, loss_name):
    model = _tiny_model(sample_from_anchor)
    generator = torch.Generator().manual_seed(1)
    batch = {
        "hidden_states": torch.randn(1, 32, 32, generator=generator),
        "input_ids": torch.randint(0, 64, (1, 32), generator=generator),
        "loss_mask": torch.ones(1, 32),
        "verifier_last_hidden_states": torch.randn(1, 32, 16, generator=generator),
        "document_ids": torch.zeros(1, 32, dtype=torch.long),
    }

    _, loss, metrics = model(
        **batch,
        max_anchors=8,
        loss_config=resolve_loss_config(f'{{"ce": 0.1, "{loss_name}": 0.9}}', "eager"),
    )

    assert torch.isfinite(loss)
    assert "eal_sum" in metrics
    assert "full_acc_sum" in metrics


def test_real_config_fields():
    parameters: dict[str, Any] = {
        "realv_lambda": 0.01,
        "realk_k": 2,
        "real_guard_on": True,
        "real_guard_mu0": 4.362,
        "real_guard_delta": 0.1,
        "real_guard_rho": 0.5,
    }
    for config_type in (DFlashSpeculatorConfig, DFlashArgs):
        config = config_type(**parameters)
        for key, expected in parameters.items():
            assert getattr(config, key) == expected


@pytest.mark.parametrize("config_type", [DFlashSpeculatorConfig, DFlashArgs])
def test_legacy_config_aliases(config_type):
    config = config_type(
        ealv_lambda=0.01,
        ealk_k=2,
        eal_guard_on=True,
        eal_guard_mu0=4.362,
        eal_guard_delta=0.1,
        eal_guard_rho=0.5,
    )
    assert config.realv_lambda == 0.01
    assert config.realk_k == 2
    assert config.real_guard_on is True
    assert config.real_guard_mu0 == 4.362
    assert config.real_guard_delta == 0.1
    assert config.real_guard_rho == 0.5


@pytest.mark.parametrize("legacy", [False, True])
def test_real_trainer_kwargs(legacy):
    parameters: dict[str, Any] = {
        "realv_lambda": 0.01,
        "realk_k": 2,
        "real_guard_on": True,
        "real_guard_mu0": 4.362,
        "real_guard_delta": 0.1,
        "real_guard_rho": 0.5,
    }
    if legacy:
        parameters = {
            key.replace("realv", "ealv")
            .replace("realk", "ealk")
            .replace("real_guard", "eal_guard"): value
            for key, value in parameters.items()
        }
    train, val = DFlashDraftModel.get_trainer_kwargs(
        loss_fn="ealk" if legacy else "realk",
        loss_implementation="eager",
        **parameters,
    )
    for key, expected in {
        "realv_lambda": 0.01,
        "realk_k": 2,
        "real_guard_on": True,
        "real_guard_mu0": 4.362,
        "real_guard_delta": 0.1,
        "real_guard_rho": 0.5,
    }.items():
        assert train[key] == expected
        assert val[key] == expected


def test_legacy_real_parameters_reach_forward():
    model = _tiny_model(False)
    generator = torch.Generator().manual_seed(1)
    batch = {
        "hidden_states": torch.randn(1, 32, 32, generator=generator),
        "input_ids": torch.randint(0, 64, (1, 32), generator=generator),
        "loss_mask": torch.ones(1, 32),
        "verifier_last_hidden_states": torch.randn(1, 32, 16, generator=generator),
        "document_ids": torch.zeros(1, 32, dtype=torch.long),
    }
    _, loss, metrics = model(
        **batch,
        max_anchors=8,
        loss_config=resolve_loss_config('{"ce": 0.1, "ealv": 0.9}', "eager"),
        ealv_lambda=0.01,
        ealk_k=2,
        eal_guard_on=True,
        eal_guard_mu0=4.362,
        eal_guard_delta=0.1,
        eal_guard_rho=0.5,
    )
    assert torch.isfinite(loss)
    assert metrics["ealv_guard_sum"].item() > 0
    assert "ealv_var_sum" in metrics
    assert "realv_var_sum" not in metrics


def test_guard_reference_is_required_by_both_config_types():
    with pytest.raises(ValueError, match="real_guard_mu0 is required"):
        DFlashSpeculatorConfig(real_guard_on=True)
    with pytest.raises(ValueError, match="real_guard_mu0 is required"):
        DFlashArgs(real_guard_on=True)


def test_real_parameters_reject_invalid_values():
    with pytest.raises(ValueError, match="non-negative"):
        DFlashSpeculatorConfig(realv_lambda=-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        DFlashArgs(realv_lambda=-0.1)
    with pytest.raises(ValueError, match="positive integer"):
        DFlashSpeculatorConfig(realk_k=0)
    with pytest.raises(ValueError, match="positive integer"):
        DFlashArgs(realk_k=0)
    with pytest.raises(ValueError, match="valid integer"):
        DFlashSpeculatorConfig(realk_k=True)
    with pytest.raises(ValueError, match="valid integer"):
        DFlashArgs(realk_k=True)


@pytest.mark.parametrize("loss_name", ["wtv", "realk"])
def test_default_anchor_layout_has_nonzero_block_loss_gradient(loss_name):
    model = _tiny_model(False)
    generator = torch.Generator().manual_seed(1)
    batch = {
        "hidden_states": torch.randn(1, 32, 32, generator=generator),
        "input_ids": torch.randint(0, 64, (1, 32), generator=generator),
        "loss_mask": torch.ones(1, 32),
        "verifier_last_hidden_states": torch.randn(1, 32, 16, generator=generator),
        "document_ids": torch.zeros(1, 32, dtype=torch.long),
    }
    _, loss, _ = model(
        **batch,
        max_anchors=8,
        loss_config=resolve_loss_config(loss_name, "eager"),
        realk_k=1,
    )
    (gradient,) = torch.autograd.grad(loss, model.fc.weight)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_forward_stops_real_trajectory_at_document_boundary(monkeypatch):
    model = _tiny_model(False)
    monkeypatch.setattr(
        "speculators.models.dflash.core.select_anchors",
        lambda *_args: (torch.tensor([0]), torch.tensor([True])),
    )
    generator = torch.Generator().manual_seed(1)
    document_ids = torch.zeros(1, 32, dtype=torch.long)
    document_ids[:, 1:] = 1
    _, loss, metrics = model(
        hidden_states=torch.randn(1, 32, 32, generator=generator),
        input_ids=torch.randint(0, 64, (1, 32), generator=generator),
        loss_mask=torch.ones(1, 32),
        verifier_last_hidden_states=torch.randn(1, 32, 16, generator=generator),
        document_ids=document_ids,
        max_anchors=1,
        loss_config=resolve_loss_config("real", "eager"),
    )
    assert loss.item() == 0.0
    assert metrics["eal_j_b_total"].item() == 0


def _tiny_reduced_vocab_model():
    torch.manual_seed(0)
    layer_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )
    model = DFlashDraftModel(
        DFlashSpeculatorConfig(
            transformer_layer_config=layer_config,
            draft_vocab_size=32,
            block_size=4,
            aux_hidden_state_layer_ids=[0, 1],
            mask_token_id=0,
            sample_from_anchor=False,
        )
    )
    torch.nn.init.normal_(model.embed_tokens.weight)
    torch.nn.init.normal_(model.lm_head.weight)
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    torch.nn.init.ones_(model.verifier_norm.weight)
    return model.eval()


def test_reduced_vocab_ce_works_and_real_has_clear_error():
    model = _tiny_reduced_vocab_model()
    generator = torch.Generator().manual_seed(1)
    batch = {
        "hidden_states": torch.randn(1, 16, 32, generator=generator),
        "input_ids": torch.full((1, 16), 63, dtype=torch.long),
        "loss_mask": torch.ones(1, 16),
        "verifier_last_hidden_states": torch.randn(1, 16, 16, generator=generator),
        "document_ids": torch.zeros(1, 16, dtype=torch.long),
    }
    _, loss, _ = model(
        **batch,
        max_anchors=4,
        loss_config=resolve_loss_config("ce", "eager"),
    )
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="equal target and draft vocabularies"):
        model(
            **batch,
            max_anchors=4,
            loss_config=resolve_loss_config("real", "eager"),
        )


def test_documented_real_config_passes_validation():
    config = TrainConfig(
        speculator_type="dflash",
        loss=LossArgs(loss_fn='{"ce": 0.1, "real": 0.9}'),
        dflash=DFlashArgs(per_position_loss_weight="fixed-exp-decay", block_size=8),
    )
    assert config.loss.loss_fn == '{"ce": 0.1, "real": 0.9}'
    assert config.dflash.per_position_loss_weight == "fixed-exp-decay"
    assert config.dflash.block_size == 8
