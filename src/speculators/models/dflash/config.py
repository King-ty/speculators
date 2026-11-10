from typing import Literal

from pydantic import AliasChoices, Field, StrictInt, model_validator
from transformers import PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators import SpeculatorModelConfig

__all__ = [
    "DFlashSpeculatorConfig",
]


@SpeculatorModelConfig.register("dflash")
class DFlashSpeculatorConfig(SpeculatorModelConfig):
    """
    Configuration for DFlash speculator with vocabulary mapping.

    DFlash features vocabulary mapping between draft (64K) and target (128K)
    vocabularies, enabling cross-tokenizer speculation.

    :param transformer_layer_config: Configuration for the transformer decoder layer
    :param draft_vocab_size: Size of draft model vocabulary for speculation
    """

    speculators_model_type: Literal["dflash"] = "dflash"
    architectures: list[str] = Field(
        default_factory=lambda: ["DFlashSpeculator"],
        description="Model architectures that can load these weights",
    )

    transformer_layer_config: PretrainedConfig = Field(
        default_factory=Qwen3Config,
        description="Configuration for the transformer decoder layer",
    )

    draft_vocab_size: int = Field(
        default=32000,
        description="Size of draft model vocabulary for speculation",
    )

    block_size: int = Field(
        default=8,
        description=(
            "Default size of the draft block predicted with a forward pass of the model"
        ),
    )

    target_hidden_size: int | None = Field(
        default=None,
        description="Hidden size of the target model (if different from draft model)",
    )

    aux_hidden_state_layer_ids: list[int] | None = Field(
        default=None,
        description="Layer IDs of the DFlash auxiliary hidden state layers",
    )

    mask_token_id: int | None = Field(
        default=None,
        description="Token ID used for masking",
    )

    sliding_window_non_causal: bool = Field(
        default=False,
        description="Use non-causal (bidirectional) masking in sliding-window layers. "
        "Full-attention layers remain bidirectional by default. "
        "Overridden by transformer_layer_config.is_causal when set.",
    )

    sample_from_anchor: bool = Field(
        default=False,
        description=(
            "Whether to sample from the anchor position. "
            "False: anchor is the bonus token, only mask tokens predict "
            "(block_size-1 speculative tokens). "
            "True: sample from anchor and all mask positions "
            "(block_size speculative tokens). "
        ),
    )

    realv_lambda: float = Field(
        default=0.05,
        validation_alias=AliasChoices("realv_lambda", "ealv_lambda"),
        description="Weight of the document-grouped variance penalty in REALV.",
    )
    realk_k: StrictInt | None = Field(
        default=None,
        validation_alias=AliasChoices("realk_k", "ealk_k"),
        description=(
            "Positions rewarded by REALK. None derives k from the reference or "
            "current mean."
        ),
    )
    real_guard_on: bool = Field(
        default=False,
        validation_alias=AliasChoices("real_guard_on", "eal_guard_on"),
        description="Enable the mean-preservation guard. Requires real_guard_mu0.",
    )
    real_guard_mu0: float | None = Field(
        default=None,
        validation_alias=AliasChoices("real_guard_mu0", "eal_guard_mu0"),
        description=(
            "Reference acceptance-length mean used by the guard and K derivation."
        ),
    )
    real_guard_delta: float = Field(
        default=0.05,
        validation_alias=AliasChoices("real_guard_delta", "eal_guard_delta"),
        description="Absolute mean drop tolerated before the guard activates.",
    )
    real_guard_rho: float = Field(
        default=0.9,
        validation_alias=AliasChoices("real_guard_rho", "eal_guard_rho"),
        description=(
            "Guard strength per unit of mean shortfall, normalized by block_size."
        ),
    )

    @model_validator(mode="after")
    def validate_real_parameters(self):
        if (
            self.realv_lambda < 0
            or self.real_guard_delta < 0
            or self.real_guard_rho < 0
        ):
            raise ValueError(
                "realv_lambda, real_guard_delta and real_guard_rho must be non-negative"
            )
        if self.realk_k is not None and (
            isinstance(self.realk_k, bool) or self.realk_k < 1
        ):
            raise ValueError("realk_k must be a positive integer or None")
        if self.real_guard_on and self.real_guard_mu0 is None:
            raise ValueError("real_guard_mu0 is required when real_guard_on=True")
        return self

    @property
    def target_vocab_size(self) -> int:
        """Get target vocabulary size from transformer config."""
        return self.transformer_layer_config.vocab_size
