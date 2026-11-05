"""Compatibility shims for the pinned dependency stack (vLLM 0.8.5 + transformers 5.3)."""

from __future__ import annotations


def patch_vllm_tokenizer_compat() -> None:
    """Let vLLM 0.8.5 load transformers-5 tokenizers.

    vLLM 0.8.5's get_cached_tokenizer reads `all_special_tokens_extended`, which
    transformers 5 removed from TokenizersBackend. vLLM only caches and re-exposes
    the value (typed list[str]); nothing in generation consumes it, so aliasing it
    to `all_special_tokens` does not change model outputs.
    """
    try:
        from transformers import TokenizersBackend
    except ImportError:  # transformers 4.x still defines the attribute
        return
    if "all_special_tokens_extended" not in vars(TokenizersBackend):
        TokenizersBackend.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens)
