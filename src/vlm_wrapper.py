"""Utilities for accessing and intervening on pretrained VLMs.

This module provides a :class:`VLM_Wrapper` class that loads a
Hugging Face vision–language model and exposes helper methods for
extracting internal activations and performing interventions during
generation. It is designed to work out-of-the-box with LLaVA-style
models (e.g., LlavaForConditionalGeneration with SigLIP vision tower
and Qwen2 language model) while remaining generic for other VLMs.

Key improvements for LLaVA compatibility:
- Robust layer path resolution (e.g., "language_model.layers.0" will
  resolve to "model.language_model.layers.0" when needed)
- Chat template support (via tokenizer/processor.apply_chat_template)
- Safer forward-hook handling for tuple outputs
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import logging

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


class VLM_Wrapper:
    """Thin wrapper around a Hugging Face VLM model.

    Parameters
    ----------
    model_name:
        Identifier of the pretrained model on the Hugging Face hub.
    device:
        Torch device onto which the model and inputs will be moved.
    use_chat_template:
        If True and available, use the model's chat template for prompt
        formatting (recommended for LLaVA-family models). If None, this
        is auto-detected.
    chat_system_prompt:
        Optional system prompt to prepend when using chat templates.
    model_kwargs:
        Additional keyword arguments passed to
        :func:`transformers.AutoModelForCausalLM.from_pretrained`.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_chat_template: Optional[bool] = None,
        chat_system_prompt: Optional[str] = None,
        **model_kwargs,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.chat_system_prompt = chat_system_prompt
        model_kwargs.setdefault("trust_remote_code", True)

        # Load model and accompanying processors/tokenizers.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        ).to(device)

        # Processor is preferred for multimodal models
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=model_kwargs["trust_remote_code"]
            )
        except Exception:
            self.processor = None

        # Tokenizer is always required for decoding and text-only fallback
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=model_kwargs["trust_remote_code"]
        )

        # Determine chat template availability and usage
        self._apply_chat_fn = None
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            self._apply_chat_fn = self.processor.apply_chat_template  # type: ignore[attr-defined]
        elif hasattr(self.tokenizer, "apply_chat_template"):
            self._apply_chat_fn = self.tokenizer.apply_chat_template  # type: ignore[attr-defined]

        self.use_chat_template = (
            use_chat_template if use_chat_template is not None else self._apply_chat_fn is not None
        )

        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------
    # Module resolution and hook management
    # ------------------------------------------------------------------
    def _iter_named_modules(self) -> Iterable[Tuple[str, torch.nn.Module]]:
        # Include full module tree from the model root
        yield from self.model.named_modules()

    def list_layers(self, pattern: Optional[str] = None) -> List[str]:
        """List all submodule paths. Optionally filter by substring pattern."""
        names = [name for name, _ in self._iter_named_modules()]
        if pattern:
            names = [n for n in names if pattern in n]
        return names

    def _resolve_layer_name(self, layer_spec: str) -> str:
        """Resolve a possibly shortened layer path to an exact module path.

        Examples
        --------
        - "language_model.layers.0" -> "model.language_model.layers.0" (for LLaVA)
        - If the spec already matches an exact path, it is returned unchanged.
        - If exactly one module path endswith the spec, that path is returned.
        """
        # Exact match first
        for name, _ in self._iter_named_modules():
            if name == layer_spec:
                return layer_spec

        # Endswith unique match
        candidates = [name for name, _ in self._iter_named_modules() if name.endswith(layer_spec)]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            msg = f"Layer spec '{layer_spec}' is ambiguous; candidates: {candidates[:10]}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Heuristic prefixes commonly seen in VLMs
        prefixes = ["model", "base_model.model", "base_model", "transformer", "language_model"]
        for prefix in prefixes:
            candidate = f"{prefix}.{layer_spec}" if prefix else layer_spec
            for name, _ in self._iter_named_modules():
                if name == candidate:
                    return candidate

        # As a last resort, raise a helpful error
        available = [name for name, _ in self._iter_named_modules()]
        hint = "; e.g., try 'model." + layer_spec + "'" if f"model.{layer_spec}" in ",".join(available) else ""
        raise ValueError(
            f"Could not resolve layer spec '{layer_spec}'.{hint} Consider calling list_layers() to inspect available paths."
        )

    def _register_hooks(self, layers_to_hook: List[str]) -> Dict[str, torch.Tensor]:
        """Register forward hooks on the specified layers and return a dict that
        will be populated with activations during the next forward pass."""

        activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

        def get_activation_hook(public_name: str):
            def hook(_module, _inp, output):
                # Normalize output to a tensor
                out = output[0] if isinstance(output, tuple) else output
                if isinstance(out, torch.Tensor):
                    activations[public_name] = out.detach()
                else:
                    # If the module returns non-tensor (rare), skip recording
                    self.logger.warning("Hooked module '%s' returned non-tensor output; skipping.", public_name)
            return hook

        for spec in layers_to_hook:
            resolved = self._resolve_layer_name(spec)
            layer = self.model.get_submodule(resolved)
            handle = layer.register_forward_hook(get_activation_hook(spec))
            self.hooks.append(handle)

        return activations

    def _clear_hooks(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------
    def _build_chat_texts(
        self, texts: Sequence[str], images: Optional[Sequence[Optional[Image.Image]]]
    ) -> Sequence[str]:
        if not self.use_chat_template or self._apply_chat_fn is None:
            return texts

        # Build per-sample conversation following HF multimodal schema
        conversations: List[List[dict]] = []
        n = len(texts)
        imgs = list(images) if images is not None else [None] * n
        if len(imgs) != n:
            raise ValueError("Number of images must match number of texts when using chat templates.")

        for i in range(n):
            content: Union[str, List[dict]]
            if imgs[i] is None:
                content = texts[i]
            else:
                content = [
                    {"type": "text", "text": texts[i]},
                    {"type": "image"},
                ]
            msgs: List[dict] = []
            if self.chat_system_prompt:
                msgs.append({"role": "system", "content": self.chat_system_prompt})
            msgs.append({"role": "user", "content": content})
            conversations.append(msgs)

        rendered = self._apply_chat_fn(  # type: ignore[misc]
            conversations, add_generation_prompt=True, tokenize=False
        )
        # HF apply_chat_template may return a single string when given one conv
        if isinstance(rendered, str):
            rendered = [rendered]
        return rendered

    def _prepare_inputs(
        self,
        text_prompts: Sequence[str],
        images: Optional[Sequence[Optional[Image.Image]]] = None,
    ) -> Dict[str, torch.Tensor]:
        texts = list(text_prompts)
        imgs = list(images) if images is not None else None

        # Prefer processor when available (multimodal-ready)
        if self.processor is not None:
            # Use chat template if available for models like LLaVA
            chat_texts = self._build_chat_texts(texts, imgs)
            inputs = self.processor(
                text=chat_texts,
                images=imgs,
                return_tensors="pt",
                padding=True,
            )
            return inputs.to(self.device)

        # Fallback to tokenizer only (text-only models)
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        return inputs.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_activations(
        self,
        text_prompts: Union[str, Sequence[str]],
        images: Optional[Union[Image.Image, Sequence[Optional[Image.Image]]]],
        layers: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Extract activations from specific layers in batch.

        Notes
        -----
        For LLaVA-style models, typical layer specs include e.g.:
        - "model.language_model.layers.0"
        - "model.multi_modal_projector"
        - "model.vision_tower.vision_model.encoder.layers.0"
        Shorter forms like "language_model.layers.0" are also accepted.
        """

        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        if images is None:
            images_seq: Optional[Sequence[Optional[Image.Image]]] = None
        elif isinstance(images, Image.Image):
            images_seq = [images]
        else:
            images_seq = images

        inputs = self._prepare_inputs(text_prompts, images_seq)
        activations = self._register_hooks(layers)

        with torch.no_grad():
            _ = self.model(**inputs)

        self._clear_hooks()
        return activations

    def generate(
        self, text_prompt: str, image: Optional[Image.Image] = None, **gen_kwargs
    ) -> str:
        """Generate text from the model without interventions."""

        inputs = self._prepare_inputs([text_prompt], [image] if image is not None else None)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_with_intervention(
        self,
        text_prompt: str,
        layer: str,
        intervention_fn: Callable[[torch.Tensor], torch.Tensor],
        image: Optional[Image.Image] = None,
        **gen_kwargs,
    ) -> str:
        """Generate text while applying a hook-based intervention.

        The intervention function should accept and return a tensor.
        If the hooked module outputs a tuple, only the first element is
        modified and the original structure is preserved.
        """

        resolved = self._resolve_layer_name(layer)
        module = self.model.get_submodule(resolved)

        def hook(_module, _inp, output):
            if isinstance(output, tuple):
                out0 = output[0]
                if isinstance(out0, torch.Tensor):
                    new0 = intervention_fn(out0)
                    # Recreate tuple with modified first element
                    return (new0,) + output[1:]
                return output
            if isinstance(output, torch.Tensor):
                return intervention_fn(output)
            return output

        handle = module.register_forward_hook(hook)
        inputs = self._prepare_inputs([text_prompt], [image] if image is not None else None)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        handle.remove()
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


__all__ = ["VLM_Wrapper"]
