# vlm_wrapper.py
"""Utilities for accessing and intervening on pretrained VLMs.

- VLM(Text+Image): AutoModelForVision2Seq
- Text-only LMs : AutoModelForCausalLM

Robust layer resolution, chat template support, safe forward hooks.
"""

from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import logging
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

VLM_MODEL_TYPES = {
    # Common HF model_type strings for VLMs
    "llava_onevision", "llava", "mllava", "phi4_multimodal",
    "fuyu", "qwen2_vl", "glm4v", "got_ocr2", "git",
    "tr_ocr", "whisper_vision", "mllama", "emu3",
}

class VLM_Wrapper:
    """Thin wrapper around a Hugging Face VLM/LM model.

    Parameters
    ----------
    model_name: str
    device: str = "cuda"
    use_chat_template: Optional[bool] = None
    chat_system_prompt: Optional[str] = None
    **model_kwargs: forwarded to .from_pretrained (trust_remote_code=True by default)
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

        # sensible defaults
        model_kwargs.setdefault("trust_remote_code", True)
        trust_remote_code = model_kwargs["trust_remote_code"]

        # --- Detect model type first
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model_type = getattr(config, "model_type", None) or ""
        self.is_vlm = self.model_type.lower() in VLM_MODEL_TYPES

        # --- Load model
        if self.is_vlm:
            self.model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)

        # --- Load processor/tokenizer
        self.processor = None
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        except Exception:
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

        # Prefer tokenizer embedded in processor if present (e.g., LLaVA)
        self.decode_tokenizer = getattr(self.processor, "tokenizer", None) or self.tokenizer

        # --- Chat template availability
        self._apply_chat_fn = None
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            self._apply_chat_fn = self.processor.apply_chat_template  # type: ignore[attr-defined]
        elif hasattr(self.tokenizer, "apply_chat_template"):
            self._apply_chat_fn = self.tokenizer.apply_chat_template  # type: ignore[attr-defined]

        self.use_chat_template = (
            use_chat_template if use_chat_template is not None else self._apply_chat_fn is not None
        )

        # generation safety (pad/eos)
        if getattr(self.decode_tokenizer, "pad_token_id", None) is None:
            # fall back to eos as pad if needed
            try:
                self.decode_tokenizer.pad_token = self.decode_tokenizer.eos_token
            except Exception:
                pass

        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------
    # Module tree & hooks
    # ------------------------------------------------------------------
    def _iter_named_modules(self) -> Iterable[Tuple[str, torch.nn.Module]]:
        yield from self.model.named_modules()

    def list_layers(self, pattern: Optional[str] = None) -> List[str]:
        names = [name for name, _ in self._iter_named_modules()]
        if pattern:
            names = [n for n in names if pattern in n]
        return names

    def _resolve_layer_name(self, layer_spec: str) -> str:
        # exact
        for name, _ in self._iter_named_modules():
            if name == layer_spec:
                return layer_spec

        # unique suffix
        candidates = [name for name, _ in self._iter_named_modules() if name.endswith(layer_spec)]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            msg = f"Layer spec '{layer_spec}' is ambiguous; candidates: {candidates[:10]}"
            self.logger.error(msg)
            raise ValueError(msg)

        # common prefixes
        prefixes = ["model", "base_model.model", "base_model", "transformer", "language_model"]
        for prefix in prefixes:
            candidate = f"{prefix}.{layer_spec}"
            for name, _ in self._iter_named_modules():
                if name == candidate:
                    return candidate

        available = [name for name, _ in self._iter_named_modules()]
        hint = "; try 'model." + layer_spec + "'" if any(n.endswith("model."+layer_spec) for n in available) else ""
        raise ValueError(
            f"Could not resolve layer spec '{layer_spec}'.{hint} Call list_layers() to inspect available paths."
        )

    def _register_hooks(self, layers_to_hook: List[str]) -> Dict[str, torch.Tensor]:
        activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

        def get_activation_hook(public_name: str):
            def hook(_module, _inp, output):
                out = output[0] if isinstance(output, tuple) else output
                if isinstance(out, torch.Tensor):
                    activations[public_name] = out.detach()
                else:
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
                content = [{"type": "text", "text": texts[i]}, {"type": "image"}]
            msgs: List[dict] = []
            if self.chat_system_prompt:
                msgs.append({"role": "system", "content": self.chat_system_prompt})
            msgs.append({"role": "user", "content": content})
            conversations.append(msgs)

        rendered = self._apply_chat_fn(conversations, add_generation_prompt=True, tokenize=False)  # type: ignore[misc]
        return [rendered] if isinstance(rendered, str) else rendered

    def _prepare_inputs(
        self,
        text_prompts: Sequence[str],
        images: Optional[Sequence[Optional[Image.Image]]] = None,
    ) -> Dict[str, torch.Tensor]:
        texts = list(text_prompts)
        imgs = list(images) if images is not None else None

        if self.processor is not None:
            chat_texts = self._build_chat_texts(texts, imgs)
            inputs = self.processor(
                text=chat_texts,
                images=imgs,
                return_tensors="pt",
                padding=True,
            )
            return inputs.to(self.device)

        # tokenizer-only fallback (text models)
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

    def generate(self, text_prompt: str, image: Optional[Image.Image] = None, **gen_kwargs) -> str:
        inputs = self._prepare_inputs([text_prompt], [image] if image is not None else None)
        # generation kwargs safety
        gen_kwargs.setdefault("pad_token_id", getattr(self.decode_tokenizer, "pad_token_id", None))
        gen_kwargs.setdefault("eos_token_id", getattr(self.decode_tokenizer, "eos_token_id", None))

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        return self.decode_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_with_intervention(
        self,
        text_prompt: str,
        layer: str,
        intervention_fn: Callable[[torch.Tensor], torch.Tensor],
        image: Optional[Image.Image] = None,
        **gen_kwargs,
    ) -> str:
        resolved = self._resolve_layer_name(layer)
        module = self.model.get_submodule(resolved)

        def hook(_module, _inp, output):
            if isinstance(output, tuple):
                out0 = output[0]
                if isinstance(out0, torch.Tensor):
                    new0 = intervention_fn(out0)
                    return (new0,) + output[1:]
                return output
            if isinstance(output, torch.Tensor):
                return intervention_fn(output)
            return output

        handle = module.register_forward_hook(hook)
        inputs = self._prepare_inputs([text_prompt], [image] if image is not None else None)

        gen_kwargs.setdefault("pad_token_id", getattr(self.decode_tokenizer, "pad_token_id", None))
        gen_kwargs.setdefault("eos_token_id", getattr(self.decode_tokenizer, "eos_token_id", None))

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        handle.remove()
        return self.decode_tokenizer.decode(output_ids[0], skip_special_tokens=True)


__all__ = ["VLM_Wrapper"]
