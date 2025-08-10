from typing import List, Callable, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

class VLMWrapper:
    """Wrapper providing standardized access to VLM internals."""

    def __init__(self, model_name: str, device: str, **model_kwargs) -> None:
        """Load a pretrained VLM model and associated processors.

        Parameters
        ----------
        model_name: str
            Hugging Face model identifier.
        device: str
            Device mapping for model weights.
        model_kwargs: dict
            Additional keyword arguments forwarded to
            :func:`~transformers.AutoModelForCausalLM.from_pretrained`.
        """

        self.device = device
        model_kwargs.setdefault("trust_remote_code", True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=model_kwargs["trust_remote_code"])
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=model_kwargs["trust_remote_code"])
        except Exception:
            self.processor = None

    def _prepare_inputs(self, prompts: List[str], image=None) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        if image is not None and self.processor is not None:
            processed = self.processor(images=image, return_tensors="pt").to(self.device)
            inputs.update(processed)
        return inputs

    def get_activations(self, prompts: List[str], layers: List[str], image=None) -> Dict[str, torch.Tensor]:
        activations: Dict[str, torch.Tensor] = {}
        handles = []
        for layer_name in layers:
            module = self.model.get_submodule(layer_name)
            def hook(module, inp, output, name=layer_name):
                activations[name] = output.detach().cpu()
            handles.append(module.register_forward_hook(hook))

        inputs = self._prepare_inputs(prompts, image)
        with torch.no_grad():
            self.model(**inputs)
        for h in handles:
            h.remove()
        return activations

    def generate_with_intervention(self, prompt: str, layer: str, intervention_fn: Callable[[torch.Tensor], torch.Tensor], image=None) -> str:
        module = self.model.get_submodule(layer)

        def hook(module, inp, output):
            return intervention_fn(output)

        handle = module.register_forward_hook(hook)

        inputs = self._prepare_inputs([prompt], image)
        with torch.no_grad():
            generated = self.model.generate(**inputs)
        handle.remove()
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text
