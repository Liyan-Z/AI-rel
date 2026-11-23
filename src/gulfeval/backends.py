"""Model backends. All expose complete(prompt) -> str. Add a new provider by subclassing Backend."""
from __future__ import annotations

import hashlib
import os
import random
import time


class Backend:
    def __init__(self, name: str, temperature: float = 0.0, max_tokens: int = 8):
        self.name, self.temperature, self.max_tokens = name, temperature, max_tokens

    def complete(self, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError


class MockBackend(Backend):
    """Deterministic fake model for tests/dry runs. Gets harder as the text contains Arabic-dialect markers."""

    def complete(self, prompt: str) -> str:
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        rnd = random.Random(h ^ int(self.temperature * 1000) ^ random.getrandbits(8 if self.temperature else 0))
        # Pretend the model knows the answer encoded in a hidden hint when present.
        if "##ANS=" in prompt:
            gold = prompt.split("##ANS=")[1][0]
            hard = "##DIALECT" in prompt
            if rnd.random() < (0.6 if hard else 0.8):
                return gold
        return rnd.choice("ABCD")


class AnthropicBackend(Backend):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def complete(self, prompt: str) -> str:
        r = self.client.messages.create(
            model=self.name, max_tokens=self.max_tokens, temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class OpenAIBackend(Backend):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        import openai

        self.client = openai.OpenAI()  # reads OPENAI_API_KEY

    def complete(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model=self.name, temperature=self.temperature, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content or ""


class HFBackend(Backend):
    """Local open model through transformers (e.g. on a free Colab GPU)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        from transformers import pipeline

        self.pipe = pipeline("text-generation", model=self.name, device_map="auto", torch_dtype="auto",
                             token=os.environ.get("HF_TOKEN"))

    def complete(self, prompt: str) -> str:
        kw = dict(max_new_tokens=self.max_tokens, return_full_text=False)
        if self.temperature > 0:
            kw.update(do_sample=True, temperature=self.temperature)
        else:
            kw.update(do_sample=False)
        msgs = [{"role": "user", "content": prompt}]
        out = self.pipe(msgs, **kw)[0]["generated_text"]
        return out if isinstance(out, str) else out[-1]["content"]


def make_backend(cfg: dict) -> Backend:
    m = cfg["model"]
    cls = {"mock": MockBackend, "anthropic": AnthropicBackend, "openai": OpenAIBackend, "hf": HFBackend}[m["backend"]]
    return cls(m["name"], m.get("temperature", 0.0), m.get("max_tokens", 8))


def with_retries(fn, tries: int = 5, base: float = 1.5):
    for i in range(tries):
        try:
            return fn()
        except Exception:  # noqa: BLE001 - provider SDKs raise many types; back off and retry
            if i == tries - 1:
                raise
            time.sleep(base ** i + random.random())
