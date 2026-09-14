"""Ollama backend: a local model over HTTP, with schema-constrained decoding.

Ollama serves quantised GGUF weights behind an HTTP API on port 11434. Since 0.5 its
`/api/generate` endpoint takes a `format` field carrying a JSON schema, and constrains decoding to
it, so the model cannot emit a token that would break the structure. That is stronger than asking
for JSON in the prompt and hoping.

Two consequences shape this file.

**The repair loop still exists and is still worth measuring.** Constrained decoding guarantees the
*shape*, not the *content*. A schema can say the action is one of `concede` or `claim` and the
model will emit one of those two strings, but nothing stops it from emitting the one the game
cannot accept this round, and a confidence field constrained to a number will not be a calibrated
one. So `constrained=False` is offered alongside the default, and the difference between the two
failure rates is the finding: it separates what the server guarantees from what the model gets
right on its own.

**No activations, ever.** There is no torch module here, only a socket, so hidden states are not
merely unavailable but meaningless as a request. `capabilities()` says so rather than returning an
empty dictionary that a results page would render as a blank plot.

Requires `ollama serve` to be running and a model pulled. Everything raises `BackendUnavailable`
with an install hint when it is not, so a test suite on a machine without Ollama skips rather than
fails.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .llm_backends import (
    DEFAULT_MAX_ATTEMPTS, Attempt, BackendUnavailable, Capabilities, Response, SchemaViolation,
    run_with_repair,
)

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2:1b"


class OllamaBackend:
    """A language model served locally by Ollama.

    Args:
        model: the tag as pulled, e.g. `llama3.2:1b`.
        host: where `ollama serve` is listening.
        constrained: pass the schema to the server for constrained decoding. Setting it False
            leaves the schema in the prompt only, which is the comparison arm: it measures what
            the model does unaided, and the gap between the two is what constrained decoding buys.
        temperature: 0.0 by default. **This does not make the backend deterministic** and
            `capabilities()` does not claim it does. It narrows the sampling distribution, while
            a different build of the same tag, a different quantisation or a different server
            version all change the output.
        timeout: seconds per request.
    """

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 constrained: bool = True, temperature: float = 0.0,
                 timeout: float = 120.0) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.constrained = constrained
        self.temperature = temperature
        self.timeout = timeout

    # --- availability -------------------------------------------------------------------

    def is_available(self) -> bool:
        """Is the server up and does it have this model? Never raises, so callers can skip."""
        try:
            return self.model in self.installed_models()
        except BackendUnavailable:
            return False

    def installed_models(self) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5.0) as fh:
                payload = json.loads(fh.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise BackendUnavailable(
                f"no Ollama server at {self.host} ({exc}). Install from ollama.com, then "
                f"`ollama serve` and `ollama pull {self.model}`") from exc
        return [m.get("name", "") for m in payload.get("models", [])]

    # --- the contract -------------------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            name="ollama",
            model=self.model,
            native_schema=self.constrained,
            activations=False,
            deterministic=False,
            note=("Ollama serves the model over HTTP, so there is no module to attach hooks to "
                  "and hidden activations cannot be recorded. Output is not reproducible either: "
                  "temperature 0 narrows sampling but the same tag can be rebuilt, requantised "
                  "or served by a different version."),
        )

    def complete(self, prompt: str, schema: dict[str, Any],
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> Response:
        data, attempts = run_with_repair(
            lambda text: self._generate(text, schema), prompt, schema, max_attempts)
        return Response(data=data, attempts=attempts, activations=None, model=self.model)

    # --- the one HTTP call --------------------------------------------------------------

    def _generate(self, prompt: str, schema: dict[str, Any]) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.constrained:
            body["format"] = schema

        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as fh:
                payload = json.loads(fh.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise BackendUnavailable(
                f"Ollama refused the request ({exc.code}): {detail}. If the model is missing, "
                f"`ollama pull {self.model}`") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise BackendUnavailable(f"could not reach Ollama at {self.host}: {exc}") from exc

        return payload.get("response", "")
