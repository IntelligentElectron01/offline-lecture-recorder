"""Local LLM calls (Ollama on loopback only), with the context guard and a per-request size log.

Every request:
- sets `num_ctx` explicitly (Ollama's server default stays at 4k) and `think=False`;
- sends `truncate: false`, so Ollama answers HTTP 400 instead of silently dropping the start of a
  prompt that is too long (measured on 0.34.4: without it a 3,217-token prompt at num_ctx=1024 was
  cut to 514 tokens with no error);
- is refused before sending if the estimated prompt is over budget, and fails after if the
  evaluated prompt is over budget, prompt + output cap does not fit in num_ctx, or the answer ended
  with done_reason "length" (output cut off);
- is logged with sizes only (num_ctx, prompt_eval_count, eval_count, done_reason), never text.

No model is ever pulled: a missing model is an error, never a download.
"""
import ipaddress
import logging
import math
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

log = logging.getLogger("lecture_recorder.llm")
CHARS_PER_TOKEN = 3.0  # conservative: measured ~3.85 for timestamped English transcript lines on qwen3
SAMPLING = {"temperature": 0.3, "top_p": 0.8, "top_k": 20, "seed": 7}  # Qwen3 non-thinking guidance; fixed seed


class LLMError(RuntimeError):
    pass


class OllamaUnavailable(LLMError):
    pass


class ContextOverflow(LLMError):
    """The prompt (or prompt + output cap) does not fit: refusing rather than truncating."""


class OutputTruncated(LLMError):
    """The model stopped because it hit the output cap (done_reason 'length')."""


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN) + 8


def check_local_host(url: str) -> str:
    """Only http://<loopback IP>:<port> is allowed. Hostnames (even 'localhost') are refused, so the
    hosts file or DNS can never send transcript text anywhere else."""
    parts = urlsplit(url)
    try:
        ok = parts.scheme == "http" and ipaddress.ip_address(parts.hostname or "").is_loopback
    except ValueError:
        ok = False
    if not ok or parts.path not in ("", "/") or parts.username or parts.query:
        raise ValueError(f"refusing Ollama host {url!r}: only a loopback address like http://127.0.0.1:11434 is allowed")
    return f"http://{parts.netloc}"


@dataclass
class LocalLLM:
    host: str
    model: str
    num_ctx: int = 8192
    max_prompt_tokens: int = 5000
    num_predict: int = 2000
    timeout_s: float = 900
    keep_alive: str = "5m"  # stays loaded between the requests of one job; unload() ends it
    requests: list[dict] = field(default_factory=list)  # the size log, one entry per request
    _http: httpx.Client | None = field(default=None, repr=False)

    def __post_init__(self):
        self.host = check_local_host(self.host)
        if self.max_prompt_tokens + self.num_predict > self.num_ctx:
            raise ValueError("max_prompt_tokens + num_predict must fit in num_ctx")

    @classmethod
    def from_config(cls, llm_cfg) -> "LocalLLM":
        return cls(host=llm_cfg.ollama_host, model=llm_cfg.summary_model, num_ctx=llm_cfg.num_ctx,
                   max_prompt_tokens=llm_cfg.max_prompt_tokens, num_predict=llm_cfg.num_predict,
                   timeout_s=llm_cfg.timeout_s)

    @property
    def http(self) -> httpx.Client:
        if self._http is None:  # no proxies from the environment, no redirects, no API keys
            self._http = httpx.Client(base_url=self.host, trust_env=False, follow_redirects=False,
                                      timeout=httpx.Timeout(self.timeout_s, connect=5))
        return self._http

    def _post(self, path: str, body: dict) -> httpx.Response:
        return self._send("POST", path, json=body)

    def _send(self, method: str, path: str, **kw) -> httpx.Response:
        """Ollama not running fails in about a second (refused) or 5 s (connect timeout), with this message."""
        try:
            return self.http.request(method, path, **kw)
        except httpx.TransportError as e:
            raise OllamaUnavailable(f"Ollama is not reachable at {self.host} ({type(e).__name__}). "
                                    "Is it installed and running?") from e

    def chat(self, system: str, user: str, *, stage: str, part: int | None = None,
             num_predict: int | None = None, fmt: dict | str | None = None) -> str:
        num_predict = num_predict or self.num_predict
        est = estimate_tokens(system) + estimate_tokens(user)
        entry = {"stage": stage, "part": part, "model": self.model, "num_ctx": self.num_ctx,
                 "num_predict": num_predict, "prompt_chars": len(system) + len(user), "est_prompt_tokens": est,
                 "prompt_eval_count": None, "eval_count": None, "done_reason": None, "seconds": None, "error": None}
        self.requests.append(entry)
        try:
            if est > self.max_prompt_tokens:
                raise ContextOverflow(f"{stage}: estimated prompt {est} tokens > budget {self.max_prompt_tokens}")
            body = {"model": self.model, "stream": False, "think": False, "truncate": False,
                    "keep_alive": self.keep_alive,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "options": {"num_ctx": self.num_ctx, "num_predict": num_predict, **SAMPLING}}
            if fmt is not None:
                body["format"] = fmt
            t0 = time.monotonic()
            r = self._post("/api/chat", body)
            entry["seconds"] = round(time.monotonic() - t0, 2)
            self._raise_for(r, stage)
            j = r.json()
            entry.update(prompt_eval_count=j.get("prompt_eval_count"), eval_count=j.get("eval_count"),
                         done_reason=j.get("done_reason"))
            self._check(entry)
            return (j.get("message") or {}).get("content", "")
        except LLMError as e:
            entry["error"] = type(e).__name__
            raise
        finally:
            log.info("llm request %s", {k: v for k, v in entry.items() if k != "model"})

    def embed(self, texts: list[str], *, stage: str, keep_alive: str | int | None = None) -> list[list[float]]:
        """Embeddings for `texts` via /api/embed, with the same guarantees as chat(): num_ctx set
        explicitly, `truncate: false` (Ollama answers HTTP 400 "the input length exceeds the context
        length" instead of silently cutting an input; measured on 0.34.4), inputs over the estimated
        per-input budget refused before sending, and a size-only log entry per request."""
        ests = [estimate_tokens(t) for t in texts]
        entry = {"stage": stage, "model": self.model, "num_ctx": self.num_ctx, "inputs": len(texts),
                 "input_chars": sum(len(t) for t in texts), "max_input_chars": max((len(t) for t in texts), default=0),
                 "max_est_tokens": max(ests, default=0), "prompt_eval_count": None, "seconds": None, "error": None}
        self.requests.append(entry)
        try:
            if not texts:
                raise LLMError(f"{stage}: nothing to embed")
            if max(ests) > self.max_prompt_tokens:
                raise ContextOverflow(f"{stage}: an input is ~{max(ests)} tokens > per-input budget {self.max_prompt_tokens}")
            body = {"model": self.model, "input": texts, "truncate": False,
                    "keep_alive": self.keep_alive if keep_alive is None else keep_alive,
                    "options": {"num_ctx": self.num_ctx}}
            t0 = time.monotonic()
            r = self._post("/api/embed", body)
            entry["seconds"] = round(time.monotonic() - t0, 2)
            self._raise_for(r, stage)
            j = r.json()
            p, vecs = j.get("prompt_eval_count"), j.get("embeddings")
            entry["prompt_eval_count"] = p
            if not isinstance(p, int):
                raise LLMError(f"{stage}: Ollama did not report token counts; cannot prove nothing was truncated")
            if p > self.num_ctx * len(texts):
                raise ContextOverflow(f"{stage}: {p} tokens evaluated for {len(texts)} inputs > num_ctx {self.num_ctx} each")
            if not isinstance(vecs, list) or len(vecs) != len(texts) or any(not v for v in vecs):
                raise LLMError(f"{stage}: expected {len(texts)} embeddings, got {len(vecs or [])}")
            return vecs
        except LLMError as e:
            entry["error"] = type(e).__name__
            raise
        finally:
            log.info("embed request %s", {k: v for k, v in entry.items() if k != "model"})

    def _raise_for(self, r: httpx.Response, stage: str) -> None:
        """Map an Ollama error answer to our errors (over-long input/prompt, model not installed)."""
        if r.status_code == 200:
            return
        detail = _error_text(r)
        if "exceed" in detail and "context" in detail:
            raise ContextOverflow(f"{stage}: Ollama refused an input longer than num_ctx {self.num_ctx}: {detail}")
        if r.status_code == 404:
            raise LLMError(f"model {self.model!r} is not installed in Ollama (install it yourself with "
                           f"`ollama pull {self.model}`; the app never downloads models)")
        raise LLMError(f"{stage}: Ollama HTTP {r.status_code}: {detail}")

    def release_models(self, models) -> bool:
        """Unload whichever of `models` (ours) Ollama still holds; True if none is left loaded."""
        loaded = self.loaded_models()
        ok = True
        for model in dict.fromkeys(models):
            if model in loaded:
                ok &= self.unload(model=model)
        return ok

    def _check(self, e: dict) -> None:
        p, n = e["prompt_eval_count"], e["eval_count"]
        if not isinstance(p, int) or not isinstance(n, int):
            raise LLMError(f"{e['stage']}: Ollama did not report token counts; cannot prove nothing was truncated")
        if p >= self.num_ctx or p + e["num_predict"] > self.num_ctx:
            raise ContextOverflow(f"{e['stage']}: prompt {p} + output cap {e['num_predict']} > num_ctx {self.num_ctx}")
        if p > self.max_prompt_tokens:
            raise ContextOverflow(f"{e['stage']}: prompt {p} tokens > budget {self.max_prompt_tokens} "
                                  "(the token estimate was too low for this text)")
        if e["done_reason"] == "length":
            raise OutputTruncated(f"{e['stage']}: output hit the {e['num_predict']}-token cap (done_reason 'length')")
        if e["done_reason"] != "stop":
            raise LLMError(f"{e['stage']}: unexpected done_reason {e['done_reason']!r}")

    def loaded_models(self) -> list[str]:
        r = self._send("GET", "/api/ps")
        r.raise_for_status()
        return [m.get("name") or m.get("model") for m in r.json().get("models", [])]

    def unload(self, wait_s: float = 30, model: str | None = None) -> bool:
        """keep_alive=0: Ollama frees the model's VRAM now. Waits until it is no longer listed."""
        model = model or self.model
        try:
            self._post("/api/generate", {"model": model, "keep_alive": 0})
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline:
                if model not in self.loaded_models():
                    return True
                time.sleep(0.5)
        except (httpx.HTTPError, LLMError):
            pass
        return False

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None


def release_llm_vram(llm_cfg) -> bool | None:
    """Called at the start of every GPU job (holding the GPU lock): if Ollama still holds our summary
    or embedding model (a job was killed before its own unload ran), unload it so it never shares the
    GPU with Whisper/pyannote or the other model. Only our models are touched. None = Ollama not
    reachable (nothing loaded by us)."""
    llm = LocalLLM.from_config(llm_cfg)
    try:
        return llm.release_models((llm_cfg.summary_model, llm_cfg.embedding_model))
    except (httpx.HTTPError, LLMError):
        return None
    finally:
        llm.close()


def embedder_from_config(llm_cfg) -> LocalLLM:
    return LocalLLM(host=llm_cfg.ollama_host, model=llm_cfg.embedding_model, num_ctx=llm_cfg.embed_num_ctx,
                    max_prompt_tokens=llm_cfg.embed_max_input_tokens, num_predict=0, timeout_s=llm_cfg.timeout_s)


def _error_text(r: httpx.Response) -> str:
    try:
        j = r.json()
        err = j.get("error", j)
        return str(err)[:500]
    except ValueError:
        return r.text[:500]
