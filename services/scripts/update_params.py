#!/usr/bin/env python3
"""Param-file generator for Amazon Bedrock.

Enumerates models from the OpenAI-compatible ``bedrock-mantle`` endpoint
(``GET /v1/models``) and writes one compact param file per AVAILABLE model
(``specs/bedrock/<model_id>.json`` = ``{parameters}``) that the ``specs``
pipeline re-renders ephemerally through ``templates/``. ``service_id``s are
preserved via ``<model_id>.service.json`` sidecars.

Bedrock is fronted BYOK: the customer supplies their own Amazon Bedrock API key
(``AWS_BEARER_TOKEN_BEDROCK``), so usage is billed by AWS directly and the
service is free through the UnitySVC gateway. The mantle endpoint is
OpenAI-compatible, so the ``anthropic_to_openai`` translator declared in the
offering template exposes every model in both OpenAI and Anthropic dialects.

``base_url`` is host-only: the gateway appends the customer's request path
(``/v1/chat/completions`` for OpenAI, ``/v1/messages`` -> ``/v1/chat/completions``
for the translated Anthropic dialect), which is exactly the mantle chat path.

The ``/v1/models`` catalog returns neither pricing nor context length, so those
are left unknown (``null``); enrich later from a maintained table if needed.

**Availability is probed, not read from the catalog.** ``GET /v1/models`` marks
*every* model ``available`` regardless of account entitlement or which API it
supports, so trusting it publishes services the gateway can't serve (they 400
and get rejected). The catalog carries NO api/endpoint/SDK-compatibility field
(only ``status``/``owned_by``/``data_retention``), so three live probes decide
what to publish, per model:

  1. ``GET /v1/models/{id}`` -- the DETAIL status is authoritative for account
     access (``unavailable`` + a ``status_reason`` for models the account can't
     invoke; the list endpoint hides this). Drops e.g. all ``anthropic.*`` and
     the unreleased ``openai.gpt-5.x`` on an account without access.
  2. ``POST /v1/chat/completions`` (1 token) -- confirms the model is reachable
     on the OpenAI Chat Completions route. A ``available`` model can still be
     Converse/InvokeModel-only ("isn't supported on this route"), e.g.
     ``google.gemma-4-*`` / ``xai.grok-*``; those are dropped too.
  3. ``POST /v1/chat/completions`` with a ``tools`` param -- sets
     ``supports_tools`` so the listing template only ships the function-calling
     example for models that actually accept tools (others 400/500 on it).

Usage: AWS_BEARER_TOKEN_BEDROCK=... python scripts/update_params.py
"""

import os
import sys
from pathlib import Path
from typing import Iterator

import httpx

from unitysvc_sellers.params_render import write_params_from_iterator

# Provider configuration
PROVIDER_NAME = "bedrock"
PROVIDER_DISPLAY_NAME = "Amazon Bedrock"
# Region-scoped mantle host. The base_url is host-only so the customer's request
# path rides through to the upstream unchanged; mantle routes cross-region, so
# us-east-1 fronts the full catalog.
AWS_REGION = "us-east-1"
API_BASE_URL = f"https://bedrock-mantle.{AWS_REGION}.api.aws"
ENV_API_KEY_NAME = "AWS_BEARER_TOKEN_BEDROCK"
MODELS_URL = f"{API_BASE_URL}/v1/models"

SCRIPT_DIR = Path(__file__).parent


def _display_name(model_id: str) -> str:
    """"anthropic.claude-opus-5" -> "Anthropic Claude Opus 5"."""
    return model_id.replace(".", " ").replace("-", " ").replace("_", " ").title()


# A one-token chat request proves the model serves the OpenAI Chat Completions
# route; a single ``get_weather`` tool proves it accepts the ``tools`` param.
# The probe body mirrors the SHAPE of the standard code-example presets —
# system + user — not just a bare user turn. Some models accept a lone user
# message but reject a system role (writer.palmyra-vision-7b 400s with
# "Conversation roles must alternate user/assistant/..."), which passed a
# bare-user probe and then failed every published example. If a model can't
# serve the example shape, it can't ship with the standard examples.
_PING = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "ping"},
]
_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def _detail_available(client: httpx.Client, model_id: str) -> tuple[bool, str]:
    """Authoritative per-account availability via ``GET /v1/models/{id}``.

    The list endpoint marks every model ``available``; the detail endpoint is the
    one that returns ``unavailable`` + a ``status_reason`` for models the account
    cannot invoke (no entitlement / unreleased). Returns (available, reason)."""
    try:
        r = client.get(f"{MODELS_URL}/{model_id}")
    except httpx.HTTPError as exc:
        return False, f"detail request failed: {exc}"
    if r.status_code != 200:
        return False, f"detail HTTP {r.status_code}"
    body = r.json()
    if body.get("status") == "available":
        return True, "available"
    return False, body.get("status_reason") or f"status={body.get('status')}"


def _chat_ok(client: httpx.Client, model_id: str, *, tools: bool) -> tuple[bool, str]:
    """POST a minimal chat completion; True iff the model serves it on the OpenAI
    route without error.

    Without ``tools`` this is the route probe — an ``available`` model can still
    be Converse/InvokeModel-only ("isn't supported on this route"). With ``tools``
    it doubles as the tool-support probe: we only care that the ``tools`` param is
    ACCEPTED (200), not that a call is emitted, so models that 400/500 on tools
    are reported unsupported. Returns (ok, detail)."""
    payload: dict = {"model": model_id, "messages": _PING, "max_tokens": 1}
    if tools:
        payload["max_tokens"] = 16  # headroom so a tool call isn't truncated pre-emit
        payload["tools"] = _TOOL
    # Reasoning models (e.g. kimi-k2-thinking) can burn seconds of thinking before
    # the first output token even at max_tokens=1, so a single tight timeout would
    # false-drop a servable model. Give it room and retry once before giving up.
    r = None
    for attempt in (1, 2):
        try:
            r = client.post(f"{API_BASE_URL}/v1/chat/completions", json=payload, timeout=90.0)
            break
        except httpx.TimeoutException:
            if attempt == 2:
                return False, "chat request timed out twice"
        except httpx.HTTPError as exc:
            return False, f"chat request failed: {exc}"
    assert r is not None
    if r.status_code == 200 and "choices" in r.json():
        return True, "ok"
    try:
        msg = r.json().get("error", {}).get("message", "") or r.text
    except ValueError:
        msg = r.text
    return False, f"HTTP {r.status_code}: {msg[:80]}"


def _native_foundation_model_ids() -> set[str]:
    """Model ids the NATIVE Bedrock runtime serves (``bedrock
    list-foundation-models``, SigV4 via the default boto3 credential chain).

    The mantle OpenAI surface and the native Converse/InvokeModel runtime have
    DIFFERENT catalogs and different spellings for the same model: mantle's
    ``qwen.qwen3-32b`` is natively ``qwen.qwen3-32b-v1:0``, mantle's
    ``moonshotai.kimi-k2-thinking`` is natively ``moonshot.kimi-k2-thinking``,
    and some mantle models (``zai.glm-4.6``, ``deepseek.v3.1``) do not exist
    natively at all. Sending a mantle id to the native runtime fails with
    "The provided model identifier is invalid", which is exactly how every
    converse-channel gateway test rejected.
    """
    # Declared in templates/config.json services_populator.requirements — it is
    # NOT pulled in by unitysvc-sellers. Adding an import here without adding it
    # there fails the populate run with ModuleNotFoundError.
    # Uses the default boto3 credential chain, so it needs AWS IAM creds in the env.
    import boto3

    bed = boto3.client("bedrock", region_name=AWS_REGION)
    # Not a paginated operation — the full catalog comes back in one response.
    resp = bed.list_foundation_models()
    return {m["modelId"] for m in resp.get("modelSummaries", []) if m.get("modelId")}


def _native_converse_id(model_id: str, native_ids: set[str]) -> str | None:
    """Map a mantle model id to its native-runtime id, or None if the model is
    not served natively (then the service ships without a converse channel).

    Candidate spellings, first match wins — derived from the observed drift:
    exact; version suffixes (``-v1:0`` / ``-1:0``); ``-instruct`` stripped
    (mantle appends it, the native catalog usually doesn't), with the same
    suffixes; and the ``moonshotai.`` -> ``moonshot.`` vendor alias applied to
    all of the above.
    """
    stems = [model_id]
    if model_id.endswith("-instruct"):
        stems.append(model_id[: -len("-instruct")])
    if model_id.startswith("moonshotai."):
        stems.extend([s.replace("moonshotai.", "moonshot.", 1) for s in list(stems)])
    for stem in stems:
        for candidate in (stem, f"{stem}-v1:0", f"{stem}-1:0"):
            if candidate in native_ids:
                return candidate
    return None


def iter_models(client: httpx.Client) -> Iterator[dict]:
    """Yield one template-variable dict per model the account can actually serve
    on the OpenAI Chat Completions route (see the module docstring for probes)."""
    print(f"Fetching models from {PROVIDER_DISPLAY_NAME} ({MODELS_URL})...")
    r = client.get(MODELS_URL)
    r.raise_for_status()
    models = r.json().get("data", [])
    print(f"Found {len(models)} catalog models; probing availability + route\n")

    native_ids = _native_foundation_model_ids()
    print(f"Native runtime catalog: {len(native_ids)} foundation models\n")

    kept = 0
    for i, m in enumerate(models, 1):
        model_id = m.get("id", "")
        if not model_id:
            continue
        print(f"[{i}/{len(models)}] {model_id}")

        available, reason = _detail_available(client, model_id)
        if not available:
            print(f"  skip (unavailable): {reason}")
            continue
        route_ok, reason = _chat_ok(client, model_id, tools=False)
        if not route_ok:
            print(f"  skip (not on OpenAI chat route): {reason}")
            continue
        supports_tools, _ = _chat_ok(client, model_id, tools=True)
        converse_model_id = _native_converse_id(model_id, native_ids)
        kept += 1
        print(
            f"  keep (tools={'yes' if supports_tools else 'no'}, "
            f"converse={converse_model_id or 'no'})"
        )

        display_name = _display_name(model_id)
        # Canonical (snake_case) metadata the platform validator requires for LLM
        # offerings. Both keys must be present; null asserts "unknown" — the
        # mantle /v1/models catalog returns neither pricing nor context length.
        details = {
            "model_name": model_id,
            "context_length": None,
            "parameter_count": None,
            "owned_by": m.get("owned_by"),
        }
        # data_retention.allowed_modes is a useful privacy signal to surface.
        data_retention = m.get("data_retention") or {}
        if data_retention.get("allowed_modes"):
            # Sorted: the API returns the list in nondeterministic order, which
            # would dirty every param file on every regen.
            details["data_retention_modes"] = sorted(data_retention["allowed_modes"])

        # BYOK: the customer's own key pays AWS directly, so the service is free
        # through the UnitySVC gateway. Keep the price cell short ("Free (BYOK)").
        pricing = {"type": "constant", "price": "0", "description": "Free (BYOK)"}

        yield {
            # Path / identity (stripped from the written parameters).
            "name": f"{PROVIDER_NAME}/{model_id}",
            "provider_name": PROVIDER_NAME,
            # Offering fields
            "offering_name": model_id,
            "display_name": display_name,
            "description": f"{display_name} served through Amazon Bedrock",
            "service_type": "llm",
            "status": "ready",
            "details": details,
            # Gates the function-calling example in the listing template — models
            # that reject the ``tools`` param would otherwise fail that one doc.
            "supports_tools": supports_tools,
            # The NATIVE Bedrock runtime id for the converse channel (differs
            # from the mantle id: version suffixes, -instruct stripping, vendor
            # aliases), or None when the model isn't served natively — then the
            # templates omit the converse channel/interface/examples entirely.
            "converse_model_id": converse_model_id,
            "payout_price": pricing,
            # Listing / channel fields
            "list_price": pricing,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "api_base_url": API_BASE_URL,
            "env_api_key_name": ENV_API_KEY_NAME,
        }
    print(f"\nKept {kept}/{len(models)} servable models")


def _prune_dropped(specs_dir: Path, kept_ids: set[str]) -> None:
    """Delete param files (+ ``.service.json`` sidecars) for models no longer in
    the servable set.

    ``write_params_from_iterator(prune_missing=True)`` prunes expanded *folders*,
    not param *files*, so a model that drops off the servable set would otherwise
    linger and keep getting uploaded and rejected. We enumerate the models
    ourselves here, so prune against that set explicitly."""
    if not specs_dir.is_dir():
        return
    removed = 0
    for param in sorted(specs_dir.glob("*.json")):
        if param.name.endswith(".service.json"):
            continue
        model_id = param.stem  # "<model_id>.json" -> "<model_id>"
        if model_id in kept_ids:
            continue
        param.unlink()
        sidecar = specs_dir / f"{model_id}.service.json"
        if sidecar.exists():
            sidecar.unlink()
        removed += 1
        print(f"  pruned dropped model: {model_id}")
    if removed:
        print(f"Pruned {removed} dropped model param file(s)")


def main() -> None:
    api_key = os.environ.get(ENV_API_KEY_NAME)
    if not api_key:
        print(f"Error: {ENV_API_KEY_NAME} not set")
        sys.exit(1)

    specs_dir = SCRIPT_DIR.parent / "specs"
    kept_ids: set[str] = set()

    def _tracking(client: httpx.Client) -> Iterator[dict]:
        for params in iter_models(client):
            kept_ids.add(params["offering_name"])
            yield params

    with httpx.Client(
        headers={"Authorization": f"Bearer {api_key}"}, timeout=30.0
    ) as client:
        stats = write_params_from_iterator(
            iterator=_tracking(client),
            output_dir=specs_dir,
        )
        _prune_dropped(specs_dir / PROVIDER_NAME, kept_ids)
    print(f"\nDone: {stats}")


if __name__ == "__main__":
    main()
