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

The ``/v1/models`` catalog returns neither pricing nor context length. Context
length is left unknown (``null``). Pricing is enriched from the litellm public
rate table (``model_prices_and_context_window.json``) — see ``_pricing_note()``,
which matches ONLY Bedrock-provider entries under exact id spellings, so a model
litellm does not carry gets no note rather than a neighbouring provider's rate.
A model with no entry legitimately has no note, permanently — that is an answer,
not a failure. Two things are failures, because a null no longer overwrites the
committed value and would silently re-ship yesterday's rate as freshly derived:
the TABLE itself failing to load or coming back empty (fatal up front, it breaks
every lookup at once), and a model that ALREADY HAS a committed rate deriving
none this run (fatal at that model).

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

A probe that FAILS (timeout, transport error, 429, 5xx) answers nothing, so the
model is neither published nor treated as gone: it is recorded as inconclusive
and the whole run then writes with ``deprecate_missing=False``. Under
unitysvc-sellers 0.3.1 a committed service the run does not yield is marked
``status="deprecated"``, and a throttled probe must never be able to retire a
model AWS is still serving.

Usage: AWS_BEARER_TOKEN_BEDROCK=... python scripts/update_params.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Iterator

import httpx

from unitysvc_sellers.model_data import ModelDataFetcher
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


def _committed_param(service_name: str, key: str):
    """What ``key`` currently holds in the committed param file, or ``None``.

    Read the param file DIRECTLY rather than through ``load_param_data``: that
    merges the ``.override.json`` companion, and an override's value is not
    something this script derived, so it must not make a lookup look successful.
    """
    path = SCRIPT_DIR.parent / "specs" / f"{service_name}.json"
    try:
        return ((json.loads(path.read_text()) or {}).get("parameters") or {}).get(key)
    except (OSError, json.JSONDecodeError):
        return None


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


def _detail_available(client: httpx.Client, model_id: str) -> tuple[bool | None, str]:
    """Authoritative per-account availability via ``GET /v1/models/{id}``.

    The list endpoint marks every model ``available``; the detail endpoint is the
    one that returns ``unavailable`` + a ``status_reason`` for models the account
    cannot invoke (no entitlement / unreleased).

    Returns (available, reason), where ``available`` is ``None`` for an
    INCONCLUSIVE probe — a transport error, a 429, a 5xx. Absence now retires a
    service, and "AWS timed out" is not "AWS stopped serving this model", so the
    caller must be able to tell a failed probe from a negative one."""
    try:
        r = client.get(f"{MODELS_URL}/{model_id}")
    except httpx.HTTPError as exc:
        return None, f"detail request failed: {exc}"
    if r.status_code == 429 or r.status_code >= 500:
        return None, f"detail HTTP {r.status_code}"
    if r.status_code != 200:
        return False, f"detail HTTP {r.status_code}"
    body = r.json()
    if body.get("status") == "available":
        return True, "available"
    return False, body.get("status_reason") or f"status={body.get('status')}"


def _chat_ok(client: httpx.Client, model_id: str, *, tools: bool) -> tuple[bool | None, str]:
    """POST a minimal chat completion; True iff the model serves it on the OpenAI
    route without error.

    Without ``tools`` this is the route probe — an ``available`` model can still
    be Converse/InvokeModel-only ("isn't supported on this route"). With ``tools``
    it doubles as the tool-support probe: we only care that the ``tools`` param is
    ACCEPTED (200), not that a call is emitted, so models that 400/500 on tools
    are reported unsupported.

    Returns (ok, detail), where ``ok`` is ``None`` for an INCONCLUSIVE probe —
    both timeouts spent, a transport error, a 429, a 5xx. A model whose probe
    merely failed must not be read as one the account can no longer serve, now
    that not being yielded retires it."""
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
                return None, "chat request timed out twice"
        except httpx.HTTPError as exc:
            return None, f"chat request failed: {exc}"
    assert r is not None
    if r.status_code == 200 and "choices" in r.json():
        return True, "ok"
    try:
        msg = r.json().get("error", {}).get("message", "") or r.text
    except ValueError:
        msg = r.text
    if r.status_code == 429 or r.status_code >= 500:
        # Throttled or upstream-side error: says nothing about the model.
        return None, f"HTTP {r.status_code}: {msg[:80]}"
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


def _format_price(price: float) -> str:
    """Render a per-1M-token price without a pointless trailing ``.0``."""
    return str(int(price)) if price == int(price) else str(round(price, 4))


def _pricing_note(
    model_id: str, converse_model_id: str | None, litellm: dict
) -> str | None:
    """AWS's published on-demand rate for a model, as a one-line rate card, or
    None when litellm carries no Bedrock entry for it.

    The mantle ``/v1/models`` catalog has no pricing at all, so this is the only
    machine-readable source we have; hand-maintaining a table of AWS list prices
    in this repo would go stale silently.

    The lookup is deliberately STRICT rather than reusing
    ``ModelDataLookup.lookup_model_details``: that helper falls back to substring
    matching over ~3k keys, which for a Bedrock id like ``openai.gpt-oss-120b``
    happily returns Groq's or DeepInfra's rate for the same open-weights model.
    Those are real numbers for the wrong bill. So: exact keys only, in the two
    spellings litellm actually uses (bare and ``bedrock/``-prefixed) for both the
    mantle id and the native runtime id, and the entry must declare a Bedrock
    provider. A model with no match yields None and the templates degrade to a
    bare "Free ~ BYOK" — no invented rate.
    """
    if not litellm:
        return None
    candidates = [model_id, f"bedrock/{model_id}"]
    if converse_model_id:
        candidates += [converse_model_id, f"bedrock/{converse_model_id}"]
    for key in candidates:
        entry = litellm.get(key)
        if not entry:
            continue
        # "bedrock" / "bedrock_converse" — never another provider's listing that
        # happens to be filed under a Bedrock-shaped id.
        if not str(entry.get("litellm_provider", "")).startswith("bedrock"):
            continue
        if "input_cost_per_token" not in entry or "output_cost_per_token" not in entry:
            continue
        inp = float(entry["input_cost_per_token"]) * 1_000_000
        out = float(entry["output_cost_per_token"]) * 1_000_000
        cached = entry.get("cache_read_input_token_cost")
        if cached is not None:
            return (
                f"${_format_price(inp)} / ${_format_price(out)} / "
                f"${_format_price(float(cached) * 1_000_000)} "
                f"per 1M input/output/cached tokens"
            )
        return (
            f"${_format_price(inp)} / ${_format_price(out)} "
            f"per 1M input/output tokens"
        )
    return None


def iter_models(client: httpx.Client, inconclusive: list[str]) -> Iterator[dict]:
    """Yield one template-variable dict per model the account can actually serve
    on the OpenAI Chat Completions route (see the module docstring for probes).

    ``inconclusive`` collects models whose probes FAILED rather than answered
    (timeout, transport error, 429, 5xx). Those models are not yielded — their
    data would be guesswork — and the caller turns deprecation off when the list
    is non-empty, because a model missing from an incomplete run is not a model
    AWS retired."""
    print(f"Fetching models from {PROVIDER_DISPLAY_NAME} ({MODELS_URL})...")
    r = client.get(MODELS_URL)
    r.raise_for_status()
    models = r.json().get("data", [])
    print(f"Found {len(models)} catalog models; probing availability + route\n")
    # An empty catalog is a failed enumeration, not an emptied Bedrock. Exiting 0
    # here would write nothing, open no PR, and look exactly like "no changes
    # today" — and with deprecation on it would try to retire everything.
    if not models:
        print(f"Error: {PROVIDER_DISPLAY_NAME} returned no catalog models")
        sys.exit(1)

    native_ids = _native_foundation_model_ids()
    print(f"Native runtime catalog: {len(native_ids)} foundation models\n")

    # Public rate table for the BYOK price note (the mantle catalog has none).
    # NOT best-effort any more: `pricing_note` is the only rate customers see,
    # nothing downstream requires it, and a null no longer overwrites the
    # committed value — so degrading to `{}` here would republish every model's
    # PREVIOUS rate while looking like a clean, freshly-derived run. A rate we
    # cannot derive must fail the run, not quietly persist.
    try:
        litellm = ModelDataFetcher().fetch_litellm_model_data(quiet=True)
    except Exception as exc:  # noqa: BLE001 - re-raised as a fatal below
        print(f"Error: litellm rate table unavailable ({exc}) — cannot derive prices")
        sys.exit(1)
    if not litellm:
        print("Error: litellm rate table came back empty — cannot derive prices")
        sys.exit(1)

    kept = 0
    for i, m in enumerate(models, 1):
        model_id = m.get("id", "")
        if not model_id:
            continue
        print(f"[{i}/{len(models)}] {model_id}")

        available, reason = _detail_available(client, model_id)
        if available is None:
            print(f"  INCONCLUSIVE (availability probe failed): {reason}")
            inconclusive.append(model_id)
            continue
        if not available:
            print(f"  skip (unavailable): {reason}")
            continue
        route_ok, reason = _chat_ok(client, model_id, tools=False)
        if route_ok is None:
            print(f"  INCONCLUSIVE (route probe failed): {reason}")
            inconclusive.append(model_id)
            continue
        if not route_ok:
            print(f"  skip (not on OpenAI chat route): {reason}")
            continue
        supports_tools, tools_reason = _chat_ok(client, model_id, tools=True)
        if supports_tools is None:
            # Don't guess `supports_tools`: False would drop the function-calling
            # example from a model that does support tools, and it is a real
            # value so null-preservation would not save the committed one.
            print(f"  INCONCLUSIVE (tool probe failed): {tools_reason}")
            inconclusive.append(model_id)
            continue
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
        # through the UnitySVC gateway. This plain description is what
        # payout_price keeps (seller-facing). The customer-facing listing cell is
        # composed in listing.json.j2 from pricing_note, into the
        # "<amount> ~<PILL> | <note>" grammar; do not build it here, since this
        # dict feeds payout_price too.
        pricing = {"type": "constant", "price": "0", "description": "Free (BYOK)"}
        pricing_note = _pricing_note(model_id, converse_model_id, litellm)
        service_name = f"{PROVIDER_NAME}/{model_id}"
        # `list_price` here is the constant "Free (BYOK)" — the customer's own
        # key pays AWS — so the field actually DERIVED from
        # `input_cost_per_token` is `pricing_note`, the rate card on the listing.
        #
        # A model litellm carries no Bedrock entry for legitimately has no note
        # (4 of the committed 37 are exactly that), and that null is permanent
        # and correct — failing on it would break every run forever. What must
        # NOT pass is a model that HAS a committed rate and now derives none:
        # under 0.3.1 a yielded null no longer overwrites the committed value,
        # so that would silently re-ship yesterday's rate as if re-derived today.
        if pricing_note is None:
            committed_note = _committed_param(service_name, "pricing_note")
            if committed_note is not None:
                print(
                    f"Error: no litellm Bedrock rate for {model_id}, but "
                    f"{service_name}.json already carries one ({committed_note!r}). "
                    "A failed lookup must not silently re-ship the committed rate."
                )
                sys.exit(1)

        yield {
            # Path / identity (stripped from the written parameters). Required
            # by unitysvc-sellers 0.3.1: it is the service's name AND its path
            # under specs/, and what deprecation matches committed services by.
            "service_name": service_name,
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
            # AWS's on-demand rate, for the BYOK price-cell note and the closing
            # pricing paragraph (both template-rendered). None where litellm has
            # no Bedrock entry — then neither surface states a rate.
            "pricing_note": pricing_note,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "api_base_url": API_BASE_URL,
            "env_api_key_name": ENV_API_KEY_NAME,
        }
    print(f"\nKept {kept}/{len(models)} servable models")


def main() -> None:
    api_key = os.environ.get(ENV_API_KEY_NAME)
    if not api_key:
        print(f"Error: {ENV_API_KEY_NAME} not set")
        sys.exit(1)

    specs_dir = SCRIPT_DIR.parent / "specs"
    # Models whose probes failed rather than answered. Collected during the
    # enumeration, which is why it is drained EAGERLY below: whether the run was
    # complete has to be known before the writer decides what to deprecate.
    inconclusive: list[str] = []

    with httpx.Client(
        headers={"Authorization": f"Bearer {api_key}"}, timeout=30.0
    ) as client:
        params = list(iter_models(client, inconclusive))

    # A model that drops off the servable set is now RETIRED by the writer
    # (``status="deprecated"``), not deleted from the repo — deprecation is a
    # publishable state the platform acts on, whereas deleting the param file
    # left the service live upstream with nothing here to explain it. So there
    # is no local pruning step any more.
    #
    # But absence only means "retired" on a COMPLETE run: a model we could not
    # probe is missing for a reason that has nothing to do with AWS's catalog.
    if inconclusive:
        print(
            f"\nIncomplete run ({len(inconclusive)} model(s) could not be probed: "
            f"{', '.join(sorted(inconclusive)[:5])}"
            f"{'…' if len(inconclusive) > 5 else ''}) — skipping deprecation"
        )
    stats = write_params_from_iterator(
        iterator=iter(params),
        output_dir=specs_dir,
        deprecate_missing=not inconclusive,
    )
    print(f"\nDone: {stats}")
    print(f"New: {stats['new']}, deprecated: {stats['deprecated']}")


if __name__ == "__main__":
    main()
