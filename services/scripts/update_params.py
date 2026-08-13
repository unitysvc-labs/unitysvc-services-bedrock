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


def iter_models(api_key: str) -> Iterator[dict]:
    """Yield one template-variable dict per available Bedrock (mantle) model."""
    print(f"Fetching models from {PROVIDER_DISPLAY_NAME} ({MODELS_URL})...")
    r = httpx.get(MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=30.0)
    r.raise_for_status()
    models = r.json().get("data", [])
    print(f"Found {len(models)} models\n")

    for i, m in enumerate(models, 1):
        model_id = m.get("id", "")
        if not model_id:
            continue
        status = m.get("status")
        print(f"[{i}/{len(models)}] {model_id} ({status})")
        # Skip models the account can't invoke (access not granted / retired).
        if status != "available":
            print("  Skipped: not available")
            continue

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
            details["data_retention_modes"] = data_retention["allowed_modes"]

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
            "payout_price": pricing,
            # Listing / channel fields
            "list_price": pricing,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "api_base_url": API_BASE_URL,
            "env_api_key_name": ENV_API_KEY_NAME,
        }
        print("  OK")


def main() -> None:
    api_key = os.environ.get(ENV_API_KEY_NAME)
    if not api_key:
        print(f"Error: {ENV_API_KEY_NAME} not set")
        sys.exit(1)

    stats = write_params_from_iterator(
        iterator=iter_models(api_key),
        output_dir=SCRIPT_DIR.parent / "specs",
    )
    print(f"\nDone: {stats}")


if __name__ == "__main__":
    main()
