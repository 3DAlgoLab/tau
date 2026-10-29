"""Starting a new session from the provider/model Tau last remembers using.

Tau has always started a new session from the catalog: the default provider's
default model, or the first provider with usable credentials when that one cannot
serve a call. On a machine with many providers that reads as a hard-wired
choice -- a bare ``tau`` lands on the same model every time, no matter how often
the user picks another one. This module is the memory that fixes it without
touching the provider catalog: the pair a session settled on goes into
``settings.json``, and a new session started with no selection flags begins from
it.

The pair is applied the way explicit ``--provider``/``--model`` flags are, which
is what lets a remembered model that lives only in an account-specific inventory
(a dynamic provider) survive a restart. It is deliberately not validated against
durable ``provider.models`` here: the pair came out of a session that actually
served it, and the durable catalog does not hold those inventories.
"""

from __future__ import annotations

from tau_coding.paths import TauPaths
from tau_coding.provider_config import (
    CredentialReader,
    ProviderConfigError,
    ProviderSettings,
    provider_has_usable_credentials,
)
from tau_coding.shell_config import (
    LastModelPreference,
    ShellSettings,
    remember_model_choice,
)


def remember_last_model(
    provider_name: str,
    model: str,
    *,
    paths: TauPaths | None = None,
) -> None:
    """Record the provider/model pair a session settled on.

    Called when a session starts from an explicitly requested pair, and whenever
    the user switches model or provider inside a session, so the next session can
    start where this one ended.
    """
    if not provider_name or not model:
        return
    remember_model_choice(
        LastModelPreference(provider_name=provider_name, model=model),
        paths=paths,
    )


def last_model_selection(
    settings: ProviderSettings,
    shell_settings: ShellSettings,
    *,
    credential_reader: CredentialReader | None = None,
) -> tuple[str, str] | None:
    """Return the remembered ``(provider, model)`` pair when it is still usable.

    ``None`` means "start the way Tau always has": the default provider, falling
    back to the first provider with usable credentials. A remembered pair is
    dropped rather than raised, for the same reason the settings reader ignores
    keys it does not understand -- a provider that was removed, a model that left
    an account inventory, or credentials that were revoked must never make
    ``tau`` refuse to start.
    """
    preference = shell_settings.last_model
    if preference is None:
        return None
    try:
        provider = settings.get_provider(preference.provider_name)
    except ProviderConfigError:
        return None
    # A provider with a durable inventory is checked against it, so a typo written
    # by one bad run cannot make every later bare ``tau`` fail. A provider whose
    # models come from an account at runtime has no durable inventory to check,
    # and its remembered model is taken as-is -- exactly as ``--model`` would be.
    if provider.models and preference.model not in provider.models:
        return None
    if credential_reader is not None and not provider_has_usable_credentials(
        provider,
        credential_reader=credential_reader,
    ):
        return None
    return provider.name, preference.model


def apply_last_model(
    settings: ProviderSettings,
    shell_settings: ShellSettings,
    *,
    provider_name: str | None,
    model: str | None,
    explicit_resume: bool = False,
    credential_reader: CredentialReader | None = None,
) -> tuple[str | None, str | None]:
    """One entry point for a session start: apply the memory, or update it.

    Called by each entry point (print, RPC, TUI) with whatever the command line
    asked for. An explicit provider *and* model are remembered, so the next bare
    run starts there. With neither, and no transcript being resumed, the
    remembered pair is returned as though it had been passed as flags -- which is
    what lets a resumed transcript and an explicit ``-m`` outrank memory.
    """
    if provider_name is not None and model is not None:
        remember_last_model(provider_name, model)
        return provider_name, model
    if provider_name is not None or model is not None or explicit_resume:
        return provider_name, model
    pair = last_model_selection(settings, shell_settings, credential_reader=credential_reader)
    return pair if pair is not None else (None, None)
