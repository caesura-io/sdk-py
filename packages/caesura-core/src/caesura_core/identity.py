"""Diagnostic SDK provenance. Versions come only from installed distribution metadata."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal


def distribution_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "0+unknown"


@dataclass(frozen=True)
class SdkIdentity:
    language: str
    core_package: str
    core_version: str
    transport: Literal["http", "in-process"]
    integration_package: str | None = None
    integration_version: str | None = None
    runtime_name: str | None = None
    runtime_version: str | None = None

    def http_headers(self) -> dict[str, str]:
        fields = {
            "language": self.language,
            "core-name": self.core_package,
            "core-version": self.core_version,
            "integration-name": self.integration_package,
            "integration-version": self.integration_version,
            "transport": self.transport,
            "runtime-name": self.runtime_name,
            "runtime-version": self.runtime_version,
        }
        return {f"x-caesura-sdk-{key}": value for key, value in fields.items() if value is not None}


def sdk_identity(transport: Literal["http", "in-process"], integration: str | None = None) -> SdkIdentity:
    return SdkIdentity(
        language="python",
        core_package="caesura-io-core",
        core_version=distribution_version("caesura-io-core"),
        transport=transport,
        integration_package=integration,
        integration_version=distribution_version(integration) if integration else None,
        runtime_name=platform.python_implementation(),
        runtime_version=platform.python_version(),
    )
