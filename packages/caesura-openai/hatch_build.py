"""Hatchling metadata hook: provides dependencies with a pinned caesura-io-core version.

Both packages are released in lockstep, so the published wheel for
caesura-io-openai requires at least its own version of caesura-io-core,
within the same major. During local development uv resolves from the
workspace, so this constraint only matters for published wheels.
"""

from __future__ import annotations

from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface


class PinCoreDependencyHook(MetadataHookInterface):
    PLUGIN_NAME = "pin-core"

    def update(self, metadata: dict[str, Any]) -> None:
        import re
        from pathlib import Path

        # Use this distribution's release-managed version. A standalone sdist
        # has no sibling workspace package to read.
        core_pyproject = Path(__file__).parent / "pyproject.toml"
        content = core_pyproject.read_text("utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        if not match:
            raise ValueError(f"Could not find version in {core_pyproject}")

        core_version = match.group(1)
        major = core_version.split(".")[0]
        pinned_core = f"caesura-io-core>={core_version},<{int(major) + 1}.0.0"

        metadata["dependencies"] = [
            pinned_core,
            "openai>=2.0.0,<4.0.0",
        ]
