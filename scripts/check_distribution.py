"""Install built wheels or sdists into a fresh environment and run the full suite."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--openai", type=int, choices=(2, 3), required=True)
    parser.add_argument("--kind", choices=("wheel", "sdist"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifacts = sorted(args.dist.resolve().glob("*.whl" if args.kind == "wheel" else "*.tar.gz"))
    if len(artifacts) != 2:
        raise SystemExit(f"Expected exactly two {args.kind} artifacts, found {len(artifacts)}")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="caesura-install-") as directory:
        work = Path(directory)
        venv = work / "venv"
        subprocess.run(["uv", "venv", str(venv)], check=True, env=env)
        python = str(venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                *map(str, artifacts),
                f"openai>={args.openai},<{args.openai + 1}",
                "pytest>=8",
                "pytest-asyncio>=0.25",
                "respx>=0.22",
            ],
            check=True,
            env=env,
        )
        subprocess.run(["uv", "pip", "check", "--python", python], check=True, env=env)
        subprocess.run(
            [
                python,
                "-I",
                "-c",
                (
                    "import sys, pathlib, caesura_core, caesura_openai, openai; "
                    "assert all(pathlib.Path(m.__file__).is_relative_to(sys.prefix) "
                    "for m in (caesura_core, caesura_openai)); "
                    "print('Installed OpenAI', openai.__version__)"
                ),
            ],
            check=True,
            cwd=work,
            env=env,
        )
        subprocess.run(
            [
                python,
                "-I",
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                "-c",
                str(root / "pyproject.toml"),
                "-o",
                f"cache_dir={work / 'pytest-cache'}",
                str(root / "packages/caesura-core/tests"),
                str(root / "packages/caesura-openai/tests"),
            ],
            check=True,
            cwd=work,
            env=env,
        )


if __name__ == "__main__":
    main()
