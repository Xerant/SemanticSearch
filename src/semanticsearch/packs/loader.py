"""Load a domain pack directory: parse `pack.yaml`, validate it, resolve file paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from semanticsearch.core.errors import PackError
from semanticsearch.packs.manifest import PackManifest

MANIFEST_NAME = "pack.yaml"
TEMPLATE_PACK_DIR = Path(__file__).parent / "template"


@dataclass(frozen=True)
class DomainPack:
    root: Path
    manifest: PackManifest

    @property
    def version_stamp(self) -> str:
        """Written to every point as `pack_version`."""
        return f"{self.manifest.name}@{self.manifest.version}"

    def resolve(self, rel: Path) -> Path:
        return rel if rel.is_absolute() else self.root / rel

    @property
    def lexicon_path(self) -> Path | None:
        lex = self.manifest.lexicon
        return self.resolve(lex.source) if lex else None

    @property
    def cluster_model_path(self) -> Path | None:
        m = self.manifest.clusters.model
        return self.resolve(m) if m else None

    @property
    def gold_path(self) -> Path | None:
        g = self.manifest.eval.gold
        return self.resolve(g) if g else None


def load_pack(root: Path | str) -> DomainPack:
    """Load and validate the pack at `root`. Raises PackError on any validation error;
    warnings are left for the caller to fetch with `validate_pack`."""
    from semanticsearch.packs.validator import Severity, validate_pack

    root = Path(root)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise PackError(f"no {MANIFEST_NAME} in {root}")

    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise PackError(f"{manifest_path}: invalid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise PackError(f"{manifest_path}: expected a mapping at the top level")

    try:
        manifest = PackManifest.model_validate(raw)
    except ValidationError as e:
        raise PackError(f"{manifest_path}: {e}") from e

    pack = DomainPack(root=root.resolve(), manifest=manifest)
    errors = [i for i in validate_pack(pack) if i.severity is Severity.ERROR]
    if errors:
        raise PackError(f"{manifest_path}:\n" + "\n".join(f"  - {i.message}" for i in errors))
    return pack
