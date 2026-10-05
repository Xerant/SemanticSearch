"""Checks on a pack that the manifest schema alone can't express: referenced files
exist, and settings that have no effect in combination are flagged."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from semanticsearch.packs.loader import DomainPack


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class PackIssue:
    severity: Severity
    message: str


def validate_pack(pack: DomainPack) -> list[PackIssue]:
    m = pack.manifest
    issues: list[PackIssue] = []

    def missing(label: str, path) -> None:
        if path is not None and not path.is_file():
            issues.append(PackIssue(Severity.ERROR, f"{label} not found: {path}"))

    missing("lexicon.source", pack.lexicon_path)
    missing("eval.gold", pack.gold_path)
    if m.clusters.enabled:
        missing("clusters.model", pack.cluster_model_path)

    if m.lexicon is None and m.weights.gamma_concept > 0:
        issues.append(
            PackIssue(
                Severity.WARNING,
                "weights.gamma_concept > 0 but no lexicon is configured; the concept signal will be 0",
            )
        )
    if m.eval.gold is None:
        issues.append(
            PackIssue(Severity.WARNING, "no eval.gold set; weights in this pack cannot be tuned or checked")
        )
    return issues
