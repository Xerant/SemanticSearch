from pathlib import Path

import pytest

from semanticsearch.core import ExpansionPolicy, PackError
from semanticsearch.packs import TEMPLATE_PACK_DIR, Severity, load_pack, validate_pack


def write_pack(root: Path, body: str, files: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pack.yaml").write_text(body, encoding="utf-8")
    for name, content in (files or {}).items():
        (root / name).write_text(content, encoding="utf-8")
    return root


def test_template_pack_loads():
    pack = load_pack(TEMPLATE_PACK_DIR)
    assert pack.version_stamp == "template@0.1.0"
    assert pack.manifest.lexicon is None
    assert pack.lexicon_path is None


def test_minimal_pack_is_valid_baseline(tmp_path):
    pack = load_pack(write_pack(tmp_path, "name: tiny\nversion: 1.0.0\n"))
    assert pack.manifest.clusters.enabled is False
    assert pack.manifest.embedder is None
    # Defaults include a nonzero gamma, which is pointless without a lexicon.
    warnings = [i.message for i in validate_pack(pack) if i.severity is Severity.WARNING]
    assert any("gamma_concept" in w for w in warnings)


def test_full_pack_resolves_paths(tmp_path):
    body = """
name: aemo-energy
version: 0.3.0
lexicon:
  source: lexicon.ttl
  match: {stopword_labels: [gas, cap]}
sectioning:
  heading_regexes: ['^\\d+(\\.\\d+)*\\s+\\S']
embedder: {id: BAAI/bge-m3, dim: 1024}
weights:
  alpha_dense: 0.55
  beta_sparse: 0.30
  gamma_concept: 0.15
  expansion: {related: 0.5}
eval: {gold: gold.jsonl}
"""
    pack = load_pack(write_pack(tmp_path, body, {"lexicon.ttl": "", "gold.jsonl": ""}))
    assert pack.lexicon_path == tmp_path.resolve() / "lexicon.ttl"
    assert pack.manifest.lexicon.match.stopword_labels == ("gas", "cap")
    assert pack.manifest.weights.expansion.to_policy() == ExpansionPolicy(related=0.5)
    assert validate_pack(pack) == []


@pytest.mark.parametrize(
    "body,fragment",
    [
        ("name: x\nversion: 1.0.0\nweights: {alpha_dense: 0.5, gama_concept: 0.1}\n", "gama_concept"),
        ("name: Bad Name\nversion: 1.0.0\n", "name"),
        ("name: x\nversion: v1\n", "version"),
        ("name: x\nversion: 1.0.0\nlexicon: {source: missing.ttl}\n", "lexicon.source not found"),
        ("name: x\nversion: 1.0.0\nclusters: {enabled: true}\n", "clusters.model is not set"),
        ("name: x\nversion: 1.0.0\nsectioning: {heading_regexes: ['(']}\n", "invalid heading regex"),
        ("name: x\nversion: 1.0.0\nweights: {alpha_dense: 0, beta_sparse: 0, gamma_concept: 0}\n", "cannot all be 0"),
        ("name: x\nversion: 1.0.0\nweights: {alpha_dense: -1}\n", "alpha_dense"),
        ("- just\n- a list\n", "mapping"),
    ],
)
def test_invalid_packs_raise(tmp_path, body, fragment):
    with pytest.raises(PackError) as exc:
        load_pack(write_pack(tmp_path, body))
    assert fragment in str(exc.value)


def test_missing_manifest(tmp_path):
    with pytest.raises(PackError, match="no pack.yaml"):
        load_pack(tmp_path)
