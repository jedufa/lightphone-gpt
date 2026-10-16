from __future__ import annotations

import struct
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_installation_uses_the_public_repository_url() -> None:
    installation = (ROOT / "docs" / "installation.md").read_text(encoding="utf-8")

    assert "git clone https://github.com/jedufa/lightphone-gpt.git" in installation
    assert "<private-review-repository-url>" not in installation


def test_readme_navigation_and_visual_are_not_duplicated() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "(#quick-start)" in readme
    assert "#safe-simulator-quick-start" not in readme
    assert readme.count("actions/workflows/test.yml/badge.svg") == 1
    assert readme.count("docs/figures/lightphone-bridge-overview.svg") == 1
    assert "docs/assets/lightphone-gpt-flow.svg" not in readme


def test_public_visual_assets_are_well_formed_and_social_preview_sized() -> None:
    flow = ROOT / "docs" / "figures" / "lightphone-bridge-overview.svg"
    preview = ROOT / "docs" / "assets" / "lightphone-gpt-social-preview.png"

    root = ET.parse(flow).getroot()
    assert root.attrib["viewBox"] == "0 0 1200 520"

    payload = preview.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", payload[16:24]) == (1280, 640)


def test_packaged_readme_visual_keeps_its_relative_path() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    data_files = project["tool"]["setuptools"]["data-files"]

    assert "README.md" in data_files["share/doc/lightphone-gpt"]
    assert data_files["share/doc/lightphone-gpt/docs/figures"] == [
        "docs/figures/lightphone-bridge-overview.svg"
    ]
