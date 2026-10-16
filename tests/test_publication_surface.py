from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_installation_uses_the_public_repository_url() -> None:
    installation = (ROOT / "docs" / "installation.md").read_text(encoding="utf-8")

    assert "git clone https://github.com/jedufa/lightphone-gpt.git" in installation
    assert "<private-review-repository-url>" not in installation


def test_public_visual_assets_are_well_formed_and_social_preview_sized() -> None:
    flow = ROOT / "docs" / "assets" / "lightphone-gpt-flow.svg"
    preview = ROOT / "docs" / "assets" / "lightphone-gpt-social-preview.png"

    root = ET.parse(flow).getroot()
    assert root.attrib["viewBox"] == "0 0 1280 640"

    payload = preview.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", payload[16:24]) == (1280, 640)
