"""Example demonstrating mono-modal and multi-modal queries across custom intents,
local ONNX vision, local VisionLMs, remote multimodal MoEs, and privacy/classification
context routing.

Run with:
    python examples/multimodal_routing_example.py
"""

from __future__ import annotations

import base64
import struct
import zlib
from pathlib import Path

from aurarouter.config import ConfigLoader
from aurarouter.media import ImageInput, get_model_modalities


def _make_sample_png() -> bytes:
    """Create a tiny valid 2x2 PNG in memory for demonstration."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    raw = b"\x00\xff\x00\x00\x00\xff\x00\x00\x00\x00\xff\x00\x00\xff\x00"
    idat = zlib.compress(raw)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def main() -> None:
    cfg_path = Path(__file__).with_name("multimodal_routing_config.yaml")
    config = ConfigLoader(config_path=str(cfg_path))

    print("Configured model modalities:")
    for model_id in config.get_all_model_ids():
        mcfg = config.get_model_config(model_id)
        print(f"  - {model_id}: modalities={sorted(get_model_modalities(mcfg))}, "
              f"supported_intents={mcfg.get('supported_intents', [])}")

    sample_img = ImageInput.from_bytes(_make_sample_png(), mime_type="image/png", strip_exif=True)
    img_b64 = base64.b64decode(sample_img.data)
    print(f"\nPrepared sample image ({sample_img.width}x{sample_img.height}, {len(img_b64)} bytes, exif_stripped={sample_img.exif_stripped})")

    example_queries = [
        {
            "name": "Mono-modal text query",
            "task": "Write a Python helper to parse ISO timestamps.",
            "context": "",
            "intent": "SIMPLE_CODE",
            "images": None,
        },
        {
            "name": "Local ONNX object classification",
            "task": "Classify the primary object in this frame.",
            "context": "SOURCE: coastal_camera_04",
            "intent": "object_classification",
            "images": [sample_img],
        },
        {
            "name": "Multimodal OCR with structured schema",
            "task": "Extract all visible text from this shipping manifest.",
            "context": "DOCUMENT_TYPE: manifest",
            "intent": "VISION_OCR",
            "images": [sample_img],
            "response_schema": "vision_ocr",
        },
        {
            "name": "Cloud-eligible multimodal summarization",
            "task": "Summarize the scene and key activities.",
            "context": "CLASSIFICATION: UNCLASSIFIED; PRIVACY: NONE",
            "intent": "vision_summarization",
            "images": [sample_img],
        },
        {
            "name": "Privacy-tagged multimodal query (forced local-only via SovereigntyGate)",
            "task": "Summarize the badge and text visible in this image.",
            "context": "CLASSIFICATION: RESTRICTED; PRIVACY: PII",
            "intent": "vision_summarization",
            "images": [sample_img],
        },
    ]

    print("\nExample query payloads:")
    for q in example_queries:
        print(f"  * {q['name']}: intent={q['intent']}, has_images={bool(q['images'])}, context={q['context']!r}")


if __name__ == "__main__":
    main()
