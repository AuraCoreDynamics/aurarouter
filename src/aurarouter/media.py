"""Multimodal media contracts, vision extraction schemas, and EXIF metadata stripping."""

from __future__ import annotations

import base64
import binascii
import math
import mimetypes
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
})

_EXT_TO_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def _detect_mime_from_bytes(raw: bytes, fallback: str = "image/png") -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"BM"):
        return "image/bmp"
    return fallback


def _extract_dimensions(raw: bytes) -> tuple[int | None, int | None]:
    """Parse image (width, height) from common headers without requiring Pillow."""
    try:
        # PNG: IHDR chunk at offset 16..24
        if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
            w, h = struct.unpack(">II", raw[16:24])
            return int(w), int(h)
        # GIF: logical screen descriptor at 6..10
        if (raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a")) and len(raw) >= 10:
            w, h = struct.unpack("<HH", raw[6:10])
            return int(w), int(h)
        # BMP: BITMAPINFOHEADER at 18..26
        if raw.startswith(b"BM") and len(raw) >= 26:
            w, h = struct.unpack("<ii", raw[18:26])
            return abs(int(w)), abs(int(h))
        # JPEG: scan markers for SOF0..SOF3
        if raw.startswith(b"\xff\xd8"):
            idx = 2
            while idx + 4 < len(raw):
                if raw[idx] != 0xFF:
                    idx += 1
                    continue
                marker = raw[idx + 1]
                if marker in (0xD8, 0xD9, 0x00) or (0xD0 <= marker <= 0xD7):
                    idx += 2
                    continue
                if marker == 0xDA:  # Start of Scan
                    break
                if idx + 4 > len(raw):
                    break
                seg_len = struct.unpack(">H", raw[idx + 2:idx + 4])[0]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3) and idx + 9 <= len(raw):
                    h, w = struct.unpack(">HH", raw[idx + 5:idx + 9])
                    return int(w), int(h)
                if seg_len < 2:
                    break
                idx += 2 + seg_len
        # WEBP: VP8X / VP8 / VP8L
        if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP" and len(raw) >= 30:
            chunk = raw[12:16]
            if chunk == b"VP8X" and len(raw) >= 30:
                w = 1 + int.from_bytes(raw[24:27], "little")
                h = 1 + int.from_bytes(raw[27:30], "little")
                return w, h
    except Exception:
        pass
    return None, None


def strip_image_metadata(raw: bytes, mime_type: str = "") -> tuple[bytes, bool]:
    """Strip EXIF/XMP/comment metadata chunks from JPEG and PNG images in pure Python.

    Falls back to Pillow re-encoding if available for other formats.
    Returns ``(clean_bytes, stripped_flag)``.
    """
    detected = _detect_mime_from_bytes(raw, fallback=mime_type or "image/png")

    # Pure-Python JPEG APP1 (EXIF/XMP) & COM/APP13 segment stripping
    if detected == "image/jpeg" and raw.startswith(b"\xff\xd8"):
        out = bytearray(b"\xff\xd8")
        idx = 2
        stripped = False
        n = len(raw)
        while idx < n:
            if raw[idx] != 0xFF:
                out.extend(raw[idx:])
                break
            marker_idx = idx + 1
            while marker_idx < n and raw[marker_idx] == 0xFF:
                marker_idx += 1
            if marker_idx >= n:
                break
            marker = raw[marker_idx]
            # Standalone markers (SOI, EOI, RSTn)
            if marker in (0x00, 0xD8, 0xD9) or (0xD0 <= marker <= 0xD7):
                out.extend(raw[idx:marker_idx + 1])
                idx = marker_idx + 1
                if marker == 0xD9:
                    break
                continue
            # SOS: copy entropy-coded scan until EOI
            if marker == 0xDA:
                out.extend(raw[idx:])
                break
            if marker_idx + 2 >= n:
                out.extend(raw[idx:])
                break
            seg_len = struct.unpack(">H", raw[marker_idx + 1:marker_idx + 3])[0]
            seg_end = marker_idx + 1 + seg_len
            # Strip APP1 (0xE1 EXIF/XMP), APP2 ICC/FlashPix only keep non-EXIF, APP12 (0xEC), APP13 (0xED), COM (0xFE)
            if marker in (0xE1, 0xEC, 0xED, 0xFE):
                stripped = True
                idx = seg_end
                continue
            out.extend(raw[idx:seg_end])
            idx = seg_end
        return bytes(out), stripped

    # Pure-Python PNG ancillary metadata chunk stripping (eXIf, tEXt, zTXt, iTXt, tIME)
    if detected == "image/png" and raw.startswith(b"\x89PNG\r\n\x1a\n"):
        strip_chunks = {b"eXIf", b"tEXt", b"zTXt", b"iTXt", b"tIME"}
        out = bytearray(raw[:8])
        idx = 8
        stripped = False
        n = len(raw)
        while idx + 12 <= n:
            length = struct.unpack(">I", raw[idx:idx + 4])[0]
            chunk_type = raw[idx + 4:idx + 8]
            chunk_end = idx + 12 + length
            if chunk_end > n:
                out.extend(raw[idx:])
                break
            if chunk_type in strip_chunks:
                stripped = True
            else:
                out.extend(raw[idx:chunk_end])
            idx = chunk_end
            if chunk_type == b"IEND":
                break
        return bytes(out), stripped

    # Fallback: try Pillow if installed
    try:
        import io
        from PIL import Image  # type: ignore

        with Image.open(io.BytesIO(raw)) as img:
            clean = Image.new(img.mode, img.size)
            clean.putdata(list(img.getdata()))
            buf = io.BytesIO()
            fmt = (img.format or "PNG").upper()
            clean.save(buf, format=fmt)
            return buf.getvalue(), True
    except Exception:
        return raw, False


@dataclass(frozen=True)
class ImageInput:
    """Normalized multimodal image payload carried across API, fabric, sessions, and providers."""

    data: str
    mime_type: str = "image/png"
    source_path: str | None = None
    width: int | None = None
    height: int | None = None
    exif_stripped: bool = False

    def raw_bytes(self) -> bytes:
        return base64.b64decode(self.data)

    def to_data_uri(self) -> str:
        return f"data:{self.mime_type};base64,{self.data}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "mime_type": self.mime_type,
            "source_path": self.source_path,
            "width": self.width,
            "height": self.height,
            "exif_stripped": self.exif_stripped,
        }

    def estimate_tokens(self) -> int:
        """Estimate VLM visual token cost using a 512px-tile heuristic."""
        if self.width and self.height and self.width > 0 and self.height > 0:
            tiles = max(1, math.ceil(self.width / 512) * math.ceil(self.height / 512))
            return 85 + 170 * tiles
        return 258

    def stripped_copy(self) -> "ImageInput":
        """Return a copy with EXIF/metadata stripped (preserving dimensions)."""
        if self.exif_stripped:
            return self
        try:
            raw = self.raw_bytes()
        except Exception:
            return self
        clean_bytes, _did_strip = strip_image_metadata(raw, self.mime_type)
        clean_b64 = base64.b64encode(clean_bytes).decode("ascii")
        w, h = self.width, self.height
        if w is None or h is None:
            w, h = _extract_dimensions(clean_bytes)
        return ImageInput(
            data=clean_b64,
            mime_type=self.mime_type,
            source_path=self.source_path,
            width=w,
            height=h,
            exif_stripped=True,
        )

    @classmethod
    def from_bytes(
        cls,
        raw_bytes: bytes,
        mime_type: str | None = None,
        source_path: str | None = None,
        strip_exif: bool = False,
    ) -> "ImageInput":
        detected_mime = _detect_mime_from_bytes(raw_bytes, fallback=mime_type or "image/png")
        effective_bytes = raw_bytes
        stripped = False
        if strip_exif:
            effective_bytes, stripped = strip_image_metadata(raw_bytes, detected_mime)
        w, h = _extract_dimensions(effective_bytes)
        b64 = base64.b64encode(effective_bytes).decode("ascii")
        return cls(
            data=b64,
            mime_type=detected_mime,
            source_path=source_path,
            width=w,
            height=h,
            exif_stripped=stripped,
        )

    @classmethod
    def from_path(cls, path: str | Path, strip_exif: bool = False) -> "ImageInput":
        p = Path(path)
        raw = p.read_bytes()
        ext = p.suffix.lower()
        guessed = _EXT_TO_MIME.get(ext) or mimetypes.guess_type(str(p))[0] or "image/png"
        return cls.from_bytes(
            raw,
            mime_type=guessed,
            source_path=str(p),
            strip_exif=strip_exif,
        )

    @classmethod
    def from_data_uri(cls, uri: str, strip_exif: bool = False) -> "ImageInput":
        if not uri.startswith("data:"):
            raise ValueError("Not a data URI")
        header, _, payload = uri.partition(",")
        mime = "image/png"
        if ";" in header:
            mime_part = header[5:].split(";", 1)[0].strip()
            if mime_part:
                mime = mime_part
        raw = base64.b64decode(payload)
        return cls.from_bytes(raw, mime_type=mime, strip_exif=strip_exif)

    @classmethod
    def from_any(cls, item: Any, strip_exif: bool = False) -> "ImageInput":
        if isinstance(item, ImageInput):
            return item.stripped_copy() if strip_exif and not item.exif_stripped else item

        if isinstance(item, dict):
            raw_data = item.get("data") or item.get("base64") or item.get("url") or ""
            mime = item.get("mime_type") or item.get("media_type") or "image/png"
            source_path = item.get("source_path") or item.get("path")
            if isinstance(raw_data, str) and raw_data.startswith("data:"):
                parsed = cls.from_data_uri(raw_data, strip_exif=strip_exif)
                if source_path and not parsed.source_path:
                    return ImageInput(
                        data=parsed.data,
                        mime_type=parsed.mime_type,
                        source_path=source_path,
                        width=parsed.width,
                        height=parsed.height,
                        exif_stripped=parsed.exif_stripped,
                    )
                return parsed
            if not raw_data and source_path:
                return cls.from_path(source_path, strip_exif=strip_exif)
            if isinstance(raw_data, bytes):
                return cls.from_bytes(raw_data, mime_type=mime, source_path=source_path, strip_exif=strip_exif)
            # Base64 string inside dict
            clean_b64 = str(raw_data).strip()
            w = item.get("width")
            h = item.get("height")
            exif_done = bool(item.get("exif_stripped", False))
            if strip_exif and not exif_done and clean_b64:
                return ImageInput(
                    data=clean_b64,
                    mime_type=mime,
                    source_path=source_path,
                    width=w,
                    height=h,
                    exif_stripped=False,
                ).stripped_copy()
            if (w is None or h is None) and clean_b64:
                try:
                    raw = base64.b64decode(clean_b64)
                    w2, h2 = _extract_dimensions(raw)
                    w = w if w is not None else w2
                    h = h if h is not None else h2
                    if mime == "image/png":
                        mime = _detect_mime_from_bytes(raw, fallback=mime)
                except (binascii.Error, ValueError):
                    pass
            return cls(
                data=clean_b64,
                mime_type=mime,
                source_path=source_path,
                width=w,
                height=h,
                exif_stripped=exif_done,
            )

        if isinstance(item, bytes):
            return cls.from_bytes(item, strip_exif=strip_exif)

        if isinstance(item, Path):
            return cls.from_path(item, strip_exif=strip_exif)

        if isinstance(item, str):
            s = item.strip()
            if s.startswith("data:"):
                return cls.from_data_uri(s, strip_exif=strip_exif)
            # Check if string is a filesystem path to an image
            try:
                if len(s) < 1024 and "\n" not in s:
                    p = Path(s)
                    if p.exists() and p.is_file():
                        return cls.from_path(p, strip_exif=strip_exif)
            except OSError:
                pass
            # Assume raw base64 string
            try:
                raw = base64.b64decode(s, validate=True)
                return cls.from_bytes(raw, strip_exif=strip_exif)
            except Exception:
                return cls(data=s, mime_type="image/png", exif_stripped=strip_exif)

        raise TypeError(f"Unsupported image input type: {type(item).__name__}")


def normalize_images(
    images: Sequence[Any] | Any | None,
    strip_exif: bool = False,
) -> list[ImageInput]:
    """Normalize a single image or sequence of image specs into ImageInput objects."""
    if not images:
        return []
    if isinstance(images, (ImageInput, str, bytes, Path, dict)):
        return [ImageInput.from_any(images, strip_exif=strip_exif)]
    return [ImageInput.from_any(item, strip_exif=strip_exif) for item in images if item is not None]


def extract_base64_images(images: Sequence[Any] | None) -> list[str]:
    """Extract raw base64 strings from normalized or dict-encoded image entries."""
    if not images:
        return []
    out: list[str] = []
    for img in normalize_images(images):
        if img.data:
            out.append(img.data)
    return out


def format_openai_content(text: str, images: Sequence[Any] | None) -> str | list[dict[str, Any]]:
    """Build OpenAI-compatible message content, preserving multimodal parts when images exist."""
    if not images:
        return text
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})
    for img in normalize_images(images):
        parts.append({"type": "image_url", "image_url": {"url": img.to_data_uri()}})
    return parts or text


def format_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal message dicts (supporting per-message ``images``) to OpenAI format."""
    formatted: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        imgs = msg.get("images")
        if isinstance(content, list):
            formatted.append({"role": role, "content": content})
        elif imgs:
            formatted.append({"role": role, "content": format_openai_content(str(content), imgs)})
        else:
            formatted.append({"role": role, "content": str(content)})
    return formatted


# ---------------------------------------------------------------------------
# Built-in Structured Vision Extraction JSON Schemas & Presets
# ---------------------------------------------------------------------------

VISION_OCR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "Full extracted text preserving layout and reading order.",
        },
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "block_type": {
                        "type": "string",
                        "enum": [
                            "heading",
                            "paragraph",
                            "table",
                            "list",
                            "caption",
                            "label",
                            "handwriting",
                            "other",
                        ],
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["text", "block_type"],
            },
        },
        "language": {"type": "string"},
    },
    "required": ["text", "blocks"],
}

VISION_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Concise summary of the visual scene or document.",
        },
        "extracted_text": {
            "type": "string",
            "description": "All visible text/OCR in the image.",
        },
        "scene_context": {
            "type": "object",
            "properties": {
                "setting": {"type": "string"},
                "environment": {"type": "string"},
                "time_of_day_or_lighting": {"type": "string"},
                "domain": {"type": "string"},
            },
            "required": ["setting"],
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "category": {"type": "string"},
                    "attributes": {"type": "array", "items": {"type": "string"}},
                    "location_hint": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["label"],
            },
        },
        "activities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "activity": {"type": "string"},
                    "actors": {"type": "array", "items": {"type": "string"}},
                    "objects_involved": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                },
                "required": ["activity"],
            },
        },
    },
    "required": ["summary", "extracted_text", "scene_context", "objects", "activities"],
}

VISION_SCHEMA_PRESETS: dict[str, dict[str, Any]] = {
    "vision_ocr": VISION_OCR_SCHEMA,
    "vision_extraction": VISION_EXTRACTION_SCHEMA,
}


def resolve_vision_schema(
    preset_or_schema: str | dict[str, Any] | None,
    intent: str | None = None,
) -> dict[str, Any] | None:
    """Resolve an explicit JSON schema, named preset, or intent default into a schema dict."""
    if isinstance(preset_or_schema, dict):
        return preset_or_schema
    if isinstance(preset_or_schema, str):
        key = preset_or_schema.strip().lower()
        if key in VISION_SCHEMA_PRESETS:
            return VISION_SCHEMA_PRESETS[key]
    if intent:
        intent_key = str(intent).strip().lower()
        if intent_key in VISION_SCHEMA_PRESETS:
            return VISION_SCHEMA_PRESETS[intent_key]
    return None
