"""CPU feature detection and binary validation for the BitNet backend.

All detection uses platform, struct, and ctypes only — no subprocess calls.
"""

import ctypes
import platform
import struct
import sys
from pathlib import Path


def _detect_cpu_features() -> list[str]:
    """Detect AVX2, AVX512, and NEON support via CPUID (x86) or platform hints (ARM)."""
    features: list[str] = []
    machine = platform.machine().lower()

    if machine in ("x86_64", "amd64", "x86"):
        features.extend(_detect_x86_features())
    elif machine in ("aarch64", "arm64"):
        # ARM NEON is mandatory on AArch64; no CPUID needed.
        features.append("NEON")

    return features


def _detect_x86_features() -> list[str]:
    """Use CPUID (via ctypes inline asm or OS intrinsics) to detect AVX2/AVX512."""
    features: list[str] = []

    if sys.platform == "win32":
        features = _cpuid_windows()
    elif sys.platform == "linux":
        features = _cpuid_linux()
    elif sys.platform == "darwin":
        features = _sysctl_macos()

    return features


def _cpuid_windows() -> list[str]:
    """Read CPU feature flags on Windows via kernel32 IsProcessorFeaturePresent."""
    features: list[str] = []
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40
        if kernel32.IsProcessorFeaturePresent(40):
            features.append("AVX2")
        # PF_AVX512F_INSTRUCTIONS_AVAILABLE = 41
        if kernel32.IsProcessorFeaturePresent(41):
            features.append("AVX512")
    except (AttributeError, OSError):
        pass
    return features


def _cpuid_linux() -> list[str]:
    """Read /proc/cpuinfo flags (text file, no subprocess)."""
    features: list[str] = []
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        for line in cpuinfo.splitlines():
            if line.startswith("flags"):
                tokens = line.split()
                if "avx2" in tokens:
                    features.append("AVX2")
                if "avx512f" in tokens:
                    features.append("AVX512")
                break
    except OSError:
        pass
    return features


def _sysctl_macos() -> list[str]:
    """Check macOS sysctl hw.optional via ctypes."""
    features: list[str] = []
    try:
        libc = ctypes.cdll.LoadLibrary("libSystem.B.dylib")
        buf = ctypes.c_int(0)
        buf_size = ctypes.c_size_t(ctypes.sizeof(buf))

        for name, label in [(b"hw.optional.avx2_0", "AVX2"),
                            (b"hw.optional.avx512f", "AVX512")]:
            ret = libc.sysctlbyname(
                name,
                ctypes.byref(buf),
                ctypes.byref(buf_size),
                None,
                ctypes.c_size_t(0),
            )
            if ret == 0 and buf.value == 1:
                features.append(label)
    except (OSError, AttributeError):
        pass
    return features


def _find_binary() -> bool:
    """Check whether the platform-specific llama-server binary exists."""
    machine = platform.machine().lower()

    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        plat = "win-x64"
        binary_name = "llama-server.exe"
    elif sys.platform == "linux" and machine == "x86_64":
        plat = "linux-x64"
        binary_name = "llama-server"
    elif sys.platform == "darwin" and machine in ("x86_64", "arm64"):
        plat = "macos-x64"
        binary_name = "llama-server"
    else:
        return False

    binary_path = Path(__file__).parent / "bin" / plat / binary_name
    return binary_path.is_file()


def run_diagnostic() -> dict:
    """Run CPU feature detection and binary validation.

    Returns a dict with:
        supported (bool): True if at least one optimised feature is detected and the binary exists.
        features (list[str]): Detected CPU features (AVX2, AVX512, NEON).
        binary_found (bool): Whether the platform binary is present.
        platform (str): Current platform identifier.
    """
    features = _detect_cpu_features()
    binary_found = _find_binary()
    plat = f"{sys.platform}/{platform.machine().lower()}"

    return {
        "supported": bool(features) and binary_found,
        "features": features,
        "binary_found": binary_found,
        "platform": plat,
    }
