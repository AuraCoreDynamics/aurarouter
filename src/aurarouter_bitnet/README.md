# aurarouter-bitnet

CPU backend plugin for [AuraRouter](https://github.com/auracoredynamics/aurarouter) that provides inference for **BitNet** (1-bit) large language models.

## Overview

BitNet models use 1-bit or ternary weight quantisation, enabling efficient CPU-only inference without a GPU. This plugin bundles a llama.cpp build compiled with BitNet kernel support and exposes it through the standard AuraRouter backend interface.

## CPU Feature Detection

The diagnostics module detects hardware features that accelerate BitNet inference:

| Feature | Architecture | Detection Method |
|---------|-------------|-----------------|
| AVX2 | x86_64 | CPUID / OS API |
| AVX512 | x86_64 | CPUID / OS API |
| NEON | AArch64 | Mandatory on ARM64 |

No subprocess calls are used — detection relies on `ctypes`, `platform`, and `/proc/cpuinfo` (Linux).

## Installation

```bash
pip install aurarouter-bitnet
```

Or for development:

```bash
pip install -e src/aurarouter_bitnet/
```

## Binary Setup

Place the appropriate `llama-server` binary in the platform directory:

```
src/aurarouter_bitnet/bin/
├── win-x64/llama-server.exe
├── linux-x64/llama-server
└── macos-x64/llama-server
```

Build llama.cpp with BitNet support enabled for your target platform.

## Plugin Interface

| Export | Description |
|--------|------------|
| `METADATA` | Package metadata dict (flavor, compute_type, score, etc.) |
| `run_diagnostic()` | CPU feature detection and binary validation |
| `setup_runtime_environment()` | Resolves binary path, configures DLL/PATH |

## Scoring

This backend scores **70** — above generic CPU (50) and below Vulkan (80). The Core will prefer a GPU backend when available but will select BitNet over a plain CPU fallback.

## Testing

```bash
pytest tests/test_bitnet_plugin.py -x -q
```
