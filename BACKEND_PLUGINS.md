# AuraRouter Backend Plugin Guide

This document describes the technical architecture of AuraRouter's modular backend system and provides instructions on how to develop and package new hardware-specific plugins.

## 1. Architectural Overview

AuraRouter (Core) is a platform-agnostic Python package. It does not contain any binary files. Local inference capabilities are provided by **Backend Plugins**—separate Python packages that bundle `llama.cpp` binaries and their dependencies (CUDA, Vulkan, etc.).

The Core uses a **Discovery and Scoring** mechanism to select the best available backend at runtime.

### Dynamic Discovery
When `BinaryManager.resolve_server_binary()` is called, it:
1. Scans the environment for all installed packages named `aurarouter_*` (e.g., `aurarouter_cuda13`).
2. Interrogates each package via its standard interface.
3. Runs hardware diagnostics to see if the required hardware (GPU) is actually present.
4. Scores each backend and selects the highest-scoring healthy one.

## 2. The Backend Plugin Interface

Every AuraRouter backend plugin must implement the following structure:

### `__init__.py`
Must contain `setup_runtime_environment()`. This function is called by the Core immediately after selection. It is responsible for setting up DLL search paths (`os.add_dll_directory`) and environment variables.

### `metadata.py`
Must contain a `METADATA` dictionary:
- `package_name`: The PyPI package name.
- `flavor`: User-friendly name (e.g., "CUDA 13.1").
- `compute_type`: "GPU (NVIDIA)", "GPU (Generic)", or "CPU".
- `llama_cpp_build`: The build number of the bundled binary.

### `diagnostics.py`
Must contain `run_diagnostic()`. This function performs live hardware checks (e.g., calling `nvidia-smi` or checking for Vulkan loaders). It returns a dictionary that the Core uses for scoring.

### `logging.py`
Should implement a `get_logger()` helper that uses the namespace `AuraRouter.Backend.<Name>`.

## 3. Binary Directory Structure

Plugins must store their binaries in a specific nested directory structure to allow the Core to find them across different operating systems:

```text
src/
└── aurarouter_plugin_name/
    └── bin/
        ├── win-x64/
        │   ├── llama-server.exe
        │   └── *.dll
        ├── linux-x64/
        │   ├── llama-server
        │   └── *.so
        └── macos-x64/
            ├── llama-server
            └── *.dylib
```

## 4. How to Build a New Plugin

Follow these steps to create a new backend (e.g., `aurarouter-rocm` for AMD GPUs):

1. **Scaffold the Project**: Use `setuptools` and create the directory structure above.
2. **Bundle Binaries**: Place your hardware-accelerated `llama-server` and its required libraries in the `bin/` folders.
3. **Implement Interface**: Copy and adapt the `__init__.py`, `metadata.py`, and `diagnostics.py` from an existing backend like `aurarouter-cuda13`.
4. **Configure `pyproject.toml`**: Ensure `include-package-data = true` and that `bin/**/*` is included in `tool.setuptools.package-data`.
5. **Testing**:
   - Install your plugin: `pip install -e .`
   - Run AuraRouter with debug logs: `aurarouter --help`
   - Verify that your plugin is discovered and scored correctly.

## 5. Scoring Table

The current scoring logic in `BinaryManager.py` follows these weights:

| Capability | Score |
|------------|-------|
| NVIDIA GPU (Detected) | 100 |
| Generic GPU (Vulkan/Metal) | 80 |
| CPU (Generic) | 50 |
| Hardware Missing / Diagnostic Fail | 0 |

## 6. Standalone Executable Support
...
```

## 7. PyPI Distribution (Important Note)

The CUDA backend packages (`aurarouter-cuda13`, `aurarouter-cuda12`) are very large due to the bundled NVIDIA DLLs (~400MB+ compressed). 

PyPI has a default limit of **100MB per file**. Before publishing these backends, you must:
1. Log in to your PyPI account.
2. Navigate to the project settings for each backend.
3. Request a **Size Limit Increase** (Project Request) explaining that these are hardware-specific sidecar packages containing required binary payloads for `llama.cpp`.
