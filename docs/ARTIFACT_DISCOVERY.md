# Artifact Discovery and Testing Integration

AuraRouter's **Unified Artifact Catalog** acts as a central registry for all compute resources in the AuraCore ecosystem. This document explains how to use the catalog for dynamic artifact discovery, specifically for automated testing scenarios where downstream projects (like AuraXLM) need to identify and utilize available models.

## The Problem: Brittle Test Configurations

Unit and integration tests for AI-powered systems often require specific local models (e.g., ONNX embedding models, GGUF LLMs). Hardcoding paths or model IDs in test suites leads to "it works on my machine" failures when those models are missing or stored in different locations on other developer machines or CI/CD agents.

## The Solution: AuraRouter as a Discovery Service

By registering all available models in the AuraRouter catalog, downstream projects can query AuraRouter at runtime to find models that match specific criteria (tags, capabilities, or providers).

### 1. Registering Artifacts

Artifacts are registered in the `catalog` section of `auraconfig.yaml`. Each entry can include a `spec` with arbitrary metadata like local file paths.

```yaml
catalog:
  all-minilm-l6-v2:
    kind: model
    display_name: all-minilm-l6-v2 (ONNX)
    provider: auraxlm
    tags: [onnx, embedding, rag]
    capabilities: [embedding]
    spec:
      model_path: C:/xlm_models/all-minilm-l6-v2
      onnx_execution_provider: cpu
```

### 2. Querying via CLI (JSON)

External systems and scripts can query the catalog using the `aurarouter` CLI. The `--json` flag provides machine-readable output for easy integration.

```bash
# List all models in the catalog
aurarouter catalog artifacts --kind model --json
```

**Example Output:**
```json
[
  {
    "artifact_id": "all-minilm-l6-v2",
    "kind": "model",
    "display_name": "all-minilm-l6-v2 (ONNX)",
    "provider": "auraxlm",
    "tags": ["onnx", "embedding", "rag"],
    "spec": {
      "model_path": "C:/xlm_models/all-minilm-l6-v2",
      "onnx_execution_provider": "cpu"
    }
  }
]
```

### 3. Querying via MCP (JSON-RPC)

For live integration, the `aurarouter.catalog.list` MCP tool can be called via JSON-RPC 2.0.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "id": "1",
  "params": {
    "name": "aurarouter.catalog.list",
    "arguments": {
      "kind": "model"
    }
  }
}
```

## Practical Example: C# Unit Test Integration

In the AuraXLM project, we use a discovery utility to find ONNX models registered in AuraRouter. This allows tests to run real inference only when the required models are available on the host machine.

### Discovery Utility (Simplified)

```csharp
public static class AuraRouterDiscovery
{
    public static List<Artifact> DiscoverModels()
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = "aurarouter",
            Arguments = "catalog artifacts --kind model --json",
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var process = Process.Start(startInfo);
        string output = process.StandardOutput.ReadToEnd();
        
        // Parse JSON output...
        return JsonSerializer.Deserialize<List<Artifact>>(output);
    }
}
```

### Using in a Test Case

```csharp
[Fact]
public async Task EmbedAsync_RealModel_FromDiscovery()
{
    // 1. Discover model from AuraRouter
    var models = AuraRouterDiscovery.DiscoverModels();
    var artifact = models.Find(m => m.tags.Contains("onnx"));
    
    if (artifact == null) return; // Gracefully skip if model not registered

    // 2. Use the discovered model_path
    var options = new AuraXlmOptions {
        OnnxModelStorePath = artifact.spec["model_path"].ToString()
    };
    
    // 3. Run test...
}
```

## Potential and Future Applications

- **Hardware-Aware Testing**: Tags like `cuda12` or `metal` can be used to only run GPU-intensive tests on capable hardware.
- **CI/CD Integration**: CI agents can pre-register their available local compute resources, allowing the same test suite to adapt to different environments.
- **Dynamic Dependency Injection**: Instead of configuring services with static model IDs, they can query AuraRouter for the "best" available model for a specific task.
