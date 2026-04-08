from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import yaml

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.Config")


def _default_config_path() -> Path:
    return Path.home() / ".auracore" / "aurarouter" / "auraconfig.yaml"


class ConfigLoader:
    """Finds, loads, mutates, and persists auraconfig.yaml.

    Read path (existing behaviour):
        Searches --config, AURACORE_ROUTER_CONFIG env var, then
        ~/.auracore/aurarouter/auraconfig.yaml.

    Write path (new):
        ``set_model``, ``remove_model``, ``set_role_chain``, ``remove_role``
        mutate the in-memory config.  ``save()`` atomically writes it back.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        allow_missing: bool = False,
    ):
        self._config_path: Optional[Path] = None

        if allow_missing:
            self.config: dict = {}
            return

        resolved = self._find_config(config_path)

        if not resolved:
            searched: list[str] = []
            if config_path:
                searched.append(
                    f"  - Command line (--config): {Path(config_path).resolve()}"
                )
            env_path = os.environ.get("AURACORE_ROUTER_CONFIG")
            if env_path:
                searched.append(
                    f"  - Environment variable (AURACORE_ROUTER_CONFIG): "
                    f"{Path(env_path).resolve()}"
                )
            searched.append(
                f"  - User home directory: {_default_config_path()}"
            )
            raise FileNotFoundError(
                "Could not find 'auraconfig.yaml'. "
                "Searched in the following locations:\n" + "\n".join(searched)
            )

        self._config_path = resolved
        with open(resolved, "r") as f:
            self.config = yaml.safe_load(f) or {}
        logger.info(f"Loaded configuration from: {resolved.resolve()}")

    # ------------------------------------------------------------------
    # Config discovery
    # ------------------------------------------------------------------

    def _find_config(self, config_path: Optional[str]) -> Optional[Path]:
        """Search for auraconfig.yaml in priority order."""
        # 1. Explicit --config argument
        if config_path:
            p = Path(config_path)
            logger.info(f"Attempting config from --config: {p.resolve()}")
            if p.is_file():
                return p
            logger.warning(f"Config not found at --config path: {p.resolve()}")

        # 2. AURACORE_ROUTER_CONFIG environment variable
        env = os.environ.get("AURACORE_ROUTER_CONFIG")
        if env:
            p = Path(env)
            logger.info(f"Attempting config from env var: {p.resolve()}")
            if p.is_file():
                return p
            logger.warning(f"Config not found at env var path: {p.resolve()}")

        # 3. User home directory
        home = _default_config_path()
        logger.info(f"Attempting config from home dir: {home.resolve()}")
        if home.is_file():
            return home

        return None

    @property
    def config_path(self) -> Optional[Path]:
        """The resolved path the config was loaded from (or will save to)."""
        return self._config_path

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    def get_role_chain(self, role: str) -> list[str]:
        """Get model chain for a role. Supports both flat list and nested dict formats.

        Returns a copy so callers cannot mutate internal state.
        """
        role_config = self.config.get("roles", {}).get(role, [])

        # Handle nested dict format (supports both 'chain' and 'models' keys)
        if isinstance(role_config, dict):
            return list(role_config.get("chain", role_config.get("models", [])))

        # Handle flat list format (auraconfig.yaml style)
        return list(role_config)

    def get_model_config(self, model_id: str) -> dict:
        """Return a copy of the model config dict."""
        return dict(self.config.get("models", {}).get(model_id, {}))

    def get_all_model_ids(self) -> list[str]:
        """Return all configured model IDs."""
        return list(self.config.get("models", {}).keys())

    def get_all_roles(self) -> list[str]:
        """Return all configured role names."""
        return list(self.config.get("roles", {}).keys())

    # ------------------------------------------------------------------
    # Provider catalog accessors
    # ------------------------------------------------------------------

    def get_catalog_manual_entries(self) -> list[dict]:
        """Return manually configured provider catalog entries.

        Reads from ``provider_catalog.manual`` in the config::

            provider_catalog:
              manual:
                - name: gemini
                  endpoint: http://localhost:9001
                  auto_start: true
        """
        return self.config.get("provider_catalog", {}).get("manual", [])

    def add_catalog_manual_entry(
        self,
        name: str,
        endpoint: str,
        auto_start: bool = False,
    ) -> None:
        """Add a manual provider entry to the catalog config."""
        catalog = self.config.setdefault("provider_catalog", {})
        manual: list[dict] = catalog.setdefault("manual", [])

        # Update existing entry if name matches
        for entry in manual:
            if entry.get("name") == name:
                entry["endpoint"] = endpoint
                entry["auto_start"] = auto_start
                return

        manual.append({
            "name": name,
            "endpoint": endpoint,
            "auto_start": auto_start,
        })

    def remove_catalog_manual_entry(self, name: str) -> bool:
        """Remove a manual provider entry by name. Returns True if found."""
        manual = self.config.get("provider_catalog", {}).get("manual", [])
        for i, entry in enumerate(manual):
            if entry.get("name") == name:
                manual.pop(i)
                return True
        return False

    def get_catalog_auto_start_entrypoints(self) -> bool:
        """Whether to auto-start entry-point providers on server boot."""
        return self.config.get("provider_catalog", {}).get(
            "auto_start_entrypoints", True
        )

    def get_max_review_iterations(self) -> int:
        """Return max_review_iterations from execution config, default 3."""
        return self.config.get("execution", {}).get("max_review_iterations", 3)

    def get_broadcast_timeout(self) -> float:
        """Return broker broadcast timeout from execution config, default 10.0."""
        return float(
            self.config.get("execution", {}).get("broadcast_timeout", 10.0)
        )

    @property
    def event_reporter_max_workers(self) -> int:
        """Max worker threads for the EventReporter pool (default 8)."""
        return int(self.config.get("event_reporter_max_workers", 8))

    @property
    def event_reporter_max_queue_depth(self) -> int:
        """Max in-flight + pending events before dropping (default 256)."""
        return int(self.config.get("event_reporter_max_queue_depth", 256))

    def get_model_hosting_tier(self, model_id: str) -> str | None:
        """Return the hosting_tier for a model, or None if not set."""
        model_cfg = self.config.get("models", {}).get(model_id, {})
        return model_cfg.get("hosting_tier")

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    def set_model(self, model_id: str, model_config: dict) -> None:
        """Add or update a model definition."""
        if "models" not in self.config:
            self.config["models"] = {}
        self.config["models"][model_id] = model_config

    def remove_model(self, model_id: str) -> bool:
        """Remove a model definition. Returns True if it existed."""
        models = self.config.get("models", {})
        if model_id in models:
            del models[model_id]
            return True
        return False

    def set_role_chain(self, role: str, chain: list[str]) -> None:
        """Set the model chain for a role (flat list format)."""
        if "roles" not in self.config:
            self.config["roles"] = {}
        self.config["roles"][role] = chain

    def remove_role(self, role: str) -> bool:
        """Remove a role. Returns True if it existed."""
        roles = self.config.get("roles", {})
        if role in roles:
            del roles[role]
            return True
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        """Atomically write the current config to YAML.

        Parameters
        ----------
        path:
            Target file.  Defaults to the path the config was originally
            loaded from, or ``~/.auracore/aurarouter/auraconfig.yaml``.

        Returns
        -------
        The path the file was written to.
        """
        target = path or self._config_path or _default_config_path()
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file in same directory, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent), suffix=".yaml.tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(
                    self.config,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            os.replace(tmp_path, str(target))
        except BaseException:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        self._config_path = target
        logger.info(f"Configuration saved to: {target}")
        return target

    def to_yaml(self) -> str:
        """Return the current config as a YAML string (for preview)."""
        return yaml.dump(
            self.config,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    # ------------------------------------------------------------------
    # Savings / triage / sessions / grid config accessors
    # ------------------------------------------------------------------

    def is_savings_enabled(self) -> bool:
        """Whether the savings subsystem is enabled (opt-out design, defaults to True)."""
        return self.config.get("savings", {}).get("enabled", True)

    def get_savings_config(self) -> dict:
        """Return the savings config section."""
        return self.config.get("savings", {})

    def get_pricing_overrides(self) -> dict:
        """Return per-model pricing overrides from savings config."""
        return self.config.get("savings", {}).get("pricing_overrides", {})

    def get_model_pricing(self, model_id: str) -> tuple:
        """Return pricing config for a specific model as (input_per_million, output_per_million).

        Returns (None, None) if no pricing is configured.
        """
        model_cfg = self.get_model_config(model_id)
        inp = model_cfg.get("cost_per_1m_input")
        out = model_cfg.get("cost_per_1m_output")
        return (inp, out)

    def get_privacy_config(self) -> dict:
        """Return privacy config from savings."""
        return self.config.get("savings", {}).get("privacy", {})

    def get_budget_config(self) -> dict:
        """Return budget config from savings."""
        return self.config.get("savings", {}).get("budget", {})

    def get_triage_config(self) -> dict:
        """Return triage config from savings."""
        return self.config.get("savings", {}).get("triage", {})

    def get_feedback_config(self) -> dict:
        """Return the feedback config from savings."""
        return self.config.get("savings", {}).get("feedback", {})

    def is_feedback_enabled(self) -> bool:
        """Whether routing feedback recording is enabled."""
        return self.get_feedback_config().get("enabled", False)

    def get_grid_services_config(self) -> dict:
        """Return the grid_services configuration section."""
        return self.config.get("grid_services", {})

    def get_sessions_config(self) -> dict:
        """Return the sessions configuration section."""
        return self.config.get("sessions", {})

    # ------------------------------------------------------------------
    # XLM integration accessors
    # ------------------------------------------------------------------

    def get_xlm_config(self) -> dict:
        """Return the 'xlm' section from config, or empty dict."""
        return self.config.get("xlm", {})

    def is_xlm_augmentation_enabled(self) -> bool:
        """Check xlm.features.prompt_augmentation flag."""
        return self.get_xlm_config().get("features", {}).get("prompt_augmentation", False)

    def is_xlm_usage_reporting_enabled(self) -> bool:
        """Check xlm.features.usage_reporting flag."""
        return self.get_xlm_config().get("features", {}).get("usage_reporting", False)

    def get_xlm_endpoint(self) -> str:
        """Return xlm.endpoint URL, or empty string."""
        return self.get_xlm_config().get("endpoint", "")

    @property
    def replica_count(self) -> int:
        """Number of AuraRouter replicas sharing the XLM rate-limit budget.

        Resolution order: AURACORE_REPLICA_COUNT env-var → config key
        'replica_count' → 1.  Clamped to [1, 100].
        """
        env_val = os.environ.get("AURACORE_REPLICA_COUNT", "")
        try:
            raw = int(env_val) if env_val else int(self.config.get("replica_count", 1))
        except (ValueError, TypeError):
            raw = 1
        return max(1, min(100, raw))

    # ------------------------------------------------------------------
    # RAG enrichment & sovereignty accessors
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pluggable Analyzer Pipeline config (TG1)
    # ------------------------------------------------------------------

    def get_pipeline_config(self) -> dict:
        """Return the system.analyzer_pipeline configuration section.

        Sensible defaults:
          enabled: false
          confidence_threshold: 0.85
          analyzers: ["onnx-vector", "edge-complexity", "slm-intent"]
        """
        defaults: dict = {
            "enabled": False,
            "confidence_threshold": 0.85,
            "analyzers": ["onnx-vector", "edge-complexity", "slm-intent"],
        }
        cfg = self.config.get("system", {}).get("analyzer_pipeline", {})
        return {**defaults, **cfg}

    def get_complexity_scorer_config(self) -> dict:
        """Return edge_complexity sub-section of system.analyzer_pipeline.

        Defaults: simple_ceiling=3, complex_floor=7.
        """
        defaults: dict = {"simple_ceiling": 3, "complex_floor": 7}
        cfg = (
            self.config.get("system", {})
            .get("analyzer_pipeline", {})
            .get("edge_complexity", {})
        )
        return {**defaults, **cfg}

    def get_hard_route_config(self) -> dict:
        """Return savings.hard_route configuration section.

        Defaults: reference_cloud_model='claude-3-5-haiku',
                  reference_cloud_provider='anthropic',
                  assumed_output_tokens=200.
        """
        defaults: dict = {
            "reference_cloud_model": "claude-3-5-haiku",
            "reference_cloud_provider": "anthropic",
            "assumed_output_tokens": 200,
        }
        cfg = self.config.get("savings", {}).get("hard_route", {})
        return {**defaults, **cfg}

    def is_rag_enrichment_enabled(self) -> bool:
        """Check system.rag_enrichment flag (default False)."""
        return self.config.get("system", {}).get("rag_enrichment", False)

    def is_sovereignty_enforcement_enabled(self) -> bool:
        """Check system.sovereignty_enforcement flag (default False)."""
        return self.config.get("system", {}).get("sovereignty_enforcement", False)

    def get_sovereignty_patterns(self) -> list[dict]:
        """Return custom sovereignty patterns from system.sovereignty_patterns."""
        return self.config.get("system", {}).get("sovereignty_patterns", [])

    # ------------------------------------------------------------------
    # MCP tool enable/disable
    # ------------------------------------------------------------------

    def is_mcp_tool_enabled(self, tool_name: str, default: bool = True) -> bool:
        """Check whether a specific MCP tool is enabled in config."""
        tools_cfg = self.config.get("mcp", {}).get("tools", {})
        tool_cfg = tools_cfg.get(tool_name, {})
        if isinstance(tool_cfg, dict):
            return tool_cfg.get("enabled", default)
        return default

    def set_mcp_tool_enabled(self, tool_name: str, enabled: bool) -> None:
        """Set the enabled state for a specific MCP tool."""
        mcp = self.config.setdefault("mcp", {})
        tools = mcp.setdefault("tools", {})
        tools.setdefault(tool_name, {})["enabled"] = enabled

    def get_mcp_tools_config(self) -> dict:
        """Return the full mcp.tools config section."""
        return self.config.get("mcp", {}).get("tools", {})

    # ------------------------------------------------------------------
    # Semantic verbs config
    # ------------------------------------------------------------------

    def set_semantic_verb(self, role: str, synonyms: list[str]) -> None:
        """Set custom semantic verb synonyms for a role."""
        verbs = self.config.setdefault("semantic_verbs", {})
        verbs[role] = synonyms

    def get_semantic_verbs(self) -> dict:
        """Return custom semantic verb mappings."""
        return self.config.get("semantic_verbs", {})

    # ------------------------------------------------------------------
    # Tag-to-role auto-integration
    # ------------------------------------------------------------------

    def auto_join_roles(
        self,
        model_id: str,
        tags: list[str],
        intent_registry: "IntentRegistry | None" = None,
        supported_intents: list[str] | None = None,
    ) -> list[str]:
        """Add model_id to role chains when tags match role names or synonyms.

        If *intent_registry* is provided and *supported_intents* is non-empty,
        each supported intent is resolved to a target role via the registry
        and the model is auto-joined to that role as well.

        Parameters
        ----------
        model_id:
            The model to add to role chains.
        tags:
            Freeform tags for tag-based auto-join.
        intent_registry:
            Optional :class:`IntentRegistry` for intent-to-role resolution.
        supported_intents:
            Optional list of intent names the model declares support for.

        Returns list of role names the model was added to.
        """
        from aurarouter.semantic_verbs import resolve_synonym

        custom_verbs = self.get_semantic_verbs()
        roles_joined: list[str] = []
        existing_roles = set(self.get_all_roles())

        for tag in tags:
            # Try direct role match (case-insensitive)
            tag_lower = tag.strip().lower()
            matched_role = None
            for role in existing_roles:
                if tag_lower == role.lower():
                    matched_role = role
                    break

            # Try synonym resolution
            if matched_role is None:
                resolved = resolve_synonym(tag, custom_verbs or None)
                if resolved.lower() != tag_lower and resolved in existing_roles:
                    matched_role = resolved

            if matched_role is not None:
                chain = self.get_role_chain(matched_role)
                if model_id not in chain:
                    self.set_role_chain(matched_role, chain + [model_id])
                    roles_joined.append(matched_role)

        # Intent-based auto-join: resolve each supported intent to a role
        if intent_registry is not None and supported_intents:
            for intent_name in supported_intents:
                target_role = intent_registry.resolve_role(intent_name)
                if target_role is None:
                    continue
                # Ensure the role exists in config
                if target_role not in existing_roles:
                    continue
                if target_role in roles_joined:
                    continue
                chain = self.get_role_chain(target_role)
                if model_id not in chain:
                    self.set_role_chain(target_role, chain + [model_id])
                    roles_joined.append(target_role)

        return roles_joined

    # ------------------------------------------------------------------
    # Unified Artifact Catalog CRUD
    # ------------------------------------------------------------------

    def catalog_get(self, artifact_id: str) -> dict | None:
        """Look up an artifact by ID.

        Checks ``config["catalog"][id]`` first, then falls back to
        ``config["models"][id]`` (legacy models treated as kind=model).

        Analyzer artifacts are enriched with a computed
        ``declared_intents`` field extracted from ``role_bindings`` keys.
        """
        from aurarouter.analyzer_schema import extract_declared_intents

        entry = self.config.get("catalog", {}).get(artifact_id)
        if entry is not None:
            result = dict(entry)
            if result.get("kind") == "analyzer":
                result["declared_intents"] = extract_declared_intents(result)
            return result
        # Legacy fallback: models section
        model_cfg = self.config.get("models", {}).get(artifact_id)
        if model_cfg is not None:
            result = dict(model_cfg)
            result.setdefault("kind", "model")
            result.setdefault("display_name", artifact_id)
            return result
        return None

    def catalog_list(self, kind: str | None = None) -> list[str]:
        """Return artifact IDs from catalog + legacy models.

        If *kind* is provided, only IDs matching that kind are returned.
        Legacy models are treated as ``kind="model"``.
        """
        ids: list[str] = []

        # Catalog section
        for aid, data in self.config.get("catalog", {}).items():
            if kind is None or data.get("kind", "model") == kind:
                ids.append(aid)

        # Legacy models section (kind=model)
        if kind is None or kind == "model":
            for mid in self.config.get("models", {}).keys():
                if mid not in ids:
                    ids.append(mid)

        return ids

    def catalog_set(self, artifact_id: str, data: dict) -> None:
        """Write an artifact to ``config["catalog"][artifact_id]``.

        If the artifact is an analyzer, its spec is validated and any
        warnings or errors are logged.  Validation is warn-only — the
        artifact is always written for backwards compatibility.
        """
        catalog = self.config.setdefault("catalog", {})
        catalog[artifact_id] = data

        # Validate analyzer specs at registration time (warn-only)
        if data.get("kind") == "analyzer":
            from aurarouter.analyzer_schema import validate_analyzer_spec

            available_roles = self.get_all_roles() or None
            result = validate_analyzer_spec(data, available_roles=available_roles)
            for err in result.errors:
                logger.warning(
                    "analyzer %s spec error: %s", artifact_id, err
                )
            for warn in result.warnings:
                logger.warning(
                    "analyzer %s spec warning: %s", artifact_id, warn
                )
            # declared_intents are computed dynamically in catalog_get()
            # and catalog_query() — no need to store in the raw config dict.

    def catalog_remove(self, artifact_id: str) -> bool:
        """Remove an artifact from ``config["catalog"]``. Returns True if existed."""
        catalog = self.config.get("catalog", {})
        if artifact_id in catalog:
            del catalog[artifact_id]
            return True
        return False

    def catalog_get_declared_intents(self, analyzer_id: str) -> list[str]:
        """Extract intent names from an analyzer's role_bindings keys.

        Returns an empty list if the artifact is not found, is not an
        analyzer, or has no role_bindings.
        """
        from aurarouter.analyzer_schema import extract_declared_intents

        data = self.catalog_get(analyzer_id)
        if data is None or data.get("kind") != "analyzer":
            return []
        return extract_declared_intents(data)

    def catalog_query(
        self,
        kind: str | None = None,
        tags: list[str] | None = None,
        capabilities: list[str] | None = None,
        provider: str | None = None,
        intents: list[str] | None = None,
        supported_intents: list[str] | None = None,
    ) -> list[dict]:
        """Filtered query over catalog + legacy models.

        Each result dict is enriched with ``artifact_id``.  Analyzer
        artifacts are also enriched with ``declared_intents``.

        Parameters
        ----------
        kind:
            Filter by artifact kind (model, service, analyzer).
        tags:
            All specified tags must be present on the artifact.
        capabilities:
            All specified capabilities must be present.
        provider:
            Exact provider match.
        intents:
            Filter analyzer artifacts to those whose ``role_bindings``
            keys include at least one of the requested intents.  Non-
            analyzer artifacts are excluded when this filter is active.
        supported_intents:
            Filter model artifacts to those whose ``supported_intents``
            list includes at least one of the requested intents.
            Artifacts without ``supported_intents`` are excluded.
        """
        from aurarouter.analyzer_schema import extract_declared_intents

        results: list[dict] = []

        # Gather all candidates: catalog entries + legacy models
        candidates: dict[str, dict] = {}
        for aid, data in self.config.get("catalog", {}).items():
            candidates[aid] = dict(data)
        for mid, mcfg in self.config.get("models", {}).items():
            if mid not in candidates:
                entry = dict(mcfg)
                entry.setdefault("kind", "model")
                entry.setdefault("display_name", mid)
                candidates[mid] = entry

        for aid, data in candidates.items():
            # Filter by kind
            if kind is not None and data.get("kind", "model") != kind:
                continue
            # Filter by tags (all specified tags must be present)
            if tags is not None:
                entry_tags = set(data.get("tags", []))
                if not all(t in entry_tags for t in tags):
                    continue
            # Filter by capabilities (all specified caps must be present)
            if capabilities is not None:
                entry_caps = set(data.get("capabilities", []))
                if not all(c in entry_caps for c in capabilities):
                    continue
            # Filter by provider
            if provider is not None and data.get("provider", "") != provider:
                continue
            # Filter by intents (analyzer-only)
            if intents is not None:
                if data.get("kind", "model") != "analyzer":
                    continue
                artifact_intents = set(extract_declared_intents(data))
                if not any(i in artifact_intents for i in intents):
                    continue
            # Filter by supported_intents (model supported_intents field)
            if supported_intents is not None:
                entry_supported = set(data.get("supported_intents", []))
                if not any(si in entry_supported for si in supported_intents):
                    continue

            result = dict(data)
            result["artifact_id"] = aid

            # Enrich analyzer artifacts with declared_intents
            if data.get("kind") == "analyzer":
                result["declared_intents"] = extract_declared_intents(data)

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Active analyzer
    # ------------------------------------------------------------------

    def get_active_analyzer(self) -> str | None:
        """Read the active analyzer ID from ``config["system"]["active_analyzer"]``."""
        return self.config.get("system", {}).get("active_analyzer")

    def set_active_analyzer(self, analyzer_id: str | None) -> None:
        """Write (or clear) the active analyzer in ``config["system"]``."""
        system = self.config.setdefault("system", {})
        if analyzer_id is None:
            system.pop("active_analyzer", None)
        else:
            system["active_analyzer"] = analyzer_id
