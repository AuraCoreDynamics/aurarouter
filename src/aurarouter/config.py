import copy
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

import yaml

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.Config")


def _deep_copy_dict(d: dict) -> dict:
    """Recursively copy a dict of plain YAML-safe types (dicts, lists, scalars)."""
    return copy.deepcopy(d)


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
        self._lock = threading.RLock()
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
        """Get model chain for a role. Supports both flat list and nested dict formats."""
        with self._lock:
            role_config = self.config.get("roles", {}).get(role, [])

            # Handle nested dict format (supports both 'chain' and 'models' keys)
            if isinstance(role_config, dict):
                return list(role_config.get("chain", role_config.get("models", [])))

            # Handle flat list format (auraconfig.yaml style)
            return list(role_config)

    def get_model_config(self, model_id: str) -> dict:
        with self._lock:
            cfg = self.config.get("models", {}).get(model_id, {})
            return dict(cfg)

    def get_model_tags(self, model_id: str) -> list[str]:
        """Return the tags list for a model, or ``[]`` if none."""
        with self._lock:
            cfg = self.get_model_config(model_id)
            tags = cfg.get("tags", [])
            return list(tags) if isinstance(tags, list) else []

    def get_model_pricing(self, model_id: str) -> tuple[float | None, float | None]:
        """Return (cost_per_1m_input, cost_per_1m_output) for a model.

        Returns (None, None) if the model has no explicit cost fields.
        """
        with self._lock:
            cfg = self.get_model_config(model_id)
            return (
                cfg.get("cost_per_1m_input"),
                cfg.get("cost_per_1m_output"),
            )

    def get_model_hosting_tier(self, model_id: str) -> str | None:
        """Return the ``hosting_tier`` for a model, or ``None`` if absent.

        Valid values: ``"on-prem"``, ``"cloud"``, ``"dedicated-tenant"``.
        """
        with self._lock:
            cfg = self.get_model_config(model_id)
            return cfg.get("hosting_tier")

    def get_all_model_ids(self) -> list[str]:
        """Return all configured model IDs."""
        with self._lock:
            return list(self.config.get("models", {}).keys())

    def get_all_roles(self) -> list[str]:
        """Return all configured role names."""
        with self._lock:
            return list(self.config.get("roles", {}).keys())

    # ------------------------------------------------------------------
    # Savings accessors (read-only)
    # ------------------------------------------------------------------

    def get_savings_config(self) -> dict:
        """Return the ``savings`` section, or ``{}`` if absent."""
        with self._lock:
            return dict(self.config.get("savings", {}))

    def get_budget_config(self) -> dict:
        """Return ``savings.budget``, or ``{}`` if absent."""
        with self._lock:
            return dict(self.get_savings_config().get("budget", {}))

    def get_privacy_config(self) -> dict:
        """Return ``savings.privacy``, or ``{}`` if absent."""
        with self._lock:
            return dict(self.get_savings_config().get("privacy", {}))

    def get_pricing_overrides(self) -> dict:
        """Return ``savings.pricing_overrides``, or ``{}`` if absent."""
        with self._lock:
            return dict(self.get_savings_config().get("pricing_overrides", {}))

    def get_triage_config(self) -> dict:
        """Return ``savings.triage``, or ``{}`` if absent."""
        with self._lock:
            return dict(self.get_savings_config().get("triage", {}))

    def is_savings_enabled(self) -> bool:
        """Return whether savings tracking is enabled (default ``True``)."""
        with self._lock:
            return self.get_savings_config().get("enabled", True)

    # ------------------------------------------------------------------
    # Sessions accessors
    # ------------------------------------------------------------------

    def get_sessions_config(self) -> dict:
        """Return the ``sessions`` section, or ``{}`` if absent.

        Expected keys:
        - enabled: bool (default False)
        - store_path: str | None (default None -> ~/.auracore/aurarouter/sessions.db)
        - condensation_threshold: float (default 0.8)
        - auto_gist: bool (default True)
        """
        with self._lock:
            return dict(self.config.get("sessions", {}))

    # ------------------------------------------------------------------
    # Grid services accessors
    # ------------------------------------------------------------------

    def get_grid_services_config(self) -> dict:
        """Return the ``grid_services`` section, or ``{}`` if absent.

        Expected keys:
        - endpoints: list[dict] with ``url`` and optional ``name``
        - auto_sync_models: bool (default True)
        """
        with self._lock:
            return dict(self.config.get("grid_services", {}))

    # ------------------------------------------------------------------
    # Execution accessors
    # ------------------------------------------------------------------

    def get_max_review_iterations(self) -> int:
        """Return the maximum number of review-correct cycles.

        Reads from ``execution.max_review_iterations``.  Defaults to ``3``.
        Returns ``0`` to disable the review loop entirely.
        """
        with self._lock:
            execution = self.config.get("execution", {})
            return int(execution.get("max_review_iterations", 3))

    # ------------------------------------------------------------------
    # MCP tools accessors
    # ------------------------------------------------------------------

    def get_mcp_tools_config(self) -> dict:
        """Return the ``mcp.tools`` section, or ``{}`` if absent."""
        with self._lock:
            return dict(self.config.get("mcp", {}).get("tools", {}))

    def is_mcp_tool_enabled(self, tool_name: str, default: bool = True) -> bool:
        """Check if a specific MCP tool is enabled.

        Returns *default* when the tool has no explicit config entry.
        """
        with self._lock:
            tool_cfg = self.get_mcp_tools_config().get(tool_name, {})
            return tool_cfg.get("enabled", default)

    def set_mcp_tool_enabled(self, tool_name: str, enabled: bool) -> None:
        """Set the enabled state for an MCP tool."""
        with self._lock:
            mcp = self.config.setdefault("mcp", {})
            tools = mcp.setdefault("tools", {})
            tool_entry = tools.setdefault(tool_name, {})
            tool_entry["enabled"] = enabled

    # ------------------------------------------------------------------
    # Semantic verbs
    # ------------------------------------------------------------------

    def get_semantic_verbs(self) -> dict[str, list[str]]:
        """Return ``{role: [synonym, ...]}`` from the config."""
        with self._lock:
            raw = self.config.get("semantic_verbs", {})
            result: dict[str, list[str]] = {}
            for role, value in raw.items():
                if isinstance(value, dict):
                    result[role] = list(value.get("synonyms", []))
                elif isinstance(value, list):
                    result[role] = list(value)
            return result

    def set_semantic_verb(self, role: str, synonyms: list[str]) -> None:
        """Set synonyms for a role in the ``semantic_verbs`` section."""
        with self._lock:
            section = self.config.setdefault("semantic_verbs", {})
            section[role] = {"synonyms": list(synonyms)}

    # ------------------------------------------------------------------
    # Tag-to-role auto-integration
    # ------------------------------------------------------------------

    def auto_join_roles(self, model_id: str, tags: list[str]) -> list[str]:
        """Add *model_id* to role chains matching the given tags.

        Resolves tags against known role names and semantic verb synonyms.
        Returns list of role names the model was added to.
        """
        with self._lock:
            roles_joined: list[str] = []
            known_roles = self.get_all_roles()

            # Build synonym reverse lookup from semantic verbs
            semantic_verbs = self.get_semantic_verbs()
            synonym_to_role: dict[str, str] = {}
            for role, synonyms in semantic_verbs.items():
                for syn in synonyms:
                    synonym_to_role[syn.lower()] = role

            for tag in tags:
                tag_lower = tag.lower()
                matched_role = None
                if tag_lower in known_roles:
                    matched_role = tag_lower
                elif tag_lower in synonym_to_role:
                    matched_role = synonym_to_role[tag_lower]

                if matched_role and model_id not in self.get_role_chain(matched_role):
                    chain = self.get_role_chain(matched_role)
                    self.set_role_chain(matched_role, chain + [model_id])
                    roles_joined.append(matched_role)

            return roles_joined

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    def set_model(self, model_id: str, model_config: dict) -> None:
        """Add or update a model definition."""
        with self._lock:
            if "models" not in self.config:
                self.config["models"] = {}
            self.config["models"][model_id] = model_config

    def remove_model(self, model_id: str) -> bool:
        """Remove a model definition. Returns True if it existed."""
        with self._lock:
            models = self.config.get("models", {})
            if model_id in models:
                del models[model_id]
                return True
            return False

    def set_role_chain(self, role: str, chain: list[str]) -> None:
        """Set the model chain for a role (flat list format)."""
        with self._lock:
            if "roles" not in self.config:
                self.config["roles"] = {}
            self.config["roles"][role] = list(chain)

    def remove_role(self, role: str) -> bool:
        """Remove a role. Returns True if it existed."""
        with self._lock:
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
        with self._lock:
            target = path or self._config_path or _default_config_path()
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)

            # Snapshot config under lock so the YAML serialisation sees a
            # consistent dict even if another thread mutates after we release.
            config_snapshot = _deep_copy_dict(self.config)

        # Write outside the lock — filesystem I/O is safe on the snapshot.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent), suffix=".yaml.tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(
                    config_snapshot,
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

        with self._lock:
            self._config_path = target
        logger.info(f"Configuration saved to: {target}")
        return target

    def to_yaml(self) -> str:
        """Return the current config as a YAML string (for preview)."""
        with self._lock:
            config_snapshot = _deep_copy_dict(self.config)
        return yaml.dump(
            config_snapshot,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
