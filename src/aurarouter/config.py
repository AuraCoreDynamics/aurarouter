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
        """Get model chain for a role. Supports both flat list and nested dict formats."""
        role_config = self.config.get("roles", {}).get(role, [])

        # Handle nested dict format (supports both 'chain' and 'models' keys)
        if isinstance(role_config, dict):
            return role_config.get("chain", role_config.get("models", []))

        # Handle flat list format (auraconfig.yaml style)
        return role_config

    def get_model_config(self, model_id: str) -> dict:
        return self.config.get("models", {}).get(model_id, {})

    def get_all_model_ids(self) -> list[str]:
        """Return all configured model IDs."""
        return list(self.config.get("models", {}).keys())

    def get_all_roles(self) -> list[str]:
        """Return all configured role names."""
        return list(self.config.get("roles", {}).keys())

    # ------------------------------------------------------------------
    # Savings accessors (read-only)
    # ------------------------------------------------------------------

    def get_savings_config(self) -> dict:
        """Return the ``savings`` section, or ``{}`` if absent."""
        return self.config.get("savings", {})

    def get_budget_config(self) -> dict:
        """Return ``savings.budget``, or ``{}`` if absent."""
        return self.get_savings_config().get("budget", {})

    def get_privacy_config(self) -> dict:
        """Return ``savings.privacy``, or ``{}`` if absent."""
        return self.get_savings_config().get("privacy", {})

    def get_pricing_overrides(self) -> dict:
        """Return ``savings.pricing_overrides``, or ``{}`` if absent."""
        return self.get_savings_config().get("pricing_overrides", {})

    def get_triage_config(self) -> dict:
        """Return ``savings.triage``, or ``{}`` if absent."""
        return self.get_savings_config().get("triage", {})

    def is_savings_enabled(self) -> bool:
        """Return whether savings tracking is enabled (default ``True``)."""
        return self.get_savings_config().get("enabled", True)

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
