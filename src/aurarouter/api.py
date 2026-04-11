"""Unified Python API for AuraRouter.

Provides :class:`AuraRouterAPI`, a stateful facade that owns the lifecycle
of all subsystems (config, fabric, savings, privacy, catalog, triage).
Synchronous only -- callers manage threading.

No PySide6 imports.  All public methods return typed dataclasses.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------

@dataclass
class APIConfig:
    """Configuration for :class:`AuraRouterAPI` initialization.

    Attributes:
        config_path: Path to ``auraconfig.yaml``.  ``None`` triggers the
            default search order (env var, home directory).
        environment: Execution environment -- ``"local"`` or ``"auragrid"``.
        enable_savings: Enable usage tracking, pricing, and budget subsystems.
        enable_privacy: Enable the privacy auditor and event store.
        enable_sessions: Reserved for future session support.
        models_dir: Override the local model storage directory.
    """

    config_path: Optional[str] = None
    environment: str = "local"
    enable_savings: bool = True
    enable_privacy: bool = True
    enable_sessions: bool = False
    models_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Return-value dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Result of a full task execution through the intent-plan-execute loop.

    Attributes:
        output: Final generated output text.
        intent: Classified intent (e.g. ``"SIMPLE_CODE"``).
        complexity: Complexity score (1--10).
        plan: Ordered list of plan steps.
        steps_executed: Number of steps executed.
        review_verdict: ``"PASS"`` or ``"FAIL"`` from the reviewer.
        review_feedback: Human-readable reviewer feedback.
        total_elapsed: Wall-clock seconds for the full task.
    """

    output: str
    intent: str = ""
    complexity: int = 5
    plan: list[str] = field(default_factory=list)
    steps_executed: int = 0
    review_verdict: str = ""
    review_feedback: str = ""
    total_elapsed: float = 0.0


@dataclass
class ModelInfo:
    """A configured model entry.

    Attributes:
        model_id: Unique model identifier from config.
        provider: Provider name (e.g. ``"ollama"``).
        config: Full raw model configuration dict.
    """

    model_id: str
    provider: str = ""
    config: dict = field(default_factory=dict)


@dataclass
class LocalAsset:
    """A locally stored model file (GGUF).

    Attributes:
        filename: File name on disk.
        repo: HuggingFace repo ID (or ``"unknown"``).
        path: Absolute path to the file.
        size_bytes: File size in bytes.
        downloaded_at: ISO-8601 timestamp.
        gguf_metadata: Optional extracted GGUF metadata dict.
    """

    filename: str
    repo: str = ""
    path: str = ""
    size_bytes: int = 0
    downloaded_at: str = ""
    gguf_metadata: Optional[dict] = None


@dataclass
class RoleChain:
    """A routing role with its model chain.

    Attributes:
        role: Canonical role name.
        chain: Ordered list of model IDs to try.
    """

    role: str
    chain: list[str] = field(default_factory=list)


@dataclass
class TrafficSummary:
    """Aggregated traffic / usage statistics.

    Attributes:
        total_tokens: Combined input + output token count.
        input_tokens: Total input tokens.
        output_tokens: Total output tokens.
        by_model: Per-model token breakdown.
        total_spend: Dollar cost in the time range.
        spend_by_provider: Per-provider dollar spend.
        projection: Monthly cost projection dict.
    """

    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_model: list[dict] = field(default_factory=list)
    total_spend: float = 0.0
    spend_by_provider: dict[str, float] = field(default_factory=dict)
    projection: dict = field(default_factory=dict)


@dataclass
class PrivacySummary:
    """Aggregated privacy audit statistics.

    Attributes:
        total_events: Number of privacy events in the time range.
        by_severity: Event counts keyed by severity level.
        by_pattern: Event counts keyed by pattern name.
        events: Raw event dicts.
    """

    total_events: int = 0
    by_severity: dict[str, int] = field(default_factory=dict)
    by_pattern: dict[str, int] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)


@dataclass
class ROIMetrics:
    """Aggregated return-on-investment metrics.

    Attributes:
        total_simulated_cost_avoided: USD cost avoided by routing locally.
        hard_route_percentage: Percentage (0.0-100.0) of tasks routed to local models.
        avg_pipeline_latency: Average elapsed time across all tasks.
        avg_hard_routed_latency: Average elapsed time for hard-routed (local) tasks.
        recent_hard_routed: Recent local tasks with complexity and savings data.
    """
    total_simulated_cost_avoided: float = 0.0
    hard_route_percentage: float = 0.0
    avg_pipeline_latency: float = 0.0
    avg_hard_routed_latency: float = 0.0
    recent_hard_routed: list[dict] = field(default_factory=list)

@dataclass
class HealthReport:
    """Health-check result for a single model or provider.

    Attributes:
        model_id: Model identifier that was checked.
        healthy: Whether the model responded successfully.
        message: Human-readable status message.
        latency: Response latency in seconds (0 if unreachable).
    """

    model_id: str
    healthy: bool = False
    message: str = ""
    latency: float = 0.0


@dataclass
class MCPToolStatus:
    """Status of an MCP tool exposed by the server.

    Attributes:
        name: Tool name.
        enabled: Whether the tool is currently enabled.
        description: Human-readable tool description.
    """

    name: str
    enabled: bool = True
    description: str = ""


@dataclass
class StorageInfo:
    """Summary of local model storage.

    Attributes:
        models_dir: Absolute path to the models directory.
        total_files: Number of registered model files.
        total_bytes: Combined size of all registered files.
    """

    models_dir: str
    total_files: int = 0
    total_bytes: int = 0


@dataclass
class CatalogEntry:
    """A provider known to the catalog.

    Attributes:
        name: Unique provider identifier.
        provider_type: Provider type key (e.g. ``"ollama"``).
        source: Discovery source: ``"builtin"``, ``"entrypoint"``, or ``"manual"``.
        installed: Whether the provider package is available.
        running: Whether the provider is currently reachable.
        version: Provider version string.
        description: Human-readable description.
    """

    name: str
    provider_type: str = ""
    source: str = ""
    installed: bool = True
    running: bool = False
    version: str = ""
    description: str = ""


# ---------------------------------------------------------------------------
# Main API class
# ---------------------------------------------------------------------------

class AuraRouterAPI:
    """Stateful facade over all AuraRouter subsystems.

    Owns lifecycle of config, compute fabric, savings (usage tracking,
    pricing, budget), privacy, triage, model storage, and the provider
    catalog.  Every public method returns typed dataclasses.

    Usage::

        api = AuraRouterAPI(APIConfig(config_path="my.yaml"))
        result = api.execute_direct("coding", "Write a hello world in Python")
        api.close()

    Or as a context manager::

        with AuraRouterAPI() as api:
            models = api.list_models()
    """

    def __init__(self, config: APIConfig | None = None) -> None:
        """Initialize all subsystems.

        Parameters
        ----------
        config:
            Optional API configuration.  ``None`` uses all defaults.
        """
        self._cfg = config or APIConfig()
        self._closed = False

        # -- Config ----------------------------------------------------------
        from aurarouter.config import ConfigLoader

        self._config = ConfigLoader(
            config_path=self._cfg.config_path,
            allow_missing=(self._cfg.config_path is None and True is False),
        )

        # -- Fabric ----------------------------------------------------------
        from aurarouter.fabric import ComputeFabric

        self._fabric = ComputeFabric(self._config)

        # -- Model storage ---------------------------------------------------
        from aurarouter.models.file_storage import FileModelStorage

        models_dir = Path(self._cfg.models_dir) if self._cfg.models_dir else None
        self._storage = FileModelStorage(models_dir)

        # -- Savings subsystem (opt-in) --------------------------------------
        self._usage_store: Any = None
        self._pricing_catalog: Any = None
        self._cost_engine: Any = None
        self._budget_manager: Any = None
        self._triage_router: Any = None

        if self._cfg.enable_savings:
            from aurarouter.savings.usage_store import UsageStore
            from aurarouter.savings.pricing import PricingCatalog, CostEngine
            from aurarouter.savings.budget import BudgetManager
            from aurarouter.savings.triage import TriageRouter

            self._usage_store = UsageStore()

            def _config_price_resolver(model_name: str):
                mc = self._config.get_model_config(model_name)
                inp = mc.get("cost_per_1m_input")
                out = mc.get("cost_per_1m_output")
                return (inp, out)

            self._pricing_catalog = PricingCatalog(
                config_resolver=_config_price_resolver,
            )
            self._cost_engine = CostEngine(self._pricing_catalog, self._usage_store)
            budget_cfg = self._config.config.get("savings", {}).get("budget", {})
            self._budget_manager = BudgetManager(self._cost_engine, budget_cfg)

            triage_cfg = self._config.config.get("savings", {}).get("triage", {})
            self._triage_router = TriageRouter.from_config(triage_cfg)

        # -- Privacy subsystem (opt-in) --------------------------------------
        self._privacy_auditor: Any = None
        self._privacy_store: Any = None

        if self._cfg.enable_privacy:
            from aurarouter.savings.privacy import PrivacyAuditor, PrivacyStore

            self._privacy_auditor = PrivacyAuditor()
            self._privacy_store = PrivacyStore()

        # -- Lazy session/speculative/monologue (initialized on first use) --------
        self._session_manager: Any = None
        self._speculative_orchestrator: Any = None
        self._monologue_orchestrator: Any = None
        self._feedback_store: Any = None

        # -- Provider catalog ------------------------------------------------
        from aurarouter.catalog import ProviderCatalog as _ProviderCatalog

        self._catalog = _ProviderCatalog(self._config)

    # ======================================================================
    # Lifecycle
    # ======================================================================

    def close(self) -> None:
        """Release all owned resources (DB connections, etc.)."""
        if self._closed:
            return
        self._closed = True
        if self._usage_store is not None:
            self._usage_store.close()
        if self._privacy_store is not None:
            self._privacy_store.close()
        if self._feedback_store is not None and hasattr(self._feedback_store, "close"):
            self._feedback_store.close()

    def __enter__(self) -> "AuraRouterAPI":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ======================================================================
    # Task Execution
    # ======================================================================

    def execute_task(
        self,
        task: str,
        context: str = "",
        output_format: str = "text",
        intent: Optional[str] = None,
        on_intent: Optional[Callable[[str, int], None]] = None,
        on_plan: Optional[Callable[[list[str]], None]] = None,
        on_step: Optional[Callable[[int, str], None]] = None,
        on_model_tried: Optional[Callable[[str, str, bool, float], None]] = None,
        on_review: Optional[Callable[[str, str], None]] = None,
    ) -> TaskResult:
        """Execute a full task through the intent-plan-execute-review loop.

        Parameters
        ----------
        task:
            Natural-language description of the task.
        context:
            Additional context for planning.
        output_format:
            Desired output format (``"text"`` or ``"json"``).
        intent:
            Override the auto-classified intent.  When provided the
            intent classification step is skipped and this value is
            used directly.
        on_intent:
            Callback ``(intent, complexity)`` after classification.
        on_plan:
            Callback ``(steps)`` after plan generation.
        on_step:
            Callback ``(step_index, step_description)`` before each step.
        on_model_tried:
            Callback ``(role, model_id, success, elapsed)`` per model attempt.
        on_review:
            Callback ``(verdict, feedback)`` after review.

        Returns
        -------
        TaskResult with the final output and execution metadata.
        """
        from aurarouter.routing import analyze_intent, generate_plan, review_output

        t0 = time.monotonic()

        # 1. Intent classification (skip if intent override provided)
        if intent is not None:
            from dataclasses import dataclass as _dc

            @_dc
            class _ForcedTriage:
                intent: str
                complexity: int

            triage = _ForcedTriage(intent=intent, complexity=0)
        else:
            triage = analyze_intent(self._fabric, task)
        if on_intent:
            on_intent(triage.intent, triage.complexity)

        if triage.intent == "DIRECT":
            # Fast path: single direct execution
            result = self._fabric.execute(
                "coding", task, json_mode=output_format == "json",
                on_model_tried=on_model_tried,
            )
            combined = result.text if result else ""
            elapsed = time.monotonic() - t0
            return TaskResult(
                output=combined,
                intent=triage.intent,
                complexity=triage.complexity,
                plan=[],
                steps_executed=1,
                review_verdict="PASS",
                review_feedback="Direct execution (no review)",
                total_elapsed=elapsed,
            )

        # 2. Plan generation
        plan = generate_plan(self._fabric, task, context)
        if on_plan:
            on_plan(plan)

        # 3. Execute steps
        json_mode = output_format == "json"
        outputs: list[str] = []
        for i, step in enumerate(plan):
            if on_step:
                on_step(i, step)
            result = self._fabric.execute(
                "coding", step, json_mode=json_mode,
                on_model_tried=on_model_tried,
            )
            outputs.append(result or "")

        combined = "\n\n".join(o for o in outputs if o)

        # 4. Review
        review = review_output(self._fabric, task, combined)
        if on_review:
            on_review(review.verdict, review.feedback)

        elapsed = time.monotonic() - t0

        return TaskResult(
            output=combined,
            intent=triage.intent,
            complexity=triage.complexity,
            plan=plan,
            steps_executed=len(plan),
            review_verdict=review.verdict,
            review_feedback=review.feedback,
            total_elapsed=elapsed,
        )

    def execute_direct(
        self,
        role: str,
        prompt: str,
        json_mode: bool = False,
        local_only: bool = False,
    ) -> "GenerateResult":
        """Execute a single generation call on a specific role.

        Parameters
        ----------
        role:
            Routing role to use (e.g. ``"coding"``, ``"reasoning"``).
        prompt:
            The prompt text.
        json_mode:
            Request JSON-formatted output from the model.
        local_only:
            If ``True``, skip cloud models in the chain.

        Returns
        -------
        A :class:`~aurarouter.savings.models.GenerateResult`.
        """
        from aurarouter.savings.models import GenerateResult

        raw = self._fabric.execute(role, prompt, json_mode=json_mode)
        if raw is None:
            return GenerateResult(text="", model_id="", provider="")
        if isinstance(raw, str):
            return GenerateResult(text=raw, model_id="", provider="")
        # If fabric returns a GenerateResult natively
        return raw  # type: ignore[return-value]

    async def execute_direct_stream(
        self,
        role: str,
        prompt: str,
        json_mode: bool = False,
        local_only: bool = False,
    ) -> "AsyncIterator[str]":
        """Streaming variant of :meth:`execute_direct`.

        Parameters
        ----------
        role:
            Routing role to use (e.g. ``"coding"``, ``"reasoning"``).
        prompt:
            The prompt text.
        json_mode:
            Request JSON-formatted output from the model.
        local_only:
            Reserved for future use.

        Yields
        ------
        str
            Tokens as they arrive from the provider.
        """
        async for token in self._fabric.execute_stream(
            role, prompt, json_mode=json_mode
        ):
            yield token

    def compare_models(
        self,
        prompt: str,
        model_ids: list[str],
    ) -> list["GenerateResult"]:
        """Run the same prompt through multiple models and collect results.

        Parameters
        ----------
        prompt:
            The prompt to send to each model.
        model_ids:
            List of model identifiers to compare.

        Returns
        -------
        One :class:`~aurarouter.savings.models.GenerateResult` per model,
        in the same order as *model_ids*.
        """
        from aurarouter.savings.models import GenerateResult
        from aurarouter.providers import get_provider

        results: list[GenerateResult] = []
        for mid in model_ids:
            cfg = self._config.get_model_config(mid)
            if not cfg:
                results.append(GenerateResult(
                    text="", model_id=mid, provider="",
                ))
                continue
            provider_name = cfg.get("provider", "")
            try:
                provider = get_provider(provider_name, cfg)
                raw = provider.generate(prompt)
                if isinstance(raw, str):
                    results.append(GenerateResult(
                        text=raw, model_id=mid, provider=provider_name,
                    ))
                elif raw is not None:
                    results.append(raw)  # type: ignore[arg-type]
                else:
                    results.append(GenerateResult(
                        text="", model_id=mid, provider=provider_name,
                    ))
            except Exception as exc:
                results.append(GenerateResult(
                    text=f"ERROR: {exc}", model_id=mid, provider=provider_name,
                ))
        return results

    # ======================================================================
    # Model Management
    # ======================================================================

    def list_models(self) -> list[ModelInfo]:
        """Return all configured models.

        Returns
        -------
        List of :class:`ModelInfo` for every model in the config.
        """
        result: list[ModelInfo] = []
        for mid in self._config.get_all_model_ids():
            cfg = self._config.get_model_config(mid)
            result.append(ModelInfo(
                model_id=mid,
                provider=cfg.get("provider", ""),
                config=cfg,
            ))
        return result

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Look up a single model by ID.

        Returns
        -------
        :class:`ModelInfo` or ``None`` if not found.
        """
        cfg = self._config.get_model_config(model_id)
        if not cfg:
            return None
        return ModelInfo(
            model_id=model_id,
            provider=cfg.get("provider", ""),
            config=cfg,
        )

    def add_model(self, model_id: str, model_config: dict) -> ModelInfo:
        """Add a new model to the configuration.

        Parameters
        ----------
        model_id:
            Unique model identifier.
        model_config:
            Full model configuration dict (must include ``provider``).

        Returns
        -------
        The newly created :class:`ModelInfo`.
        """
        self._config.set_model(model_id, model_config)
        return ModelInfo(
            model_id=model_id,
            provider=model_config.get("provider", ""),
            config=model_config,
        )

    def update_model(self, model_id: str, model_config: dict) -> ModelInfo:
        """Update an existing model's configuration.

        Parameters
        ----------
        model_id:
            Model identifier to update.
        model_config:
            New model configuration dict.

        Returns
        -------
        The updated :class:`ModelInfo`.
        """
        self._config.set_model(model_id, model_config)
        return ModelInfo(
            model_id=model_id,
            provider=model_config.get("provider", ""),
            config=model_config,
        )

    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the configuration.

        Returns
        -------
        ``True`` if the model existed and was removed.
        """
        return self._config.remove_model(model_id)

    def test_model_connection(self, model_id: str) -> tuple[bool, str]:
        """Test connectivity to a model's provider.

        Sends a minimal probe prompt and checks for a valid response.

        Returns
        -------
        ``(success, message)`` tuple.
        """
        from aurarouter.providers import get_provider

        cfg = self._config.get_model_config(model_id)
        if not cfg:
            return (False, f"Model '{model_id}' not found in config")

        provider_name = cfg.get("provider", "")
        try:
            provider = get_provider(provider_name, cfg)
            t0 = time.monotonic()
            result = provider.generate("Say OK.", json_mode=False)
            elapsed = time.monotonic() - t0
            if result and str(result).strip():
                return (True, f"OK ({elapsed:.2f}s)")
            return (False, "Empty response")
        except Exception as exc:
            return (False, str(exc))

    def auto_tune_model(self, model_id: str) -> Optional[dict]:
        """Run auto-tuning on a model and update its config.

        Extracts GGUF metadata and recommends optimal inference parameters
        for ``llamacpp`` models.

        Returns
        -------
        The recommended parameters dict, or ``None`` if auto-tuning
        is not applicable.
        """
        from aurarouter.tuning import auto_tune_model as _auto_tune

        cfg = self._config.get_model_config(model_id)
        if not cfg:
            return None

        provider = cfg.get("provider", "")
        tuned = _auto_tune(provider, cfg)
        if tuned is not cfg:
            self._config.set_model(model_id, tuned)
            return tuned.get("parameters")
        return None

    def list_local_assets(self) -> list[LocalAsset]:
        """Return all locally stored model files.

        Returns
        -------
        List of :class:`LocalAsset` entries from the file storage registry.
        """
        entries = self._storage.list_models()
        return [
            LocalAsset(
                filename=e.get("filename", ""),
                repo=e.get("repo", ""),
                path=e.get("path", ""),
                size_bytes=e.get("size_bytes", 0),
                downloaded_at=e.get("downloaded_at", ""),
                gguf_metadata=e.get("gguf_metadata"),
            )
            for e in entries
        ]

    def download_asset(
        self,
        repo: str,
        filename: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> LocalAsset:
        """Download a model from HuggingFace Hub.

        Parameters
        ----------
        repo:
            HuggingFace repository ID.
        filename:
            GGUF file name inside the repository.
        progress_callback:
            Optional ``(downloaded_bytes, total_bytes)`` callback.

        Returns
        -------
        :class:`LocalAsset` for the downloaded file.
        """
        from aurarouter.models.downloader import download_model

        dest = str(self._storage.models_dir) if self._cfg.models_dir else None
        path = download_model(
            repo, filename, dest=dest, progress_callback=progress_callback,
        )
        # Re-read from registry to get full metadata
        for e in self._storage.list_models():
            if e.get("filename") == filename:
                return LocalAsset(
                    filename=e["filename"],
                    repo=e.get("repo", repo),
                    path=e.get("path", str(path)),
                    size_bytes=e.get("size_bytes", 0),
                    downloaded_at=e.get("downloaded_at", ""),
                    gguf_metadata=e.get("gguf_metadata"),
                )
        return LocalAsset(filename=filename, repo=repo, path=str(path))

    def import_asset(self, file_path: str, repo: str = "local") -> LocalAsset:
        """Register an existing local model file in the storage registry.

        Parameters
        ----------
        file_path:
            Absolute path to the ``.gguf`` file.
        repo:
            Repository identifier to record (default ``"local"``).

        Returns
        -------
        :class:`LocalAsset` for the registered file.
        """
        p = Path(file_path)
        self._storage.register(repo=repo, filename=p.name, path=p)
        return LocalAsset(
            filename=p.name,
            repo=repo,
            path=str(p),
            size_bytes=p.stat().st_size if p.is_file() else 0,
        )

    def remove_asset(self, filename: str, delete_file: bool = False) -> bool:
        """Remove a model file from the storage registry.

        Parameters
        ----------
        filename:
            File name to remove.
        delete_file:
            Also delete the file from disk.

        Returns
        -------
        ``True`` if the asset was found and removed.
        """
        return self._storage.remove(filename, delete_file=delete_file)

    def get_storage_info(self) -> StorageInfo:
        """Return summary information about local model storage.

        Returns
        -------
        :class:`StorageInfo` with directory path, file count, and total size.
        """
        entries = self._storage.list_models()
        total_bytes = sum(e.get("size_bytes", 0) for e in entries)
        return StorageInfo(
            models_dir=str(self._storage.models_dir),
            total_files=len(entries),
            total_bytes=total_bytes,
        )

    def list_grid_models(self) -> list[ModelInfo]:
        """List models available on the AuraGrid network.

        Returns
        -------
        List of :class:`ModelInfo` for grid-hosted models, or empty
        list if AuraGrid integration is not available.
        """
        # Filter configured models that have grid/remote tags
        result: list[ModelInfo] = []
        for mid in self._config.get_all_model_ids():
            cfg = self._config.get_model_config(mid)
            tags = cfg.get("tags", [])
            if "remote" in tags or "grid" in tags:
                result.append(ModelInfo(
                    model_id=mid,
                    provider=cfg.get("provider", ""),
                    config=cfg,
                ))
        return result

    # ======================================================================
    # Routing
    # ======================================================================

    def list_roles(self) -> list[RoleChain]:
        """Return all configured routing roles with their model chains.

        Returns
        -------
        List of :class:`RoleChain` entries.
        """
        return [
            RoleChain(role=r, chain=self._config.get_role_chain(r))
            for r in self._config.get_all_roles()
        ]

    def get_role_chain(self, role: str) -> Optional[RoleChain]:
        """Look up the model chain for a single role.

        Returns
        -------
        :class:`RoleChain` or ``None`` if the role is not configured.
        """
        chain = self._config.get_role_chain(role)
        if not chain:
            return None
        return RoleChain(role=role, chain=chain)

    def set_role_chain(self, role: str, chain: list[str]) -> RoleChain:
        """Set or replace the model chain for a role.

        Parameters
        ----------
        role:
            Canonical role name.
        chain:
            Ordered list of model IDs.

        Returns
        -------
        The updated :class:`RoleChain`.
        """
        self._config.set_role_chain(role, chain)
        return RoleChain(role=role, chain=chain)

    def remove_role(self, role: str) -> bool:
        """Remove a routing role.

        Returns
        -------
        ``True`` if the role existed and was removed.
        """
        return self._config.remove_role(role)

    def get_missing_required_roles(self) -> list[str]:
        """Return required role names that are not configured.

        Uses the semantic verb registry to determine which roles are
        required for the routing loop to function.

        Returns
        -------
        List of missing role names.
        """
        from aurarouter.semantic_verbs import get_required_roles

        configured = set(self._config.get_all_roles())
        return [r for r in get_required_roles() if r not in configured]

    def resolve_role_synonym(self, verb: str) -> str:
        """Resolve a verb or synonym to its canonical role name.

        Parameters
        ----------
        verb:
            A role name or synonym (e.g. ``"programming"``).

        Returns
        -------
        The canonical role name (e.g. ``"coding"``).
        """
        from aurarouter.semantic_verbs import resolve_synonym

        return resolve_synonym(verb)

    def get_triage_rules(self) -> list[dict]:
        """Return the current triage routing rules.

        Returns
        -------
        List of rule dicts with ``max_complexity``, ``preferred_role``,
        and ``description`` keys.  Empty if savings is disabled.
        """
        if self._triage_router is None:
            return []
        return [
            {
                "max_complexity": r.max_complexity,
                "preferred_role": r.preferred_role,
                "description": r.description,
            }
            for r in self._triage_router.rules
        ]

    # ======================================================================
    # Monitoring
    # ======================================================================

    def get_traffic(
        self,
        time_range: Optional[tuple[str, str]] = None,
    ) -> TrafficSummary:
        """Return aggregated traffic and cost statistics.

        Parameters
        ----------
        time_range:
            Optional ``(start_iso, end_iso)`` tuple to filter by date range.

        Returns
        -------
        :class:`TrafficSummary` with token counts, spend, and projections.
        """
        if self._usage_store is None or self._cost_engine is None:
            return TrafficSummary()

        start = time_range[0] if time_range else None
        end = time_range[1] if time_range else None

        totals = self._usage_store.total_tokens(start=start, end=end)
        by_model = self._usage_store.aggregate_tokens(
            start=start, end=end, group_by="model_id",
        )
        spend = self._cost_engine.total_spend(start=start, end=end)
        spend_by = self._cost_engine.spend_by_provider(start=start, end=end)
        projection = self._cost_engine.monthly_projection()

        return TrafficSummary(
            total_tokens=totals["total_tokens"],
            input_tokens=totals["input_tokens"],
            output_tokens=totals["output_tokens"],
            by_model=by_model,
            total_spend=spend,
            spend_by_provider=spend_by,
            projection=projection,
        )

    def get_roi_metrics(self, timeframe_days: int) -> ROIMetrics:
        """Calculate and return return-on-investment metrics for the given timeframe.

        Returns zeroed metrics if savings are disabled or no data exists.
        """
        if self._usage_store is None:
            return ROIMetrics()

        from datetime import datetime, timedelta, timezone

        start_time = (datetime.now(timezone.utc) - timedelta(days=timeframe_days)).isoformat()
        records = self._usage_store.query(start=start_time)

        if not records:
            return ROIMetrics()

        total_cost_avoided = sum(r.simulated_cost_avoided for r in records)
        
        local_records = [r for r in records if not r.is_cloud]
        hard_route_percentage = (len(local_records) / len(records)) * 100.0 if records else 0.0
        
        avg_pipeline_latency = sum(r.elapsed_s for r in records) / len(records) if records else 0.0
        avg_hard_routed_latency = (
            sum(r.elapsed_s for r in local_records) / len(local_records)
            if local_records else 0.0
        )

        # Recent hard-routed tasks (top 50)
        recent_hard_routed = [
            {
                "timestamp": r.timestamp,
                "model_id": r.model_id,
                "complexity": r.complexity_score,
                "savings": r.simulated_cost_avoided,
                "latency": r.elapsed_s,
            }
            for r in reversed(local_records[-50:]) # Top 50 most recent local tasks
        ]

        return ROIMetrics(
            total_simulated_cost_avoided=total_cost_avoided,
            hard_route_percentage=hard_route_percentage,
            avg_pipeline_latency=avg_pipeline_latency,
            avg_hard_routed_latency=avg_hard_routed_latency,
            recent_hard_routed=recent_hard_routed,
        )

    def get_privacy_events(
        self,
        time_range: Optional[tuple[str, str]] = None,
        severity: Optional[str] = None,
    ) -> PrivacySummary:
        """Return aggregated privacy audit events.

        Parameters
        ----------
        time_range:
            Optional ``(start_iso, end_iso)`` tuple.
        severity:
            Minimum severity filter (``"low"``, ``"medium"``, ``"high"``).

        Returns
        -------
        :class:`PrivacySummary` with event counts and breakdowns.
        """
        if self._privacy_store is None:
            return PrivacySummary()

        start = time_range[0] if time_range else None
        end = time_range[1] if time_range else None

        events = self._privacy_store.query(
            start=start, end=end, min_severity=severity,
        )
        summary = self._privacy_store.summary(start=start, end=end)

        return PrivacySummary(
            total_events=summary["total_events"],
            by_severity=summary["by_severity"],
            by_pattern=summary["by_pattern"],
            events=events,
        )

    def check_health(self, model_id: Optional[str] = None) -> list[HealthReport]:
        """Probe model health via provider connectivity.

        Parameters
        ----------
        model_id:
            Specific model to check.  ``None`` checks all configured models.

        Returns
        -------
        List of :class:`HealthReport` entries.
        """
        from aurarouter.providers import get_provider

        ids = [model_id] if model_id else self._config.get_all_model_ids()
        reports: list[HealthReport] = []

        for mid in ids:
            cfg = self._config.get_model_config(mid)
            if not cfg:
                reports.append(HealthReport(
                    model_id=mid, healthy=False, message="Not configured",
                ))
                continue

            provider_name = cfg.get("provider", "")
            try:
                provider = get_provider(provider_name, cfg)
                t0 = time.monotonic()
                result = provider.generate("ping", json_mode=False)
                latency = time.monotonic() - t0
                ok = result is not None and str(result).strip() != ""
                reports.append(HealthReport(
                    model_id=mid,
                    healthy=ok,
                    message="OK" if ok else "Empty response",
                    latency=latency,
                ))
            except Exception as exc:
                reports.append(HealthReport(
                    model_id=mid, healthy=False, message=str(exc),
                ))

        return reports

    def get_budget_status(self) -> Optional[dict]:
        """Return current budget enforcement status.

        Returns
        -------
        Dict with ``allowed``, ``daily_spend``, ``monthly_spend``,
        ``daily_limit``, ``monthly_limit`` keys, or ``None`` if budget
        tracking is disabled.
        """
        if self._budget_manager is None:
            return None

        status = self._budget_manager.check_budget("cloud")
        return {
            "allowed": status.allowed,
            "reason": status.reason,
            "daily_spend": status.daily_spend,
            "monthly_spend": status.monthly_spend,
            "daily_limit": status.daily_limit,
            "monthly_limit": status.monthly_limit,
        }

    # ======================================================================
    # Configuration
    # ======================================================================

    def get_config_yaml(self) -> str:
        """Return the current configuration as a YAML string.

        Returns
        -------
        YAML-formatted string of the active configuration.
        """
        return self._config.to_yaml()

    def save_config(self, path: Optional[str] = None) -> str:
        """Persist the current configuration to disk.

        Parameters
        ----------
        path:
            Target file path.  ``None`` saves to the original location.

        Returns
        -------
        The path the config was written to.
        """
        target = Path(path) if path else None
        result = self._config.save(target)
        return str(result)

    def reload_config(self) -> None:
        """Reload configuration from disk and update all subsystems."""
        config_path = str(self._config.config_path) if self._config.config_path else None
        from aurarouter.config import ConfigLoader

        self._config = ConfigLoader(config_path=config_path)
        self._fabric.update_config(self._config)

        if self._budget_manager is not None:
            budget_cfg = self._config.config.get("savings", {}).get("budget", {})
            self._budget_manager.update_config(budget_cfg)

        if self._triage_router is not None:
            from aurarouter.savings.triage import TriageRouter

            triage_cfg = self._config.config.get("savings", {}).get("triage", {})
            self._triage_router = TriageRouter.from_config(triage_cfg)

    def get_mcp_tools(self) -> list[MCPToolStatus]:
        """Return the status of all MCP tools exposed by the server.

        Returns
        -------
        List of :class:`MCPToolStatus` entries.
        """
        tools_cfg = self._config.config.get("mcp_tools", {})
        return [
            MCPToolStatus(
                name=name,
                enabled=bool(info) if isinstance(info, bool) else info.get("enabled", True),
                description=info.get("description", "") if isinstance(info, dict) else "",
            )
            for name, info in tools_cfg.items()
        ]

    def set_mcp_tool(self, name: str, enabled: bool) -> MCPToolStatus:
        """Enable or disable an MCP tool.

        Parameters
        ----------
        name:
            Tool name.
        enabled:
            Whether the tool should be enabled.

        Returns
        -------
        Updated :class:`MCPToolStatus`.
        """
        tools_cfg = self._config.config.setdefault("mcp_tools", {})
        if name in tools_cfg and isinstance(tools_cfg[name], dict):
            tools_cfg[name]["enabled"] = enabled
        else:
            tools_cfg[name] = {"enabled": enabled}
        desc = tools_cfg[name].get("description", "") if isinstance(tools_cfg[name], dict) else ""
        return MCPToolStatus(name=name, enabled=enabled, description=desc)

    def get_system_settings(self) -> dict:
        """Return global system settings from the configuration.

        Returns
        -------
        Dict of system-level settings (environment, logging, etc.).
        """
        return {
            "environment": self._cfg.environment,
            "enable_savings": self._cfg.enable_savings,
            "enable_privacy": self._cfg.enable_privacy,
            "enable_sessions": self._cfg.enable_sessions,
            "config_path": str(self._config.config_path) if self._config.config_path else None,
            "models_dir": str(self._storage.models_dir),
        }

    def set_system_settings(self, settings: dict) -> dict:
        """Update system-level settings in memory.

        Parameters
        ----------
        settings:
            Dict of settings to merge.  Only recognized keys are applied.

        Returns
        -------
        Updated settings dict.
        """
        # Apply to underlying config where applicable
        for key in ("logging", "server", "discovery"):
            if key in settings:
                self._config.config[key] = settings[key]
        return self.get_system_settings()

    def get_environment(self) -> str:
        """Return the current execution environment.

        Returns
        -------
        ``"local"`` or ``"auragrid"``.
        """
        return self._cfg.environment

    def config_affects_other_nodes(self) -> bool:
        """Check whether saving the config would affect other AuraGrid nodes.

        Returns
        -------
        ``True`` if running in ``"auragrid"`` environment.
        """
        return self._cfg.environment == "auragrid"

    # ======================================================================
    # Provider Catalog
    # ======================================================================

    def list_catalog(self) -> list[CatalogEntry]:
        """Discover and return all known providers.

        Returns
        -------
        List of :class:`CatalogEntry` from all sources.
        """
        raw = self._catalog.discover()
        return [
            CatalogEntry(
                name=e.name,
                provider_type=e.provider_type,
                source=e.source,
                installed=e.installed,
                running=e.running,
                version=e.version,
                description=e.description,
            )
            for e in raw
        ]

    def add_catalog_provider(self, name: str, endpoint: str) -> CatalogEntry:
        """Register a manual provider endpoint in the catalog.

        Parameters
        ----------
        name:
            Unique provider name.
        endpoint:
            MCP endpoint URL.

        Returns
        -------
        The new :class:`CatalogEntry`.
        """
        raw = self._catalog.register_manual(name, endpoint)
        return CatalogEntry(
            name=raw.name,
            provider_type=raw.provider_type,
            source=raw.source,
            installed=raw.installed,
            running=raw.running,
            version=raw.version,
            description=raw.description,
        )

    def remove_catalog_provider(self, name: str) -> bool:
        """Remove a manual provider from the catalog.

        Returns
        -------
        ``True`` if the provider was found and removed.
        """
        return self._catalog.unregister_manual(name)

    def start_catalog_provider(self, name: str) -> bool:
        """Start a provider MCP server subprocess.

        Returns
        -------
        ``True`` if the process was started successfully.
        """
        return self._catalog.start_provider(name)

    def stop_catalog_provider(self, name: str) -> bool:
        """Stop a running provider subprocess.

        Returns
        -------
        ``True`` if the process was terminated.
        """
        return self._catalog.stop_provider(name)

    def check_catalog_provider(self, name: str) -> tuple[bool, str]:
        """Probe a catalog provider's health.

        Returns
        -------
        ``(healthy, message)`` tuple.
        """
        return self._catalog.check_provider_health(name)

    def auto_register_catalog_models(self, name: str) -> int:
        """Discover models from a catalog provider and register them.

        Returns
        -------
        Number of models added to the config.
        """
        return self._catalog.auto_register_models(name, self._config)

    # ======================================================================
    # Unified Artifact Catalog
    # ======================================================================

    def catalog_list(self, kind: str | None = None) -> list[dict]:
        """List all catalog artifacts, optionally filtered by kind (model/service/analyzer).

        Parameters
        ----------
        kind:
            Filter to a specific artifact kind.  ``None`` returns all kinds.

        Returns
        -------
        List of dicts, each with at least ``artifact_id`` and ``kind`` keys.
        """
        try:
            if hasattr(self._config, "catalog_list"):
                ids = self._config.catalog_list(kind=kind)
                results: list[dict] = []
                for aid in ids:
                    entry = self._config.catalog_get(aid)
                    if entry is not None:
                        entry["artifact_id"] = aid
                        results.append(entry)
                    else:
                        results.append({"artifact_id": aid, "kind": kind or "model"})
                return results
        except Exception:
            pass
        return []

    def catalog_get(self, artifact_id: str) -> dict | None:
        """Get a single catalog artifact by ID.

        Parameters
        ----------
        artifact_id:
            Unique artifact identifier.

        Returns
        -------
        Artifact dict with metadata, or ``None`` if not found.
        """
        try:
            if hasattr(self._config, "catalog_get"):
                return self._config.catalog_get(artifact_id)
        except Exception:
            pass
        return None

    def catalog_set(self, artifact_id: str, data: dict) -> None:
        """Register or update a catalog artifact.

        Parameters
        ----------
        artifact_id:
            Unique artifact identifier.
        data:
            Artifact data dict (must include ``kind`` and ``display_name``).
        """
        try:
            if hasattr(self._config, "catalog_set"):
                self._config.catalog_set(artifact_id, data)
        except Exception:
            pass

    def catalog_remove(self, artifact_id: str) -> bool:
        """Remove a catalog artifact. Returns True if it existed.

        Parameters
        ----------
        artifact_id:
            Unique artifact identifier to remove.

        Returns
        -------
        ``True`` if the artifact was found and removed.
        """
        try:
            if hasattr(self._config, "catalog_remove"):
                return self._config.catalog_remove(artifact_id)
        except Exception:
            pass
        return False

    def catalog_query(
        self,
        kind: str | None = None,
        tags: list[str] | None = None,
        capabilities: list[str] | None = None,
        provider: str | None = None,
    ) -> list[dict]:
        """Query catalog with filters.

        Parameters
        ----------
        kind:
            Filter by artifact kind (``"model"``, ``"service"``, ``"analyzer"``).
        tags:
            All specified tags must be present on the artifact.
        capabilities:
            All specified capabilities must be present.
        provider:
            Exact provider match.

        Returns
        -------
        List of matching artifact dicts, each enriched with ``artifact_id``.
        """
        try:
            if hasattr(self._config, "catalog_query"):
                return self._config.catalog_query(
                    kind=kind, tags=tags,
                    capabilities=capabilities, provider=provider,
                )
        except Exception:
            pass
        return []

    # ======================================================================
    # Route Analyzers
    # ======================================================================

    def get_active_analyzer(self) -> str | None:
        """Get the currently active route analyzer ID.

        Returns
        -------
        Artifact ID of the active analyzer, or ``None`` if using the
        built-in default.
        """
        try:
            if hasattr(self._config, "get_active_analyzer"):
                return self._config.get_active_analyzer()
        except Exception:
            pass
        return None

    def set_active_analyzer(self, analyzer_id: str | None) -> None:
        """Set or clear the active route analyzer.

        Parameters
        ----------
        analyzer_id:
            Catalog artifact ID of the analyzer to activate, or ``None``
            to clear (revert to built-in default).
        """
        try:
            if hasattr(self._config, "set_active_analyzer"):
                self._config.set_active_analyzer(analyzer_id)
        except Exception:
            pass

    # ======================================================================
    # T1.1 Session Management
    # ======================================================================

    def list_sessions(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """List sessions with metadata. Returns [] if sessions disabled."""
        if not self._cfg.enable_sessions:
            return []
        try:
            mgr = self._get_session_manager()
            sessions = mgr.list_sessions(limit=limit, offset=offset)
            result = []
            for s in sessions:
                if isinstance(s, dict):
                    result.append(s)
                else:
                    result.append({
                        "session_id": getattr(s, "session_id", str(s)),
                        "created_at": getattr(s, "created_at", ""),
                        "updated_at": getattr(s, "updated_at", ""),
                        "message_count": getattr(s, "message_count", 0),
                    })
            return result
        except Exception:
            return []

    def create_session(self, context_limit: int = 0) -> dict:
        """Create a new session. Returns error dict if sessions disabled."""
        if not self._cfg.enable_sessions:
            return {"error": "sessions_disabled"}
        try:
            mgr = self._get_session_manager()
            session = mgr.create_session(context_limit=context_limit)
            if isinstance(session, dict):
                return session
            return {
                "session_id": getattr(session, "session_id", ""),
                "created_at": getattr(session, "created_at", ""),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def get_session(self, session_id: str) -> dict | None:
        """Get session with history and stats. None if not found."""
        if not self._cfg.enable_sessions:
            return {"error": "sessions_disabled"}
        try:
            mgr = self._get_session_manager()
            session = mgr.get_session(session_id)
            if session is None:
                return None
            if isinstance(session, dict):
                return session
            messages = []
            if hasattr(session, "messages"):
                for m in session.messages:
                    if isinstance(m, dict):
                        messages.append(m)
                    else:
                        messages.append({
                            "role": getattr(m, "role", ""),
                            "content": getattr(m, "content", ""),
                            "model_id": getattr(m, "model_id", ""),
                            "timestamp": getattr(m, "timestamp", ""),
                        })
            return {
                "session_id": getattr(session, "session_id", session_id),
                "created_at": getattr(session, "created_at", ""),
                "updated_at": getattr(session, "updated_at", ""),
                "messages": messages,
                "message_count": len(messages),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def add_session_message(self, session_id: str, role: str, content: str,
                            model_id: str = "", tokens: int = 0) -> dict:
        """Append message to session."""
        if not self._cfg.enable_sessions:
            return {"error": "sessions_disabled"}
        try:
            mgr = self._get_session_manager()
            if hasattr(mgr, "add_message"):
                mgr.add_message(session_id=session_id, role=role, content=content)
            return self.get_session(session_id) or {"error": "session_not_found"}
        except Exception as exc:
            return {"error": str(exc)}

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted."""
        if not self._cfg.enable_sessions:
            return False
        try:
            mgr = self._get_session_manager()
            return mgr.delete_session(session_id)
        except Exception:
            return False

    def execute_in_session(self, session_id: str, task: str, context: str = "",
                           callbacks: dict | None = None) -> dict:
        """Execute task within session context."""
        if not self._cfg.enable_sessions:
            return {"error": "sessions_disabled"}
        try:
            result = self.execute_task(task=task, context=context)
            self.add_session_message(session_id, "user", task)
            self.add_session_message(session_id, "assistant", result.output,
                                     tokens=result.steps_executed)
            return {
                "result": result.output,
                "session_id": session_id,
                "intent": result.intent,
                "complexity": result.complexity,
                "review_verdict": result.review_verdict,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _get_session_manager(self) -> Any:
        """Lazy-initialize SessionManager."""
        if self._session_manager is None:
            from aurarouter.sessions.manager import SessionManager
            from aurarouter.sessions.store import SessionStore
            self._session_manager = SessionManager(store=SessionStore())
        return self._session_manager

    # ======================================================================
    # T1.2 Speculative Decoding API
    # ======================================================================

    def get_speculative_config(self) -> dict:
        """Return speculative decoding configuration."""
        try:
            cfg = self._config.config.get("speculative", {})
            return {
                "enabled": cfg.get("enabled", False),
                "complexity_threshold": cfg.get("complexity_threshold", 7),
                "notional_confidence_threshold": cfg.get("confidence_threshold", 0.85),
                "timeout": cfg.get("timeout", 60),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def get_speculative_sessions(self) -> list[dict]:
        """Return list of active speculative sessions."""
        try:
            orch = self._get_speculative_orchestrator()
            if orch is None:
                return []
            sessions = orch.get_active_sessions()
            result = []
            for s in sessions:
                if isinstance(s, dict):
                    result.append(s)
                else:
                    result.append({
                        "session_id": getattr(s, "session_id", ""),
                        "drafter_model": getattr(s, "drafter_model", ""),
                        "verifier_model": getattr(s, "verifier_model", ""),
                        "acceptance_rate": getattr(s, "acceptance_rate", 0.0),
                        "status": getattr(s, "status", "unknown"),
                    })
            return result
        except Exception:
            return []

    def get_speculative_session(self, session_id: str) -> dict | None:
        """Return single speculative session detail."""
        try:
            orch = self._get_speculative_orchestrator()
            if orch is None:
                return None
            s = orch.get_session(session_id)
            if s is None:
                return None
            if isinstance(s, dict):
                return s
            return {
                "session_id": getattr(s, "session_id", session_id),
                "drafter_model": getattr(s, "drafter_model", ""),
                "verifier_model": getattr(s, "verifier_model", ""),
                "acceptance_rate": getattr(s, "acceptance_rate", 0.0),
                "status": getattr(s, "status", "unknown"),
                "input_tokens": getattr(s, "input_tokens", 0),
                "output_tokens": getattr(s, "output_tokens", 0),
            }
        except Exception:
            return None

    def _get_speculative_orchestrator(self) -> Any:
        """Lazy-initialize SpeculativeOrchestrator."""
        if self._speculative_orchestrator is None:
            try:
                from aurarouter.speculative import SpeculativeOrchestrator
                self._speculative_orchestrator = SpeculativeOrchestrator(self._config)
            except Exception:
                return None
        return self._speculative_orchestrator

    # ======================================================================
    # T1.3 Monologue Reasoning API
    # ======================================================================

    def get_monologue_config(self) -> dict:
        """Return monologue reasoning configuration."""
        try:
            cfg = self._config.config.get("monologue", {})
            return {
                "enabled": cfg.get("enabled", False),
                "max_iterations_default": cfg.get("max_iterations", 5),
                "convergence_threshold_default": cfg.get("convergence_threshold", 0.85),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def get_monologue_sessions(self) -> list[dict]:
        """Return list of monologue sessions."""
        try:
            orch = self._get_monologue_orchestrator()
            if orch is None:
                return []
            sessions = orch.get_active_sessions()
            result = []
            for s in sessions:
                if isinstance(s, dict):
                    result.append(s)
                else:
                    result.append({
                        "session_id": getattr(s, "session_id", ""),
                        "convergence_reason": getattr(s, "convergence_reason", ""),
                        "iteration_count": getattr(s, "iteration_count", 0),
                        "status": getattr(s, "status", "unknown"),
                    })
            return result
        except Exception:
            return []

    def get_monologue_trace(self, session_id: str) -> dict | None:
        """Return full reasoning trace for a monologue session."""
        try:
            orch = self._get_monologue_orchestrator()
            if orch is None:
                return None
            session = orch.get_session(session_id)
            if session is None:
                return None
            if isinstance(session, dict):
                return session
            steps = []
            raw_steps = getattr(session, "steps", []) or getattr(session, "reasoning_steps", [])
            for step in raw_steps:
                if isinstance(step, dict):
                    steps.append(step)
                else:
                    steps.append({
                        "role": getattr(step, "role", ""),
                        "model_id": getattr(step, "model_id", ""),
                        "output_preview": str(getattr(step, "output", ""))[:200],
                        "mas_relevancy_score": getattr(step, "mas_relevancy_score", 0.0),
                        "confidence": getattr(step, "confidence", 0.0),
                        "iteration": getattr(step, "iteration", 0),
                    })
            return {
                "session_id": session_id,
                "steps": steps,
                "convergence_reason": getattr(session, "convergence_reason", ""),
                "iteration_count": len([s for s in steps if s.get("role") == "generator"]),
            }
        except Exception:
            return None

    def _get_monologue_orchestrator(self) -> Any:
        """Lazy-initialize MonologueOrchestrator."""
        if self._monologue_orchestrator is None:
            try:
                from aurarouter.monologue import MonologueOrchestrator
                self._monologue_orchestrator = MonologueOrchestrator(self._config, self._fabric)
            except Exception:
                return None
        return self._monologue_orchestrator

    # ======================================================================
    # T1.4 Sovereignty & Intent Inspection API
    # ======================================================================

    def evaluate_sovereignty(self, prompt: str) -> dict:
        """Dry-run sovereignty evaluation."""
        try:
            from aurarouter.sovereignty import SovereigntyGate
            gate = SovereigntyGate(self._config)
            result = gate.evaluate(prompt)
            if isinstance(result, dict):
                return result
            return {
                "verdict": str(getattr(result, "verdict", "OPEN")),
                "reason": getattr(result, "reason", ""),
                "matched_patterns": getattr(result, "matched_patterns", []),
            }
        except Exception as exc:
            return {"error": str(exc), "verdict": "OPEN", "matched_patterns": []}

    def get_sovereignty_config(self) -> dict:
        """Return sovereignty enforcement configuration."""
        try:
            cfg = self._config.config.get("sovereignty", {})
            custom_patterns = cfg.get("patterns", [])
            return {
                "enabled": cfg.get("enabled", False),
                "custom_patterns_count": len(custom_patterns),
                "recent_verdicts_summary": {},
            }
        except Exception as exc:
            return {"error": str(exc)}

    def list_intents(self) -> list[dict]:
        """Return all registered intents."""
        try:
            from aurarouter.intent_registry import build_intent_registry
            registry = build_intent_registry(self._config)
            result = []
            for intent in registry.get_all():
                if isinstance(intent, dict):
                    result.append(intent)
                else:
                    result.append({
                        "name": getattr(intent, "name", ""),
                        "description": getattr(intent, "description", ""),
                        "target_role": getattr(intent, "target_role", ""),
                        "source": getattr(intent, "source", "builtin"),
                        "priority": getattr(intent, "priority", 0),
                    })
            return result
        except Exception:
            return []

    def get_intent(self, name: str) -> dict | None:
        """Return single intent definition or None."""
        try:
            from aurarouter.intent_registry import build_intent_registry
            registry = build_intent_registry(self._config)
            intent = registry.get_by_name(name)
            if intent is None:
                return None
            if isinstance(intent, dict):
                return intent
            return {
                "name": getattr(intent, "name", name),
                "description": getattr(intent, "description", ""),
                "target_role": getattr(intent, "target_role", ""),
                "source": getattr(intent, "source", "builtin"),
                "priority": getattr(intent, "priority", 0),
            }
        except Exception:
            return None

    # ======================================================================
    # T1.5 Telemetry & Feedback API
    # ======================================================================

    def get_model_performance(self, window_days: int = 7) -> list[dict]:
        """Return per-model performance stats from feedback store."""
        try:
            fb = self._get_feedback_store()
            if fb is None:
                return []
            stats = fb.model_stats()
            # model_stats() returns list[dict] with "model_id" key in each entry
            if isinstance(stats, list):
                return stats
            result = []
            for model_id, data in stats.items():
                if isinstance(data, dict):
                    result.append({"model_id": model_id, **data})
                else:
                    result.append({
                        "model_id": model_id,
                        "call_count": getattr(data, "call_count", 0),
                        "success_rate": getattr(data, "success_rate", 0.0),
                        "avg_latency_ms": getattr(data, "avg_latency_ms", 0.0),
                    })
            return result
        except Exception:
            return []

    def get_savings_summary(self, start: str | None = None, end: str | None = None) -> dict:
        """Return savings summary."""
        if self._usage_store is None:
            return {"error": "savings_disabled"}
        try:
            totals = self._usage_store.total_tokens(start=start, end=end)
            cost_avoided = 0.0
            hard_route_count = 0
            total_count = 0
            if hasattr(self._usage_store, "query"):
                records = self._usage_store.query(start=start, end=end)
                total_count = len(records)
                cost_avoided = sum(getattr(r, "simulated_cost_avoided", 0.0) for r in records)
                hard_route_count = sum(1 for r in records if not getattr(r, "is_cloud", True))
            hard_route_ratio = (hard_route_count / total_count * 100.0) if total_count > 0 else 0.0
            local_routing_pct = hard_route_ratio
            projection = 0.0
            if self._cost_engine is not None:
                projection = self._cost_engine.monthly_projection()
            return {
                "total_cost_avoided": cost_avoided,
                "hard_route_ratio": hard_route_ratio,
                "local_routing_pct": local_routing_pct,
                "cloud_spend": 0.0,
                "monthly_projection": projection,
                "roi_estimate": cost_avoided,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def get_rag_status(self) -> dict:
        """Return RAG enrichment pipeline status."""
        try:
            from aurarouter.rag_enrichment import RagEnrichmentPipeline
            pipeline = RagEnrichmentPipeline(self._config)
            return {
                "enabled": pipeline.is_enabled(),
                "endpoint_configured": bool(self._config.config.get("auraxlm", {}).get("endpoint")),
                "recent_retrievals_count": 0,
                "avg_latency_ms": 0.0,
            }
        except Exception:
            return {"enabled": False, "endpoint_configured": False,
                    "recent_retrievals_count": 0, "avg_latency_ms": 0.0}

    def _get_feedback_store(self) -> Any:
        """Lazy-initialize FeedbackStore."""
        if self._feedback_store is None:
            try:
                from aurarouter.savings.feedback_store import FeedbackStore
                self._feedback_store = FeedbackStore()
            except Exception:
                return None
        return self._feedback_store
