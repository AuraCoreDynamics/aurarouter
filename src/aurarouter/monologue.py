"""AuraMonologue — Recursive multi-expert reasoning blackboard.

Treats the entire grid as a single thinking entity using a blackboard
pattern where Generator, Critic, and Refiner experts read/write
reasoning artifacts concurrently to the Sharded Reasoning WAL.

MAS-Score-Gated Node Idling: Before dispatching to an expert, the
orchestrator queries AnchorScoringMas for model relevancy. If below
threshold, the expert is skipped (node idles), saving compute.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from aurarouter._logging import get_logger

if TYPE_CHECKING:
    from aurarouter.config import ConfigLoader
    from aurarouter.fabric import ComputeFabric
    from aurarouter.mcp_client.registry import McpClientRegistry
    from aurarouter.rag_enrichment import RagEnrichmentPipeline
    from aurarouter.sovereignty import SovereigntyGate

logger = get_logger("AuraRouter.Monologue")


@dataclass
class ReasoningStep:
    """A single expert step in the monologue reasoning trace."""

    step_id: str
    role: str  # "generator", "critic", "refiner"
    model_id: str
    input_prompt: str
    output: str
    latent_anchors_used: list[str] = field(default_factory=list)
    mas_relevancy_score: float = 0.0
    confidence: float = 0.0
    iteration: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "role": self.role,
            "model_id": self.model_id,
            "input_prompt": self.input_prompt,
            "output": self.output,
            "latent_anchors_used": self.latent_anchors_used,
            "mas_relevancy_score": self.mas_relevancy_score,
            "confidence": self.confidence,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
        }


@dataclass
class MonologueResult:
    """Result of a completed monologue reasoning session."""

    session_id: str
    final_output: str
    reasoning_trace: list[ReasoningStep] = field(default_factory=list)
    total_iterations: int = 0
    convergence_reason: str = ""
    total_latency_ms: float = 0.0
    nodes_idled: int = 0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "final_output": self.final_output,
            "reasoning_trace": [s.to_dict() for s in self.reasoning_trace],
            "total_iterations": self.total_iterations,
            "convergence_reason": self.convergence_reason,
            "total_latency_ms": self.total_latency_ms,
            "nodes_idled": self.nodes_idled,
        }


class MonologueOrchestrator:
    """Recursive multi-expert reasoning on a shared blackboard WAL.

    Execution loop per iteration:
        1. Retrieve relevant latent anchors via auraxlm.retrieve_anchors
        2. Generator writes reasoning trace (skip if MAS relevancy low)
        3. Critic scores trace + writes critique (skip if MAS relevancy low)
        4. Convergence check (critic score >= threshold)
        5. Refiner produces hardened response
        6. Write latent anchors to WAL via auraxlm.write_anchor
    """

    def __init__(
        self,
        fabric: ComputeFabric,
        mcp_registry: McpClientRegistry | None,
        sovereignty_gate: SovereigntyGate | None,
        rag_pipeline: RagEnrichmentPipeline | None,
    ):
        self._fabric = fabric
        self._mcp_registry = mcp_registry
        self._sovereignty_gate = sovereignty_gate
        self._rag_pipeline = rag_pipeline
        self._sessions: dict[str, MonologueResult] = {}

    @property
    def config(self) -> ConfigLoader:
        return self._fabric.config

    def _get_config_value(self, key: str, default):
        sys_cfg = self.config.config.get("system", {})
        return sys_cfg.get(key, default)

    def is_enabled(self) -> bool:
        return bool(self._get_config_value("monologue", False))

    def get_session(self, session_id: str) -> MonologueResult | None:
        return self._sessions.get(session_id)

    def get_active_sessions(self) -> list[MonologueResult]:
        return [s for s in self._sessions.values() if not s.convergence_reason]

    def _select_experts(
        self, sovereignty_result=None,
    ) -> tuple[str | None, str | None, str | None]:
        """Select generator, critic, and refiner models from role chains.

        Returns (generator_model, critic_model, refiner_model). Any may be
        None if no suitable model is available after sovereignty filtering.
        """
        gen_chain = self.config.get_role_chain("reasoning")
        crit_chain = self.config.get_role_chain("reviewer")
        ref_chain = self.config.get_role_chain("coding")

        if sovereignty_result is not None and self._sovereignty_gate is not None:
            gen_chain = self._sovereignty_gate.enforce(
                gen_chain, self.config, sovereignty_result
            )
            crit_chain = self._sovereignty_gate.enforce(
                crit_chain, self.config, sovereignty_result
            )
            ref_chain = self._sovereignty_gate.enforce(
                ref_chain, self.config, sovereignty_result
            )

        generator = gen_chain[0] if gen_chain else None
        critic = crit_chain[0] if crit_chain else None
        refiner = ref_chain[0] if ref_chain else None

        # Diversity: critic should differ from generator when possible
        if critic == generator and len(crit_chain) > 1:
            critic = crit_chain[1]

        return generator, critic, refiner

    async def _retrieve_anchors(self, top_k: int = 5) -> list[dict]:
        """Retrieve latent anchors via auraxlm.retrieve_anchors MCP tool."""
        if self._mcp_registry is None:
            return []
        try:
            clients = self._mcp_registry.get_clients_with_capability("anchor")
            for client in clients:
                result = await asyncio.to_thread(
                    client.call_tool, "auraxlm.retrieve_anchors",
                    top_k=top_k, include_flagged=False,
                )
                if result:
                    return json.loads(result) if isinstance(result, str) else result
        except Exception as exc:
            logger.debug("Anchor retrieval failed (degraded): %s", exc)
        return []

    async def _score_anchor(self, anchor_id: str, prompt: str) -> float:
        """Score a model's relevancy via auraxlm.score_anchor MCP tool."""
        if self._mcp_registry is None:
            return 1.0  # Assume relevant when MCP unavailable
        try:
            clients = self._mcp_registry.get_clients_with_capability("anchor")
            for client in clients:
                result = await asyncio.to_thread(
                    client.call_tool, "auraxlm.score_anchor",
                    anchor_id=anchor_id, prompt=prompt,
                )
                if result:
                    data = json.loads(result) if isinstance(result, str) else result
                    return float(data.get("combined_score", 1.0))
        except Exception as exc:
            logger.debug("Anchor scoring failed (assume relevant): %s", exc)
        return 1.0

    async def _write_anchor(self, step: ReasoningStep) -> None:
        """Write a latent anchor to WAL via auraxlm.write_anchor MCP tool."""
        if self._mcp_registry is None:
            return
        try:
            clients = self._mcp_registry.get_clients_with_capability("anchor")
            for client in clients:
                await asyncio.to_thread(
                    client.call_tool, "auraxlm.write_anchor",
                    anchor_id=step.step_id, model_id=step.model_id,
                    session_id=step.step_id, content=step.output,
                )
                return
        except Exception as exc:
            logger.debug("Anchor write failed (non-fatal): %s", exc)

    def _compute_text_similarity(self, text_a: str, text_b: str) -> float:
        """Simple word-overlap similarity for convergence detection.

        Uses Jaccard similarity on word sets as a lightweight proxy
        for cosine similarity on embeddings.
        """
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    async def reason(
        self,
        task: str,
        context: str | None = None,
        max_iterations: int = 5,
        convergence_threshold: float = 0.85,
        mas_relevancy_threshold: float = 0.4,
        permissions: dict | None = None,
        event_callback: Callable[[str], None] | None = None,
    ) -> MonologueResult:
        """Execute recursive multi-expert reasoning on the blackboard.

        Returns a MonologueResult with the final output, reasoning trace,
        and convergence metadata.
        """
        from aurarouter.sovereignty import SovereigntyVerdict

        def emit(event: str):
            if event_callback:
                event_callback(event)

        # Local permissions check (Task 2.2)
        if permissions:
            logger.debug(f"Monologue execution using local permissions: {permissions}")

        session_id = uuid.uuid4().hex[:16]
        start_time = time.monotonic()
        result = MonologueResult(session_id=session_id, final_output="")
        self._sessions[session_id] = result

        emit(f"[MONOLOGUE START] Task: {task[:50]}...")

        # Sovereignty check
        sovereignty_result = None
        if self._sovereignty_gate is not None:
            sovereignty_result = self._sovereignty_gate.evaluate(task)
            if sovereignty_result.verdict == SovereigntyVerdict.BLOCKED:
                emit("[MONOLOGUE BLOCKED] Sovereignty check failed")
                result.convergence_reason = "sovereignty_blocked"
                result.total_latency_ms = (time.monotonic() - start_time) * 1000
                return result

        generator_model, critic_model, refiner_model = self._select_experts(
            sovereignty_result
        )

        if generator_model is None:
            emit("[MONOLOGUE ERROR] No models available")
            result.convergence_reason = "no_models_available"
            result.total_latency_ms = (time.monotonic() - start_time) * 1000
            return result

        full_prompt = f"{context}\n\n{task}" if context else task
        previous_output = ""

        for iteration in range(1, max_iterations + 1):
            emit(f"[MONOLOGUE ITERATION {iteration}]")
            # Step 1: Retrieve latent anchors
            anchors = await self._retrieve_anchors()
            anchor_ids = [a.get("anchor_id", "") for a in anchors if isinstance(a, dict)]
            if anchor_ids:
                emit(f"[MONOLOGUE ANCHORS] Found {len(anchor_ids)} relevant anchors")

            # Step 2: Generator
            emit(f"[MONOLOGUE STEP] Generating reasoning ({generator_model})...")
            gen_relevancy = await self._score_anchor(
                f"gen-{generator_model}", full_prompt
            )
            if gen_relevancy < mas_relevancy_threshold:
                result.nodes_idled += 1
                emit(f"[MONOLOGUE IDLE] Generator {generator_model} idled (relevancy={gen_relevancy:.2f})")
                logger.info(
                    "Monologue iter %d: generator %s idled (MAS relevancy=%.3f < %.3f)",
                    iteration, generator_model, gen_relevancy, mas_relevancy_threshold,
                )
            else:
                gen_prompt = (
                    f"You are a Generator expert. Produce a detailed reasoning trace.\n"
                    f"TASK: {full_prompt}\n"
                )
                if previous_output:
                    gen_prompt += f"PREVIOUS REASONING:\n{previous_output}\n"
                if anchor_ids:
                    gen_prompt += f"RELEVANT ANCHORS: {', '.join(anchor_ids[:5])}\n"
                gen_prompt += "Return your reasoning trace."

                gen_result = self._fabric.execute(
                    "reasoning", gen_prompt,
                    chain_override=[generator_model] if generator_model else None,
                )
                gen_output = gen_result.text if gen_result else ""

                step = ReasoningStep(
                    step_id=f"{session_id}-gen-{iteration}",
                    role="generator",
                    model_id=generator_model or "",
                    input_prompt=gen_prompt,
                    output=gen_output,
                    latent_anchors_used=anchor_ids[:5],
                    mas_relevancy_score=gen_relevancy,
                    confidence=0.0,
                    iteration=iteration,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                result.reasoning_trace.append(step)
                await self._write_anchor(step)
                previous_output = gen_output
                emit(f"[MONOLOGUE GEN] {len(gen_output)} chars produced")

            # Step 3: Critic
            critic_score = 0.0
            if critic_model is not None and previous_output:
                emit(f"[MONOLOGUE STEP] Critiquing reasoning ({critic_model})...")
                crit_relevancy = await self._score_anchor(
                    f"crit-{critic_model}", full_prompt
                )
                if crit_relevancy < mas_relevancy_threshold:
                    result.nodes_idled += 1
                    emit(f"[MONOLOGUE IDLE] Critic {critic_model} idled (relevancy={crit_relevancy:.2f})")
                    logger.info(
                        "Monologue iter %d: critic %s idled (MAS relevancy=%.3f < %.3f)",
                        iteration, critic_model, crit_relevancy, mas_relevancy_threshold,
                    )
                else:
                    crit_prompt = (
                        f"You are a Critic expert. Evaluate the reasoning trace and "
                        f"assign a confidence score 0.0-1.0.\n"
                        f"ORIGINAL TASK: {full_prompt}\n"
                        f"REASONING TRACE:\n{previous_output}\n"
                        f"Return JSON: {{\"score\": 0.85, \"feedback\": \"...\"}}"
                    )

                    crit_result = self._fabric.execute(
                        "reviewer", crit_prompt, json_mode=True,
                        chain_override=[critic_model] if critic_model else None,
                    )
                    crit_text = crit_result.text if crit_result else ""
                    try:
                        crit_data = json.loads(crit_text)
                        critic_score = float(crit_data.get("score", 0.0))
                    except (json.JSONDecodeError, ValueError, TypeError):
                        critic_score = 0.0

                    crit_step = ReasoningStep(
                        step_id=f"{session_id}-crit-{iteration}",
                        role="critic",
                        model_id=critic_model,
                        input_prompt=crit_prompt,
                        output=crit_text,
                        latent_anchors_used=anchor_ids[:5],
                        mas_relevancy_score=crit_relevancy,
                        confidence=critic_score,
                        iteration=iteration,
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )
                    result.reasoning_trace.append(crit_step)
                    await self._write_anchor(crit_step)
                    emit(f"[MONOLOGUE CRITIC] Score: {critic_score:.2f}")

            # Step 4: Convergence check — critic score
            if critic_score >= convergence_threshold:
                emit(f"[MONOLOGUE CONVERGED] Score {critic_score:.2f} >= {convergence_threshold}")
                result.convergence_reason = "confidence_threshold"
                break

            # Step 4b: Convergence check — output similarity
            if iteration > 1 and len(result.reasoning_trace) >= 2:
                gen_steps = [
                    s for s in result.reasoning_trace if s.role == "generator"
                ]
                if len(gen_steps) >= 2:
                    sim = self._compute_text_similarity(
                        gen_steps[-1].output, gen_steps[-2].output
                    )
                    if sim > 0.95:
                        emit(f"[MONOLOGUE CONVERGED] Output similarity={sim:.3f}")
                        result.convergence_reason = "output_similarity"
                        break

            # Step 5: Refiner
            if refiner_model is not None and previous_output:
                emit(f"[MONOLOGUE STEP] Refining reasoning ({refiner_model})...")
                ref_prompt = (
                    f"You are a Refiner expert. Produce a hardened, final response.\n"
                    f"ORIGINAL TASK: {full_prompt}\n"
                    f"REASONING TRACE:\n{previous_output}\n"
                )
                if crit_text := (
                    result.reasoning_trace[-1].output
                    if result.reasoning_trace and result.reasoning_trace[-1].role == "critic"
                    else ""
                ):
                    ref_prompt += f"CRITIC FEEDBACK:\n{crit_text}\n"
                ref_prompt += "Return the final hardened response."

                ref_result = self._fabric.execute(
                    "coding", ref_prompt,
                    chain_override=[refiner_model] if refiner_model else None,
                )
                ref_output = ref_result.text if ref_result else previous_output

                ref_step = ReasoningStep(
                    step_id=f"{session_id}-ref-{iteration}",
                    role="refiner",
                    model_id=refiner_model,
                    input_prompt=ref_prompt,
                    output=ref_output,
                    latent_anchors_used=anchor_ids[:5],
                    mas_relevancy_score=1.0,
                    confidence=critic_score,
                    iteration=iteration,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                result.reasoning_trace.append(ref_step)
                await self._write_anchor(ref_step)
                previous_output = ref_output
                emit(f"[MONOLOGUE REFINE] Improved reasoning produced")

            result.total_iterations = iteration

        # Final output from last refiner or generator step
        if not result.convergence_reason:
            result.convergence_reason = "max_iterations"
            emit("[MONOLOGUE END] Max iterations reached")

        result.total_iterations = max(
            result.total_iterations,
            max((s.iteration for s in result.reasoning_trace), default=0),
        )
        result.final_output = previous_output
        result.total_latency_ms = (time.monotonic() - start_time) * 1000
        emit(f"[MONOLOGUE FINISHED] {len(result.final_output)} chars final output")

        logger.info(
            "Monologue %s converged: reason=%s, iterations=%d, idled=%d, latency=%.1fms",
            session_id, result.convergence_reason, result.total_iterations,
            result.nodes_idled, result.total_latency_ms,
        )

        return result
