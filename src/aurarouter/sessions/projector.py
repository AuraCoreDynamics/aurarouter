"""Zero-LLM ephemeral context projection for emergency model failover.

AuraRouter v2.1 §5

Projects a canonical multi-turn session or raw message history into a target
model's smaller context budget in O(N) CPU time (< 1 ms) without making any
LLM summarizer calls and without mutating ``Session.history``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aurarouter.sessions.gisting import GIST_INSTRUCTION
from aurarouter.tokens import count_tokens

if TYPE_CHECKING:
    from aurarouter.sessions.models import Session

_MSG_FRAMING_OVERHEAD = 4


@dataclass(frozen=True)
class TargetContextBudget:
    """Context budget specification for a candidate model."""

    model_id: str
    total_context_limit: int          # e.g. 8192 for local Llama-3-8B (0 = unlimited)
    reserved_output_tokens: int = 2048
    safety_margin_ratio: float = 0.08  # 8% buffer for chat template & BPE variance

    @property
    def max_usable_input_tokens(self) -> int:
        if self.total_context_limit <= 0:
            return 0
        effective_reserved = min(
            self.reserved_output_tokens,
            int(self.total_context_limit * 0.35),
        )
        usable = int(self.total_context_limit * (1.0 - self.safety_margin_ratio)) - effective_reserved
        return max(64, usable)


@dataclass(frozen=True)
class ProjectedPayload:
    """Ephemeral projected message payload fitted to a target model's context window."""

    messages: list[dict[str, str]]
    system_prompt: str
    estimated_input_tokens: int
    was_projected: bool
    turns_kept_full: int
    turns_gisted: int
    turns_dropped: int
    active_turn_truncated: bool = False


def _fast_estimate_tokens(text: str, cached_tokens: int = 0, model_id: str = "") -> int:
    """Fast token estimator: uses cached count when present, conservative O(1) heuristic on huge blobs."""
    if cached_tokens > 0:
        return cached_tokens
    if not text:
        return 0
    if len(text) > 8192:
        # Conservative ~3.5 chars/token avoids expensive multi-millisecond BPE on 128k pastes
        return max(1, int(len(text) / 3.5))
    return count_tokens(text, model_id=model_id)


class SessionProjector:
    """Deterministic O(N) ephemeral context projector."""

    def __init__(
        self,
        safety_margin_ratio: float = 0.08,
        active_turn_head_keep_ratio: float = 0.35,
        active_turn_tail_keep_ratio: float = 0.45,
        inject_failover_header: bool = True,
    ) -> None:
        self._safety_margin_ratio = safety_margin_ratio
        self._head_ratio = active_turn_head_keep_ratio
        self._tail_ratio = active_turn_tail_keep_ratio
        self._inject_failover_header = inject_failover_header

    def _elide_raw_body(self, text: str, max_tokens: int, model_id: str = "") -> tuple[str, int]:
        """Internal middle-out elider on plain text body (without trailing gist footer)."""
        orig_tokens = _fast_estimate_tokens(text, model_id=model_id)
        if max_tokens <= 0 or orig_tokens <= max_tokens:
            return text, orig_tokens

        total_keep_weight = max(0.01, self._head_ratio + self._tail_ratio)
        head_frac = self._head_ratio / total_keep_weight

        # Conservative char budget (~3.2 chars/token) leaving ~24 tokens for the elision notice
        usable_tokens_for_body = max(16, max_tokens - 24)
        char_budget = max(48, int(usable_tokens_for_body * 3.2))
        head_char_budget = max(24, int(char_budget * head_frac))
        tail_char_budget = max(24, char_budget - head_char_budget)

        lines = text.splitlines(keepends=True)
        if len(lines) >= 4:
            head_parts: list[str] = []
            acc = 0
            for ln in lines:
                if acc + len(ln) > head_char_budget and head_parts:
                    break
                head_parts.append(ln)
                acc += len(ln)

            tail_parts: list[str] = []
            acc_tail = 0
            for ln in reversed(lines[len(head_parts):]):
                if acc_tail + len(ln) > tail_char_budget and tail_parts:
                    break
                tail_parts.append(ln)
                acc_tail += len(ln)
            tail_parts.reverse()

            head_str = "".join(head_parts).rstrip("\r\n")
            tail_str = "".join(tail_parts).lstrip("\r\n")
        else:
            head_str = text[:head_char_budget]
            tail_str = text[-tail_char_budget:] if len(text) > head_char_budget + tail_char_budget else ""

        kept_tokens = _fast_estimate_tokens(head_str, model_id=model_id) + _fast_estimate_tokens(
            tail_str, model_id=model_id
        )
        omitted = max(1, orig_tokens - kept_tokens)
        marker = f"\n... [{omitted} tokens omitted during context projection] ...\n"
        combined = f"{head_str}{marker}{tail_str}" if tail_str else f"{head_str}{marker}"

        est = _fast_estimate_tokens(combined, model_id=model_id)
        if est > max_tokens:
            # Final hard slice guarantee so budget is strictly respected
            hard_chars = max(32, int(max_tokens * 3.0) - len(marker))
            h_len = int(hard_chars * head_frac)
            t_len = max(0, hard_chars - h_len)
            combined = (
                f"{head_str[:h_len]}{marker}{tail_str[-t_len:]}"
                if t_len > 0 and tail_str
                else f"{head_str[:hard_chars]}{marker}"
            )
            est = min(max_tokens, _fast_estimate_tokens(combined, model_id=model_id))

        return combined, est

    def elide_middle_out(self, text: str, max_tokens: int, model_id: str = "") -> tuple[str, int]:
        """Truncate oversized text preserving head + tail lines and any trailing GIST_INSTRUCTION."""
        orig_tokens = _fast_estimate_tokens(text, model_id=model_id)
        if max_tokens <= 0 or orig_tokens <= max_tokens:
            return text, orig_tokens

        # Protect trailing GIST_INSTRUCTION from ever being sliced off or corrupted (QA 1.6)
        if GIST_INSTRUCTION and text.endswith(GIST_INSTRUCTION):
            body = text[: -len(GIST_INSTRUCTION)]
            gist_tok = _fast_estimate_tokens(GIST_INSTRUCTION, model_id=model_id)
            body_budget = max(16, max_tokens - gist_tok)
            elided_body, body_est = self._elide_raw_body(body, body_budget, model_id=model_id)
            preserved = f"{elided_body}{GIST_INSTRUCTION}"
            total_est = min(max(max_tokens, gist_tok + 1), body_est + gist_tok)
            return preserved, total_est

        return self._elide_raw_body(text, max_tokens, model_id=model_id)

    def project(
        self,
        messages: list[dict] | None = None,
        system_prompt: str = "",
        budget: TargetContextBudget | None = None,
        session: Session | None = None,
        failover_from: str | None = None,
        failover_reason: str = "",
    ) -> ProjectedPayload:
        """Project ``messages`` (or ``session``) into ``budget`` without mutating inputs."""
        if messages is None:
            messages = session.get_messages_as_dicts(include_metadata=True) if session is not None else []

        # Separate leading system messages from conversational turns without mutating caller list
        extra_sys_parts: list[str] = []
        existing_prior_gists: list[str] = []
        conv_turns: list[dict[str, Any]] = []

        for m in messages:
            role = str(m.get("role", "user"))
            content = str(m.get("content", ""))
            if role == "system" and not conv_turns:
                if content.lstrip().startswith("[Prior Context]"):
                    for ln in content.splitlines():
                        stripped = ln.strip()
                        if stripped.startswith("- "):
                            existing_prior_gists.append(stripped[2:].strip())
                else:
                    extra_sys_parts.append(content)
            else:
                turn_copy: dict[str, Any] = {"role": role, "content": content}
                if "gist" in m and m["gist"]:
                    turn_copy["gist"] = str(m["gist"])
                if "tokens" in m and m["tokens"]:
                    try:
                        turn_copy["tokens"] = int(m["tokens"])
                    except (TypeError, ValueError):
                        pass
                if "gist_tokens" in m and m["gist_tokens"]:
                    try:
                        turn_copy["gist_tokens"] = int(m["gist_tokens"])
                    except (TypeError, ValueError):
                        pass
                if "tombstoned" in m:
                    turn_copy["tombstoned"] = bool(m["tombstoned"])
                conv_turns.append(turn_copy)

        merged_system = "\n\n".join(
            p for p in ([system_prompt] if system_prompt else []) + extra_sys_parts if p
        )

        # Align optional Session.history metadata with conv_turns (matching from newest turn backwards)
        hist = list(session.history) if session is not None else []
        hist_offset = len(hist) - len(conv_turns)

        def _turn_meta(turn_idx: int) -> tuple[int, bool, str | None, int]:
            """Return (cached_tokens, tombstoned, gist, gist_tokens) for conv_turns[turn_idx]."""
            h_idx = hist_offset + turn_idx
            turn = conv_turns[turn_idx]
            dict_tok = int(turn.get("tokens", 0) or 0)
            dict_tomb = bool(turn.get("tombstoned", False)) or turn["content"].startswith("[Tombstoned:")
            dict_gist = turn.get("gist")
            dict_gist_tok = int(turn.get("gist_tokens", 0) or 0)

            if 0 <= h_idx < len(hist):
                hm = hist[h_idx]
                is_tomb = hm.tombstoned or dict_tomb
                t_tok = (hm.tokens or dict_tok) if not is_tomb else 0
                return (
                    t_tok,
                    is_tomb,
                    hm.gist or dict_gist,
                    hm.gist_tokens or dict_gist_tok,
                )
            return (0 if dict_tomb else dict_tok), dict_tomb, dict_gist, dict_gist_tok

        target_model = budget.model_id if budget else ""
        max_usable = budget.max_usable_input_tokens if budget else 0

        # Estimate baseline unprojected token count
        sys_tokens = _fast_estimate_tokens(merged_system, model_id=target_model)
        prior_ctx_block = ""
        if existing_prior_gists:
            prior_ctx_block = (
                "[Prior Context]\n"
                + "\n".join(f"- {g}" for g in existing_prior_gists)
                + "\n[End Prior Context]"
            )
        prior_ctx_tokens = _fast_estimate_tokens(prior_ctx_block, model_id=target_model)

        turn_token_counts: list[int] = []
        for idx, t in enumerate(conv_turns):
            c_tok, _, _, _ = _turn_meta(idx)
            # If gist instruction was appended to last user turn, session.history[-1].tokens excludes it
            if idx == len(conv_turns) - 1 and GIST_INSTRUCTION and t["content"].endswith(GIST_INSTRUCTION):
                c_tok = 0
            turn_token_counts.append(
                _fast_estimate_tokens(t["content"], cached_tokens=c_tok, model_id=target_model)
                + _MSG_FRAMING_OVERHEAD
            )

        raw_total = sys_tokens + prior_ctx_tokens + sum(turn_token_counts)

        # Fast-path: unlimited context or already fits comfortably inside target budget
        if max_usable <= 0 or raw_total <= max_usable:
            out_msgs = [{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))} for m in messages]
            return ProjectedPayload(
                messages=out_msgs,
                system_prompt=system_prompt,
                estimated_input_tokens=raw_total,
                was_projected=False,
                turns_kept_full=len(conv_turns),
                turns_gisted=0,
                turns_dropped=0,
                active_turn_truncated=False,
            )

        # --- Projection Required ---
        remaining = max_usable
        header_str = ""
        if self._inject_failover_header and failover_from:
            reason_tag = f" ({failover_reason})" if failover_reason else ""
            header_str = (
                f"[FAILOVER CONTEXT: Routed from {failover_from}{reason_tag} to "
                f"{budget.model_id} ({budget.total_context_limit} ctx). "
                f"Prior turns condensed via pre-computed gists.]"
            )

        # 1. Pin system prompt (+ failover header), capping at 35% of max_usable
        effective_sys = (
            f"{merged_system}\n\n{header_str}".strip() if merged_system or header_str else ""
        )
        max_sys_budget = max(32, int(max_usable * 0.35))
        if effective_sys:
            eff_sys_tokens = _fast_estimate_tokens(effective_sys, model_id=target_model)
            if eff_sys_tokens > max_sys_budget:
                effective_sys, eff_sys_tokens = self.elide_middle_out(
                    effective_sys, max_sys_budget, model_id=target_model
                )
            remaining -= eff_sys_tokens

        if not conv_turns:
            return ProjectedPayload(
                messages=[],
                system_prompt=effective_sys,
                estimated_input_tokens=max_usable - remaining,
                was_projected=True,
                turns_kept_full=0,
                turns_gisted=0,
                turns_dropped=0,
            )

        # 2. Pin active final turn (middle-out eliding if single turn exceeds remaining budget)
        active_idx = len(conv_turns) - 1
        active_role = str(conv_turns[active_idx]["role"])
        active_content = str(conv_turns[active_idx]["content"])
        active_tokens = turn_token_counts[active_idx]
        active_truncated = False

        if active_tokens > remaining:
            elided_content, elided_tok = self.elide_middle_out(
                active_content,
                max(32, remaining - _MSG_FRAMING_OVERHEAD),
                model_id=target_model,
            )
            active_msg = {"role": active_role, "content": elided_content}
            remaining = max(0, remaining - (elided_tok + _MSG_FRAMING_OVERHEAD))
            active_truncated = True
        else:
            active_msg = {"role": active_role, "content": active_content}
            remaining -= active_tokens

        # 3. Walk prior turns newest-to-oldest (active_idx - 1 down to 0)
        kept_full_reversed: list[dict[str, str]] = []
        gisted_Reversed: list[str] = []
        turns_kept_full = 1  # active turn is kept
        turns_gisted = 0
        turns_dropped = 0
        switched_to_gists = active_truncated

        prior_frame_overhead = 10  # "[Prior Context]\n...\n[End Prior Context]"

        for idx in range(active_idx - 1, -1, -1):
            _, is_tombstoned, gist_text, gist_tok_cached = _turn_meta(idx)
            if is_tombstoned:
                turns_dropped += 1
                continue

            t_tok = turn_token_counts[idx]
            if not switched_to_gists and t_tok <= remaining:
                kept_full_reversed.append(
                    {"role": str(conv_turns[idx]["role"]), "content": str(conv_turns[idx]["content"])}
                )
                remaining -= t_tok
                turns_kept_full += 1
                continue

            switched_to_gists = True
            if gist_text:
                g_tok = _fast_estimate_tokens(
                    gist_text, cached_tokens=gist_tok_cached, model_id=target_model
                ) + 3
                needed = g_tok + (prior_frame_overhead if not gisted_Reversed else 0)
                if needed <= remaining:
                    gisted_Reversed.append(gist_text)
                    remaining -= needed
                    turns_gisted += 1
                else:
                    turns_dropped += 1
            else:
                # User turn or assistant turn without explicit gist
                turns_dropped += 1

        # 4. Also pack any session-level shared_context / existing_prior_gists not already included
        seen_gists = set(gisted_Reversed)
        extra_archive_gists: list[str] = []
        if session is not None and session.shared_context:
            extra_archive_gists.extend(g.summary for g in reversed(session.shared_context))
        extra_archive_gists.extend(reversed(existing_prior_gists))

        for g_summary in extra_archive_gists:
            if not g_summary or g_summary in seen_gists:
                continue
            g_tok = _fast_estimate_tokens(g_summary, model_id=target_model) + 3
            needed = g_tok + (prior_frame_overhead if not gisted_Reversed else 0)
            if needed <= remaining:
                gisted_Reversed.append(g_summary)
                seen_gists.add(g_summary)
                remaining -= needed
                if turns_dropped > 0:
                    turns_gisted += 1
                    turns_dropped -= 1

        kept_full_reversed.reverse()
        gisted_Reversed.reverse()

        projected_messages: list[dict[str, str]] = []
        if gisted_Reversed:
            gist_block = (
                "[Prior Context]\n"
                + "\n".join(f"- {g}" for g in gisted_Reversed)
                + "\n[End Prior Context]"
            )
            projected_messages.append({"role": "system", "content": gist_block})

        projected_messages.extend(kept_full_reversed)
        projected_messages.append(active_msg)

        estimated_used = max_usable - remaining
        return ProjectedPayload(
            messages=projected_messages,
            system_prompt=effective_sys,
            estimated_input_tokens=estimated_used,
            was_projected=True,
            turns_kept_full=turns_kept_full,
            turns_gisted=turns_gisted,
            turns_dropped=turns_dropped,
            active_turn_truncated=active_truncated,
        )
