"""Claude tool-calling agent: orchestrates detect_bias → compute_belief → allocation tools, then reflects periodically."""

from __future__ import annotations

import json

from roulette_agent.tools import TOOLS, dispatch_tool

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are a statistically-rigorous roulette betting advisor. Your goal is to decide the best \
bets for the next spin using the tools available.

## Session context
- Wheel type  : {wheel_type}
- Bankroll    : {bankroll}
- Bet unit    : {bet_unit}
- Excluded dozens: {excluded_dozens}
- Recent history length: {n_history} spins
{custom_config_note}

## Rules you must follow
1. Always call detect_bias first with the full recent_history.
2. Call compute_belief with the resulting bias_report before any allocation tool.
3. Choose at most ONE allocation strategy per decision — follow the priority order below.
4. NEVER recommend bets that cover any number in the excluded dozens.
5. If the final bet list is empty, that is a valid recommendation — output rationale explaining why.

## Preferred workflow (2 tool calls)
Step 1: detect_bias(recent_history)          → bias_report
Step 2: adaptive_gaussian_kelly_allocation(recent_history, bias_report, bankroll, bet_unit,
                                            excluded_dozens, wheel_type)  → bets

adaptive_gaussian_kelly_allocation is the PRIMARY allocator for all situations:
  • Early history (N < 100): large stake + wide Gaussian coverage (exploration phase).
    Even with no bias signal, returns amplified outside bets for broad floor coverage.
  • Growing history: blends in temporal recency bias and trend multipliers.
  • Mature history (N ≥ 200): converges to tight, hotspot-focused true portfolio Kelly.
  Amounts always scale with bankroll (true Kelly behaviour).

## Fallback (only when adaptive_gaussian_kelly_allocation is unavailable)
  • gaussian_kelly_allocation — pure Kelly, no adaptive schedule, requires pre-built p
  • fixed_baseline_allocation — flat outside-bet hedge
  • No bet — when verdict is no_evidence AND you judge exploration unwarranted

## On external stats
External stats (display-screen percentages) are provided for REFERENCE ONLY.
They can corroborate a bias signal but CANNOT upgrade your confidence or increase bet size. \
A consistent external signal merely confirms; an inconsistent one lowers trust.

## Agent notes (accumulated experience)
{notes}

When you have finished calling tools and are ready to give your final answer, stop calling tools \
and output a JSON object with exactly two keys:
  "bets": [ {{"type": "...", "numbers": [...] or null, "amount": ...}}, ... ]
  "rationale": "one or two sentences explaining the decision"

Output ONLY that JSON object as your final message — no surrounding text.
"""

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_MAX_TOOL_ROUNDS = 10


class RouletteAgent:
    def __init__(self, anthropic_client, model: str = "claude-sonnet-4-5-20250929",
                 context: dict | None = None):
        self._client = anthropic_client
        self._model = model
        self._context = context or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, session_state: dict) -> dict:
        """Run a full tool-use loop and return {"bets": [...], "rationale": "..."}.

        session_state keys:
          bankroll, bet_unit, wheel_type, excluded_dozens,
          recent_history, external_stats, external_stats_n_estimate,
          hyperparams, notes
        """
        custom_payouts = self._context.get("custom_payouts")
        custom_wheel_order = self._context.get("custom_wheel_order")
        config_lines = []
        if custom_payouts:
            config_lines.append(f"- Custom payouts active: {custom_payouts} (tools use these automatically)")
        if custom_wheel_order:
            config_lines.append(f"- Custom wheel order loaded ({len(custom_wheel_order)} pockets); sector analysis uses your specific wheel layout")
        custom_config_note = "\n".join(config_lines)

        system = _SYSTEM_TEMPLATE.format(
            wheel_type=session_state.get("wheel_type", "american"),
            bankroll=session_state.get("bankroll", 0),
            bet_unit=session_state.get("bet_unit", 10),
            excluded_dozens=session_state.get("excluded_dozens", []),
            n_history=len(session_state.get("recent_history", [])),
            notes=session_state.get("notes", "No accumulated notes yet."),
            custom_config_note=custom_config_note,
        )

        # Inject external_stats summary into user message if present
        ext = session_state.get("external_stats")
        ext_note = ""
        if ext:
            ext_note = f"\nExternal display-screen stats (reference only): {json.dumps(ext)}"
            n_est = session_state.get("external_stats_n_estimate")
            if n_est:
                ext_note += f" (N estimate: {n_est})"

        user_msg = (
            "Based on the current session state, decide the best bet(s) for the next spin.\n"
            f"Recent history: {session_state.get('recent_history', [])}{ext_note}"
        )

        messages: list[dict] = [{"role": "user", "content": user_msg}]

        for _round in range(_MAX_TOOL_ROUNDS):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system,
                tools=TOOLS,
                messages=messages,
            )

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                return self._parse_final(response.content)

            if response.stop_reason != "tool_use":
                raise RuntimeError(
                    f"Unexpected stop_reason: {response.stop_reason!r}"
                )

            # Execute all tool calls in this turn
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                try:
                    result = dispatch_tool(block.name, block.input, context=self._context)
                    result_content = json.dumps(result, default=str)
                    is_error = False
                except Exception as exc:
                    result_content = f"Error: {exc}"
                    is_error = True

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                    "is_error": is_error,
                })

            messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(
            f"Tool loop exceeded {_MAX_TOOL_ROUNDS} rounds without end_turn."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_final(self, content) -> dict:
        """Extract the JSON bets+rationale from the final assistant message."""
        text = ""
        for block in content:
            if hasattr(block, "text"):
                text += block.text

        text = text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Graceful fallback: extract JSON block from mixed text
            import re
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                result = json.loads(m.group())
            else:
                result = {"bets": [], "rationale": text}

        # Ensure required keys exist
        result.setdefault("bets", [])
        result.setdefault("rationale", "")
        return result
