"""Tests for roulette_agent.tools and roulette_agent.agent (mocked Anthropic client)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from roulette_agent.agent import RouletteAgent, _MAX_TOOL_ROUNDS
from roulette_agent.tools import TOOLS, dispatch_tool


# ---------------------------------------------------------------------------
# A. Tool dispatch correctness
# ---------------------------------------------------------------------------

class TestDispatchTool:
    def test_detect_bias_empty_history(self):
        result = dispatch_tool("detect_bias", {"recent_history": []})
        assert result["verdict"] == "no_evidence"
        assert result["weight"] == pytest.approx(0.0)

    def test_detect_bias_with_wheel_type(self):
        result = dispatch_tool("detect_bias", {
            "recent_history": list(range(37)) * 2,
            "wheel_type": "european",
        })
        assert result["wheel_type"] == "european"

    def test_compute_belief_returns_string_keys(self):
        bias = dispatch_tool("detect_bias", {"recent_history": []})
        p = dispatch_tool("compute_belief", {
            "recent_history": [],
            "bias_report": bias,
        })
        assert all(isinstance(k, str) for k in p.keys())
        assert len(p) == 38  # american default

    def test_compute_belief_sums_to_one(self):
        bias = dispatch_tool("detect_bias", {"recent_history": [17] * 50})
        p = dispatch_tool("compute_belief", {
            "recent_history": [17] * 50,
            "bias_report": bias,
        })
        assert abs(sum(p.values()) - 1.0) < 1e-6

    def test_kelly_allocation_uniform_returns_empty(self):
        bias = dispatch_tool("detect_bias", {"recent_history": []})
        p = dispatch_tool("compute_belief", {"recent_history": [], "bias_report": bias})
        result = dispatch_tool("kelly_allocation", {
            "p": p, "bankroll": 800, "bet_unit": 10, "excluded_dozens": [],
        })
        assert result["bets"] == []

    def test_greedy_ev_uniform_returns_empty(self):
        bias = dispatch_tool("detect_bias", {"recent_history": []})
        p = dispatch_tool("compute_belief", {"recent_history": [], "bias_report": bias})
        result = dispatch_tool("greedy_ev_allocation", {
            "p": p, "bankroll": 800, "bet_unit": 10, "excluded_dozens": [],
        })
        assert result["bets"] == []

    def test_fixed_baseline_returns_bet(self):
        bias = dispatch_tool("detect_bias", {"recent_history": []})
        p = dispatch_tool("compute_belief", {"recent_history": [], "bias_report": bias})
        result = dispatch_tool("fixed_baseline_allocation", {
            "p": p, "bankroll": 800, "bet_unit": 10, "excluded_dozens": [],
        })
        assert len(result["bets"]) >= 1

    def test_settle_straight_hit(self):
        result = dispatch_tool("settle", {
            "bets": [{"type": "straight", "numbers": [17], "amount": 10}],
            "result_number": 17,
        })
        assert result["pnl"] == pytest.approx(350.0)

    def test_settle_straight_miss(self):
        result = dispatch_tool("settle", {
            "bets": [{"type": "straight", "numbers": [17], "amount": 10}],
            "result_number": 5,
        })
        assert result["pnl"] == pytest.approx(-10.0)

    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            dispatch_tool("nonexistent", {})

    def test_tools_list_has_eight_entries(self):
        assert len(TOOLS) == 8

    def test_all_tools_have_required_schema_fields(self):
        for t in TOOLS:
            assert "name" in t
            assert "description" in t
            assert "input_schema" in t
            assert t["input_schema"]["type"] == "object"


# ---------------------------------------------------------------------------
# B. RouletteAgent tool-use loop (mocked client)
# ---------------------------------------------------------------------------

def _make_tool_use_block(tool_id: str, name: str, input_dict: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_dict
    return block


def _make_text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_response(content, stop_reason: str):
    resp = MagicMock()
    resp.content = content
    resp.stop_reason = stop_reason
    return resp


def _build_session(history=None):
    return {
        "bankroll": 800.0,
        "bet_unit": 10.0,
        "wheel_type": "american",
        "excluded_dozens": [],
        "recent_history": history or [],
        "external_stats": None,
        "external_stats_n_estimate": None,
        "hyperparams": {},
        "notes": "",
    }


class TestRouletteAgentLoop:
    def _make_agent(self) -> tuple[RouletteAgent, MagicMock]:
        client = MagicMock()
        agent = RouletteAgent(client, model="claude-test-model")
        return agent, client

    # --- single-turn: end immediately ---
    def test_end_turn_no_tools(self):
        agent, client = self._make_agent()
        final_json = json.dumps({"bets": [], "rationale": "No evidence of bias."})
        client.messages.create.return_value = _make_response(
            [_make_text_block(final_json)], "end_turn"
        )
        result = agent.decide(_build_session())
        assert result["bets"] == []
        assert "rationale" in result
        assert client.messages.create.call_count == 1

    # --- one tool round then end_turn ---
    def test_one_tool_round(self):
        agent, client = self._make_agent()

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: ask to run detect_bias
                return _make_response(
                    [_make_tool_use_block("tid1", "detect_bias", {"recent_history": []})],
                    "tool_use",
                )
            else:
                # Second call: final answer
                return _make_response(
                    [_make_text_block(json.dumps({"bets": [], "rationale": "Fair wheel."}))],
                    "end_turn",
                )

        client.messages.create.side_effect = side_effect
        result = agent.decide(_build_session())
        assert result["bets"] == []
        assert call_count == 2

    # --- full realistic sequence: detect → belief → kelly → end ---
    def test_full_detect_belief_kelly_sequence(self):
        agent, client = self._make_agent()
        call_count = 0

        history = [17] * 100

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_response(
                    [_make_tool_use_block("t1", "detect_bias", {"recent_history": history})],
                    "tool_use",
                )
            elif call_count == 2:
                bias_report = dispatch_tool("detect_bias", {"recent_history": history})
                return _make_response(
                    [_make_tool_use_block("t2", "compute_belief", {
                        "recent_history": history,
                        "bias_report": bias_report,
                    })],
                    "tool_use",
                )
            elif call_count == 3:
                bias_report = dispatch_tool("detect_bias", {"recent_history": history})
                p = dispatch_tool("compute_belief", {
                    "recent_history": history, "bias_report": bias_report,
                })
                return _make_response(
                    [_make_tool_use_block("t3", "kelly_allocation", {
                        "p": p, "bankroll": 800, "bet_unit": 10, "excluded_dozens": [],
                    })],
                    "tool_use",
                )
            else:
                return _make_response(
                    [_make_text_block(json.dumps({
                        "bets": [{"type": "red", "numbers": None, "amount": 10}],
                        "rationale": "Weak bias detected on red.",
                    }))],
                    "end_turn",
                )

        client.messages.create.side_effect = side_effect
        result = agent.decide(_build_session(history))
        assert len(result["bets"]) == 1
        assert result["bets"][0]["type"] == "red"
        assert call_count == 4

    # --- tool error is forwarded gracefully ---
    def test_tool_error_does_not_crash_loop(self):
        agent, client = self._make_agent()
        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_response(
                    [_make_tool_use_block("tid1", "unknown_tool_xyz", {})],
                    "tool_use",
                )
            else:
                return _make_response(
                    [_make_text_block(json.dumps({"bets": [], "rationale": "Error handled."}))],
                    "end_turn",
                )

        client.messages.create.side_effect = side_effect
        result = agent.decide(_build_session())
        assert result["bets"] == []
        assert call_count == 2

    # --- loop limit raises RuntimeError ---
    def test_loop_limit_raises(self):
        agent, client = self._make_agent()

        def always_tool(**kwargs):
            return _make_response(
                [_make_tool_use_block("t", "detect_bias", {"recent_history": []})],
                "tool_use",
            )

        client.messages.create.side_effect = always_tool
        with pytest.raises(RuntimeError, match="exceeded"):
            agent.decide(_build_session())
        assert client.messages.create.call_count == _MAX_TOOL_ROUNDS

    # --- JSON wrapped in markdown fences is parsed ---
    def test_markdown_fenced_json_parsed(self):
        agent, client = self._make_agent()
        fenced = "```json\n" + json.dumps({"bets": [], "rationale": "ok"}) + "\n```"
        client.messages.create.return_value = _make_response(
            [_make_text_block(fenced)], "end_turn"
        )
        result = agent.decide(_build_session())
        assert result["bets"] == []
        assert result["rationale"] == "ok"

    # --- unexpected stop_reason raises ---
    def test_unexpected_stop_reason_raises(self):
        agent, client = self._make_agent()
        client.messages.create.return_value = _make_response([], "max_tokens")
        with pytest.raises(RuntimeError, match="stop_reason"):
            agent.decide(_build_session())
