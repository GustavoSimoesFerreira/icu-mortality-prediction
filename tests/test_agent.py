"""Offline tests for the agent loop — mocks Ollama, no server needed."""

from src.agent import agent as agent_mod


class FakeClient:
    """Simulates an Ollama tool-calling conversation without a server."""

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def chat_messages(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            # First turn: the "model" decides to call a tool.
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_patient_risk", "arguments": {"patient_id": 0}}}
                ],
            }
        # Second turn: the model writes the final answer (no more tool calls).
        return {"role": "assistant", "content": "Patient 0 has elevated risk [1]."}


def test_agent_loop_executes_tool_then_answers(monkeypatch):
    # Replace the client and the tool so no model/data is needed.
    monkeypatch.setattr(agent_mod, "OllamaClient", FakeClient)
    monkeypatch.setattr(
        agent_mod, "get_patient_risk",
        lambda pid, cfg: {"patient_id": pid, "risk_28d_mortality": 0.42, "top_drivers": []},
    )

    cfg = {"agent": {"host": "x", "gen_model": "m", "timeout": 1, "max_iterations": 4}}
    result = agent_mod.run_agent(0, cfg, verbose=False)
    assert "elevated risk" in result


def test_dispatch_unknown_tool():
    out = agent_mod.dispatch("does_not_exist", {}, cfg={})
    assert "Unknown tool" in out
