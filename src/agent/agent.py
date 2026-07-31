"""An agent that writes a clinical risk summary by orchestrating two tools.

Given a patient, the LLM autonomously decides to (1) fetch the patient's model
risk and drivers, (2) search PubMed for evidence on the main driver, and
(3) write a concise, cited summary for a clinician. Tool calls are handled via
Ollama's function-calling API.

Usage:
    python -m src.agent.run --config configs/agent.yaml --patient 0

Requires a running Ollama server with a tool-capable model (e.g. llama3.1) and
a RAG index already built (src.rag.ingest).
"""

from __future__ import annotations

import json

import yaml

from src.agent.tools import get_patient_risk, search_medical_evidence
from src.rag.ollama_client import OllamaClient

SYSTEM = (
    "You are a clinical decision-support assistant. Your goal is to produce a "
    "short risk summary for the given ICU patient. Follow these steps in order:\n"
    "1. Call get_patient_risk with the patient's integer id to obtain the "
    "predicted 28-day mortality risk and its top drivers.\n"
    "2. Identify the single most important risk driver from the result, then "
    "call search_medical_evidence with a SPECIFIC, non-empty query about that "
    "driver (for example, the clinical factor's name together with 'ICU "
    "mortality', such as 'SAPS score ICU mortality'). Never call "
    "search_medical_evidence with an empty query.\n"
    "3. Write a concise summary (5-8 sentences) stating the risk, the main "
    "drivers, and what the evidence says, citing sources as [1], [2].\n"
    "Only use facts returned by the tools; never invent data or citations."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_patient_risk",
            "description": "Get the model's predicted 28-day mortality risk and top "
            "contributing factors for an ICU patient, by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "integer",
                        "description": "Patient index in the held-out test set.",
                    }
                },
                "required": ["patient_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_medical_evidence",
            "description": "Search PubMed literature for evidence on a clinical topic "
            "and return relevant snippets with citation numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Clinical search query."}
                },
                "required": ["query"],
            },
        },
    },
]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dispatch(name: str, args: dict, cfg: dict) -> str:
    """Execute a tool call and return its result as a string."""
    if name == "get_patient_risk":
        return json.dumps(get_patient_risk(int(args["patient_id"]), cfg))
    if name == "search_medical_evidence":
        return search_medical_evidence(str(args["query"]), cfg)
    return f"Unknown tool: {name}"


def run_agent(patient_id: int, cfg: dict, verbose: bool = True) -> str:
    client = OllamaClient(
        cfg["agent"]["host"],
        embed_model="",  # not used for chat
        gen_model=cfg["agent"]["gen_model"],
        timeout=cfg["agent"]["timeout"],
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Write the risk summary for patient {patient_id}."},
    ]

    for _ in range(cfg["agent"]["max_iterations"]):
        message = client.chat_messages(messages, tools=TOOLS)
        messages.append(message)
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return message.get("content", "")
        for call in tool_calls:
            fn = call["function"]
            args = fn["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            if verbose:
                print(f"  [agent] calling {fn['name']}({args}) ...")
            result = dispatch(fn["name"], args, cfg)
            messages.append({"role": "tool", "content": result})

    return messages[-1].get("content", "(max iterations reached without a final answer)")
