"""Skill instructions stay separate from analysis placement and provider auditing."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from caesura_core import AnalysisContext, AnalyzeMessage, AnalyzeResult, CustomerTurn, InjectConfig
from caesura_openai import CaesuraOpenAIOptions, create_async_caesura, create_caesura

if TYPE_CHECKING:
    from caesura_core.types import InjectAs

SKILL = "Exact skill: שלום."


class Local:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def analyze(self, body: Any, **kwargs: Any) -> AnalyzeResult:
        self.calls.append(deepcopy(body))
        return AnalyzeResult("guidance")


class Provider:
    def __init__(self, asynchronous: bool) -> None:
        self.calls: list[Any] = []
        create = self.acreate if asynchronous else self.create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.responses = SimpleNamespace(create=create)
        self.close = self.aclose if asynchronous else self.sclose

    def sclose(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(deepcopy(kwargs))
        return SimpleNamespace(id="response", usage={"tokens": 3})

    async def acreate(self, **kwargs: Any) -> Any:
        return self.create(**kwargs)


# Each expected instruction prefix is explicit: no instruction-count special case.
CHAT_CASES = [
    pytest.param([], [{"role": "system", "content": SKILL}], id="missing"),
    pytest.param(
        [{"role": "developer", "content": "first"}, {"role": "system", "content": "last", "name": "policy"}],
        [
            {"role": "developer", "content": "first"},
            {"role": "system", "content": f"last\n\n{SKILL}", "name": "policy"},
        ],
        id="last-system",
    ),
    pytest.param(
        [{"role": "system", "content": "first"}, {"role": "developer", "content": "last"}],
        [{"role": "system", "content": "first"}, {"role": "developer", "content": f"last\n\n{SKILL}"}],
        id="last-developer",
    ),
    pytest.param(
        [{"role": "system", "content": f"before {SKILL} after"}, {"role": "developer", "content": "last"}],
        [{"role": "system", "content": f"before {SKILL} after"}, {"role": "developer", "content": "last"}],
        id="duplicate-earlier-system",
    ),
    pytest.param(
        [{"role": "developer", "content": [{"type": "text", "text": SKILL}]}, {"role": "system", "content": "last"}],
        [{"role": "developer", "content": [{"type": "text", "text": SKILL}]}, {"role": "system", "content": "last"}],
        id="duplicate-earlier-structured-developer",
    ),
    pytest.param(
        [
            {
                "role": "developer",
                "name": "policy",
                "content": [{"type": "text", "text": "existing", "cache_control": {"type": "ephemeral"}}],
            }
        ],
        [
            {
                "role": "developer",
                "name": "policy",
                "content": [
                    {"type": "text", "text": "existing", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": f"\n\n{SKILL}"},
                ],
            }
        ],
        id="structured-preserves-metadata",
    ),
    pytest.param(
        [{"role": "system", "content": SKILL.lower()}],
        [{"role": "system", "content": f"{SKILL.lower()}\n\n{SKILL}"}],
        id="exact-case-required",
    ),
]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("analysis_role", ["user", "developer"])
@pytest.mark.parametrize("prefix,expected_prefix", CHAT_CASES)
async def test_chat_skill_reuse_placement_and_audit(
    asynchronous: bool, analysis_role: InjectAs, prefix: list[Any], expected_prefix: list[Any]
) -> None:
    local, provider = Local(), Provider(asynchronous)
    audits: list[Any] = []
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            conversation_id="id",
            transport=local,
            mode="sync",
            inject=InjectConfig(skill_prompt=SKILL, as_role=analysis_role, template="{analysis}"),
            on_provider_request=audits.append,
        ),
    )
    # The prompt in dialogue must not suppress instruction insertion. Tool pairs,
    # multimodal content, and application-generated user content remain untouched.
    customer = {"role": "user", "name": "Visitor", "speakerIndex": 42, "content": SKILL}
    tail = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call", "content": "result"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "RAG"},
                {"type": "image_url", "image_url": {"url": "https://example.test/image"}},
            ],
        },
    ]
    history = deepcopy([*prefix, customer, *tail])
    expected = [*expected_prefix, customer, {"role": analysis_role, "content": "guidance"}, *tail]
    try:
        for position in (len(prefix), len(expected_prefix)):
            before = deepcopy(history)
            context = AnalysisContext(
                turn=CustomerTurn("turn", AnalyzeMessage("user", SKILL, speaker_name="Visitor", speaker_index=42)),
                dialogue_positions=[position],
            )
            call = client.chat.completions.create(
                model="fixture", messages=history, temperature=0.2, tools=[], caesura_context=context
            )
            if asynchronous:
                await call
            assert history == before
            assert provider.calls[-1] == {"model": "fixture", "messages": expected, "temperature": 0.2, "tools": []}
            assert audits[-1].parameters == provider.calls[-1]
            assert len(audits[-1].recommendation_ids) == 1
            history = deepcopy(provider.calls[-1]["messages"])
        assert len(local.calls) == 1  # Reusing the turn and injected history is idempotent.
        assert len(audits) == len(provider.calls) == 2
        snapshot = deepcopy(audits[0].parameters)
        history[0]["content"] = "caller mutation"
        assert audits[0].parameters == snapshot
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("skill", ["", " \t\n", SKILL])
@pytest.mark.parametrize("existing", [None, "existing", f"before {SKILL} after"])
async def test_skill_disabled_or_reused_instructions(
    asynchronous: bool, responses: bool, skill: str, existing: str | None
) -> None:
    local, provider = Local(), Provider(asynchronous)
    audits: list[Any] = []
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            conversation_id="id",
            transport=local,
            mode="sync",
            inject=InjectConfig(skill_prompt=skill, as_role="developer"),
            on_provider_request=audits.append,
        ),
    )
    key = "input" if responses else "messages"
    kwargs: dict[str, Any] = {
        "model": "fixture",
        key: "fresh dialogue" if responses else [{"role": "user", "content": "fresh dialogue"}],
    }
    if existing is not None:
        if responses:
            kwargs["instructions"] = existing
        else:
            kwargs["messages"].insert(0, {"role": "developer", "content": existing})
    expected = deepcopy(kwargs)
    if skill.strip() and (existing is None or skill not in existing):
        text = f"{existing}\n\n{skill}" if existing else skill
        if responses:
            expected["instructions"] = text
        elif existing is None:
            expected["messages"].insert(0, {"role": "system", "content": text})
        else:
            expected["messages"][0]["content"] = text
    try:
        for _ in range(2):
            before = deepcopy(kwargs)
            resource = client.responses if responses else client.chat.completions
            result = resource.create(**kwargs, caesura_context=AnalysisContext())
            if asynchronous:
                await result
            assert kwargs == before
            assert provider.calls[-1] == expected
            assert audits[-1].parameters == expected
            assert audits[-1].recommendation_ids == []
            kwargs = deepcopy(provider.calls[-1])
        assert local.calls == []
        assert len(audits) == len(provider.calls) == 2
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("role", ["system", "developer", "user", "assistant"])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(
    "instructions",
    [
        {},
        {"instructions": None},
        {"instructions": ""},
        {"instructions": "existing"},
        {"instructions": f"before {SKILL} after"},
    ],
)
async def test_responses_skill_checks_all_instruction_sources(
    asynchronous: bool, role: str, structured: bool, instructions: dict[str, Any]
) -> None:
    local, provider = Local(), Provider(asynchronous)
    audits: list[Any] = []
    client: Any = (create_async_caesura if asynchronous else create_caesura)(
        provider,
        CaesuraOpenAIOptions(
            conversation_id="id",
            transport=local,
            mode="sync",
            inject=InjectConfig(skill_prompt=SKILL, template="{analysis}", as_role="developer"),
            on_provider_request=audits.append,
        ),
    )
    content: Any = f"before {SKILL} after"
    if structured:
        content = [{"type": "input_text", "text": content}]
    history = [
        {"type": "message", "role": role, "content": content, "id": "original-message"},
        {"role": "developer", "content": "Last instruction has no skill"},
        {"role": "user", "content": "Customer asks"},
        {"role": "user", "content": [{"type": "input_text", "text": "RAG background"}]},
    ]
    kwargs = {"model": "fixture", "input": deepcopy(history), "metadata": {"request": "original"}, **instructions}
    expected = deepcopy(kwargs)
    expected["input"] = [*history[:3], {"role": "developer", "content": "guidance"}, history[3]]
    if role not in ("system", "developer"):
        existing = instructions.get("instructions") or ""
        if SKILL not in existing:
            expected["instructions"] = f"{existing}\n\n{SKILL}" if existing else SKILL
    try:
        for _ in range(2):
            before = deepcopy(kwargs)
            result = client.responses.create(
                **kwargs,
                caesura_context=AnalysisContext(
                    turn=CustomerTurn("turn", AnalyzeMessage("user", "Customer asks")),
                    dialogue_positions=[2],
                ),
            )
            if asynchronous:
                await result
            assert kwargs == before
            assert provider.calls[-1] == expected
            assert audits[-1].parameters == provider.calls[-1]
            assert len(audits[-1].recommendation_ids) == 1
            kwargs = deepcopy(provider.calls[-1])
        assert len(local.calls) == 1
        assert len(audits) == len(provider.calls) == 2
        snapshot = deepcopy(audits[0].parameters)
        kwargs["input"][0]["content"] = "caller mutation"
        kwargs["metadata"]["request"] = "changed"
        assert audits[0].parameters == snapshot
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()
