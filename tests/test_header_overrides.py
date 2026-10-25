import asyncio
from typing import AsyncIterator

import pytest

from chatkit.server import (
    agents_sdk_user_agent_override,
    chat_completions_headers_override,
    responses_headers_override,
)
from chatkit.types import (
    InferenceOptions,
    ProgressUpdateEvent,
    ThreadCreateParams,
    ThreadsCreateReq,
    ThreadStreamEvent,
    UserMessageInput,
    UserMessageTextContent,
)
from tests.test_chatkit_server import make_server


@pytest.fixture(params=[None, {"User-Agent": "caller", "X-Request": "existing"}])
def original_headers(request):
    headers = request.param
    chat_token = chat_completions_headers_override.set(headers)
    responses_token = responses_headers_override.set(headers)
    try:
        yield headers
    finally:
        chat_completions_headers_override.reset(chat_token)
        responses_headers_override.reset(responses_token)


def test_restores_headers_after_normal_exit(original_headers):
    with agents_sdk_user_agent_override():
        headers = chat_completions_headers_override.get()
        assert headers is not None
        assert "ChatKit/Python" in headers["User-Agent"]
        assert responses_headers_override.get() == headers

    assert chat_completions_headers_override.get() is original_headers
    assert responses_headers_override.get() is original_headers


@pytest.mark.parametrize(
    "error_type", [ValueError, asyncio.CancelledError, GeneratorExit]
)
def test_restores_headers_after_exception(original_headers, error_type):
    error = error_type("interrupted")

    with pytest.raises(error_type) as caught:
        with agents_sdk_user_agent_override():
            raise error

    assert caught.value is error
    assert chat_completions_headers_override.get() is original_headers
    assert responses_headers_override.get() is original_headers


def test_nested_failure_restores_outer_overrides(original_headers):
    with agents_sdk_user_agent_override():
        outer_chat = chat_completions_headers_override.get()
        outer_responses = responses_headers_override.get()
        with pytest.raises(ValueError, match="inner failure"):
            with agents_sdk_user_agent_override():
                raise ValueError("inner failure")
        assert chat_completions_headers_override.get() is outer_chat
        assert responses_headers_override.get() is outer_responses

    assert chat_completions_headers_override.get() is original_headers
    assert responses_headers_override.get() is original_headers


async def test_task_cancellation_restores_headers_before_outer_cleanup():
    entered = asyncio.Event()
    restored = []
    parent_chat = chat_completions_headers_override.get()
    parent_responses = responses_headers_override.get()

    async def worker():
        chat_headers = {"User-Agent": "worker-chat"}
        response_headers = {"User-Agent": "worker-responses"}
        chat_token = chat_completions_headers_override.set(chat_headers)
        response_token = responses_headers_override.set(response_headers)
        try:
            try:
                with agents_sdk_user_agent_override():
                    entered.set()
                    await asyncio.Event().wait()
            finally:
                restored.append(
                    chat_completions_headers_override.get() is chat_headers
                    and responses_headers_override.get() is response_headers
                )
        finally:
            chat_completions_headers_override.reset(chat_token)
            responses_headers_override.reset(response_token)

    task = asyncio.create_task(worker())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert restored == [True]
    assert chat_completions_headers_override.get() is parent_chat
    assert responses_headers_override.get() is parent_responses


async def test_closing_generator_in_another_context_does_not_raise(original_headers):
    chat_token = chat_completions_headers_override.set(original_headers)
    responses_token = responses_headers_override.set(original_headers)

    async def stream():
        with agents_sdk_user_agent_override():
            yield "event"

    events = stream()
    try:
        assert await anext(events) == "event"
        await asyncio.create_task(events.aclose())
    finally:
        await events.aclose()
        chat_completions_headers_override.reset(chat_token)
        responses_headers_override.reset(responses_token)


async def test_respond_failure_restores_headers_after_streaming(original_headers):
    async def responder(*args) -> AsyncIterator[ThreadStreamEvent]:
        yield ProgressUpdateEvent(text="Starting")
        raise ValueError("respond failed")

    with make_server(responder) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Hello")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )

    assert any(event.type == "error" for event in events)
    assert chat_completions_headers_override.get() is original_headers
    assert responses_headers_override.get() is original_headers
