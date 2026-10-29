"""Kernel process tests: the boundary a code-mode harness stands on.

These spawn a real Python child, which is the point -- the guarantees under test
(credential scrubbing, byte-bounded output, kill-on-overrun, the token handshake)
only exist at the process boundary, and a fake child would test the fake.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from tau_coding.kernel.frames import encode_frame
from tau_coding.kernel.process import ProcessKernel, looks_secret
from tau_coding.kernel.protocol import KernelLimits, KernelRejectedError, SubcallOutcome

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeSubcall:
    """A stand-in for the provider call a real code-mode tool will inject."""

    def __init__(self, failures: tuple[str, ...] = ()) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.failures = failures

    async def __call__(self, prompt: str, model: str | None) -> SubcallOutcome:
        self.calls.append((prompt, model))
        if prompt in self.failures:
            raise RuntimeError(f"no route for {prompt!r}")
        return SubcallOutcome(text=f"<<{prompt}>>", model=model or "fake", metadata={"tokens": 7})


@asynccontextmanager
async def _running(tmp_path: Path, **kwargs: Any) -> AsyncIterator[ProcessKernel]:
    kernel = ProcessKernel(cwd=tmp_path, **kwargs)
    try:
        await kernel.start()
        yield kernel
    finally:
        await kernel.close()


async def test_the_namespace_persists_across_cells(tmp_path: Path) -> None:
    async with _running(tmp_path) as kernel:
        first = await kernel.execute("rows = [f'r{i}' for i in range(500)]\nprint(len(rows))")
        assert first.ok
        assert first.stdout == "500\n"
        second = await kernel.execute("print(rows[0], rows[-1])")
        assert second.ok
        assert second.stdout == "r0 r499\n"


async def test_ret_carries_data_and_names_why_the_rest_cannot(tmp_path: Path) -> None:
    async with _running(tmp_path) as kernel:
        good = await kernel.execute("ret({'partitions': 4, 'ok': True})")
        assert good.returned
        assert good.ret == {"partitions": 4, "ok": True}
        assert good.ret_error is None

        bad = await kernel.execute("import io\nret({'handle': io.StringIO()})")
        assert bad.returned
        assert bad.ret is None
        assert "needs JSON data" in (bad.ret_error or "")
        assert "StringIO" in (bad.ret_error or "")


async def test_a_raising_cell_does_not_end_the_kernel(tmp_path: Path) -> None:
    async with _running(tmp_path) as kernel:
        boom = await kernel.execute("raise ValueError('boom')")
        assert not boom.ok
        assert not boom.state_lost
        assert "ValueError: boom" in (boom.error or "")
        assert kernel.alive
        keeper = await kernel.execute("answer = 7")
        assert keeper.ok


async def test_output_is_capped_and_the_loss_is_counted(tmp_path: Path) -> None:
    async with _running(tmp_path, max_cell_output_bytes=1024) as kernel:
        result = await kernel.execute("print('x' * 5000)")
        assert len(result.stdout.encode("utf-8")) == 1024
        assert result.stdout_omitted_bytes == 5001 - 1024
        assert result.omitted_bytes == result.stdout_omitted_bytes + result.stderr_omitted_bytes


async def test_an_overrun_cell_is_killed_and_says_state_is_lost(tmp_path: Path) -> None:
    async with _running(tmp_path, cell_timeout_seconds=1.0) as kernel:
        before = await kernel.execute("precious = 'kept'")
        assert before.ok

        hung = await kernel.execute("while True:\n    pass")
        assert hung.timed_out
        assert hung.state_lost
        assert not hung.ok
        assert kernel.restarts == 0  # the replacement restart is lazy
        assert not kernel.alive

        after = await kernel.execute("print('precious' in SHOW_VARS())")
        assert after.ok
        assert after.stdout == "False\n"
        assert kernel.restarts == 1


async def test_subcalls_round_trip_from_inside_a_cell(tmp_path: Path) -> None:
    subcall = _FakeSubcall()
    async with _running(tmp_path, subcall=subcall, max_concurrent_subcalls=3) as kernel:
        result = await kernel.execute(
            "print(llm_query('summarise chunk one'))\n"
            "print(llm_query_batched(['a', 'b'], model='small'))"
        )
        assert result.ok, result.error
        assert "<<summarise chunk one>>" in result.stdout
        assert "['<<a>>', '<<b>>']" in result.stdout
        assert [(prompt, model) for prompt, model in subcall.calls] == [
            ("summarise chunk one", None),
            ("a", "small"),
            ("b", "small"),
        ]
        assert kernel.subcalls_used == 3


async def test_one_failing_subcall_leaves_a_hole_not_a_crash(tmp_path: Path) -> None:
    """A batch must not sink: ``None`` marks the prompt the host could not serve."""
    subcall = _FakeSubcall(failures=("bad",))
    async with _running(tmp_path, subcall=subcall) as kernel:
        result = await kernel.execute("print(llm_query_batched(['good', 'bad']))")
        assert result.ok, result.error
        assert result.stdout == "['<<good>>', None]\n"

        single = await kernel.execute("llm_query('bad')")
        assert not single.ok
        assert "sub-call failed" in (single.error or "")
        assert "no route for" in (single.error or "")
        assert kernel.alive


async def test_a_kernel_without_a_subcall_handler_says_so(tmp_path: Path) -> None:
    async with _running(tmp_path) as kernel:
        result = await kernel.execute("llm_query('anything')")
        assert not result.ok
        assert "no sub-call handler is wired" in (result.error or "")


async def test_credentials_do_not_cross_into_the_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-live-secret")
    monkeypatch.setenv("MYAPP_SESSION_TOKEN", "token-secret")
    monkeypatch.setenv("DATABASE_PASSWORD", "pw-secret")
    monkeypatch.setenv("TAU_HOME", "/should/not/travel")
    monkeypatch.setenv("MY_BUILD_DIR", str(tmp_path))

    async with _running(tmp_path) as kernel:
        result = await kernel.execute(
            "import os\n"
            "keys = ['ANTHROPIC_API_KEY', 'MYAPP_SESSION_TOKEN', 'DATABASE_PASSWORD', "
            "'TAU_HOME']\n"
            "print([os.environ.get(k) for k in keys])\n"
            "print(os.environ.get('MY_BUILD_DIR') is not None)"
        )
        assert result.ok, result.error
        assert "[None, None, None, None]" in result.stdout
        assert "True" in result.stdout
        assert "sk-live-secret" not in result.stdout


@pytest.mark.parametrize(
    ("name", "secret"),
    [
        ("OPENAI_API_KEY", True),
        ("XAI_API_KEY", True),
        ("AWS_SECRET_ACCESS_KEY", True),
        ("MYAPP_SESSION_TOKEN", True),
        ("DATABASE_PASSWORD", True),
        ("GITHUB_TOKEN", True),
        ("PATH", False),
        ("HOME", False),
        ("MY_BUILD_DIR", False),
        ("PYTHONPATH", False),
    ],
)
def test_the_credential_filter_names_what_it_refuses(name: str, secret: bool) -> None:
    assert looks_secret(name) is secret


async def test_show_vars_inventories_names_without_values(tmp_path: Path) -> None:
    async with _running(tmp_path) as kernel:
        await kernel.execute("haystack = ['needle-%d' % i for i in range(3000)]")
        result = await kernel.execute("print(SHOW_VARS())")
        assert "haystack" in result.stdout
        assert "needle-1" not in result.stdout

        state = await kernel.state()
        rows = {row.name: row for row in state.variables}
        assert rows["haystack"].length == 3000
        assert rows["haystack"].type == "list"
        assert rows["haystack"].shallow_bytes > 0


async def test_bind_hands_the_kernel_a_path_not_a_payload(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("line one\nneedle here\nline three\n", encoding="utf-8")
    async with _running(tmp_path) as kernel:
        await kernel.bind("corpus", corpus)
        result = await kernel.execute(
            "with open(corpus, encoding='utf-8') as handle:\n"
            "    text = handle.read()\n"
            "ret([line for line in text.splitlines() if 'needle' in line])"
        )
        assert result.ok, result.error
        assert result.ret == ["needle here"]

        with pytest.raises(KernelRejectedError, match="cannot bind"):
            await kernel.bind("llm_query", corpus)


async def test_reset_drops_the_namespace_but_keeps_the_bindings(tmp_path: Path) -> None:
    async with _running(tmp_path, subcall=_FakeSubcall()) as kernel:
        await kernel.execute("doomed = 1")
        await kernel.reset()
        gone = await kernel.execute("print(doomed)")
        assert not gone.ok
        assert "NameError" in (gone.error or "")
        survivor = await kernel.execute("print(llm_query('still here'))")
        assert survivor.ok
        assert survivor.stdout == "<<still here>>\n"


async def test_a_foreign_connection_is_refused(tmp_path: Path) -> None:
    """The port is reachable by any local process, so the token is the gate."""
    async with _running(tmp_path) as kernel:
        reader, writer = await asyncio.open_connection("127.0.0.1", kernel.port)
        try:
            writer.write(encode_frame({"op": "hello", "token": "guessed"}))
            await writer.drain()
            assert await reader.read(64) == b""
        finally:
            writer.close()
            await writer.wait_closed()

        assert kernel.alive
        kept = await kernel.execute("still = 'working'")
        assert kept.ok


async def test_limits_are_enforced_inside_the_child(tmp_path: Path) -> None:
    async with _running(tmp_path, limits=KernelLimits(cpu_seconds=7, open_files=64)) as kernel:
        result = await kernel.execute(
            "import resource\n"
            "print(resource.getrlimit(resource.RLIMIT_CPU))\n"
            "print(resource.getrlimit(resource.RLIMIT_NOFILE)[0])"
        )
        assert result.ok, result.error
        fields = result.stdout.split()
        assert fields[:2] == ["(7,", "7)"]
        assert fields[-1] == "64"
        assert kernel.limits_notes == ()


async def test_close_releases_the_process(tmp_path: Path) -> None:
    kernel = ProcessKernel(cwd=tmp_path)
    await kernel.start()
    pid = kernel.pid
    assert pid is not None
    await kernel.close()

    assert not kernel.alive
    assert kernel.pid is None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)

    state = await kernel.state()
    assert not state.alive
    assert state.note
