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

from tau_coding.kernel import process as process_module
from tau_coding.kernel.frames import encode_frame
from tau_coding.kernel.process import MAX_CONTEXT_INDEX_BYTES, ProcessKernel, looks_secret
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


class _SlowSubcall:
    """Answers like a provider would: correctly, and not instantly."""

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self.calls = 0

    async def __call__(self, prompt: str, model: str | None) -> SubcallOutcome:
        self.calls += 1
        await asyncio.sleep(self._delay)
        return SubcallOutcome(text=f"<<{prompt}>>", model=model or "fake", metadata={})


class _SlowRecursion:
    """A child kernel takes turns, and turns take time."""

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self.calls = 0

    async def __call__(self, task: str, model: str | None) -> SubcallOutcome:
        self.calls += 1
        await asyncio.sleep(self._delay)
        return SubcallOutcome(text=f"<<{task}>>", model=model or "fake", metadata={})


async def test_the_host_model_latency_is_not_charged_to_the_cell(tmp_path: Path) -> None:
    """A cell is billed for its own code, not for how long the provider takes.

    Two *sequential* sub-calls served in 0.6 s each used to kill a 1 s cell with
    ``timed_out``/``state_lost`` and take the namespace down with them, even though
    the cell never ran past its own code. Recursion is serial by construction, so
    on the 30 s default a handful of slow turns did the same thing to a real run.
    """

    slow = _SlowSubcall(0.6)
    async with _running(tmp_path, subcall=slow) as kernel:
        warm = await kernel.execute("corpus = 'x' * 10\nprint('ready')")
        assert warm.ok

        cell = await kernel.execute(
            'a = llm_query("one")\nb = llm_query("two")\nprint(len(a), len(b))', timeout=1.0
        )
        assert cell.ok, cell.error
        assert not cell.timed_out and not cell.state_lost
        assert cell.stdout == "7 7\n"
        assert slow.calls == 2

        survivor = await kernel.execute("print(len(corpus))")
        assert survivor.ok
        assert survivor.stdout == "10\n"


async def test_a_childs_turns_are_not_charged_to_the_parents_cell(tmp_path: Path) -> None:
    slow = _SlowRecursion(0.6)
    async with _running(tmp_path, rlm=slow) as kernel:
        cell = await kernel.execute(
            'first = rlm_query("scan")\nsecond = rlm_query("verify")\nprint(first, second)',
            timeout=1.0,
        )
        assert cell.ok, cell.error
        assert not cell.timed_out and not cell.state_lost
        assert cell.stdout == "<<scan>> <<verify>>\n"
        assert slow.calls == 2


async def test_host_time_is_credited_only_up_to_a_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The credit keeps a cell alive, it does not make it immortal."""

    monkeypatch.setattr(process_module, "MAX_CREDITED_HOST_SECONDS", 0.5)
    slow = _SlowSubcall(0.6)
    async with _running(tmp_path, subcall=slow) as kernel:
        cell = await kernel.execute(
            'a = llm_query("one")\n'
            'b = llm_query("two")\n'
            'c = llm_query("three")\n'
            'd = llm_query("four")\n'
            "print(a, b, c, d)",
            timeout=1.0,
        )
        assert not cell.ok
        assert cell.timed_out and cell.state_lost
        assert "not counted" in (cell.error or "")


async def test_a_kernel_without_a_subcall_handler_says_so(tmp_path: Path) -> None:
    async with _running(tmp_path) as kernel:
        result = await kernel.execute("llm_query('anything')")
        assert not result.ok
        assert "no sub-call handler is wired" in (result.error or "")


class _FakeContext:
    """A stand-in for the session rows behind a ``context_index``/``open`` handle.

    The point of the fake is what it *does not* do: it hands the kernel a path, never
    the history itself, so the tests can watch the bytes arrive through the file.
    """

    def __init__(
        self,
        *,
        path: Path = Path("/tmp/tau-context-fake.jsonl"),
        index_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.index_query: list[str | None] = []
        self.open_refs: list[tuple[str, ...]] = []
        self.path = path
        self.index_rows = (
            index_rows
            if index_rows is not None
            else [
                {
                    "address": 1,
                    "entry_id": "e1",
                    "role": "user",
                    "bytes": 11,
                    "preview": "first ask",
                },
                {
                    "address": 2,
                    "entry_id": "e2",
                    "role": "assistant",
                    "bytes": 13,
                    "preview": "first answer",
                },
            ]
        )

    async def index(self, query: str | None) -> dict[str, Any]:
        self.index_query.append(query)
        return {"rows": list(self.index_rows), "total": len(self.index_rows), "omitted": 0}

    async def open(self, refs: tuple[str, ...]) -> dict[str, Any]:
        self.open_refs.append(refs)
        path = self.path
        path.write_text(
            '{"address": 1, "text": "the transcript, host-written"}\n',
            encoding="utf-8",
        )
        return {"path": str(path), "written": 1, "omitted": 0, "note": "1 entry"}


async def test_context_handles_round_trip_from_inside_a_cell(tmp_path: Path) -> None:
    context = _FakeContext(path=tmp_path / "slice.jsonl")
    async with _running(tmp_path, context=context) as kernel:
        result = await kernel.execute(
            "info = context_index('answer')\n"
            "print(info['total'])\n"
            "path = context_open('#1-#2')\n"
            "print(open(path).read().strip())\n"
            "print(SHOW_VARS()['context_open'] == 'function')"
        )
        assert result.ok, result.error
        assert "2\n" in result.stdout
        assert "the transcript, host-written" in result.stdout
        assert context.index_query == ["answer"]
        assert context.open_refs == [("#1-#2",)]
        assert kernel.context_reads == 2


async def test_a_kernel_without_a_context_source_says_so(tmp_path: Path) -> None:
    async with _running(tmp_path) as kernel:
        result = await kernel.execute("context_index()")
        assert not result.ok
        assert "no session context is wired" in (result.error or "")
        assert kernel.alive


async def test_context_refs_are_refused_before_the_source_sees_them(tmp_path: Path) -> None:
    context = _FakeContext()
    async with _running(tmp_path, context=context) as kernel:
        result = await kernel.execute("context_open(*['#1'] * 40)")
        assert not result.ok
        assert "the limit is 32" in (result.error or "")
        assert context.open_refs == []
        assert kernel.alive

        refused = await kernel.execute("context_open(7)")
        assert not refused.ok
        assert "references such as" in (refused.error or "")
        assert context.open_refs == []


async def test_an_oversized_index_is_trimmed_with_the_drop_counted(tmp_path: Path) -> None:
    context = _FakeContext(
        index_rows=[
            {
                "address": index,
                "entry_id": f"e{index}",
                "role": "user",
                "bytes": 5,
                "preview": "z" * 200,
            }
            for index in range(1, 1201)
        ]
    )
    async with _running(tmp_path, context=context) as kernel:
        assert len(context.index_rows) * 200 > MAX_CONTEXT_INDEX_BYTES
        result = await kernel.execute(
            "info = context_index()\n"
            "print(len(info['rows']), info['omitted'], info.get('note', ''))"
        )
        assert result.ok, result.error
        listed, dropped = result.stdout.split()[0], int(result.stdout.split()[1])
        assert int(listed) < 1200
        assert dropped >= 1200 - int(listed)
        assert "trimmed" in result.stdout


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


class _FakeRecursion:
    """A stand-in for the host-side recursion runner the code-mode tool injects.

    The transport is what is under test here: a recursion request must travel the
    same channel as a sub-call, and a failure inside one must not take the kernel
    with it.
    """

    def __init__(self, failures: tuple[str, ...] = ()) -> None:
        self.tasks: list[tuple[str, str | None]] = []
        self.failures = failures

    async def __call__(self, task: str, model: str | None = None) -> SubcallOutcome:
        self.tasks.append((task, model))
        if task in self.failures:
            raise RuntimeError(f"no child for {task!r}")
        return SubcallOutcome(text=f"<<{task}>>", model=model or "fake", metadata={"rlm_depth": 1})


async def test_recursion_names_exist_only_when_a_handler_is_wired(tmp_path: Path) -> None:
    async with _running(tmp_path) as unwired:
        assert "rlm_query" not in unwired.binding_names
        result = await unwired.execute("print(rlm_query('do the thing'))")
        assert not result.ok
        assert "NameError" in (result.error or "")

    async with _running(tmp_path, rlm=_FakeRecursion()) as wired:
        state = await wired.state()
        assert "rlm_query" in state.bindings
        assert "rlm_query_batched" in state.bindings
        answered = await wired.execute("print(rlm_query('do the thing'))")
        assert answered.ok, answered.error
        assert answered.stdout == "<<do the thing>>\n"
        assert wired.recursions_used == 1


async def test_a_recursion_failure_costs_the_call_not_the_kernel(tmp_path: Path) -> None:
    recursion = _FakeRecursion(failures=("impossible",))
    async with _running(tmp_path, rlm=recursion) as kernel:
        failed = await kernel.execute("print(rlm_query('impossible'))")
        assert not failed.ok
        assert "recursion failed" in (failed.error or "")
        assert kernel.alive
        survivor = await kernel.execute("print('still here')")
        assert survivor.ok


async def test_a_batch_of_recursions_keeps_its_slots(tmp_path: Path) -> None:
    recursion = _FakeRecursion(failures=("bad",))
    async with _running(tmp_path, rlm=recursion) as kernel:
        result = await kernel.execute("print(rlm_query_batched(['good', 'bad']))")
        assert result.ok, result.error
        assert result.stdout == "['<<good>>', None]\n"
        assert kernel.recursions_used == 2


async def test_recursion_and_subcalls_share_one_channel(tmp_path: Path) -> None:
    async with _running(tmp_path, subcall=_FakeSubcall(), rlm=_FakeRecursion()) as kernel:
        mixed = await kernel.execute("print(llm_query('ask'), rlm_query('delegate'))")
        assert mixed.ok, mixed.error
        assert mixed.stdout == "<<ask>> <<delegate>>\n"
        assert kernel.subcalls_used == 1
        assert kernel.recursions_used == 1


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
