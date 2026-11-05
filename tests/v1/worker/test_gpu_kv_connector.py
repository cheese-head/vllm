# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu.kv_connector as kv_connector_module
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


def _make_connector(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> ActiveKVConnector:
    backend = Mock()
    backend.handle_preemptions.side_effect = lambda _: events.append("handle")
    backend.record_load_destination_ready.side_effect = lambda _: events.append("ready")
    backend.bind_connector_metadata.side_effect = lambda _: events.append("bind")
    backend.start_load_kv.side_effect = lambda *_args, **_kwargs: events.append("start")
    backend.wait_for_save.side_effect = lambda: events.append("wait")
    backend.get_transfer_results.return_value = KVConnectorTransferResults()
    backend.get_block_ids_with_load_errors.return_value = set()
    backend.get_kv_connector_stats.return_value = None
    backend.get_kv_connector_kv_cache_events.return_value = None
    backend.build_connector_worker_meta.return_value = None
    backend.clear_connector_metadata.side_effect = lambda: events.append("clear")
    monkeypatch.setattr(kv_connector_module, "get_kv_transfer_group", lambda: backend)
    monkeypatch.setattr(
        kv_connector_module, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(kv_connector_module, "get_forward_context", object)

    kv_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_consumer",
        kv_buffer_device="cpu",
    )
    connector = ActiveKVConnector(SimpleNamespace(kv_transfer_config=kv_config), {})
    events.clear()
    return connector


def _scheduler_output(has_sync_kv_loads: bool) -> SimpleNamespace:
    return SimpleNamespace(
        kv_connector_metadata=object(),
        finished_req_ids=set(),
        has_sync_kv_loads=has_sync_kv_loads,
    )


@pytest.mark.parametrize("has_sync_kv_loads", [False, True])
def test_load_start_phase(
    monkeypatch: pytest.MonkeyPatch,
    has_sync_kv_loads: bool,
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)
    output = _scheduler_output(has_sync_kv_loads)

    request_indices = torch.tensor([3, 1])
    request_ids = ["first", "second"]
    attn_metadata = {"layer": object()}
    connector.before_cache_update(output)
    events.append("cache-writers")
    connector.after_cache_update(output)
    connector.pre_forward(
        output,
        request_state_indices=request_indices,
        request_ids=request_ids,
        attn_metadata=attn_metadata,
    )
    assert events == (
        ["handle", "cache-writers", "ready", "bind", "start"]
        if has_sync_kv_loads
        else ["handle", "cache-writers", "ready", "bind"]
    )

    connector.post_forward(set())
    assert events == [
        "handle",
        "cache-writers",
        "ready",
        "bind",
        "start",
        "wait",
        "clear",
    ]

    kwargs = connector.kv_connector.start_load_kv.call_args.kwargs
    assert kwargs["request_state_indices"] is request_indices
    assert kwargs["request_ids"] is request_ids
    assert kwargs["attn_metadata"] is attn_metadata

    # A subsequent step without a forward must not reuse the prior batch.
    connector.no_forward(_scheduler_output(False))
    assert connector.kv_connector.start_load_kv.call_count == 2
    assert connector.kv_connector.start_load_kv.call_args.kwargs == {}


def test_no_forward_starts_deferred_load_once(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)

    output = _scheduler_output(False)
    connector.before_cache_update(output)
    connector.after_cache_update(output)
    connector.no_forward(output)

    assert events == ["handle", "ready", "bind", "start", "wait", "clear"]


@pytest.mark.parametrize("writer", ["zero", "cow"])
@pytest.mark.parametrize("has_sync_kv_loads", [False, True])
def test_actual_v2_execute_protects_cache_before_writers(
    monkeypatch, writer, has_sync_kv_loads
):
    """Exercise the real runner call order without allocating a GPU runner."""
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)
    output = _scheduler_output(has_sync_kv_loads)
    output.total_num_scheduled_tokens = 0
    source = Path(__file__).resolve().parents[3] / "vllm/v1/worker/gpu/model_runner.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner"
    )
    execute = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "execute_model"
    )
    execute.decorator_list = []
    namespace: dict[str, Any] = {}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            execute,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    runner = SimpleNamespace(
        kv_connector=connector,
        update_pp_decode_requests=lambda: None,
        finish_requests=lambda _: None,
        free_states=lambda _: None,
        add_requests=lambda _: None,
        update_requests=lambda _: events.append(writer),
        block_tables=SimpleNamespace(apply_staged_writes=lambda: None),
        aux_output_connector=None,
        _merge_ec_connector_no_forward=lambda _, result: result,
    )
    namespace["execute_model"](runner, output)
    assert events == ["handle", writer, "ready", "bind", "start", "wait", "clear"]
