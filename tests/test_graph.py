"""Regression suite for the frame-graph compiler.

Every test calls at least one public stub in :mod:`fg.graph`.  The suite is
the specification: pass dependencies, barrier minimality, alias liveness and
pool capacity, dead-code reachability, state derivation, cycle reporting and
batch isolation are all exact (tolerance zero).
"""

import pytest

from fg.graph import (
    add_pass,
    alias_memory,
    compile_batch,
    cull_graph,
    derive_state,
    insert_barriers,
    toposort,
    validate_graph,
)


def _pass(name, reads=(), writes=()):
    return {"name": name, "reads": list(reads), "writes": list(writes)}


def _graph(passes, resources):
    return {"passes": list(passes), "resources": dict(resources)}


def _resource(size=8, access=None):
    entry = {"size": size}
    if access is not None:
        entry["access"] = access
    return entry


def test_toposort_uses_read_write_dependencies_not_registration_order():
    # "shade" is registered first, but it consumes the frame written by
    # "raster".  Registration order is therefore not a valid compile order.
    graph = _graph(
        [
            _pass("shade", reads=["frame"], writes=["color"]),
            _pass("raster", writes=["frame"]),
        ],
        {"frame": _resource(), "color": _resource()},
    )

    assert toposort(graph) == ["raster", "shade"]


def test_read_then_write_requires_a_barrier():
    order = ["sample", "writeback"]
    accesses = {
        "sample": {"reads": ["depth"], "writes": []},
        "writeback": {"reads": [], "writes": ["depth"]},
    }

    barriers = insert_barriers(order, accesses)

    assert len(barriers) == 1
    assert barriers[0] == {
        "before": "writeback",
        "resource": "depth",
        "from": "read",
        "to": "write",
    }


def test_consecutive_reads_do_not_add_extra_barriers():
    # read -> read is free; write -> read is a real transition.  A correct
    # compiler emits barriers only before "write_c" and "read_d".
    order = ["read_a", "read_b", "write_c", "read_d"]
    accesses = {
        "read_a": {"reads": ["color"], "writes": []},
        "read_b": {"reads": ["color"], "writes": []},
        "write_c": {"reads": [], "writes": ["color"]},
        "read_d": {"reads": ["color"], "writes": []},
    }

    barriers = insert_barriers(order, accesses)

    assert [barrier["before"] for barrier in barriers] == ["write_c", "read_d"]
    assert [(barrier["from"], barrier["to"]) for barrier in barriers] == [
        ("read", "write"),
        ("write", "read"),
    ]


def test_alias_keeps_writable_history_live_until_its_last_read():
    # "history" is written at pass 0 and still read at pass 3.  "scratch"
    # writes at pass 1, inside that read-after-write window, so the two must
    # not share bytes even though their *write* points differ.
    resources = [
        {"name": "history", "size": 8, "reads": [3], "writes": [0]},
        {"name": "scratch", "size": 8, "reads": [], "writes": [1]},
    ]

    allocation = alias_memory(resources, pool_size=16)

    history = allocation["history"]
    scratch = allocation["scratch"]
    history_end = history["offset"] + history["size"]
    scratch_end = scratch["offset"] + scratch["size"]
    assert history["offset"] != scratch["offset"]
    assert history_end <= scratch["offset"] or scratch_end <= history["offset"]


def test_alias_pool_capacity_is_hard():
    # Both resources are live at pass 1, so they need disjoint 8-byte spans.
    # Eight bytes cannot hold a 16-byte peak; no offset may exceed the pool.
    resources = [
        {"name": "a", "size": 8, "reads": [0, 1], "writes": []},
        {"name": "b", "size": 8, "reads": [1, 2], "writes": []},
    ]

    with pytest.raises(ValueError, match="pool"):
        alias_memory(resources, pool_size=8)


def test_cull_keeps_source_and_drops_unused_output_pass():
    graph = _graph(
        [
            _pass("source", writes=["gbuffer"]),
            _pass("shade", reads=["gbuffer"], writes=["unused"]),
            _pass("output", reads=["gbuffer"], writes=["final"]),
        ],
        {
            "gbuffer": _resource(),
            "unused": _resource(),
            "final": _resource(),
        },
    )

    culled = cull_graph(graph, used=["final"])
    names = [item["name"] for item in culled["passes"]]

    assert names == ["source", "output"]
    assert "shade" not in names
    # The input graph is not mutated.
    assert [item["name"] for item in graph["passes"]] == [
        "source",
        "shade",
        "output",
    ]


def test_derive_state_read_plus_write_is_readwrite():
    accesses = {
        "sample": {"reads": ["albedo"], "writes": []},
        "resolve": {"reads": ["albedo"], "writes": []},
        "light": {"reads": [], "writes": ["albedo"]},
        "clear": {"reads": ["depth"], "writes": ["depth"]},
        "readonly": {"reads": ["mask"], "writes": []},
        "writeonly": {"reads": [], "writes": ["target"]},
    }

    states = derive_state(accesses)

    assert states["albedo"] == "readwrite"
    assert states["depth"] == "readwrite"
    assert states["mask"] == "read"
    assert states["target"] == "write"


def test_validate_rejects_cycle_and_names_cycle_nodes():
    graph = _graph(
        [
            _pass("ping", reads=["b"], writes=["a"]),
            _pass("pong", reads=["a"], writes=["b"]),
        ],
        {"a": _resource(), "b": _resource()},
    )

    diagnostics = validate_graph(graph)
    joined = "\n".join(diagnostics)

    assert diagnostics
    assert "cycle" in joined.lower()
    assert "ping" in joined
    assert "pong" in joined

    # Validation also reports the structural problems that are not cycles.
    structural = _graph(
        [
            _pass("dup", reads=["ghost"]),
            _pass("dup"),
            _pass("writer", writes=["readonly"]),
        ],
        {"readonly": _resource(access="read")},
    )
    structural_diagnostics = validate_graph(structural)
    assert any(
        item.startswith("duplicate pass: dup") for item in structural_diagnostics
    )
    assert any(
        "undeclared resource: ghost" in item for item in structural_diagnostics
    )
    assert any(
        "read/write conflict" in item for item in structural_diagnostics
    )


def test_compile_batch_resets_resource_state_per_graph():
    graph_a = _graph(
        [_pass("read_frame", reads=["tex"])],
        {"tex": _resource()},
    )
    graph_b = _graph(
        [_pass("write_frame", writes=["tex"])],
        {"tex": _resource()},
    )

    compiled = compile_batch({"a": graph_a, "b": graph_b}, {"pool_size": 32})

    assert compiled["a"]["order"] == ["read_frame"]
    assert compiled["b"]["order"] == ["write_frame"]
    # Graph B starts with an empty state table: a write as first use needs no
    # barrier.  Reusing graph A's trailing "read" state would insert one.
    assert compiled["a"]["barriers"] == []
    assert compiled["b"]["barriers"] == []


def test_add_pass_rejects_undeclared_write_resource():
    graph = _graph([], {"shadow_map": _resource(size=16)})

    with pytest.raises(KeyError):
        add_pass(
            graph,
            "shadow",
            reads=["shadow_map"],
            writes=["shadow_mpa"],  # typo: not declared
        )

    assert "shadow_mpa" not in graph["resources"]
    assert graph["passes"] == []


def test_simple_chain_toposort_is_unique():
    graph = _graph(
        [
            _pass("sink", reads=["b"]),
            _pass("middle", reads=["a"], writes=["b"]),
            _pass("source", writes=["a"]),
        ],
        {"a": _resource(), "b": _resource()},
    )

    assert toposort(graph) == ["source", "middle", "sink"]


def test_read_only_sequence_has_zero_barriers():
    order = ["r1", "r2", "r3"]
    accesses = {
        name: {"reads": ["color"], "writes": []}
        for name in order
    }

    assert insert_barriers(order, accesses) == []


def test_alias_reuses_memory_across_disjoint_lifetimes():
    resources = [
        {"name": "a", "size": 8, "reads": [0], "writes": [1]},
        {"name": "b", "size": 8, "reads": [2], "writes": [3]},
    ]

    allocation = alias_memory(resources, pool_size=16)

    assert allocation == {
        "a": {"offset": 0, "size": 8},
        "b": {"offset": 0, "size": 8},
    }
    # Alias is legal only because the closed live intervals do not overlap.
    a_live_end = max(resources[0]["reads"] + resources[0]["writes"])
    b_live_start = min(resources[1]["reads"] + resources[1]["writes"])
    assert a_live_end < b_live_start
    for entry in allocation.values():
        assert entry["offset"] >= 0
        assert entry["offset"] + entry["size"] <= 16


def test_toposort_raises_on_cycle_and_names_nodes():
    graph = _graph(
        [
            _pass("alpha", reads=["b"], writes=["a"]),
            _pass("beta", reads=["a"], writes=["b"]),
        ],
        {"a": _resource(), "b": _resource()},
    )

    with pytest.raises(ValueError) as excinfo:
        toposort(graph)

    message = str(excinfo.value)
    assert "cycle" in message.lower()
    assert "alpha" in message
    assert "beta" in message


def test_compile_batch_equals_individual_compilations():
    config = {"pool_size": 32}
    graph_a = _graph(
        [
            _pass("write_a", writes=["a"]),
            _pass("read_a", reads=["a"]),
        ],
        {"a": _resource()},
    )
    graph_b = _graph(
        [
            _pass("write_b", writes=["b"]),
            _pass("read_b", reads=["b"]),
        ],
        {"b": _resource()},
    )

    together = compile_batch({"a": graph_a, "b": graph_b}, config)

    assert together["a"] == compile_batch({"a": graph_a}, config)["a"]
    assert together["b"] == compile_batch({"b": graph_b}, config)["b"]
