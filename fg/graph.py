"""Frame-graph compiler stubs.

Graph model
-----------

``graph`` is a plain dict::

    {
        "passes": [
            {"name": "gbuffer", "reads": ["camera"], "writes": ["normal"]},
            ...
        ],
        "resources": {
            "normal": {"size": 8, "access": "readwrite"},
            ...
        },
    }

``resources[*]["access"]`` is optional.  It is one of ``"read"``,
``"write"`` or ``"readwrite"`` and defaults to ``"readwrite"``.  It describes
what passes are allowed to do to the resource; it is not a description of the
runtime state (that state is derived by :func:`derive_state`).

Conventions
-----------
* Pass names and resource names are strings.  Pass registration order is the
  order of ``graph["passes"]``.
* Pass indices used by :func:`alias_memory` are zero-based positions in a
  compiled pass order.  Live intervals are closed: ``[start, end]``; two
  resources that touch at one pass overlap at that pass.
* There is no tolerance anywhere.  Offsets, sizes, barrier counts and states
  are exact values; comparisons must not use epsilon.
"""

import heapq


def _dedupe(names):
    """Return ``names`` as a list, duplicates removed, first-seen order kept."""
    seen = set()
    result = []
    for name in names:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def add_pass(graph, name, reads, writes):
    """Register one pass and its resource accesses in ``graph``.

    ``reads`` and ``writes`` are iterables of resource names.  They are stored
    as lists with duplicates removed while preserving first-seen order.  The
    function mutates ``graph`` and returns that same dict.

    A pass may reference only resources already present in
    ``graph["resources"]``.  This function never creates a resource entry: an
    undeclared name usually means a typo, and silently manufacturing the
    resource would hide the mistake and change the graph's memory plan.

    Raises:
        ValueError: ``name`` is empty or a pass with that name already exists.
        KeyError: any name in ``reads`` or ``writes`` is undeclared.

    The graph must be unchanged when registration fails.
    """
    if not name:
        raise ValueError("pass name must not be empty")
    if any(existing["name"] == name for existing in graph["passes"]):
        raise ValueError("duplicate pass: %s" % name)

    read_list = _dedupe(reads)
    write_list = _dedupe(writes)

    declared = graph["resources"]
    for resource in read_list + write_list:
        if resource not in declared:
            raise KeyError("undeclared resource: %s" % resource)

    graph["passes"].append(
        {"name": name, "reads": read_list, "writes": write_list}
    )
    return graph


def _dependency_edges(passes):
    """Build data-dependency edges between pass names.

    Returns ``(edges, registration)`` where ``edges`` maps a pass name to the
    set of pass names that must come after it, and ``registration`` maps each
    pass name to its registration index (first occurrence wins).
    """
    registration = {}
    for index, item in enumerate(passes):
        registration.setdefault(item["name"], index)

    edges = {name: set() for name in registration}

    writers = {}
    readers = {}
    for item in passes:
        name = item["name"]
        for resource in item.get("writes", ()):
            writers.setdefault(resource, []).append(name)
        for resource in item.get("reads", ()):
            readers.setdefault(resource, []).append(name)

    for resource, resource_writers in writers.items():
        # Every writer must come before every reader of the resource.
        for reader in readers.get(resource, ()):
            for writer in resource_writers:
                if writer != reader:
                    edges[writer].add(reader)
        # Writers of the same resource keep their registration order.
        ordered = sorted(set(resource_writers), key=registration.get)
        for earlier, later in zip(ordered, ordered[1:]):
            edges[earlier].add(later)

    return edges, registration


def _find_cycle(edges, nodes):
    """Return one cycle as a list of names ``[a, b, ..., a]`` within ``nodes``.

    ``nodes`` restricts the search to a subgraph (e.g. the passes Kahn's
    algorithm could not emit).  Returns ``None`` when no cycle exists there.
    """
    allowed = set(nodes)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {name: WHITE for name in allowed}
    stack = []

    def visit(node):
        color[node] = GRAY
        stack.append(node)
        for nxt in sorted(edges.get(node, ())):
            if nxt not in allowed:
                continue
            if color[nxt] == GRAY:
                start = stack.index(nxt)
                return stack[start:] + [nxt]
            if color[nxt] == WHITE:
                found = visit(nxt)
                if found is not None:
                    return found
        stack.pop()
        color[node] = BLACK
        return None

    for name in sorted(allowed):
        if color[name] == WHITE:
            found = visit(name)
            if found is not None:
                return found
    return None


def toposort(graph):
    """Return a deterministic topological order of pass names.

    Dependencies are data dependencies, not registration order:

    * every pass that writes resource ``R`` must come before every pass that
      reads ``R``;
    * writers of the same resource keep their registration order.

    Among currently ready passes, the lexicographically smallest name is
    emitted first, so a chain has one unique result independent of the input
    list order (a heap is the intended tie-breaker).

    Raises:
        ValueError: duplicate pass names, or a dependency cycle.  A cycle
            error must name every pass on the reported cycle, for example
            ``"cycle detected: ping -> pong -> ping"``.
    """
    passes = graph["passes"]
    names = [item["name"] for item in passes]
    if len(names) != len(set(names)):
        seen = set()
        for name in names:
            if name in seen:
                raise ValueError("duplicate pass: %s" % name)
            seen.add(name)

    edges, _ = _dependency_edges(passes)

    indegree = {name: 0 for name in edges}
    for name, targets in edges.items():
        for target in targets:
            indegree[target] += 1

    ready = [name for name, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)

    order = []
    while ready:
        name = heapq.heappop(ready)
        order.append(name)
        for target in edges[name]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)

    if len(order) != len(edges):
        remaining = [name for name, degree in indegree.items() if degree > 0]
        cycle = _find_cycle(edges, remaining) or remaining
        raise ValueError("cycle detected: %s" % " -> ".join(cycle))

    return order


def insert_barriers(order, accesses):
    """Insert the minimum set of resource-transition barriers.

    ``order`` is the compiled pass order and ``accesses`` maps each pass name
    to ``{"reads": [...], "writes": [...]}``.  The state of a resource is one
    of ``"read"``, ``"write"`` or ``"readwrite"`` (a pass that both reads and
    writes a resource leaves it in ``"readwrite"``).

    A barrier is required before the current pass only when the resource
    state actually changes in a hazardous way:

    * first use of a resource: no barrier;
    * ``read`` -> ``read``: no barrier;
    * ``write`` -> ``write``: no barrier;
    * every other transition, including read -> write and write -> read,
      requires exactly one barrier.

    This makes consecutive reads free and prevents both read-write and
    write-read hazards.  One barrier is emitted per resource per pass at most.

    Returns a list of dicts in execution order::

        {"before": <pass>, "resource": <name>,
         "from": <previous state>, "to": <new state>}

    Raises:
        ValueError: ``order`` contains duplicates, has names missing from
            ``accesses``, or ``accesses`` contains names absent from ``order``.
    """
    if len(order) != len(set(order)):
        raise ValueError("order contains duplicate pass names")
    missing = [name for name in order if name not in accesses]
    if missing:
        raise ValueError("passes missing from accesses: %s" % ", ".join(missing))
    extra = [name for name in accesses if name not in set(order)]
    if extra:
        raise ValueError("accesses contain passes absent from order: %s" % ", ".join(extra))

    states = {}
    barriers = []
    for name in order:
        entry = accesses[name]
        reads = set(entry.get("reads", ()))
        writes = set(entry.get("writes", ()))
        for resource in sorted(reads | writes):
            if resource in reads and resource in writes:
                new_state = "readwrite"
            elif resource in writes:
                new_state = "write"
            else:
                new_state = "read"

            old_state = states.get(resource)
            if old_state is None:
                states[resource] = new_state
                continue
            if old_state == new_state and new_state in ("read", "write"):
                continue
            if old_state == new_state:
                # readwrite -> readwrite: both directions already hazard.
                states[resource] = new_state
                continue
            barriers.append(
                {
                    "before": name,
                    "resource": resource,
                    "from": old_state,
                    "to": new_state,
                }
            )
            states[resource] = new_state
    return barriers


def alias_memory(resources, pool_size):
    """Assign byte offsets in one memory pool, aliasing dead resources.

    ``resources`` is an iterable of dicts::

        {"name": "shadow", "size": 8, "reads": [3], "writes": [0]}

    ``reads`` and ``writes`` are the pass indices at which the resource is
    accessed.  The live interval is the closed interval from the smallest to
    the largest access index, **including reads**.  A writable resource with a
    read after its last write stays live until that read; ending its life at
    the final write is a read-after-write aliasing bug.  A write-only resource
    ends at its last write.  A resource with no access has no lifetime and is
    invalid.

    Two resources may share physical bytes only when their live intervals are
    disjoint.  Resources whose live intervals overlap must receive disjoint
    byte spans, even if one of them is read-only.  Allocation is deterministic
    first-fit: try the lowest non-negative offset first, and reuse an already
    used offset when the live intervals permit it.

    ``pool_size`` is a hard capacity: every ``offset + size`` must be less
    than or equal to it.  There is no slack and no rounding.

    Returns ``{name: {"offset": int, "size": int}}`` in input order.

    Raises:
        ValueError: duplicate/empty names, non-positive sizes, no accesses, an
            invalid ``pool_size``, or a layout that does not fit in the pool.
            The capacity error message mentions ``pool_size``.
    """
    if isinstance(pool_size, bool) or not isinstance(pool_size, int) or pool_size <= 0:
        raise ValueError("invalid pool_size: %r" % (pool_size,))

    entries = list(resources)
    seen = set()
    prepared = []
    for entry in entries:
        name = entry["name"]
        if not name:
            raise ValueError("resource name must not be empty")
        if name in seen:
            raise ValueError("duplicate resource: %s" % name)
        seen.add(name)
        size = entry["size"]
        if not isinstance(size, int) or size <= 0:
            raise ValueError("resource %s has non-positive size: %r" % (name, size))
        accesses = list(entry.get("reads", ())) + list(entry.get("writes", ()))
        if not accesses:
            raise ValueError("resource %s has no accesses" % name)
        prepared.append(
            {
                "name": name,
                "size": size,
                "start": min(accesses),
                "end": max(accesses),
            }
        )

    placed = []  # dicts with name/size/start/end plus "offset"
    result = {}
    for item in prepared:
        offset = 0
        while True:
            span_end = offset + item["size"]
            blocker = None
            for other in placed:
                intervals_overlap = (
                    item["start"] <= other["end"] and other["start"] <= item["end"]
                )
                if not intervals_overlap:
                    continue
                other_end = other["offset"] + other["size"]
                if not (span_end <= other["offset"] or other_end <= offset):
                    blocker = other
                    break
            if blocker is None:
                break
            offset = blocker["offset"] + blocker["size"]
        if offset + item["size"] > pool_size:
            raise ValueError(
                "layout exceeds pool_size %d: resource %s needs offset %d + size %d"
                % (pool_size, item["name"], offset, item["size"])
            )
        item["offset"] = offset
        placed.append(item)
        result[item["name"]] = {"offset": offset, "size": item["size"]}

    return result


def cull_graph(graph, used):
    """Return a new graph containing only passes needed for ``used`` outputs.

    ``used`` is an iterable of resource names that must remain available.
    Compute the backward slice: a pass is live when it writes a needed
    resource; all resources that pass reads then become needed; repeat to a
    fixpoint.  Passes outside that slice are dead, even when they have inputs
    and even when they produce their own unused outputs.  A source pass (one
    with no reads) that produces a needed resource must be kept.

    The graph is not mutated.  Passes in the result keep their original order,
    and the resource declaration table is copied unchanged.  Used names must
    be declared resources; an unknown name raises ``KeyError`` rather than
    silently producing an empty graph.
    """
    declared = graph["resources"]
    needed = set()
    for name in used:
        if name not in declared:
            raise KeyError("undeclared resource: %s" % name)
        needed.add(name)

    passes = graph["passes"]
    live = set()
    changed = True
    while changed:
        changed = False
        for index, item in enumerate(passes):
            if index in live:
                continue
            if any(resource in needed for resource in item.get("writes", ())):
                live.add(index)
                for resource in item.get("reads", ()):
                    if resource not in needed:
                        needed.add(resource)
                changed = True

    return {
        "passes": [
            {"name": item["name"], "reads": list(item.get("reads", ())),
             "writes": list(item.get("writes", ()))}
            for index, item in enumerate(passes)
            if index in live
        ],
        "resources": dict(declared),
    }


def derive_state(accesses):
    """Derive the aggregate access state of every resource.

    ``accesses`` maps pass names to ``{"reads": [...], "writes": [...]}``.
    The result maps every referenced resource to exactly one of:

    * ``"read"``  -- read somewhere, never written;
    * ``"write"`` -- written somewhere, never read;
    * ``"readwrite"`` -- both a read and a write occur, in any passes,
      including a single pass that both reads and writes the resource.

    A read plus a write must never collapse to ``"read"``.  Return ``{}`` for
    empty input.
    """
    read_resources = set()
    written_resources = set()
    for entry in accesses.values():
        read_resources.update(entry.get("reads", ()))
        written_resources.update(entry.get("writes", ()))

    states = {}
    for resource in read_resources | written_resources:
        if resource in read_resources and resource in written_resources:
            states[resource] = "readwrite"
        elif resource in written_resources:
            states[resource] = "write"
        else:
            states[resource] = "read"
    return states


def validate_graph(graph):
    """Return deterministic, human-readable diagnostics for ``graph``.

    An empty list means the graph is valid.  Otherwise the returned messages
    are sorted lexicographically and include every problem found; validation
    never raises for a bad graph.  The checks are:

    * duplicate pass names -- ``"duplicate pass: <name>"``;
    * reads or writes of resources absent from ``graph["resources"]`` --
      ``"undeclared resource: <resource> (pass <pass>)"``;
    * access conflicts with a resource's declared ``access`` mode, for example
      writing a resource declared ``"read"`` or reading one declared
      ``"write"`` -- the message contains ``"read/write conflict"``.  Legal
      read-modify-write use of a ``"readwrite"`` resource is not a conflict;
    * dependency cycles, including the names of the passes on the cycle --
      the message contains ``"cycle"``.

    The cycle check must run on the data-dependency graph, not on pass
    registration order.  A graph with a cycle is invalid even if every
    resource is declared and no other rule is violated.
    """
    diagnostics = []
    passes = graph["passes"]
    declared = graph["resources"]

    seen = set()
    for item in passes:
        name = item["name"]
        if name in seen:
            diagnostics.append("duplicate pass: %s" % name)
        seen.add(name)

    for item in passes:
        name = item["name"]
        for resource in item.get("reads", ()):
            if resource not in declared:
                diagnostics.append(
                    "undeclared resource: %s (pass %s)" % (resource, name)
                )
            elif declared[resource].get("access", "readwrite") == "write":
                diagnostics.append(
                    "read/write conflict: pass %s reads %s declared write"
                    % (name, resource)
                )
        for resource in item.get("writes", ()):
            if resource not in declared:
                diagnostics.append(
                    "undeclared resource: %s (pass %s)" % (resource, name)
                )
            elif declared[resource].get("access", "readwrite") == "read":
                diagnostics.append(
                    "read/write conflict: pass %s writes %s declared read"
                    % (name, resource)
                )

    edges, _ = _dependency_edges(passes)
    cycle = _find_cycle(edges, edges.keys())
    if cycle is not None:
        diagnostics.append("cycle detected: %s" % " -> ".join(cycle))

    return sorted(diagnostics)


def compile_batch(graphs, config):
    """Compile each graph in ``graphs`` independently.

    ``graphs`` maps batch key -> graph.  ``config`` must contain a positive
    integer ``"pool_size"``.  Keys are processed in mapping order.

    For each graph, validation and compilation start from empty local state.
    No resource state, barrier list or alias offset may leak from one graph to
    the next; the second graph must compile exactly as if it were alone.

    A valid graph produces::

        {
            "order": [...],       # toposort names
            "barriers": [...],    # insert_barriers result
            "aliases": {...},     # alias_memory result
            "errors": [],         # empty
        }

    An invalid graph is not compiled; it produces empty ``order``,
    ``barriers`` and ``aliases`` plus the diagnostics from
    ``validate_graph``.  A pool-capacity failure is appended to ``errors`` and
    leaves ``aliases`` empty.  Inputs are not mutated.

    Returns ``{batch_key: compiled_result}``.
    """
    pool_size = config.get("pool_size")
    if isinstance(pool_size, bool) or not isinstance(pool_size, int) or pool_size <= 0:
        raise ValueError("config must contain a positive integer pool_size")

    results = {}
    for key, graph in graphs.items():
        errors = validate_graph(graph)
        if errors:
            results[key] = {
                "order": [],
                "barriers": [],
                "aliases": {},
                "errors": list(errors),
            }
            continue

        order = toposort(graph)
        accesses = {
            item["name"]: {
                "reads": list(item.get("reads", ())),
                "writes": list(item.get("writes", ())),
            }
            for item in graph["passes"]
        }
        barriers = insert_barriers(order, accesses)

        position = {name: index for index, name in enumerate(order)}
        usage = {}
        for item in graph["passes"]:
            index = position[item["name"]]
            for resource in item.get("reads", ()):
                usage.setdefault(resource, {"reads": [], "writes": []})["reads"].append(index)
            for resource in item.get("writes", ()):
                usage.setdefault(resource, {"reads": [], "writes": []})["writes"].append(index)

        alias_input = [
            {
                "name": name,
                "size": graph["resources"][name]["size"],
                "reads": sorted(access["reads"]),
                "writes": sorted(access["writes"]),
            }
            for name, access in usage.items()
        ]

        aliases = {}
        try:
            aliases = alias_memory(alias_input, pool_size)
        except ValueError as exc:
            errors = [str(exc)]

        results[key] = {
            "order": order,
            "barriers": barriers,
            "aliases": aliases,
            "errors": errors,
        }

    return results
