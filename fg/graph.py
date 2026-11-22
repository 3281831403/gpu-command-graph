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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
