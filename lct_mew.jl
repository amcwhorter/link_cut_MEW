# lct_mew.jl — Marked Edge Walk using Link-Cut Trees
#
# Drop-in algorithmic replacement for beano2.3.jl.
# Key changes vs. the SimpleGraph version (see profiling notes):
#
#   • Eliminates deepcopy(tree) each iteration (~45 % of runtime in profile2.jlprof).
#     Proposals are applied in-place to the LCT; undone with inverse link/cut on reject.
#   • find_cycle_edges: O(log n) LCT path query instead of O(n) cycle_basis().
#   • Node-to-district lookup: O(1) array instead of O(n) connected_components().
#   • calculate_score caches the MH weight across accepted steps, so only k Cholesky
#     calls are needed per iteration (vs. 2k in the original).
#   • all_parts_edgelists: O(|E(g)|) single pass instead of O(n_district^2) combinations.

include("splaytrees.jl")
include("linkcuttrees.jl")

using Graphs, StatsBase, LinearAlgebra, SparseArrays, DataFrames

const Edge64 = Graphs.SimpleGraphs.SimpleEdge{Int64}

# ── canonical edge ─────────────────────────────────────────────────────────

simple_edge(u::Int, v::Int) = u < v ? Edge64(u, v) : Edge64(v, u)
simple_edge(e::Edge64)      = src(e) < dst(e) ? e : Edge64(dst(e), src(e))

# ── state ──────────────────────────────────────────────────────────────────

"""
Full MCMC state for the LCT-based Marked Edge Walk.

  g            — dual graph (immutable throughout the chain)
  lct          — link-cut tree representing the current spanning tree T
  tree_adj     — adjacency-list copy of T; kept in sync with lct for O(1)
                 degree queries and BFS without touching LCT internals
  nontree_edges — E(g) \\ E(T); sampled to pick edge_plus each step
  marked_edges — the k-1 tree edges whose removal partitions T into k districts
  node_to_dist — node index → district label (1..k); updated on accept
  k, n         — number of districts / nodes
"""
mutable struct LCTState
    g             :: SimpleGraph{Int64}
    lct           :: LinkCutTree{Int64}
    tree_adj      :: Vector{Set{Int64}}
    nontree_edges :: Set{Edge64}
    marked_edges  :: Vector{Edge64}
    node_to_dist  :: Vector{Int64}
    k             :: Int64
    n             :: Int64
end

# ── LCT link / cut wrappers ───────────────────────────────────────────────

function link_edge!(s::LCTState, e::Edge64)
    u, v = src(e), dst(e)
    evert!(s.lct.nodes[u])
    link!(s.lct.nodes[u], s.lct.nodes[v])
    push!(s.tree_adj[u], v)
    push!(s.tree_adj[v], u)
    delete!(s.nontree_edges, e)
end

function cut_edge!(s::LCTState, e::Edge64)
    u, v = src(e), dst(e)
    evert!(s.lct.nodes[u])
    cut!(s.lct.nodes[v])
    delete!(s.tree_adj[u], v)
    delete!(s.tree_adj[v], u)
    push!(s.nontree_edges, e)
end

# ── cycle finding via LCT path query ─────────────────────────────────────

"""
Return the edges on the unique tree path from u to v.
After conceptually adding edge_plus=(u,v), these edges plus edge_plus form
the cycle.  Uses O(log n) LCT expose operations instead of O(n) cycle_basis.
"""
function find_cycle_edges(lct::LinkCutTree, u::Int, v::Int)
    evert!(lct.nodes[u])          # re-root at u
    expose!(lct.nodes[v])         # preferred path = u → v in one splay tree
    path  = traverseSubtree(lct.nodes[v], "in-order")   # shallowest → deepest
    verts = [nd.vertex for nd in path]
    return [simple_edge(verts[i], verts[i+1]) for i in 1:length(verts)-1]
end

# ── partition from tree adjacency ────────────────────────────────────────

"""
BFS over tree_adj, treating marked edges as district boundaries.
Returns a node_to_dist vector (labels 1..k).
Replaces connected_components(tree) + rem_edge!/add_edge! dance.
"""
function compute_node_to_dist(
    tree_adj   :: Vector{Set{Int64}},
    marked_set :: Set{Edge64},
    n          :: Int64
)
    node_to_dist = zeros(Int64, n)
    dist = 0
    for start in 1:n
        node_to_dist[start] != 0 && continue
        dist += 1
        queue = [start]
        while !isempty(queue)
            v = pop!(queue)
            node_to_dist[v] != 0 && continue
            node_to_dist[v] = dist
            for w in tree_adj[v]
                node_to_dist[w] == 0 &&
                    simple_edge(v, w) ∉ marked_set &&
                    push!(queue, w)
            end
        end
    end
    return node_to_dist
end

# ── spanning-tree count (Cholesky) ───────────────────────────────────────

"""
Build per-district edge lists in a single O(|E(g)|) pass.
Much faster than the old combinations(part, 2) approach for large districts.
"""
function all_parts_edgelists(
    g            :: SimpleGraph{Int64},
    node_to_dist :: Vector{Int64},
    k            :: Int64
)
    lists = [Edge64[] for _ in 1:k]
    for e in edges(g)
        u, v = src(e), dst(e)
        d = node_to_dist[u]
        d == node_to_dist[v] && push!(lists[d], simple_edge(u, v))
    end
    return lists
end

"""
Log number of spanning trees of the subgraph defined by edgelist.
Uses your log_τ_sparse approach: stays fully sparse (no Matrix() conversion),
builds COO → SparseMatrixCSC directly, then sparse Cholesky + logdet.
Vertices are remapped to 1..m so the matrix is always district-sized.
"""
function log_τ_sparse(edgelist::Vector{Edge64})
    isempty(edgelist) && return 0.0

    unique_nodes = Set{Int64}()
    for e in edgelist
        push!(unique_nodes, src(e), dst(e))
    end
    m = length(unique_nodes)
    m <= 1 && return 0.0

    node_map = Dict(node => i for (i, node) in enumerate(unique_nodes))

    rows = Int64[]; cols = Int64[]; vals = Int64[]
    for e in edgelist
        i, j = node_map[src(e)], node_map[dst(e)]
        push!(rows, i, j); push!(cols, j, i); push!(vals, 1, 1)
    end

    A  = sparse(rows, cols, vals, m, m)
    L  = spdiagm(0 => vec(sum(A, dims=2))) - A
    Λ  = cholesky(L[2:end, 2:end])
    return logdet(Λ)
end

"""
Log spanning tree count of the quotient graph.
The quotient graph has k nodes; districts i and j are connected by
border_count_{ij} parallel edges (equivalently, a weighted edge of that weight).
For k=2 this reduces to log(border_count), matching the original formula.
For k>2 this is the correct Matrix-Tree computation — sum(log(border_counts))
would be wrong there.
"""
function log_τ_quotient(borders::Dict{Tuple{Int64,Int64},Int64}, k::Int64)
    k <= 1 && return 0.0
    isempty(borders) && return 0.0
    L = zeros(Float64, k, k)
    for ((i, j), w) in borders
        L[i, i] += w;  L[j, j] += w
        L[i, j] -= w;  L[j, i] -= w
    end
    k == 2 && return log(L[1, 1])   # fast path: 1×1 reduced Laplacian
    Λ = cholesky(Symmetric(L[2:end, 2:end]))
    return logdet(Λ)
end

"""
Count border edges between every adjacent district pair.
Returns Dict{(di, dj), count} with di < dj.
"""
function compute_borders(g::SimpleGraph{Int64}, node_to_dist::Vector{Int64})
    b = Dict{Tuple{Int64, Int64}, Int64}()
    for e in edges(g)
        di, dj = node_to_dist[src(e)], node_to_dist[dst(e)]
        di == dj && continue
        key = di < dj ? (di, dj) : (dj, di)
        b[key] = get(b, key, 0) + 1
    end
    return b
end

"""
Combined MH score: Σ log τ(district_i) + log τ(quotient graph).
The quotient graph term is the correct k-district generalisation of the border
length correction — for k=2 it equals log(border_count), for k>2 it is the
full Matrix-Tree determinant on the k-node contracted graph.
Cached across iterations; only recomputed for the proposed state each step.
"""
function calculate_score(
    g            :: SimpleGraph{Int64},
    node_to_dist :: Vector{Int64},
    k            :: Int64
)
    lists    = all_parts_edgelists(g, node_to_dist, k)
    τ_trees  = sum(log_τ_sparse(el) for el in lists)
    borders  = compute_borders(g, node_to_dist)
    τ_quotient = log_τ_quotient(borders, k)
    return τ_trees + τ_quotient
end

# ── transition probability ────────────────────────────────────────────────

"""
Metropolis-Hastings forward/reverse proposal ratio.
Uses degree arrays (O(1) lookup) instead of neighbors(tree, v) (O(degree)).
Signature matches the original transition_probability in beano2.3.jl.
"""
function transition_probability(
    cycle_edges      :: Vector{Edge64},
    edge_plus        :: Edge64,
    old_marked       :: Edge64,
    new_marked       :: Edge64,
    old_marked_edges :: Vector{Edge64},
    new_marked_edges :: Vector{Edge64},
    degree_old       :: Vector{Int64},
    degree_new       :: Vector{Int64}
)
    new_marked == edge_plus && return 0.0

    w, x = src(old_marked), dst(old_marked)
    u, v = src(new_marked), dst(new_marked)

    if Set([u, v]) == Set([w, x])
        d_u  = degree_old[u];  d_u_p = degree_new[u]
        d_v  = degree_old[v];  d_v_p = degree_new[v]
        pm   = (d_u + d_v) / (d_u_p + d_v_p) * d_u_p / d_u * d_v_p / d_v
    else
        shared = intersect([u, v], [w, x])[1]
        pm     = degree_new[shared] / degree_old[shared]
    end

    cycle = Set(cycle_edges)
    l     = length(setdiff(cycle, Set(old_marked_edges)))
    l_p   = length(setdiff(cycle, Set(new_marked_edges)))
    l_p == 0 && return 0.0

    return (l / l_p) * pm
end

# ── helpers ───────────────────────────────────────────────────────────────

function tally(df::DataFrame, col::String, node_to_dist::Vector{Int64}, k::Int64)
    vals   = df[:, col]
    totals = zeros(k)
    for i in eachindex(node_to_dist)
        totals[node_to_dist[i]] += vals[i]
    end
    return totals
end

function within_percent_of_ideal(vals::Vector{Float64}, ideal::Float64, epsilon::Float64)
    maximum(abs.((vals .- ideal) ./ ideal)) < epsilon
end

# ── energy function ───────────────────────────────────────────────────────

"""
Log ratio log(E_new / E_old) for the Metropolis step.

Default: cut-count energy  E ∝ exp(-beta * (n_cuts - target_cuts)^2).

Replace or extend this with your multivariate_energy_function logic.
The function receives both new and old node_to_dist so you can compute
any district-level statistic (vote shares, compactness, etc.) without
re-running connected_components.
"""
function log_energy_ratio(
    g           :: SimpleGraph{Int64},
    new_ntd     :: Vector{Int64},
    old_ntd     :: Vector{Int64},
    beta        :: Float64,
    target_cuts :: Int64
)
    cuts_new = count(e -> new_ntd[src(e)] != new_ntd[dst(e)], edges(g))
    cuts_old = count(e -> old_ntd[src(e)] != old_ntd[dst(e)], edges(g))
    return -beta * ((cuts_new - target_cuts)^2 - (cuts_old - target_cuts)^2)
end

# ── initialisation ────────────────────────────────────────────────────────

"""
Wilson's algorithm for a uniform random spanning tree.
Returns a Set{Edge64} of tree edges.
"""
function wilsons(g::SimpleGraph{Int64})
    n       = nv(g)
    visited = falses(n)
    spanning = Set{Edge64}()
    start    = rand(1:n)
    visited[start] = true

    while count(visited) < n
        current = rand(findall(.!visited))
        path    = [current]
        while !visited[current]
            next = rand(all_neighbors(g, current))
            idx  = findfirst(==(next), path)
            isnothing(idx) ? push!(path, next) : resize!(path, idx)
            current = next
        end
        for i in 1:length(path)-1
            visited[path[i]] = true
            push!(spanning, simple_edge(path[i], path[i+1]))
        end
    end
    return spanning
end

"""
Construct an LCTState from a dual graph, an initial spanning tree edge set,
and k-1 marked edges.  Links tree edges into the LCT in BFS order to avoid
same-component link errors.
"""
function build_lct_state(
    g             :: SimpleGraph{Int64},
    tree_edge_set :: Set{Edge64},
    marked_edges  :: Vector{Edge64},
    k             :: Int64
)
    n = nv(g)

    # adjacency list for the spanning tree
    tree_adj = [Set{Int64}() for _ in 1:n]
    for e in tree_edge_set
        push!(tree_adj[src(e)], dst(e))
        push!(tree_adj[dst(e)], src(e))
    end

    # build LCT in BFS order so link! never joins an already-connected pair
    lct     = LinkCutTree{Int64}(n)
    visited = falses(n)
    visited[1] = true
    queue   = [1]
    while !isempty(queue)
        v = popfirst!(queue)
        for w in tree_adj[v]
            visited[w] && continue
            visited[w] = true
            evert!(lct.nodes[w])
            link!(lct.nodes[w], lct.nodes[v])
            push!(queue, w)
        end
    end

    # non-tree edges for sampling edge_plus
    nontree = Set{Edge64}()
    for e in edges(g)
        se = simple_edge(src(e), dst(e))
        se ∉ tree_edge_set && push!(nontree, se)
    end

    node_to_dist = compute_node_to_dist(tree_adj, Set(marked_edges), n)

    return LCTState(g, lct, tree_adj, nontree, copy(marked_edges), node_to_dist, k, n)
end

# ── single MCMC step ──────────────────────────────────────────────────────

"""
One Metropolis-Hastings step.  Returns (updated_score, accepted::Bool).

No deepcopy of the tree: proposals are applied in-place to the LCT and
tree_adj, then reversed with inverse link/cut operations on rejection.
old_node_to_dist (a Vector{Int64} copy, ~n × 8 bytes) is the only
allocation that scales with graph size, replacing the O(n × degree) copy
of the full SimpleGraph adjacency list in the original code.
"""
function mcmc_step!(
    s           :: LCTState,
    df          :: DataFrame,
    epsilon     :: Float64,
    beta        :: Float64,
    target_cuts :: Int64,
    score_cur   :: Float64,
    pop_ideal   :: Float64;
    max_tries   :: Int = 200
)
    n, k = s.n, s.k

    # snapshot degrees and partition BEFORE any proposal
    degree_old    = [length(s.tree_adj[v]) for v in 1:n]
    old_node_to_dist = copy(s.node_to_dist)   # ~n×8 bytes; much cheaper than copy(tree)

    local cycle_edges, edge_plus, edge_minus, old_marked, new_marked, new_ntd, m_idx

    found = false
    for _ in 1:max_tries

        # ── cycle basis step ──────────────────────────────────────────────
        edge_plus   = rand(s.nontree_edges)
        u, v        = src(edge_plus), dst(edge_plus)
        cycle_edges = find_cycle_edges(s.lct, u, v)
        marked_set  = Set(s.marked_edges)
        possible    = setdiff(Set(cycle_edges), marked_set)
        if isempty(possible)
            continue
        end
        edge_minus = rand(possible)
        cut_edge!(s, edge_minus)
        link_edge!(s, edge_plus)

        # ── marked edge step ──────────────────────────────────────────────
        old_marked = rand(s.marked_edges)
        v1, v2     = src(old_marked), dst(old_marked)
        chosen_v   = rand((v1, v2))
        new_marked = simple_edge(chosen_v, rand(s.tree_adj[chosen_v]))
        m_idx      = findfirst(==(old_marked), s.marked_edges)
        s.marked_edges[m_idx] = new_marked

        # ── population balance check ──────────────────────────────────────
        new_ntd = compute_node_to_dist(s.tree_adj, Set(s.marked_edges), n)
        pops    = tally(df, "TOTPOP", new_ntd, k)
        if within_percent_of_ideal(pops, pop_ideal, epsilon)
            found = true
            break
        end

        # ── undo this attempt ─────────────────────────────────────────────
        s.marked_edges[m_idx] = old_marked
        cut_edge!(s, edge_plus)
        link_edge!(s, edge_minus)
    end

    !found && return score_cur, false

    # ── compute acceptance probability ────────────────────────────────────
    degree_new = [length(s.tree_adj[v]) for v in 1:n]

    # reconstruct old_marked_edges vector (before the marked step)
    old_m_vec        = copy(s.marked_edges)
    old_m_vec[m_idx] = old_marked

    # original cycle_basis_step includes edge_plus in cycle_edges (it's a vertex
    # cycle, so edge_plus closes the loop); replicate that here so the l/l_p
    # ratio in transition_probability matches exactly
    a1 = transition_probability(
        [cycle_edges; edge_plus], edge_plus, old_marked, new_marked,
        old_m_vec, s.marked_edges, degree_old, degree_new
    )

    if a1 == 0.0
        s.marked_edges[m_idx] = old_marked
        cut_edge!(s, edge_plus)
        link_edge!(s, edge_minus)
        return score_cur, false
    end

    score_new = calculate_score(s.g, new_ntd, k)
    a2        = score_cur - score_new
    a3_log    = log_energy_ratio(s.g, new_ntd, old_node_to_dist, beta, target_cuts)

    # ── accept / reject ───────────────────────────────────────────────────
    if log(rand(Float64)) < log(a1) + a2 + a3_log
        s.node_to_dist = new_ntd
        return score_new, true
    else
        s.marked_edges[m_idx] = old_marked
        cut_edge!(s, edge_plus)
        link_edge!(s, edge_minus)
        return score_cur, false
    end
end

# ── main chain loop ───────────────────────────────────────────────────────

"""
Run the LCT-based Marked Edge Walk for num_iterations steps.
Returns a Vector of node_to_dist snapshots (one per step).

Arguments:
  s              — LCTState built by build_lct_state()
  df             — DataFrame with at least a "TOTPOP" column
  num_iterations — number of MCMC steps
  epsilon        — population balance tolerance (fraction of ideal)
  beta           — inverse temperature for the energy function
  target_cuts    — desired number of cut edges (for log_energy_ratio)
"""
function run_chain!(
    s              :: LCTState,
    df             :: DataFrame,
    num_iterations :: Int;
    epsilon        :: Float64 = 0.05,
    beta           :: Float64 = 1.0,
    target_cuts    :: Int     = 0
)
    pop_ideal  = sum(df[:, "TOTPOP"]) / s.k
    score_cur  = calculate_score(s.g, s.node_to_dist, s.k)

    partitions = Vector{Vector{Int64}}(undef, num_iterations)
    accepts    = 0

    for i in 1:num_iterations
        score_cur, accepted = mcmc_step!(
            s, df, epsilon, beta, target_cuts, score_cur, pop_ideal
        )
        accepted && (accepts += 1)
        partitions[i] = copy(s.node_to_dist)
    end

    println("Acceptance rate: $(round(accepts / num_iterations, digits = 4))")
    return partitions
end

# ── convenience: build state from a SimpleGraph tree ─────────────────────

"""
Convert a Graphs.SimpleGraph spanning tree + marked edges (the format used
in beano2.3.jl) into an LCTState.  Useful for warm-starting from an existing
chain or testing against the original code.
"""
function from_simplegraph(
    g            :: SimpleGraph{Int64},
    tree         :: SimpleGraph{Int64},
    marked_edges :: Vector{Edge64},
    k            :: Int64
)
    tree_edge_set = Set(simple_edge(src(e), dst(e)) for e in edges(tree))
    return build_lct_state(g, tree_edge_set, marked_edges, k)
end
