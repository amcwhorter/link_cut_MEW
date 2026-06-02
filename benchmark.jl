# benchmark.jl — compare beano2.2_WI (SimpleGraph) vs lct_mew (LinkCutTree)
#
# Uses a synthetic grid graph + uniform population so no external data is needed.
# Run with:   julia --project=. benchmark.jl
#
# K=2: for k=2 both codes use the identical τ correction formula
# (log(border_count) == log_τ_quotient), so acceptance rates should match.


include("lct_mew.jl")

# ── pull in the original algorithm under a separate module namespace ──
# beano2.2_WI.jl is the most complete version: has proposal, calculate_τs,
# log_τ_sparse, all_parts_edgelists, wilsons, and the full transition_probability.
module Beano
    include(joinpath(@__DIR__, "..", "Marked_edges", "beano2.2_WI.jl"))
end

using DataFrames, Statistics

# ────────────────────────────────────────────────────────────────────────────
# 1.  Build a synthetic test case
# ────────────────────────────────────────────────────────────────────────────

const GRID_M   = 22        # grid dimensions → 400 nodes
const GRID_N   = 22
const K        = 2         # number of districts — k=2 keeps τ formulas identical
const EPSILON  = 0.01      # population balance tolerance
const BETA     = 0.0       # set to 0 so energy = 1 (pure spanning-tree walk)
const N_WARMUP = 500        # iterations discarded before timing
const N_TIMED  = 20000       # iterations used for timing

println("Grid: $(GRID_M)×$(GRID_N)  ($(GRID_M*GRID_N) nodes, k=$K)")

g = Graphs.grid((GRID_M, GRID_N))
n = nv(g)

# Uniform population: every node has population 1
df = DataFrame(TOTPOP = ones(Float64, n))

# ────────────────────────────────────────────────────────────────────────────
# 2.  Find a valid initial partition (balanced, k districts)
# ────────────────────────────────────────────────────────────────────────────

# Find k-1 marked edges giving a balanced k-way partition.
# Strategy: root the spanning tree, compute subtree populations bottom-up,
# then look for edges whose subtree pop ≈ m*ideal for m=1..k-1.
# For a uniform population tree these subtrees are always nested, so
# removing the k-1 found edges gives exactly k balanced components.
function find_initial_partition(g, df, k, epsilon)
    ideal = sum(df.TOTPOP) / k

    for _ in 1:10_000
        tree = Beano.wilsons(g)

        # ── root at node 1, DFS to get pre-order and parents ──────────────
        parent    = zeros(Int, n)
        sub_pop   = Float64.(df.TOTPOP)
        visited   = falses(n)
        pre_order = Int[]

        stack = [(1, 0)]
        while !isempty(stack)
            v, p = pop!(stack)
            visited[v] && continue
            visited[v] = true
            parent[v]  = p
            push!(pre_order, v)
            for w in Graphs.neighbors(tree, v)
                !visited[w] && push!(stack, (w, v))
            end
        end

        # ── accumulate subtree populations bottom-up (reverse pre-order) ──
        for v in Iterators.reverse(pre_order)
            parent[v] != 0 && (sub_pop[parent[v]] += sub_pop[v])
        end

        # ── find k-1 cut edges at subtree pops ≈ m*ideal ─────────────────
        targets = Set(1:k-1)
        cuts    = Edge64[]
        for v in pre_order
            v == 1 && continue
            for m in copy(targets)
                if abs(sub_pop[v] - m * ideal) / ideal <= epsilon / 2
                    push!(cuts, simple_edge(v, parent[v]))
                    delete!(targets, m)
                    break
                end
            end
            isempty(targets) && break
        end
        length(cuts) != k - 1 && continue

        # ── verify resulting partition is balanced ─────────────────────────
        pd   = Beano.partition(tree, cuts)
        length(unique(values(pd))) != k && continue
        pops = [sum(df.TOTPOP[i] for (i, d) in pd if d == dist) for dist in 1:k]
        Beano.within_percent_of_ideal(pops, ideal, epsilon) && return tree, cuts
    end

    error("Could not find balanced $(k)-way partition after 10,000 tries")
end

println("Finding initial partition...")
orig_tree, orig_marked = find_initial_partition(g, df, K, EPSILON)
println("  Found $(K)-district partition.")

# ────────────────────────────────────────────────────────────────────────────
# 3.  Shared helper: run N steps and return elapsed seconds
# ────────────────────────────────────────────────────────────────────────────

function time_original(g, df, tree, marked, n_steps, k, epsilon)
    t_start = time()
    tree    = deepcopy(tree)
    marked  = deepcopy(marked)
    accepts = 0

    for _ in 1:n_steps
        tree_old          = copy(tree)
        marked_edges_old  = copy(marked)

        result = Beano.proposal(g, df, tree, marked, epsilon, k)
        cycle_edges, edge_plus, tree_new, old_edge, new_edge, marked_new, _ = result

        a1  = Beano.transition_probability(
                cycle_edges, edge_plus, old_edge, new_edge,
                marked_edges_old, marked_new, tree_old, tree_new)

        a2  = Beano.calculate_τs(g, tree_old, tree_new, marked_edges_old, marked_new)

        a3  = 1.0   # beta=0 → flat energy

        log_a = log(max(a1, 1e-300)) + a2 + log(a3)

        if log(rand()) < log_a
            tree   = tree_new
            marked = marked_new
            accepts += 1
        else
            tree   = copy(tree_old)
            marked = copy(marked_edges_old)
        end
    end
    elapsed = time() - t_start
    return elapsed, accepts / n_steps
end


function time_lct(g, df, tree_sg, marked, n_steps, k, epsilon, beta)
    tree_edges = Set(simple_edge(src(e), dst(e)) for e in edges(tree_sg))
    state      = build_lct_state(g, tree_edges, marked, k)
    pop_ideal  = sum(df.TOTPOP) / k
    score_cur  = calculate_score(g, state.node_to_dist, k)
    target_c   = 0   # ignored when beta=0

    t_start = time()
    accepts = 0

    for _ in 1:n_steps
        score_cur, accepted = mcmc_step!(
            state, df, epsilon, beta, target_c, score_cur, pop_ideal
        )
        accepted && (accepts += 1)
    end
    elapsed = time() - t_start
    return elapsed, accepts / n_steps
end

# ────────────────────────────────────────────────────────────────────────────
# 4.  Warm up (JIT compile), then time
# ────────────────────────────────────────────────────────────────────────────

println("\nWarming up ($(N_WARMUP) steps each)...")
time_original(g, df, orig_tree, orig_marked, N_WARMUP, K, EPSILON)
time_lct(     g, df, orig_tree, orig_marked, N_WARMUP, K, EPSILON, BETA)
println("  Done.")

println("\nTiming $(N_TIMED) steps each...")
t_orig, ar_orig = time_original(g, df, orig_tree, orig_marked, N_TIMED, K, EPSILON)
t_lct,  ar_lct  = time_lct(     g, df, orig_tree, orig_marked, N_TIMED, K, EPSILON, BETA)

# ────────────────────────────────────────────────────────────────────────────
# 5.  Report
# ────────────────────────────────────────────────────────────────────────────

ms_orig = t_orig / N_TIMED * 1000
ms_lct  = t_lct  / N_TIMED * 1000
speedup = t_orig / t_lct

println()
println("─────────────────────────────────────────────────")
println("  Algorithm        ms/step   accept rate")
println("─────────────────────────────────────────────────")
println("  beano2.3 (orig)  $(rpad(round(ms_orig, digits=2), 9))  $(round(ar_orig*100, digits=1))%")
println("  lct_mew  (new)   $(rpad(round(ms_lct,  digits=2), 9))  $(round(ar_lct *100, digits=1))%")
println("─────────────────────────────────────────────────")
println("  Speedup:  $(round(speedup, digits=2))×")
println()

if abs(ar_orig - ar_lct) > 0.1
    @warn "Acceptance rates differ by >10% — check that both chains are using the same energy."
end
