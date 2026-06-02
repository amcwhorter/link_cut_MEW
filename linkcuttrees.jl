# Link-cut tree implementation — adapted from CycleWalk.jl (jonmjonm/CycleWalk.jl)
# Only the operations needed by lct_mew.jl are included.

# ── struct ───────────────────────────────────────────────────────────────────

struct LinkCutTree{T <: Integer}
    nodes :: AbstractArray{Union{Node, Nothing}}

    function LinkCutTree{T}(s::Integer) where {T <: Integer}
        f = new(Vector{Union{Node, Nothing}}(undef, s))
        for n in 1:length(f.nodes)
            f.nodes[n] = Node(convert(T, n))
        end
        return f
    end
end

Base.getindex(lct::LinkCutTree, i::Int) = lct.nodes[i]

# ── core operations ──────────────────────────────────────────────────────────

function replaceRightSubtree!(n::Node, r::Union{Node, Nothing} = nothing)
    c = n.children[2]
    if c isa Node
        c.pathParent = n
        c.parent = nothing
        push!(n.pathChildren, c)
    end
    setRight!(n, r)
    if r isa Node
        r.pathParent = nothing
        delete!(n.pathChildren, r)
    end
end

"Make the preferred path go from the root of the represented tree down to n."
function expose!(n::Node)
    splay!(n)
    replaceRightSubtree!(n)
    while n.pathParent isa Node
        p = n.pathParent
        splay!(p)
        replaceRightSubtree!(p, n)
        splay!(n)
    end
end

"Link two represented trees: u (must be root of its tree) becomes a child of v."
function link!(u::Node, v::Node)
    expose!(u)
    u.children[1] isa Node && throw(ArgumentError("u must be root of its tree"))
    expose!(v)
    (u.parent isa Node || u.pathParent isa Node) &&
        throw(ArgumentError("u and v are already in the same tree"))
    u.pathParent = v
    push!(v.pathChildren, u)
end

"Cut u from its parent in the represented tree (u must not be root)."
function cut!(u::Node)
    expose!(u)
    !(u.children[1] isa Node) && throw(ArgumentError("u is the root; cannot cut"))
    v = u.children[1]
    v.parent = nothing
    setLeft!(u, nothing)
end

"Re-root the represented tree at u."
function evert!(u::Node)
    expose!(u)
    u.reversed = true
end

"Return the root node of the represented tree containing u."
function find_root!(u::Node)
    expose!(u)
    while u.children[1] !== nothing
        u = u.children[1]
    end
    return u
end

# ── component queries ─────────────────────────────────────────────────────────

"Count nodes in the same represented tree as `node`."
function nv_cc(node::Node, start = true)
    start && expose!(node)
    count = 1
    for ii in 1:2
        node.children[ii] !== nothing && (count += nv_cc(node.children[ii], false))
    end
    for n in node.pathChildren
        count += nv_cc(n, false)
    end
    return count
end

"Collect vertex indices of all nodes in the same represented tree as `node`."
function cc(node::Node, start = true, vec::Vector{Int} = Vector{Int}(undef, 0))
    start && expose!(node)
    push!(vec, node.vertex)
    for ii in 1:2
        node.children[ii] !== nothing && cc(node.children[ii], false, vec)
    end
    for n in node.pathChildren
        cc(n, false, vec)
    end
    return vec
end
