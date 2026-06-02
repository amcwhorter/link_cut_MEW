# Splay tree implementation — adapted from CycleWalk.jl (jonmjonm/CycleWalk.jl)
# Used as the auxiliary data structure underlying the link-cut tree.

mutable struct Node{T}
    vertex      :: T
    parent      :: Union{Node, Nothing}
    pathParent  :: Union{Node, Nothing}
    children    :: Vector{Union{Node, Nothing}}
    reversed    :: Bool
    pathChildren :: Set{Node}

    function Node{T}(
        vertex,
        parent      :: Union{Node, Nothing},
        leftchild   :: Union{Node, Nothing},
        rightchild  :: Union{Node, Nothing},
        pathParent  :: Union{Node, Nothing},
    ) where {T}
        n = new(vertex, parent, pathParent)
        n.reversed = false
        n.children = Vector{Union{Node, Nothing}}(undef, 2)
        setLeft!(n, leftchild)
        setRight!(n, rightchild)
        n.pathChildren = Set{Node}()
        return n
    end
end

Node(vertex::T) where {T} = Node{T}(vertex, nothing, nothing, nothing, nothing)

# ── getters / setters ────────────────────────────────────────────────────────

function setParent!(n::Union{Node, Nothing}, p::Union{Node, Nothing})
    n === nothing && return nothing
    n.parent = p
end

function setChild!(n::Node, i::Int, c::Union{Node, Nothing})
    n.children[i] = c
    setParent!(c, n)
end

setLeft!(n::Node, l::Union{Node, Nothing})  = setChild!(n, 1, l)
setRight!(n::Node, r::Union{Node, Nothing}) = setChild!(n, 2, r)

# ── utility ──────────────────────────────────────────────────────────────────

function sameNode(n1::Union{Node, Nothing}, n2::Union{Node, Nothing})
    n1 isa Node && n2 isa Node && return n1.vertex == n2.vertex
    return n1 == n2
end

function childIndex(n::Node)
    findfirst(x -> sameNode(n, x), n.parent.children)
end

function findSplayRoot(n::Node)
    r = n
    while r.parent isa Node
        r = r.parent
    end
    return r
end

function findExtreme(n::Node, largest::Bool)
    r = findSplayRoot(n)
    ci = largest ? 2 : 1
    while r.children[ci] isa Node
        r = r.children[ci]
    end
    return r
end

# ── subtree traversal ────────────────────────────────────────────────────────

function traverseSubtree!(A::Array, n::Node, order::Int, rev::Bool)
    order == 1 && append!(A, [n])
    for i in 0:1
        i == 1 && order == 2 && append!(A, [n])
        c = n.children[(i ⊻ n.reversed ⊻ rev) + 1]
        c isa Node && traverseSubtree!(A, c, order, n.reversed ⊻ rev)
    end
    order == 3 && append!(A, [n])
end

function traverseSubtree(n::Node, order::String = "in-order")
    A = Vector{Node}(undef, 0)
    pre, in_, post = false, false, false
    (pre = (order == "pre-order")) || (in_ = (order == "in-order")) || (post = (order == "post-order"))
    ord = pre + in_ * 2 + post * 3
    traverseSubtree!(A, n, ord, false)
    return A
end

# ── splay rotations ──────────────────────────────────────────────────────────

function rotateUp(n::Node)
    i = childIndex(n)
    p = n.parent
    g = p.parent

    setParent!(n, g)
    if g isa Node
        j = childIndex(p)
        setChild!(g, j, n)
    else
        n.pathParent = p.pathParent
        p.pathParent = nothing
        if n.pathParent !== nothing
            delete!(n.pathParent.pathChildren, p)
            push!(n.pathParent.pathChildren, n)
        end
    end

    setChild!(p, i, n.children[3 - i])
    setChild!(n, 3 - i, p)
end

function pushReversed!(n::Node)
    n.parent isa Node && pushReversed!(n.parent)
    if n.reversed
        n.children[1], n.children[2] = n.children[2], n.children[1]
        for c in n.children
            c isa Node && (c.reversed = !c.reversed)
        end
        n.reversed = false
    end
end

function splay!(n::Node)
    pushReversed!(n)
    while n.parent isa Node
        p = n.parent
        if p.parent === nothing
            rotateUp(n)
        elseif childIndex(n) == childIndex(p)
            rotateUp(p); rotateUp(n)
        else
            rotateUp(n); rotateUp(n)
        end
    end
end
