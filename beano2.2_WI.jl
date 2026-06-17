using Graphs, StatsBase, Combinatorics, LinearAlgebra, SparseArrays
    #contains helpful functions - log metropolis hastings version

function simple_edge(edge::Graphs.SimpleGraphs.SimpleEdge)
    u, v = src(edge), dst(edge)
    if u < v
        return edge
    else 
        return Graphs.SimpleGraphs.SimpleEdge(v, u)
    end
end

function simple_edge(u::Int, v::Int)
    if u < v
        return Graphs.SimpleGraphs.SimpleEdge(u, v)
    else
        return Graphs.SimpleGraphs.SimpleEdge(v, u)
    end
end



function instantiate(m::Int, n::Int)
    g = Graphs.grid((m, n))
    min_tree_edges = kruskal_mst(g, Graphs.weights(g))
    tree = SimpleGraph(m * n)

    for edge in min_tree_edges 
        add_edge!(tree, edge)
    end

    tree = SimpleGraph(tree)
    marked_edges = sample(min_tree_edges, 1, replace=false)

    return g, tree, marked_edges
end


function instantiate(g::Graphs.SimpleGraph)
    min_tree_edges = kruskal_mst(g, Graphs.weights(g))
    tree = SimpleGraph(length(collect(1:nv(g))))

    for edge in min_tree_edges
        add_edge!(tree, simple_edge(edge))
    end

    tree = SimpleGraph(tree)
    marked_edges = sample(min_tree_edges, 4, replace=false)
    
    return g, tree, marked_edges
end


function grid_layout(g)
    xlocs = Vector{Int64}()
    ylocs = Vector{Int64}()
    for i in 1:m
        for j in 1:n
            push!(xlocs, i)
            push!(ylocs, j)
        end
    end
    return xlocs, ylocs
end

function tally(df, column_name, partition_dict)
    to_add = df[:, column_name]

    totals = zeros(maximum(values(partition_dict)))

    for i in keys(partition_dict)
        j = partition_dict[i]
        totals[j] += to_add[i]
    end 
    return totals
end

function within_percent_of_ideal(vals, ideal, epsilon)
    ideals = ideal * ones(length(vals))
    epsilons = abs.((vals - ideals)/ideal)
    return maximum(epsilons) < epsilon
end

function single_within_percent_of_ideal(vals, ideal, epsilon)
    ideals = ideal * ones(length(vals))
    epsilons = abs.((vals - ideals)/ideal)
    return minimum(epsilons) < epsilon
end


function find_balanced_edge_cut(tree, df)
    for edge in edges(tree)
        part_dict = partition(tree, [edge])

        totals = tally(df, "TOTPOP", part_dict)

        ideal = sum(df[:,"TOTPOP"])/2

        if within_percent_of_ideal(totals, ideal, 0.15)
            return edge
        end
    end

    println("no balanced edge cuts")

    return 

end

function find_all_balanced_edge_cuts(tree, df)
    returns = []
    for edge in edges(tree)
        part_dict = partition(tree, [edge])
        totals = tally(df, "TOTPOP", part_dict)
        ideal = sum(df[:,"TOTPOP"])/2
        if within_percent_of_ideal(totals, ideal, 0.15)
            push!(returns, edge)
        end
    end
    return returns
end

function cycle_basis_step(g, tree, marked_edges)
    edges_g = Set(edges(g))
    edges_tree = Set(edges(tree))
    difference_set = setdiff(edges_g, edges_tree)

    edge_plus = rand(difference_set)

    tree_copy = deepcopy(tree)
    add_edge!(tree_copy, edge_plus)

    cycle = Graphs.cycle_basis(tree_copy)[1]
    cycle_edges = simple_edge.(cycle, circshift(cycle, -1))

    possible_cuts = setdiff(cycle_edges, marked_edges)
    edge_minus = rand(possible_cuts)

    rem_edge!(tree_copy, edge_minus)

    return cycle_edges, edge_plus, tree_copy
end

function marked_edge_step(tree, marked_edges)
    old_edge = rand(marked_edges)
    v1, v2 = src(old_edge), dst(old_edge)
    chosen_v = rand((v1, v2))
    new_edge = simple_edge(chosen_v, rand(Graphs.neighbors(tree, chosen_v)))

    marked_copy = deepcopy(marked_edges)
    setdiff!(marked_copy, [old_edge])
    push!(marked_copy, new_edge)
    return old_edge, new_edge, marked_copy
end

function proposal(g, df, tree, marked_edges, epsilon, k)
    reject = 0
    tree_old = deepcopy(tree)
    marked_edges_old = deepcopy(marked_edges)
    returns = []

    while true
        cycle_edges, edge_plus, tree_new = cycle_basis_step(g, tree, marked_edges)
        old_edge, new_edge, marked_edges_new = marked_edge_step(tree, marked_edges)

        partition_dict = partition(tree_new, marked_edges_new)
        pops = tally(df, "TOTPOP", partition_dict)
        ideal = sum(pops)/k

        
        if within_percent_of_ideal(pops, ideal, epsilon)
            push!(returns, [cycle_edges, edge_plus, tree_new, old_edge, new_edge, marked_edges_new, reject])
            break
        else
            reject +=1 
            tree = deepcopy(tree_old)
            marked_edges = deepcopy(marked_edges_old)
        end
    end
    return returns[1]
end

function flip_proposal(df, part, bounds)
    reject = 0
    part_old = copy(part)
    bounds_old = copy(bounds)
    returns = []

    while true
        chosen = rand(bounds_old)
        affil = part[chosen]
        if affil == 1
            part[chosen] = 2
        else
            part[chosen] = 1
        end

        pops = tally(df, "TOTPOP", part)
        ideal = sum(pops) / 2
        if within_percent_of_ideal(pops, ideal, 0.05)
            push!(returns, part)
            break
        end
        reject += 1
        part = copy(part_old)
    end
    return returns[1]
end

function transition_probability(cycle_edges, edge_plus, marked_edge_old, marked_edge_new, marked_edges_old, marked_edges_new, tree_old, tree_new)
    if marked_edge_new == edge_plus
        return 0
    end

    w, x = src(marked_edge_old), dst(marked_edge_old)
    u, v = src(marked_edge_new), dst(marked_edge_new)

    if Set([u, v]) == Set([w, x])
        d_u = length(neighbors(tree_old, u))
        d_u_p = length(neighbors(tree_new, u))
        d_v = length(neighbors(tree_old, v))
        d_v_p = length(neighbors(tree_new, v))

        pm = (d_u + d_v)/(d_u_p + d_v_p) * d_u_p/d_u * d_v_p/d_v
    else
        u = intersect([u, v], [w, x])[1]
        d_u = length(neighbors(tree_old, u))
        d_u_p = length(neighbors(tree_new, u))
        pm = d_u_p/d_u
    end


    cycle = Set(cycle_edges)
    m_old = Set(marked_edges_old)
    m_new = Set(marked_edges_new)

    l = length(setdiff(cycle, m_old))
    l_p = length(setdiff(cycle, m_new))

    pt = l/l_p

    return pt * pm
end


function log_τ(edgelist)

    unique_nodes = Set{Int}()
    for edge in edgelist
        u, v = src(edge), dst(edge)
        push!(unique_nodes, u, v)
    end

    node_map = Dict(node => i for (i, node) in enumerate(unique_nodes))
    num_nodes = length(unique_nodes)

    A = zeros(Int, num_nodes, num_nodes)
    
    for edge in edgelist
        u, v = src(edge), dst(edge)
        i, j = node_map[u], node_map[v]

        A[i, j] += 1
        A[j, i] += 1  
    end

    D = Diagonal(vec(sum(A, dims = 2)))
    L = D - A

    Λ = cholesky(L[2:end,2:end])
    U = Λ.L
    c = 2 * sum(log.([U[i, i] for i in 1:(length(unique_nodes) - 1)]))

    return c
end


function log_τ_sparse(edgelist)
    unique_nodes = Set{Int}()
    for edge in edgelist
        u, v = src(edge), dst(edge)
        push!(unique_nodes, u, v)
    end

    node_map = Dict(node => i for (i, node) in enumerate(unique_nodes))
    num_nodes = length(unique_nodes)

    rows = Int[]
    cols = Int[]
    vals = Int[]

    for e in edgelist
        u, v = src(e), dst(e)
        i, j = node_map[u], node_map[v]

        push!(rows, i); push!(cols, j); push!(vals, 1)
        push!(rows, j); push!(cols, i); push!(vals, 1)
    end

    A = sparse(rows, cols, vals, num_nodes, num_nodes)

    degs = vec(sum(A, dims=2))
    L = spdiagm(0 => degs) - A
    inds = 2:num_nodes
    Lr = L[inds, inds]

    Λ = cholesky(Lr)

    c = logdet(Λ)
    return c
end



# function nodes_to_edgelist(g, part)
#     connected_edges = [simple_edge(u, v) for (u, v) in combinations(part, 2)]
#     part_edges = intersect(connected_edges, edges(g))
#     return part_edges
# end

function nodes_to_edgelist(g, part)
    part_set = Set(part)
    [e for e in edges(g) if (src(e) in part_set && dst(e) in part_set)]
end


function all_parts_edgelists(g, parts)

    node_to_part = Dict()
    for (i, part) in enumerate(parts)
        for v in part
            node_to_part[v] = i
        end
    end
    part_edges = [Graphs.SimpleGraphs.SimpleEdge{Int64}[] for _ in 1:length(parts)]
    for e in edges(g)
        u, v = src(e), dst(e)
        if haskey(node_to_part, u) && haskey(node_to_part, v)
            pu, pv = node_to_part[u], node_to_part[v]
            if pu == pv
                push!(part_edges[pu], e)
            end
        end
    end
    return part_edges
end

function cut_edges(ptition, g)
    cuts = []
    for e in edges(g)
        u, v = src(e), dst(e)

        if ptition[u] != ptition[v]
            push!(cuts, simple_edge(u, v))
        end
    end
    return cuts
end

function boundary_nodes(cuts)
    returns = Set{Int}()
    for e in cuts
        u, v = src(e), dst(e)
        push!(returns, u)
        push!(returns, v)
    end
    return collect(returns)
end



# function cut_edges(g, tree, marked_edges)

#     for m in marked_edges
#         rem_edge!(tree, m)
#     end

#     parts = connected_components(tree)
#     es = [nodes_to_edgelist(g, part) for part in parts]

#     cuts = setdiff(edges(g), Set(Iterators.flatten(es)))

#     for m in marked_edges
#         add_edge!(tree, m)
#     end

#     return length(cuts)
# end

function partition(parts::Vector{Any})
    partition_dict = Dict{Int, Int}()
    for (i, part) in enumerate(parts)
        for node in part
            partition_dict[node] = i
        end
    end
    return partition_dict
end 


function partition(parts::Vector{Vector{Int64}})
    partition_dict = Dict{Int, Int}()
    for (i, part) in enumerate(parts)
        for node in part
            partition_dict[node] = i
        end
    end
    return partition_dict
end 


function partition(tree::SimpleGraph{Int64}, marked_edges::Vector{Graphs.SimpleGraphs.SimpleEdge{Int64}})
    for m in marked_edges
        rem_edge!(tree, m)
    end

    parts = connected_components(tree)

    for m in marked_edges
        add_edge!(tree, m)
    end

    return partition(parts)
end



function borders_dict(cut_edges, p_dict)
    borders_dict = Dict()
    for edge in cut_edges
        u, v = src(edge), dst(edge)
        key = Set([p_dict[u], p_dict[v]])
        if !(key in keys(borders_dict))
            borders_dict[key] = 1
        else
            borders_dict[key] += 1
        end
    end
    return borders_dict
end

function get_neighboring_part_pairs(parts, g)
    district_neighbors = Dict{Tuple{Int, Int}, Int}()
    for e in edges(g)
        u, v = src(e), dst(e)
        pu, pv = parts[u], parts[v]
        if pu != pv
            key = pu < pv ? (pu, pv) : (pv, pu)
            district_neighbors[key] = get(district_neighbors, key, 0) + 1
        end
    end
    return district_neighbors
end



function calculate_τs(g, tree_old, tree_new, marked_edges_old, marked_edges_new)

    tree_new_c = deepcopy(tree_new)
    tree_old_c = deepcopy(tree_old)

    for m in marked_edges_old
        rem_edge!(tree_old_c, m)
    end

    for m in marked_edges_new
        rem_edge!(tree_new_c, m)
    end


    #calculate τ relating to border lengths
    old_parts = connected_components(tree_old_c)
    new_parts = connected_components(tree_new_c)

    p_dict_old = partition(old_parts)
    p_dict_new = partition(new_parts)

    cut_dict_old = get_neighboring_part_pairs(p_dict_old, g)
    cut_dict_new = get_neighboring_part_pairs(p_dict_new, g)

    τ_border_lengths = sum(log.(values(cut_dict_old))) - sum(log.(values(cut_dict_new)))

    old_edges = all_parts_edgelists(g, old_parts)
    new_edges = all_parts_edgelists(g, new_parts)

    # old_edges = [nodes_to_edgelist(g, part) for part in old_parts]
    # new_edges = [nodes_to_edgelist(g, part) for part in new_parts]

    old_τs = log_τ_sparse.(old_edges)
    new_τs = log_τ_sparse.(new_edges)

    τ_spanning_trees = sum(old_τs) - sum(new_τs)
    
    τ = τ_border_lengths + τ_spanning_trees

    return τ
end


# function energy_function(g, tree, marked_edges, beta)

#     ptition = partition(tree, marked_edges)
#     cuts = length(cut_edges(ptition, g))

#     J = (25 - cuts) ^ 2 
    
#     return exp(-beta * J)
# end


function energy_function_cuts(a::Number, bet, m2)
    J = (m2 - a) ^ 2
    return exp(-bet * J)
end

function energy_function_percents(a::Number, alph, m1)
    J = (m1 - a) ^ 2
    return exp(-alph * J)
end

function energy_function(v::Vector)
    d = [56, 56] - v
    a = norm(d)
    return exp(-10 * a)
end

function binomial_prob(a, n, p)

    binomial_coefficient = binomial(n, a)

    binomial_prob = binomial_coefficient * p^a * (1 - p)^(n - a)

    return binomial_prob
end



function energy_function_cuts(g, tree, marked_edges, bet, m2)
    ptition = partition(tree, marked_edges)
    cuts = length(cut_edges(ptition, g))

    return energy_function_cuts(cuts, bet, m2)
end

function wilsons(g)
    spanning_tree = Set()

    visited = Set()

    start = rand(1:nv(g))
    push!(visited, start)

    while length(visited) < nv(g)
        unvisited = setdiff(1:nv(g), visited)
        current = rand(unvisited)
        path = [current]

        while current ∉ visited

            next = rand(all_neighbors(g, current))

            loop_index = findfirst(==(next), path)
            
            if !isnothing(loop_index)
                path = path[1:loop_index]
            else
                push!(path, next)
            end

            current = next
        end

        for i in 1:(length(path) - 1)
            u, v = path[i], path[i + 1]
            push!(visited, u)
            push!(spanning_tree, simple_edge(u, v))
        end
    end

    tree = Graphs.SimpleGraph(nv(g))
    
    for edge in spanning_tree
        add_edge!(tree, edge)
    end


    return tree

end


function political_energy_function(df, tree, marked_edges, alph, m1)
    ptition = partition(tree, marked_edges)
    votes = tally(df, "PRES16D", ptition)
    total_votes = tally(df, "PRES16R", ptition) + votes
    percents = votes ./ total_votes * 100
    return energy_function_percents(percents[2], alph, m1)
end

function top_percent_from_partitions(ptition, df)
    votes = tally(df, "PRES16D", ptition)
    total_votes = tally(df, "PRES16R", ptition) + votes
    percents = votes ./ total_votes * 100
    return percents
end


function polsby_popper(area, perimeter)
    return 4 * pi * area / (perimeter ^ 2)
end


function polsby_popper(areas, boundary_lengths, perim_dict, tree, marked_edges, g)
    rem_edge!(tree, marked_edges[1])
    con_parts = connected_components(tree)
    ptition = partition(con_parts)
    cuts = cut_edges(ptition, g)
    add_edge!(tree, marked_edges[1])

    areas = [sum(areas[con_part]) for con_part in con_parts]
    boundaries = [sum(boundary_lengths[con_part]) for con_part in con_parts]
    border_length = sum([perim_dict[cut] for cut in cuts])

    pp1 = polsby_popper(areas[1], boundaries[1] + border_length)
    pp2 = polsby_popper(areas[2], boundaries[2] + border_length)

    return (pp1 + pp2)/2
end

function polsby_popper(areas, boundary_lengths, perim_dict, ptition, g)
    cuts = cut_edges(ptition, g)
    border_length = sum([perim_dict[cut] for cut in cuts])

    ones, twos = [], []
    for (key, val) in ptition
        if val == 1
            push!(ones, key)
        else
            push!(twos, key)
        end
    end

    con_parts = [ones, twos]
    boundaries = [sum(boundary_lengths[con_part]) for con_part in con_parts]
    areas = [sum(areas[con_part]) for con_part in con_parts]

    pp1 = polsby_popper(areas[1], boundaries[1] + border_length)
    pp2 = polsby_popper(areas[2], boundaries[2] + border_length)

    return (pp1 + pp2) / 2
end

function polsby_popper_energy_function(p)
    
    J = (1/3 - p) ^ 2
    
    return exp(- 1000 * J)
    
end

function multivariate_energy_function(df, g, tree, marked_edges, alph, bet, m1, m2)
    return political_energy_function(df, tree, marked_edges, alph, m1) * energy_function_cuts(g, tree, marked_edges, bet, m2)
end


function multivariate_energy_function(cuts, percents, alph, bet, m1, m2)
    return energy_function_cuts(length(cuts), bet, m2) * energy_function_percents(percents, alph, m1)

end



function find_one_sided_cut(tree, df, target_pop, epsilon)
    for edge in edges(tree)
        part_dict = partition(tree, [edge])
        totals = tally(df, "TOTPOP", part_dict)
        if single_within_percent_of_ideal(totals, target_pop, epsilon)
            return edge
        end
    end
    return 
end


function separate_district(tree, cut_edge, target_pop, df)
    tree_copy = copy(tree)
    rem_edge!(tree_copy, cut_edge)
    components = connected_components(tree_copy)
    component_pops = [sum(df[component, "TOTPOP"]) for component in components]
    target_diffs = abs.(component_pops .- target_pop)
    idx = argmin(target_diffs)

    district_vertices = components[idx]
    remaining_vertices = setdiff(vertices(tree), district_vertices)
    return district_vertices, remaining_vertices
end

function find_k_partition(g, df, k, pop_col, epsilon)
    remaining_graph = copy(g)
    remaining_pop = sum(df[!, pop_col])
    remaining_df = copy(df)
    districts = []

    current_to_original = collect(1:nv(g))


    for i in ProgressBar(1:(k - 1))
        target_pop = remaining_pop / (k - i + 1)

        b = []
        c = []
        counter = 1
        while true
            tree = wilsons(remaining_graph)
            cut_edge = find_one_sided_cut(tree, remaining_df, target_pop, epsilon)

            if !isnothing(cut_edge)
                push!(b, cut_edge)
                push!(c, tree)
                break
            end
            counter += 1
            if counter > 10
                println("couldn't get it done")
                return
            end
        end
        cut_edge = b[1]
        tree = c[1]
        district, remaining_nodes = separate_district(tree, cut_edge, target_pop, remaining_df)

        og_district = [current_to_original[node] for node in district]
        push!(districts, og_district)
        remaining_pop -= sum(df[og_district, pop_col])

        remaining_graph, node_map = induced_subgraph(remaining_graph, remaining_nodes)

        current_to_original = [current_to_original[node_map[i]] for i in 1:length(node_map)]
        remaining_df = remaining_df[node_map, :]

        if i == k - 1
            og_district = [current_to_original[node] for node in vertices(remaining_graph)]
            push!(districts, og_district)
        end
    end
    

    return districts
end



function partition_to_tree_marked_edges(g, districts)
    k = length(districts)
    
    parts = partition(districts)
    tree = SimpleGraph(nv(g))

    for district in districts 
        district_subgraph, _ = induced_subgraph(g, district)
        district_tree = wilsons(district_subgraph)

        for edge in edges(district_tree)
            og_src = district[src(edge)]
            og_dst = district[dst(edge)]
            add_edge!(tree, simple_edge(og_src, og_dst))
        end
    end

    cut_edges = []
    for edge in edges(g)
        dist1 = parts[src(edge)]
        dist2 = parts[dst(edge)]

        if  dist1 != dist2
            push!(cut_edges, simple_edge(edge))
        end
    end

    connected_districts = Set([1])
    marked_edges = Graphs.SimpleGraphs.SimpleEdge{Int64}[]

    while (length(connected_districts)) < k
        for edge in cut_edges
            dist1 = parts[src(edge)]
            dist2 = parts[dst(edge)]

            if (dist1 in connected_districts) ⊻ (dist2 in connected_districts)
                push!(marked_edges, edge)
                push!(connected_districts, dist1)
                push!(connected_districts, dist2)

                add_edge!(tree, simple_edge(edge))

                filter!(e -> e != edge, cut_edges)
                break
            end
        end
    end
    return tree, marked_edges
end

function energy_function_MVAP(df, tree, marked_edges, alph, m1)
    ptition = partition(tree, marked_edges)
    pop = tally(df, "TOTPOP", ptition)
    hvap = tally(df, "HVAP", ptition)

    prop = hvap ./ pop
    good = [0.3 < p < 0.6 ? 1 : 0 for p in prop]
    num = sum(good)

    return  num
end

function MVAP_analyser(ptition, df)
    pop = tally(df, "TOTPOP", ptition)
    hvap = tally(df, "HVAP", ptition)
    
    prop = hvap ./ pop
    good = [0.4 < p ? 1 : 0 for p in prop]
    num = sum(good)
    return num
end


function MVAP_maximizer(df, g, tree, marked_edges, T)
    ptition = partition(tree, marked_edges)

    HVAP_ginger_seats = MVAP_analyser(ptition, df)
    return HVAP_ginger_seats/T
end


function tx_energy_function_cuts(g, tree, marked_edges, bet, m2)
    ptition = partition(tree, marked_edges)
    cuts = length(cut_edges(ptition, g))

    return -bet*(cuts - m2)^2
end


function energy_function_mm(g, df, tree, marked_edges, alph, m1)
    ptition = partition(tree, marked_edges)
    mm = mean_median_score(ptition, df)
    return - 100000 * (mm - 0)^2
end

function seats(ptition, df)
    tot = tally(df, "PRES16D", ptition) + tally(df, "PRES16R", ptition) + tally(df, "PRE216R", ptition)
    dem = tally(df, "PRES16D", ptition)

    perc = dem ./ tot
    won = [p > 0.5 ? 1 : 0 for p in perc]
    return sum(won)
end

function mean_median_score(ptition, df)
    tot = tally(df, "PRES16D", ptition) + tally(df, "PRES16R", ptition) + tally(df, "PRES16R", ptition)
    dem = tally(df, "PRES16D", ptition)

    perc = dem ./ tot

    return median(perc) - mean(perc)
end

function TX_energy_function(df, g, tree, marked_edges, alph, bet, m1, m2)   
    E1 = tx_energy_function_cuts(g, tree, marked_edges, bet, m2)
    # E2 = energy_function_mm(g, df, tree, marked_edges, alph, m1)

    return E1
end

function Rosen(df, g, tree, marked_edges, T)
    ptition = partition(tree, marked_edges)
    mins = MVAP_analyser(ptition, df)
    won = seats(ptition, df)
    return -Rosen(mins, won)/T
end

function Rosen(cuts, dem_seats)
    x1 = (dem_seats - 18)/3
    x2 = (cuts - 10)/3

    E = (1 - x1)^2 + 100 * (x2 - x1^2)^2
    return E
end


function Rosen_analysis(part, df, g)
    mins = MVAP_analyser(part, df)
    won = seats(part, df)

    return Rosen(mins, won)
end



function mono_energy(df, tree, marked_edges)
    part = partition(tree, marked_edges)
    wins = seats(part, df)
    target = 13
    return abs(wins - target)
end

function minimizer(df, g, tree, marked_edges, T)
    ptition = partition(tree, marked_edges)
    c = length(cut_edges(ptition, g))
    return -c / T
end

function sorted_percents(df, tree, marked_edges)
    ptition = partition(tree, marked_edges)
    tot = tally(df, "PRE16D", ptition) + tally(df, "PRE16R", ptition)
    dem = tally(df, "PRE16D", ptition)

    percs = dem ./ tot

    return sort!(percs)
end


function sorted_percents(df, ptition)
    tot = tally(df, "PRE16D", ptition) + tally(df, "PRE16R", ptition)
    dem = tally(df, "PRE16D", ptition)

    percs = dem ./ tot

    return sort!(percs)
end

function sorted_MVAP(df, ptition)
    tot = tally(df, "VAP", ptition)
    mvap = tally(df, "BVAP", ptition) + tally(df, "HVAP", ptition) + tally(df, "ASIANVAP", ptition)
    percs = mvap ./ tot

    return sort!(percs)
end





        