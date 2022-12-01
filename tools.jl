
mutable struct SimData
    ps::Any
    cE::Vector{Float64}
    cR::Vector{Float64}
    cA::Vector{Float64}
    d::Vector{Float64}
    w::Vector{Float64} # injections from uncertain resources [pu]
    w_cap::Vector{Float64} # installed capacity of uncertain resources [pu]
    support_width::Float64 # width of robust interval for uncertain resources [0..1]
    gen2bus::Union{SparseMatrixCSC, Matrix}
    wind2bus::Union{SparseMatrixCSC, Matrix}
    ptdf::Union{SparseMatrixCSC, Matrix}
end


function create_sample_data(simdata::SimData, Nj::Vector{Int64}, rel_stdv::Vector{Float64})
# create data with different N for each feature

    # unpack data
    D = length(simdata.w)
    w = simdata.w
    w_cap = simdata.w_cap
    support_width = simdata.support_width

    # build support
    omega_min = support_width .* (-w)
    omega_max = support_width .* (w_cap .- w)

    # create distributions and sample
    w_dist = [truncated(Normal(0, rel_stdv[j]*w[j]), omega_min[j], omega_max[j]) for j in 1:D] # (unknown) distribution of forecast errors
    ω_hat_sampled =  [rand(w_dist[j], Nj[j]) for j in 1:D] # samples from each data source
    ω_hat = [ω_hat_sampled[j] .- mean(ω_hat_sampled[j]) for j in 1:D] # center samples
    for j in 1:D # truncate to fit into support
        for (i,v) in enumerate(ω_hat[j]) 
            if v < omega_min[j]
                ω_hat[j][i] = omega_min[j]
            elseif  v > omega_max[j]
                ω_hat[j][i] = omega_max[j]
            end
        end
    end

    # empirical support is all possible combinations of observerd samples
    emp_support = vec(collect(Base.product(ω_hat...)))
    emp_support = [collect(tup) for tup in emp_support]

    # polyhedral support vertices
    support_corners = [[omega_min[j], omega_max[j]] for j in 1:D]
    poly_support_vertices = vec(collect(Base.product(support_corners...)))
    poly_support_vertices = [collect(tup) for tup in poly_support_vertices]
    
    return (samples=ω_hat, emp_support=emp_support, poly_support_vertices=poly_support_vertices, 
            omega_min=omega_min, omega_max=omega_max, Nj=Nj)
end

function create_sample_data_standardized(simdata::SimData, Nprime::Int64, rel_stdv::Vector{Float64})
# create data with same N for each feature
    D = length(simdata.w)
    return create_sample_data(simdata, repeat([Nprime], D), rel_stdv)
end
