using PSDataIO

using Random, Distributions
using LinearAlgebra
using JuMP, Gurobi
using SparseArrays
using CSV, Tables
using DataFrames

import PyPlot as plt

##
include("models.jl")

mutable struct SimData
    ps::Any
    cE::Vector{Float64}
    cR::Vector{Float64}
    cA::Vector{Float64}
    d::Vector{Float64}
    gen2bus::Union{SparseMatrixCSC, Matrix}
    wind2bus::Union{SparseMatrixCSC, Matrix}
    ptdf::Union{SparseMatrixCSC, Matrix}
end

# power system data
case_file = "data/matpower/case5.m"
matdata = read_matpower_m_file(case_file)
ps = create_topology_data_from_matpower_data(matdata)

ps.branch_smax = [4.; 1.9; 2.2; 1.; 1.; 2.4;] # only 1 and 6 are limited in the case5 dataset
ps.wind_loc = [3,5]
ps.Nwind = 2

# additonal data
d = matdata.bus[:,3] ./ ps.basemva
cE = ps.gen_cost_lin
cR = cA = [80; 80; 15; 30; 80]

gen2bus = sparse(ps.gen_loc, 1:ps.Ngen, ones(ps.Ngen), ps.Nbus, ps.Ngen)
wind2bus = sparse(ps.wind_loc, 1:ps.Nwind, ones(ps.Nwind), ps.Nbus, ps.Nwind)
ptdf = create_ptdf_matrix(ps)

##
# include("models.jl")

##
# Experiment 1
# Test with different epsilon
# set support width
simdat = SimData(ps, cE, cR, cA, d, gen2bus, wind2bus, ptdf)
support_width = 1
eps_set = [1., 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
lam_res = Dict()
ener_mp_res = Dict()
bal_mp_res = Dict()
for (ei,e) in enumerate(eps_set)
    for (eei,ee) in enumerate(eps_set)
        act_eps = [e, ee]
        lam = [0., 0.]
        ener_mp = 0
        bal_mp = [0., 0.]
        Random.seed!(42) # reset seed here so that every set of parameters has same set of samples
        for i in 1:10
            # run 10 times and average to reduce effects from samples
            res = run_robust_wc(simdat, act_eps, support_width; sample_support=false)
            lam .+= res.lambdas
            ener_mp += res.enerbal_dual
            bal_mp += res.balbal_dual
        end
        lam_res[(ei,eei)] = 0.1 .* lam
        ener_mp_res[(ei,eei)] = 0.1 .* ener_mp
        bal_mp_res[(ei,eei)] = 0.1 .* bal_mp
    end
end

data = []
for (k,v) in lam_res
    push!(data, Dict(
    "eps1" => eps_set[k[1]],
    "eps2" => eps_set[k[2]],
    "lam1" => v[1],
    "lam2" => v[2],
    "e_price"  => ener_mp_res[k],
    "bal_price1" => bal_mp_res[k][1],
    "bal_price2" => bal_mp_res[k][2])
    )
end
lam_df = DataFrame(data)
lam_df = lam_df[!, ["eps1", "eps2", "lam1", "lam2", "e_price", "bal_price1", "bal_price2"]]
sort!(lam_df, [:eps2, :eps1], rev=true)
CSV.write("lam_res.csv", lam_df)

##

## 
# Single run
include("models.jl")
Random.seed!(42)
simdat = SimData(ps, cE, cR, cA, d, gen2bus, wind2bus, ptdf)
support_width = 0.5

res = run_robust_wc(simdat, [0.1, 0.1], support_width; sample_support=false, Nj=[5,10]);

dual_df = DataFrame(
    j = [k[1] for (k,v) in res.s_up_dual],
    i = [k[2] for (k,v) in res.s_up_dual],
    up = [v for (k,v) in res.s_up_dual],
    lo = [v for (k,v) in res.s_lo_dual],
    av = [v for (k,v) in res.s_av_dual],
)
CSV.write("dual_res.csv", dual_df)
##

# for i in 1:1000
#     res = run_robust_wc(simdat, [0.1, 0.1], 0.25; sample_support=false, Nj=[5,10])
#     if sum([res.s_up_dual[k] * v for (k,v) in res.s_lo_dual]) != 0
#         dual_df = DataFrame(
#             j = [k[1] for (k,v) in res.s_up_dual],
#             i = [k[2] for (k,v) in res.s_up_dual],
#             up = [v for (k,v) in res.s_up_dual],
#             lo = [v for (k,v) in res.s_lo_dual],
#             av = [v for (k,v) in res.s_av_dual],
#         )
#         CSV.write("dual_res.csv", dual_df)
#         println("NONZERO")
#         break
#     end
# end

##

