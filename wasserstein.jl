using PSDataIO

using Random, Distributions
using LinearAlgebra
using JuMP, Gurobi
using SparseArrays
using CSV, Tables
using DataFrames

import PyPlot as plt

Random.seed!(42)

##
include("models.jl")

##
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

##
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

# Simple DCOPF for analysis
mopf = Model(Gurobi.Optimizer)
@variable(mopf, p[g=1:ps.Ngen] >= 0)
@constraint(mopf, sum(p) == sum(d))
@constraint(mopf, p .<= ps.gen_pmax)
@constraint(mopf, p .>= ps.gen_pmin)
flow = ptdf*(gen2bus*p - d)
@constraint(mopf,  flow .<= ps.branch_smax)
@constraint(mopf, -flow .<= ps.branch_smax)
gen_cost = cE'*(p.*ps.basemva)
@objective(mopf, Min, gen_cost)
optimize!(mopf)
value.(p)

##
# Experiment 1
# Test with different epsilon
# set support width
simdat = SimData(ps, cE, cR, cA, d, gen2bus, wind2bus, ptdf)
support_width = 0.1
eps_set = [1., 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
lam_res = Dict()
for (ei,e) in enumerate(eps_set)
    for (eei,ee) in enumerate(eps_set)
        act_eps = [e, ee]
        lam = [0., 0.]
        for i in 1:10
            # run 10 times and average to reduce effects from samples
            lam .+= run_robust_wc(simdat, eps, support_width)
        end
        lam_res[(ei,eei)] = 0.1 .* lam
    end
end

data = []
for (k,v) in lam_res
    push!(data, Dict(
    "eps1" => eps_set[k[1]],
    "eps2" => eps_set[k[2]],
    "lam1" => v[1],
    "lam2" => v[2])
    )
end
lam_df = DataFrame(data)
CSV.write("lam_res.csv", lam_df)

##







# # Create some plots

# lam1_grid = zeros(length(eps_set), length(eps_set))
# lam2_grid = zeros(length(eps_set), length(eps_set))
# for (k,v) in lam_res
#     lam1_grid[k[1], k[2]] = v[1]
#     lam2_grid[k[1], k[2]] = v[2]
# end

# fig, axs = plt.subplots(2, 1)
# axs[1].pcolor(lam1_grid)
# axs[2].pcolor(lam2_grid)
# for i in size(lam1_grid, 1)
#     for j in size(lam1_grid, 2)
#         axs[1].text(j, i, "$(lam1_grid[i,j])", ha="center", va="center")
#     end
# end
# gcf()

# CSV.write("lam1.csv",  Tables.table(lam1_grid), writeheader=false)


#






# look at some results

dual.(enerbal)

value(expcost)
value(gencost)
value(rescost)

value.(p)
value.(A)

value.(rp)
value.(rm)

value.(fRAMp)
value.(fRAMm)

value.(flow)