using PSDataIO

using Random, Distributions
using LinearAlgebra
using JuMP, Gurobi
using SparseArrays
using CSV, Tables
using DataFrames

import PyPlot as plt

##

include("tools.jl")
include("models.jl")

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

##
# Experiment 1
# Test with different epsilon
# 
w = [100, 150] ./ ps.basemva # wind forecast
w_cap = [200, 300] ./ ps.basemva # installed capacity of resource

support_width = 1

simdat = SimData(ps, cE, cR, cA, d, w, w_cap, support_width, gen2bus, wind2bus, ptdf)

eps_set = [1., 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
lam_cost_res = Dict()
lam_cc_res = Dict()
ener_mp_res = Dict()
bal_mp_res = Dict()
for (ei,e) in enumerate(eps_set)
    for (eei,ee) in enumerate(eps_set)
        act_eps = [e, ee]
        lam_cost = [0., 0.]
        lam_cc = [0., 0.]
        ener_mp = 0
        bal_mp = [0., 0.]
        Random.seed!(42) # reset seed here so that every set of parameters has same set of samples
        for i in 1:10
            # run 10 times and average to reduce effects from samples
            samples = create_sample_data_standardized(simdat, 10, [0.15, 0.15])
            res = run_cvar_wc(simdat, samples, act_eps; gamma=0.1, mileage_cost=true)
            lam_cost .+= res.lambdas_cost
            lam_cc .+= res.lambdas_cc
            ener_mp += res.enerbal_dual
            bal_mp += res.balbal_dual
        end
        lam_cost_res[(ei,eei)] = 0.1 .* lam_cost
        lam_cc_res[(ei,eei)] = 0.1 .* lam_cc
        ener_mp_res[(ei,eei)] = 0.1 .* ener_mp
        bal_mp_res[(ei,eei)] = 0.1 .* bal_mp
    end
end

data = []
for (k,v) in lam_cost_res
    push!(data, Dict(
    "eps1" => eps_set[k[1]],
    "eps2" => eps_set[k[2]],
    "lam1_cost" => v[1],
    "lam2_cost" => v[2],
    "lam1_cc" => lam_cc_res[k][1],
    "lam2_cc" => lam_cc_res[k][2],
    "e_price"  => ener_mp_res[k],
    "bal_price1" => bal_mp_res[k][1],
    "bal_price2" => bal_mp_res[k][2])
    )
end
lam_df = DataFrame(data)
lam_df = lam_df[!, ["eps1", "eps2", "lam1_cost", "lam2_cost", "lam1_cc", "lam2_cc", "e_price", "bal_price1", "bal_price2"]]
sort!(lam_df, [:eps2, :eps1], rev=true)
CSV.write("lam_res.csv", lam_df)

##


##
# single run 
w = [100, 150] ./ ps.basemva 
w_cap = [200, 300] ./ ps.basemva 

support_width = 0.25

simdat = SimData(ps, cE, cR, cA, d, w, w_cap, support_width, gen2bus, wind2bus, ptdf)
Random.seed!(42)
samples = create_sample_data_standardized(simdat, 10, [0.15, 0.15])

epsilons = [1, 1] # both lambdas zero

res = run_cvar_wc(simdat, samples, epsilons; gamma=0.1)
objective_value(res.model)

##

# look at some data


b_vec[2]


value.(flow)
ps.branch_smax