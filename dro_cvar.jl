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
# single run 
w = [100, 150] ./ ps.basemva 
w_cap = [200, 300] ./ ps.basemva 

support_width = 0.25

simdat = SimData(ps, cE, cR, cA, d, w, w_cap, support_width, gen2bus, wind2bus, ptdf)
Random.seed!(42)
samples = create_sample_data_standardized(simdat, 10, [0.15, 0.15])

epsilons = [1, 1] # both lambdas zero

res = run_cvar_wc(simdat, samples, epsilons; gamma=0.1)
objective_value(res)

##

# look at some data


b_vec[2]


value.(flow)
ps.branch_smax