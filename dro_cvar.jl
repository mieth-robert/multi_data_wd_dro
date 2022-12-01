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
# wind power data and distributions

ϵj = [0.01, 0.01] # wasserstein budget for each data source
support_width = 0.25
Random.seed!(42)

D = 2 # number of features/data vendors
w = [100, 150] ./ ps.basemva # wind forecast
w_dist = [Normal(0, f*0.15) for f in w] # (unknown) distribution of forecast errors
w_cap = [200, 300] ./ ps.basemva # installed capacity of resource
omega_min = support_width .* (-w)
omega_max = support_width .* (w_cap .- w)
Nprime = 10 # number of standardized samples
ω_hat_sampled = [rand(w_dist[j], Nprime) for j in 1:D] # samples from each data source
ω_hat = [ω_hat_sampled[j] .- mean(ω_hat_sampled[j]) for j in 1:D] # center samples

# basic support
for j in 1:D
    for (i,v) in enumerate(ω_hat[j]) 
        if v < omega_min[j]
            ω_hat[j][i] = omega_min[j]
        elseif  v > omega_max[j]
            ω_hat[j][i] = omega_max[j]
        end
    end
end

##

# some settings
gamma = 0.1 # risk level for chance constraint
FR = 0.8 # factor for flow limits
ω_hat_max = omega_max
ω_hat_min = omega_min

##

# CVAR DCOPF with WD-DR Cost
m = Model(Gurobi.Optimizer)
set_optimizer_attribute(m, "OutputFlag", 1)
@variable(m, p[g=1:ps.Ngen] >=0)
@variable(m, rp[g=1:ps.Ngen] >=0)
@variable(m, rm[g=1:ps.Ngen] >=0)
@variable(m, A[g=1:ps.Ngen, j=1:D] >=0)
@variable(m, fRAMp[l=1:ps.Nbranch] >=0)
@variable(m, fRAMm[l=1:ps.Nbranch] >=0)

# auxillary variables for wc exp. cost reformulation
@variable(m, λ_cost[j=1:D] >=0)
@variable(m, s_cost[j=1:D, i=1:Nprime])

# auxillary variables for CVaR reformulation
@variable(m, τ)
a_mat = [-A; A; ptdf*(wind2bus - gen2bus*A); -ptdf*(wind2bus - gen2bus*A); zeros(D)']
b_vec = [-rp .- τ; -rm .- τ; -fRAMp .- τ; -fRAMm .- τ; 0]
K = size(b_vec)[1] # number of constraints WITH additonal row for CVaR reformulation
@variable(m, v) 
@variable(m, λ_cc[j=1:D] >=0)
@variable(m, s_cc[j=1:D, i=1:Nprime])
@variable(m, z[j=1:D, i=1:Nprime, k=1:K])
@variable(m, u[j=1:D, i=1:Nprime, k=1:K] >= 0)
@variable(m, l[j=1:D, i=1:Nprime, k=1:K] >= 0)

# basic constraints
@constraint(m, enerbal, sum(p) + sum(w) == sum(d))
@constraint(m, p .+ rp .<= ps.gen_pmax)
@constraint(m, p .- rm .>= ps.gen_pmin)
@constraint(m, balbal, A'ones(ps.Ngen) .== ones(ps.Nwind))
flow = ptdf*(gen2bus*p + wind2bus*w - d)
@constraint(m,  flow .== (ps.branch_smax .* FR) .- fRAMp)
@constraint(m, -flow .== (ps.branch_smax .* FR) .- fRAMm)

# # CVaR reformulation of chance constraints
@constraint(m, 0 >= τ + v) 
@constraint(m, gamma * v >= sum(λ_cc[j] * ϵj[j] for j in 1:D ) + (1/Nprime) * sum(s_cc[i] for i in 1:Nprime))
@constraint(m, [i=1:Nprime, k=1:K], s_cc[i] >= b_vec[k] + sum(z[j,i,k]*ω_hat[j][i] + u[j,i,k]*ω_hat_max[j] - l[j,i,k]*ω_hat_min[j] for j in 1:D))
@constraint(m, [j=1:D, i=1:Nprime, k=1:K], u[j,i,k] - l[j,i,k] == a_mat[k,j] - z[j,i,k])
@constraint(m, [j=1:D, i=1:Nprime, k=1:K], λ_cc[j] >=  z[j,i,k])
@constraint(m, [j=1:D, i=1:Nprime, k=1:K], λ_cc[j] >= -z[j,i,k])

# wasserstein worst case cost
@constraint(m, [j=1:D, i=1:Nprime], s_cost[j,i] >= sum((cA[g] .* ps.basemva)*A[g,j]*ω_hat_max[j] for g in 1:ps.Ngen) - λ_cost[j]*(ω_hat_max[j] - ω_hat[j][i]))
@constraint(m, [j=1:D, i=1:Nprime], s_cost[j,i] >= sum((cA[g] .* ps.basemva)*A[g,j]*ω_hat_min[j] for g in 1:ps.Ngen) + λ_cost[j]*(ω_hat_min[j] - ω_hat[j][i]))
@constraint(m, [j=1:D, i=1:Nprime], s_cost[j,i] >= sum((cA[g] .* ps.basemva)*A[g,j]*ω_hat[j][i] for g in 1:ps.Ngen))

# objective
gencost = cE' * (p .* ps.basemva)
rescost = cR' * ((rp .+ rm) .* ps.basemva)
expcost = sum(λ_cost[j]*ϵj[j]  + 1/Nprime * sum(s_cost[j,i] for i in 1:Nprime) for j in 1:D)
@objective(m, Min, gencost + rescost + expcost)
optimize!(m)


##

# look at some data


b_vec[2]


value.(flow)
ps.branch_smax