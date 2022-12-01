using PSDataIO

using Random, Distributions
using LinearAlgebra
using JuMP, Gurobi
using SparseArrays
using CSV, Tables
using DataFrames

import PyPlot as plt

##

# power system data
case_file = "data/matpower/case5.m"
matdata = read_matpower_m_file(case_file)
ps = create_topology_data_from_matpower_data(matdata)

ps.branch_smax = [4.; 1.9; 2.2; 1.; 1.; 2.4;] # only 1 and 6 are limited in the case5 dataset
ps.wind_loc = [3, 5]
ps.Nwind = 2

# additonal data
d = matdata.bus[:,3] ./ ps.basemva
cE = ps.gen_cost_lin
cR = cA = [15; 80; 30; 30; 80]

gen2bus = sparse(ps.gen_loc, 1:ps.Ngen, ones(ps.Ngen), ps.Nbus, ps.Ngen)
wind2bus = sparse(ps.wind_loc, 1:ps.Nwind, ones(ps.Nwind), ps.Nbus, ps.Nwind)
ptdf = create_ptdf_matrix(ps)

w = [100, 150] ./ ps.basemva # wind forecast
w_cap = [200, 300] ./ ps.basemva

##

# Simple DCOPF for analysis
mopf = Model(Gurobi.Optimizer)
@variable(mopf, p[g=1:ps.Ngen] >= 0)
@constraint(mopf, sum(p) + sum(w) == sum(d))
@constraint(mopf, p .<= ps.gen_pmax)
@constraint(mopf, p .>= ps.gen_pmin)
flow = ptdf*(gen2bus*p + wind2bus*w - d)
@constraint(mopf,  flow .<= ps.branch_smax)
@constraint(mopf, -flow .<= ps.branch_smax)
gen_cost = cE'*(p.*ps.basemva)
@objective(mopf, Min, gen_cost)
optimize!(mopf)

objective_value(mopf)
value.(p)


##

D=2
CR = 0.8

# Identify maximum wind capacity
# Maximum for sum(w) = [4.16, 4.37] for locations [3,5] (CR=1)
# 3.55 and 3.67 for CR = 0.8

mopf = Model(Gurobi.Optimizer)
@variable(mopf, p[g=1:ps.Ngen] >= 0)
@variable(mopf, wcap[j=1:D] >=0 )
@constraint(mopf, sum(p) + sum(wcap) == sum(d))
@constraint(mopf, p .<= ps.gen_pmax)
@constraint(mopf, p .>= ps.gen_pmin)
# @constraint(mopf, wcap[1] <= 2)
# @constraint(mopf, wcap[2] <= 3)
flow = ptdf*(gen2bus*p + wind2bus*wcap - d)
@constraint(mopf,  flow .<= ps.branch_smax .* CR)
@constraint(mopf, -flow .<= ps.branch_smax .* CR)
gen_cost = cE'*(p.*ps.basemva)
@objective(mopf, Max, sum(wcap))
optimize!(mopf)

value.(p)
value.(wcap)


##
# Simple robust model to test polyhedral formulation
FR = 0.8
support_width = 0.5
omega_min = support_width .* (-w)
omega_max = support_width .* (w_cap .- w)
support_corners = [[omega_min[j], omega_max[j]] for j in 1:D]
emp_support_alt = vec(collect(Base.product(support_corners...)))
emp_support_alt = [collect(tup) for tup in emp_support_alt]

m = Model(Gurobi.Optimizer)
@variable(m, p[g=1:ps.Ngen] >=0)
@variable(m, rp[g=1:ps.Ngen] >=0)
@variable(m, rm[g=1:ps.Ngen] >=0)
@variable(m, A[g=1:ps.Ngen, j=1:D] >=0)
@variable(m, fRAMp[l=1:ps.Nbranch] >=0)
@variable(m, fRAMm[l=1:ps.Nbranch] >=0)

@constraint(m, enerbal, sum(p) + sum(w) == sum(d))
@constraint(m, p .+ rp .<= ps.gen_pmax)
@constraint(m, p .- rm .>= ps.gen_pmin)
@constraint(m, balbal, A' * ones(ps.Ngen) .== ones(ps.Nwind))
flow = ptdf*(gen2bus*p + wind2bus*w - d)
@constraint(m,  flow .== (ps.branch_smax .* FR) .- fRAMp)
@constraint(m, -flow .== (ps.branch_smax .* FR) .- fRAMm)

# # robust with empirical support
# for ωi in emp_support_alt
#     @constraint(m, -A*ωi .<= rp)
#     @constraint(m,  A*ωi .<= rm)
#     delta_flow = ptdf*(wind2bus - gen2bus*A)*ωi 
#     @constraint(m,  delta_flow .<= fRAMp)
#     @constraint(m,  -delta_flow .<= fRAMm)
# end

# robust with polyhedral support
Cs = zeros(2*D, 2)
Cs[collect(1:2:2*D), :] = Diagonal(-ones(D))
Cs[collect(2:2:2*D), :] = Diagonal(ones(D))
ds = zeros(2*D)
ds[collect(1:2:2*D)] = -omega_min
ds[collect(2:2:2*D)] = omega_max
@variable(m, yp[jj=1:(2*D), l=1:ps.Nbranch] >= 0)
@variable(m, ym[jj=1:(2*D), l=1:ps.Nbranch] >= 0)
@constraint(m,  -A*omega_min .<= rp)
@constraint(m,  A*omega_max .<= rm)
@constraint(m,  [l=1:ps.Nbranch], ds'*yp[:,l] <=  fRAMp[l])
@constraint(m,  [l=1:ps.Nbranch], Cs'*yp[:,l] .== vec(ptdf[l,:]' * (wind2bus - gen2bus*A)))
@constraint(m,  [l=1:ps.Nbranch], ds'*ym[:,l] <=  fRAMm[l])
@constraint(m,  [l=1:ps.Nbranch], Cs'*ym[:,l] .== vec(-ptdf[l,:]' * (wind2bus - gen2bus*A)))


gen_cost = cE'*(p.*ps.basemva)
rescost = cR' * ((rp .+ rm) .* ps.basemva)
@objective(m, Min, gen_cost + rescost)
optimize!(m)
@show value.(m[:rp])
@show value.(m[:rm]);

##
