using PSDataIO

using Random, Distributions
using LinearAlgebra
using JuMP, Gurobi
using SparseArrays

Random.seed!(42)

# power system data
case_file = "data/matpower/case5.m"
matdata = read_matpower_m_file(case_file)
ps = create_topology_data_from_matpower_data(matdata)

ps.branch_smax = [4.; 1.9; 2.2; 1.; 1.; 2.4;] # only 1 and 6 are defined in the case5 dataset

# wind power data and distributions
D = 2 # number of features/data vendors
w = [100, 150] ./ ps.basemva # wind forecast
w_dist = [Normal(0, f*0.15) for f in w] # (unknown) distribution of forecast errors
Nj = [5, 10] # number of samples from vendor
ω_hat =  [rand(w_dist[j], Nj[j]) for j in 1:D] # samples from each data source
ϵj = [0.1, 0.001] # wasserstein budget for each data source
ps.wind_loc = [3,5]
ps.Nwind = 2

N_total = prod(Nj)
emp_support = zeros((D,N_total))
for j in 1:D
    for i in 0:N_total-1
        emp_support[j, i+1] = ω_hat[j][mod(i,Nj[j])+1]
    end
end 
ω_hat_max = [maximum(ω_hat[j]) for j in 1:D]
ω_hat_min = [minimum(ω_hat[j]) for j in 1:D]
 
# additonal data
d = matdata.bus[:,3] ./ ps.basemva
cE = ps.gen_cost_lin
cR = cA = [100; 100; 10; 10; 100]

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


# Robust DCOPF with WD-DR Cost
m = Model(Gurobi.Optimizer)
@variable(m, p[g=1:ps.Ngen] >=0)
@variable(m, rp[g=1:ps.Ngen] >=0)
@variable(m, rm[g=1:ps.Ngen] >=0)
@variable(m, λ[j=1:D] >=0)
@variable(m, s[j=1:D, i=1:Nj[j]])
@variable(m, A[g=1:ps.Ngen, j=1:D] >=0)
@variable(m, fRAMp[l=1:ps.Nbranch] >=0)
@variable(m, fRAMm[l=1:ps.Nbranch] >=0)

@constraint(m, enerbal, sum(p) + sum(w) == sum(d))
@constraint(m, p .+ rp .<= ps.gen_pmax)
@constraint(m, p .- rm .>= ps.gen_pmin)
@constraint(m, A'ones(ps.Ngen) .== ones(ps.Nwind))
flow = ptdf*(gen2bus*p + wind2bus*w - d)
@constraint(m,  flow .== ps.branch_smax .- fRAMp)
@constraint(m, -flow .== ps.branch_smax .- fRAMm)

for ω_hat in emp_support
    @constraint(m, -A*ω_hat .<= rp)
    @constraint(m,  A*ω_hat .<= rm)
    delta_flow = ptdf*(wind2bus - gen2bus*A)*ω_hat
    @constraint(m,  delta_flow .<= fRAMp)
    @constraint(m, -delta_flow .<= fRAMm)
end

for j in 1:D
    for i in 1:Nj[j]
        @constraint(m, s[j,i] >= sum((cA[g].*ps.basemva)*A[g,j]*ω_hat_max[j] for g in 1:ps.Ngen) - λ[j]*(ω_hat_max[j] - ω_hat[j][i]))
        @constraint(m, s[j,i] >= sum((cA[g].*ps.basemva)*A[g,j]*ω_hat_min[j] for g in 1:ps.Ngen) + λ[j]*(ω_hat_min[j] - ω_hat[j][i]))
        @constraint(m, s[j,i] >= sum((cA[g].*ps.basemva)*A[g,j]*ω_hat[j][i] for g in 1:ps.Ngen))
    end
end

gencost = cE' * (p .* ps.basemva)
rescost = cR' * ((rp .+ rm) .* ps.basemva)
expcost = sum(λ[j]*ϵj[j] + 1/Nj[j] * sum(s[j,i] for i in 1:Nj[j]) for j in 1:D)
@objective(m, Min, gencost + rescost + expcost)
optimize!(m)

value.(λ)
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