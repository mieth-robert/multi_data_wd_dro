
function run_robust_wc(simdata, epsilon, support_width; sample_support=false)

    ps = simdata.ps
    cE = simdata.cE
    cR = simdata.cR
    cA = simdata.cA
    d = simdata.d
    gen2bus = simdata.gen2bus
    wind2bus = simdata.wind2bus
    ptdf = simdata.ptdf

    # wind power data and distributions
    D = 2 # number of features/data vendors
    w = [100, 150] ./ ps.basemva # wind forecast
    w_dist = [Normal(0, f*0.15) for f in w] # (unknown) distribution of forecast errors
    Nj = [5, 10] # number of samples from vendor
    ω_hat_sampled =  [rand(w_dist[j], Nj[j]) for j in 1:D] # samples from each data source
    ω_hat = [ω_hat_sampled[j] .- mean(ω_hat_sampled[j]) for j in 1:D] # center samples
  
    # ϵj = [0.1, 0.005] # wasserstein budget for each data source
    ϵj = epsilon


    N_total = prod(Nj)
    emp_support = zeros((D,N_total))
    for j in 1:D
        for i in 0:N_total-1
            emp_support[j, i+1] = ω_hat[j][mod(i,Nj[j])+1]
        end
    end 

    if sample_support
        ω_hat_max = [maximum(ω_hat[j]) for j in 1:D]
        ω_hat_min = [minimum(ω_hat[j]) for j in 1:D]
    else
        ω_hat_max =  w .* support_width
        ω_hat_min = -w .* support_width
    end

    # some tweaks
    FR = 0.8 # factor for flow limits

    # Robust DCOPF with WD-DR Cost
    m = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m, "OutputFlag", 0)
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
    @constraint(m,  flow .== (ps.branch_smax .* FR) .- fRAMp)
    @constraint(m, -flow .== (ps.branch_smax .* FR) .- fRAMm)

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
    @show termination_status(m)
    if termination_status(m)==OPTIMAL
        return value.(λ)
    else
        return false
    end
end