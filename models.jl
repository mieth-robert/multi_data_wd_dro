
function run_robust_wc(simdata, samples, ϵj, empirical_support=false)

    ps = simdata.ps
    cE = simdata.cE
    cR = simdata.cR
    cA = simdata.cA
    d = simdata.d
    w = simdata.w

    gen2bus = simdata.gen2bus
    wind2bus = simdata.wind2bus
    ptdf = simdata.ptdf

    Nj = samples.Nj
    D = length(Nj)
    ω_hat = samples.samples
    if empirical_support
        omega_max = samples.omega_max_emp
        omega_min = samples.omega_min_emp
    else
        omega_max = samples.omega_max
        omega_min = samples.omega_min
    end

    # some tweaks
    FR = 0.8 # factor for flow limits

    # Robust DCOPF with WD-DR Cost
    m = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m, "OutputFlag", 0)
    @variable(m, p[g=1:ps.Ngen] >=0)
    @variable(m, rp[g=1:ps.Ngen] >=0)
    @variable(m, rm[g=1:ps.Ngen] >=0)
    @variable(m, λ[j=1:D])
    @variable(m, s[j=1:D, i=1:Nj[j]])
    @variable(m, A[g=1:ps.Ngen, j=1:D] >=0)
    @variable(m, fRAMp[l=1:ps.Nbranch] >=0)
    @variable(m, fRAMm[l=1:ps.Nbranch] >=0)

    @constraint(m, lam_nonneg, λ .>= 0)

    @constraint(m, enerbal, sum(p) + sum(w) == sum(d))
    @constraint(m, p .+ rp .<= ps.gen_pmax)
    @constraint(m, p .- rm .>= ps.gen_pmin)
    @constraint(m, balbal, A' * ones(ps.Ngen) .== ones(ps.Nwind))
    flow = ptdf*(gen2bus*p + wind2bus*w - d)
    @constraint(m, flowlim_up,  flow .== (ps.branch_smax .* FR) .- fRAMp)
    @constraint(m, flowlim_dn, -flow .== (ps.branch_smax .* FR) .- fRAMm)

    # # robust with empirical support
    # for ωi in samples.poly_support_vertices
    #     println(ωi)
    #     @constraint(m, -A*ωi .<= rp)
    #     @constraint(m,  A*ωi .<= rm)
    #     delta_flow = ptdf*(wind2bus - gen2bus*A)*ωi 
    #     @constraint(m,  delta_flow .<= fRAMp)
    #     @constraint(m, -delta_flow .<= fRAMm)
    # end

    # robust with polyhedral support
    C = zeros(2*D, 2)
    C[collect(1:2:2*D), :] = Diagonal(-ones(D))
    C[collect(2:2:2*D), :] = Diagonal(ones(D))
    d = zeros(2*D)
    d[collect(1:2:2*D)] = -omega_min
    d[collect(2:2:2*D)] = omega_max
    @variable(m, yp[jj=1:(2*D), l=1:ps.Nbranch] >= 0)
    @variable(m, ym[jj=1:(2*D), l=1:ps.Nbranch] >= 0)
    @constraint(m,  -A*omega_min .<= rp)
    @constraint(m,  A*omega_max .<= rm)
    @constraint(m,  [l=1:ps.Nbranch], d'*yp[:,l] <=  fRAMp[l])
    @constraint(m,  [l=1:ps.Nbranch], C'*yp[:,l] .== vec(ptdf[l,:]' * (wind2bus - gen2bus*A))) # vec necessary here, to make shape fit
    @constraint(m,  [l=1:ps.Nbranch], d'*ym[:,l] <=  fRAMm[l])
    @constraint(m,  [l=1:ps.Nbranch], C'*ym[:,l] .== vec(-ptdf[l,:]' * (wind2bus - gen2bus*A)))

    # wc cost formulation
    ji_tuples = []
    s_up = []
    s_lo = []
    s_av = []
    for j in 1:D
        for i in 1:Nj[j]
            act_s_up = @constraint(m, s[j,i] >= sum((cA[g]*ps.basemva)*A[g,j]*omega_max[j] for g in 1:ps.Ngen) - λ[j]*(omega_max[j] - ω_hat[j][i]))
            act_s_lo = @constraint(m, s[j,i] >= sum((cA[g]*ps.basemva)*A[g,j]*omega_min[j] for g in 1:ps.Ngen) + λ[j]*(omega_min[j] - ω_hat[j][i]))
            act_s_av = @constraint(m, s[j,i] >= sum((cA[g]*ps.basemva)*A[g,j]*ω_hat[j][i] for g in 1:ps.Ngen))
            push!(s_up, act_s_up)
            push!(s_lo, act_s_lo)
            push!(s_av, act_s_av)
            push!(ji_tuples, (j,i))
        end
    end

    # define objective
    gencost = cE' * (p .* ps.basemva)
    rescost = cR' * ((rp .+ rm) .* ps.basemva)
    expcost = sum(λ[j]*ϵj[j] + 1/Nj[j] * sum(s[j,i] for i in 1:Nj[j]) for j in 1:D)
    @objective(m, Min, gencost + rescost + expcost)
    optimize!(m)
    @show termination_status(m)

    # return 
    if termination_status(m)==OPTIMAL
        s_up_dual = dual.(s_up)
        s_lo_dual = dual.(s_lo)
        s_av_dual = dual.(s_av)
        return (model = m, lambdas = value.(λ), p = value.(p), A = value.(A), lam_nonneg_dual = dual.(m[:lam_nonneg]),
            s_up_dual = Dict(zip(ji_tuples, s_up_dual)), s_lo_dual = Dict(zip(ji_tuples, s_lo_dual)), 
            s_av_dual = Dict(zip(ji_tuples, s_av_dual)), enerbal_dual = dual.(enerbal), balbal_dual = dual.(balbal), 
            expcost = value(expcost), rescost = value(rescost), gencost = value(gencost))
    else
        return false
    end
end


function run_robust_wc_milage(simdata, samples, ϵj)
# same as run_robust_wc but with milage cost

    ps = simdata.ps
    cE = simdata.cE
    cR = simdata.cR
    cA = simdata.cA
    d = simdata.d
    w = simdata.w

    gen2bus = simdata.gen2bus
    wind2bus = simdata.wind2bus
    ptdf = simdata.ptdf

    Nj = samples.Nj
    D = length(Nj)
    ω_hat = samples.samples
    if empirical_support
        omega_max = samples.omega_max_emp
        omega_min = samples.omega_min_emp
    else
        omega_max = samples.omega_max
        omega_min = samples.omega_min
    end

    # some tweaks
    FR = 0.8 # factor for flow limits

    # Robust DCOPF with WD-DR Cost
    m = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m, "OutputFlag", 0)
    @variable(m, p[g=1:ps.Ngen] >=0)
    @variable(m, rp[g=1:ps.Ngen] >=0)
    @variable(m, rm[g=1:ps.Ngen] >=0)
    @variable(m, λ[j=1:D])
    @variable(m, s[j=1:D, i=1:Nj[j]])
    @variable(m, A[g=1:ps.Ngen, j=1:D] >=0)
    @variable(m, fRAMp[l=1:ps.Nbranch] >=0)
    @variable(m, fRAMm[l=1:ps.Nbranch] >=0)

    @constraint(m, lam_nonneg, λ .>= 0)

    @constraint(m, enerbal, sum(p) + sum(w) == sum(d))
    @constraint(m, p .+ rp .<= ps.gen_pmax)
    @constraint(m, p .- rm .>= ps.gen_pmin)
    @constraint(m, balbal, A' * ones(ps.Ngen) .== ones(ps.Nwind))
    flow = ptdf*(gen2bus*p + wind2bus*w - d)
    @constraint(m,  flow .== (ps.branch_smax .* FR) .- fRAMp)
    @constraint(m, -flow .== (ps.branch_smax .* FR) .- fRAMm)

    # robust with polyhedral support
    C = zeros(2*D, 2)
    C[collect(1:2:2*D), :] = Diagonal(-ones(D))
    C[collect(2:2:2*D), :] = Diagonal(ones(D))
    d = zeros(2*D)
    d[collect(1:2:2*D)] = -omega_min
    d[collect(2:2:2*D)] = omega_max
    @variable(m, yp[jj=1:(2*D), l=1:ps.Nbranch] >= 0)
    @variable(m, ym[jj=1:(2*D), l=1:ps.Nbranch] >= 0)
    @constraint(m,  -A*omega_min .<= rp)
    @constraint(m,  A*omega_max .<= rm)
    @constraint(m,  [l=1:ps.Nbranch], d'*yp[:,l] <=  fRAMp[l])
    @constraint(m,  [l=1:ps.Nbranch], C'*yp[:,l] .== vec(ptdf[l,:]' * (wind2bus - gen2bus*A))) # vec necessary here, to make shape fit
    @constraint(m,  [l=1:ps.Nbranch], d'*ym[:,l] <=  fRAMm[l])
    @constraint(m,  [l=1:ps.Nbranch], C'*ym[:,l] .== vec(-ptdf[l,:]' * (wind2bus - gen2bus*A)))

    # wc cost formulation
    ji_tuples = []
    s_up = []
    s_lo = []
    s_av = []
    for j in 1:D
        for i in 1:Nj[j]
            act_s_up = @constraint(m, s[j,i] >= sum((cA[g].*ps.basemva)*A[g,j]*omega_max[j] for g in 1:ps.Ngen) - λ[j]*(omega_max[j] - ω_hat[j][i]))
            act_s_lo = @constraint(m, s[j,i] >= sum((-cA[g].*ps.basemva)*A[g,j]*omega_min[j] for g in 1:ps.Ngen) + λ[j]*(omega_min[j] - ω_hat[j][i]))
            act_s_avpos = @constraint(m, s[j,i] >= sum((cA[g].*ps.basemva)*A[g,j]*ω_hat[j][i] for g in 1:ps.Ngen))
            act_s_avneg = @constraint(m, s[j,i] >= sum((-cA[g].*ps.basemva)*A[g,j]*ω_hat[j][i] for g in 1:ps.Ngen))
            push!(s_up, act_s_up)
            push!(s_lo, act_s_lo)
            push!(s_av, act_s_avpos)
            push!(s_av, act_s_avneg)
            push!(ji_tuples, (j,i))
        end
    end

    # define objective
    gencost = cE' * (p .* ps.basemva)
    rescost = cR' * ((rp .+ rm) .* ps.basemva)
    expcost = sum(λ[j]*ϵj[j] + 1/Nj[j] * sum(s[j,i] for i in 1:Nj[j]) for j in 1:D)
    @objective(m, Min, gencost + rescost + expcost)
    optimize!(m)
    @show termination_status(m)

    # return
    if termination_status(m)==OPTIMAL
        s_up_dual = dual.(s_up)
        s_lo_dual = dual.(s_lo)
        s_av_dual = dual.(s_av)
        return (model = m, lambdas = value.(λ), p = value.(p), A = value.(A), lam_nonneg_dual = dual.(m[:lam_nonneg]),
            s_up_dual = Dict(zip(ji_tuples, s_up_dual)), s_lo_dual = Dict(zip(ji_tuples, s_lo_dual)), 
            s_av_dual = Dict(zip(ji_tuples, s_av_dual)), enerbal_dual = dual.(enerbal), balbal_dual = dual.(balbal), 
            expcost = value(expcost), rescost = value(rescost), gencost = value(gencost))
    else
        return false
    end
end

function run_cvar_wc(simdata, samples, ϵj; gamma=0.1, mileage_cost=false)
# run with cvar approximated joint chance constraint

    ps = simdata.ps
    cE = simdata.cE
    cR = simdata.cR
    cA = simdata.cA
    d = simdata.d
    w = simdata.w

    gen2bus = simdata.gen2bus
    wind2bus = simdata.wind2bus
    ptdf = simdata.ptdf

    Nprime = samples.Nj[1]
    D = length(samples.Nj)
    ω_hat = samples.samples
    omega_max = samples.omega_max
    omega_min = samples.omega_min

    # some settings
    FR = 0.8 # factor for flow limits

    # CVAR DCOPF with WD-DR Cost
    m = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m, "OutputFlag", 0)
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
    @constraint(m, balbal, A' * ones(ps.Ngen) .== ones(ps.Nwind))
    flow = ptdf*(gen2bus*p + wind2bus*w - d)
    @constraint(m, flowlim_up,  flow .== (ps.branch_smax .* FR) .- fRAMp)
    @constraint(m, flowlim_dn, -flow .== (ps.branch_smax .* FR) .- fRAMm)

    # # CVaR reformulation of chance constraints
    @constraint(m, 0 >= τ + v) 
    @constraint(m, gamma * v >= sum(λ_cc[j] * ϵj[j] for j in 1:D ) + (1/Nprime) * sum(s_cc[i] for i in 1:Nprime))
    @constraint(m, [i=1:Nprime, k=1:K], s_cc[i] >= b_vec[k] + sum(z[j,i,k]*ω_hat[j][i] + u[j,i,k]*omega_max[j] - l[j,i,k]*omega_min[j] for j in 1:D))
    @constraint(m, [j=1:D, i=1:Nprime, k=1:K], u[j,i,k] - l[j,i,k] == a_mat[k,j] - z[j,i,k])
    @constraint(m, [j=1:D, i=1:Nprime, k=1:K], λ_cc[j] >=  z[j,i,k])
    @constraint(m, [j=1:D, i=1:Nprime, k=1:K], λ_cc[j] >= -z[j,i,k])

    # wasserstein worst case cost
    if mileage_cost 
    # calculate cost of reserve activation as c^A*abs(r(ω))
        @constraint(m, [j=1:D, i=1:Nprime], s_cost[j,i] >= sum(( cA[g]*ps.basemva)*A[g,j]*omega_max[j] for g in 1:ps.Ngen) - λ_cost[j]*(omega_max[j] - ω_hat[j][i]))
        @constraint(m, [j=1:D, i=1:Nprime], s_cost[j,i] >= sum((-cA[g]*ps.basemva)*A[g,j]*omega_min[j] for g in 1:ps.Ngen) + λ_cost[j]*(omega_min[j] - ω_hat[j][i]))
        @constraint(m, [j=1:D, i=1:Nprime], s_cost[j,i] >= sum(( cA[g]*ps.basemva)*A[g,j]*ω_hat[j][i] for g in 1:ps.Ngen))
        @constraint(m, [j=1:D, i=1:Nprime], s_cost[j,i] >= sum((-cA[g]*ps.basemva)*A[g,j]*ω_hat[j][i] for g in 1:ps.Ngen))
    else
        @constraint(m, [j=1:D, i=1:Nprime], s_cost[j,i] >= sum((cA[g]*ps.basemva)*A[g,j]*omega_max[j] for g in 1:ps.Ngen) - λ_cost[j]*(omega_max[j] - ω_hat[j][i]))
        @constraint(m, [j=1:D, i=1:Nprime], s_cost[j,i] >= sum((cA[g]*ps.basemva)*A[g,j]*omega_min[j] for g in 1:ps.Ngen) + λ_cost[j]*(omega_min[j] - ω_hat[j][i]))
        @constraint(m, [j=1:D, i=1:Nprime], s_cost[j,i] >= sum((cA[g]*ps.basemva)*A[g,j]*ω_hat[j][i] for g in 1:ps.Ngen))
    end

    # objective
    gencost = cE' * (p .* ps.basemva)
    rescost = cR' * ((rp .+ rm) .* ps.basemva)
    expcost = sum(λ_cost[j]*ϵj[j]  + 1/Nprime * sum(s_cost[j,i] for i in 1:Nprime) for j in 1:D)
    @objective(m, Min, gencost + rescost + expcost)
    optimize!(m)
    @show termination_status(m)

     # return 
     if termination_status(m)==OPTIMAL
        return (model = m, lambdas_cost = value.(λ_cost), lambdas_cc = value.(λ_cc), 
                p = value.(p), A = value.(A), enerbal_dual = dual.(enerbal), balbal_dual = dual.(balbal), 
                expcost = value(expcost), rescost = value(rescost), gencost = value(gencost))
    else
        return false
    end
end


