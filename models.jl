
function run_robust_wc(simdata, epsilon, support_width; sample_support=false, Nj=[5,10], rel_stv=[0.15,0.15])

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
    w_cap = [200, 300] ./ ps.basemva # installed capacity of resource
    omega_min = support_width .* (-w)
    omega_max = support_width .* (w_cap .- w)
    w_dist = [truncated(Normal(0, rel_stv[j]*w[j]), omega_min[j], omega_max[j]) for j in 1:D] # (unknown) distribution of forecast errors
    # Nj = [8, 10] # number of samples from vendor
    ω_hat_sampled =  [rand(w_dist[j], Nj[j]) for j in 1:D] # samples from each data source
    ω_hat = [ω_hat_sampled[j] .- mean(ω_hat_sampled[j]) for j in 1:D] # center samples
    for j in 1:D
        for (i,v) in enumerate(ω_hat[j]) 
            if v < omega_min[j]
                ω_hat[j][i] = omega_min[j]
            elseif  v > omega_max[j]
                ω_hat[j][i] = omega_max[j]
            end
        end
    end

    # ϵj = [0.1, 0.005] # wasserstein budget for each data source
    ϵj = epsilon
    
    # empirical support is all possible combinations of observerd samples
    emp_support = vec(collect(Base.product(ω_hat...)))
    emp_support = [collect(tup) for tup in emp_support]
    support_corners = [[omega_min[j], omega_max[j]] for j in 1:D]
    emp_support_alt = vec(collect(Base.product(support_corners...)))
    emp_support_alt = [collect(tup) for tup in emp_support_alt]
    
    if sample_support
        ω_hat_max = [maximum(ω_hat[j]) for j in 1:D]
        ω_hat_min = [minimum(ω_hat[j]) for j in 1:D]
    else
        # ω_hat_max =  w .* support_width
        # ω_hat_min = -w .* support_width
        ω_hat_max = omega_max
        ω_hat_min = omega_min
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

    # # robust with empirical support
    # for ωi in emp_support_alt
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
            act_s_up = @constraint(m, s[j,i] >= sum((cA[g].*ps.basemva)*A[g,j]*ω_hat_max[j] for g in 1:ps.Ngen) - λ[j]*(ω_hat_max[j] - ω_hat[j][i]))
            act_s_lo = @constraint(m, s[j,i] >= sum((cA[g].*ps.basemva)*A[g,j]*ω_hat_min[j] for g in 1:ps.Ngen) + λ[j]*(ω_hat_min[j] - ω_hat[j][i]))
            act_s_av = @constraint(m, s[j,i] >= sum((cA[g].*ps.basemva)*A[g,j]*ω_hat[j][i] for g in 1:ps.Ngen))
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


function run_robust_wc_milage(simdata, epsilon, support_width; sample_support=false, Nj=[5,10], rel_stv=[0.15,0.15])
# same as run_robust_wc but with milage cost

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
    w_cap = [200, 300] ./ ps.basemva # installed capacity of resource
    omega_min = support_width .* (-w)
    omega_max = support_width .* (w_cap .- w)
    w_dist = [truncated(Normal(0, rel_stv[j]*w[j]), omega_min[j], omega_max[j]) for j in 1:D] # (unknown) distribution of forecast errors
    # Nj = [8, 10] # number of samples from vendor
    ω_hat_sampled =  [rand(w_dist[j], Nj[j]) for j in 1:D] # samples from each data source
    ω_hat = [ω_hat_sampled[j] .- mean(ω_hat_sampled[j]) for j in 1:D] # center samples
    for j in 1:D
        for (i,v) in enumerate(ω_hat[j]) 
            if v < omega_min[j]
                ω_hat[j][i] = omega_min[j]
            elseif  v > omega_max[j]
                ω_hat[j][i] = omega_max[j]
            end
        end
    end

    # ϵj = [0.1, 0.005] # wasserstein budget for each data source
    ϵj = epsilon
    
    # empirical support is all possible combinations of observerd samples
    emp_support = vec(collect(Base.product(ω_hat...)))
    emp_support = [collect(tup) for tup in emp_support]
    support_corners = [[omega_min[j], omega_max[j]] for j in 1:D]
    emp_support_alt = vec(collect(Base.product(support_corners...)))
    emp_support_alt = [collect(tup) for tup in emp_support_alt]
    
    if sample_support
        ω_hat_max = [maximum(ω_hat[j]) for j in 1:D]
        ω_hat_min = [minimum(ω_hat[j]) for j in 1:D]
    else
        # ω_hat_max =  w .* support_width
        # ω_hat_min = -w .* support_width
        ω_hat_max = omega_max
        ω_hat_min = omega_min
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
            act_s_up = @constraint(m, s[j,i] >= sum((cA[g].*ps.basemva)*A[g,j]*ω_hat_max[j] for g in 1:ps.Ngen) - λ[j]*(ω_hat_max[j] - ω_hat[j][i]))
            act_s_lo = @constraint(m, s[j,i] >= sum((-cA[g].*ps.basemva)*A[g,j]*ω_hat_min[j] for g in 1:ps.Ngen) + λ[j]*(ω_hat_min[j] - ω_hat[j][i]))
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
