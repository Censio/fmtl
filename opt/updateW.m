function[rmse, primal_objs, dual_objs, W] = updateW(Xtrain, Xtest, Ytrain, Ytest, n, Sigma, rho, W, tr, lambda, opts, alpha)
    s = size(W);
    d = s(1);
    
    m = length(Xtrain);
    
    if(~opts.w_update)
        curr_err = compute_rmse(Xtest, Ytest, W, opts);
        rmse = curr_err;
    	primal_objs = compute_primal_new(Xtrain, Ytrain, W, tr, lambda);
    	dual_objs = compute_dual_new(alpha, Ytrain, W, tr, lambda);     
    end
    
    % update W
    for hh = 1:opts.mocha_inner_iters
        rng(hh*1000);
        if(opts.sys_het)
            sys_iters = (opts.top - opts.bottom) .* rand(m,1) + opts.bottom;
        end
        
        if(opts.w_update)
            % compute RMSE
            rmse(hh) = compute_rmse(Xtest, Ytest, W, opts);
            primal_objs(hh) = compute_primal_new(Xtrain, Ytrain, W, tr, lambda);
            dual_objs(hh) = compute_dual_new(alpha, Ytrain, W, tr, lambda);
        end
        
        % loop over tasks (in parallel)
        deltaW = zeros(d, m);
        deltaB = zeros(d, m);

        for t = 1:m
            tperm = randperm(n(t));
            alpha_t = alpha{t};
            curr_sig = Sigma(t,t);
            if(opts.sys_het)
                local_iters = n(t) * sys_iters(t);
            else
                local_iters = n(t) * opts.mocha_sdca_frac;
            end
            
            % run SDCA locally
            for s=1:local_iters
                % select random coordinate
                idx = tperm(mod(s, n(t)) + 1);
                alpha_old = alpha_t(idx);
                curr_y = Ytrain{t}(idx);
                curr_x = Xtrain{t}(idx, :);

                % compute update
                update = (curr_y * curr_x * (W(:,t) + rho * deltaW(:, t)));
                grad = lambda * n(t) * (1.0 - update) / (curr_sig * rho * (curr_x * curr_x')) + (alpha_old * curr_y);
                alpha_t(idx) = curr_y * max(0.0, min(1.0, grad));
                deltaW(:, t) = deltaW(:, t) + Sigma(t, t) * (alpha_t(idx) - alpha_old) * curr_x' / (lambda * n(t));
                deltaB(:, t) = deltaB(:, t) + (alpha_t(idx) - alpha_old) * curr_x' / n(t);
                alpha{t} = alpha_t;              
            end            
            
        end
        % combine updates globally
        for t = 1:m
            for tt = 1:m
                W(:, t) = W(:, t) + deltaB(:, tt) * Sigma(t, tt) * (1.0 / lambda);
            end
        end
       
    end
end