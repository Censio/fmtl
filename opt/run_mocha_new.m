function [rmse, primal_objs, dual_objs, W, Sigma, rho] = run_mocha_new(Xtrain, Ytrain, Xtest, Ytest, lambda, opts, W, Sigma, tr)
% Mocha Method
% Inputs
%   Xtrain: input training data
%   Ytrain: output training data
%   Xtest: input test data
%   Ytest: output test data
%   lambda: regularization parameter
%   opts: optional arguments
% Output
%   Average RMSE across tasks, primal and dual objectives

%% intialize variables
fprintf('===Running MOCHA===\n');
m = length(Xtrain); % # of tasks
d = size(Xtrain{1}, 2); % # of features
%Omega = inv(Sigma);
%tr = trace(W * Omega * W');

totaln = 0; n = zeros(m, 1);
for t = 1:m
    n(t) = length(Ytrain{t});
    totaln = totaln + n(t);
    alpha{t} = zeros(n(t), 1);
end

%% intialize counters
rho = 1.0;
if(opts.w_update)
    rmse = zeros(opts.mocha_inner_iters, 1);
    dual_objs = zeros(opts.mocha_inner_iters, 1); 
    primal_objs = zeros(opts.mocha_inner_iters, 1);
else
    rmse = zeros(opts.mocha_outer_iters, 1);
    dual_objs = zeros(opts.mocha_outer_iters, 1); 
    primal_objs = zeros(opts.mocha_outer_iters, 1);
end

for h = 1:opts.mocha_outer_iters
    [rmse_, primal_obj, dual_obj, W] = updateW(Xtrain, Xtest, Ytrain, Ytest, n, Sigma, rho, W, tr, lambda, opts, alpha);
    rmse(h) = rmse_;
    primal_objs(h) = primal_obj;
    dual_objs(h) = dual_obj;
    
    %% make sure eigenvalues are positive
    A = W'*W;
    if(any(eig(A) < 0))
        [V,Dmat] = eig(A);
        dm= diag(Dmat);
        dm(dm <= 1e-7) = 1e-7;
        D_c = diag(dm);
        A = V*D_c*V';
    end
    
    %% update Omega, Sigma
    sqm = sqrtm(A);
    Sigma = sqm / trace(sqm);
    Omega = inv(Sigma);
    rho = max(sum(abs(Sigma),2)./ diag(Sigma));

    %%
    fprintf('matrix norm: %d, cond: %d, rank: %d\n', norm(W), cond(Sigma), rank(Sigma))
    
    if opts.save
        name = strcat(opts.name, "_result", mat2str(h), ".mat");
        save("-v6", name, "W", "Xtrain", "Xtest", "Ytrain", "Ytest", "Sigma", "Omega", "A", "sqm");
    end
end

end