%% load dataset
datarepo = 'data/'; % location of data folder
name = 'small'; % toy dataset
load([datarepo name]); % load data

%% set parameters
addpath('opt/'); addpath('util/'); % add functions
ntrials = 1; % number of trials to run
training_percent = 0.75; % percentage of data for training
opts.obj='C'; % classification
opts.avg = 1; % compute average error across tasks
opts.save = false;

err_mtl = zeros(ntrials, 1);
%%
for trial = 1:ntrials
    rng(trial+2, 'twister'); 
    [Xtrain, Ytrain, Xtest, Ytest] = split_data(X, Y, training_percent);
    
    mocha_lambda = 1.000000e-05 ;
    opts.mocha_outer_iters = 10;
    opts.mocha_inner_iters = 100;
    opts.mocha_sdca_frac = 0.5;
    opts.w_update = 0; % do a full run, not just one w-update
    opts.sys_het = 0; % not messing with systems heterogeneity
    opts.name = "data/batch";
    opts.save = true;
    d = 9;
    m = length(X);
    W = zeros(d, m);
    Sigma = eye(m) * (1/m);  
    
    if true
        fprintf('===\n');
        opts.save = false;
        opts.name = "iterative";
        opts.precision = 'best';
        
        %
        m = m-1;
        W = zeros(d, m);
        Sigma = eye(m) * (1/m);
        xtrain = cell(m,1);
        ytrain = cell(m,1);
        xtest = cell(m,1);
        ytest = cell(m,1);
        for t = 1:m
            xtrain{t} = Xtrain{t};
            xtest{t} = Xtest{t};
            ytrain{t} = Ytrain{t};
            ytest{t} = Ytest{t};
        end
        Right = Sigma\W';
        tr = trace(W * Right);
        [rmse_mocha_reg, primal_mocha_reg, dual_mocha_reg, W, Sigma, rho] = run_mocha_new(xtrain, ytrain, xtest, ytest, mocha_lambda, opts, W, Sigma, tr);
        err_mtl(trial) = rmse_mocha_reg(end);
        fprintf("update1 error: %d\n", rmse_mocha_reg(end));

        %%
        fprintf('===\n');
        m = length(X);
        w = zeros(d,1);
        xtrain = {Xtrain{m}};
        ytrain = {Ytrain{m}};
        xtest = {Xtest{m}};
        ytest = {Ytest{m}};
        n = [length(ytrain{1})];
        alpha = {zeros(length(ytrain{1}), 1)};
        s = 0.05;
        Sigma_new = eye(m) .* s;
        Sigma_new(1:end-1,1:end-1) = (1-s).*Sigma;
        Wnew = horzcat(W, w);
        Right = Sigma_new\Wnew';
        tr = trace(Wnew * Right);
        
        for t = 1:100
            [rmse_, primal_obj, dual_obj, w] = updateW(xtrain, xtest, ytrain, ytest, n, s, rho, w, tr, mocha_lambda, opts, alpha);
            Wnew = horzcat(W, w);
            Sigma_new = add_task(Sigma, Wnew, opts);
            s = Sigma_new(end, end);
            Right = Sigma_new\Wnew';
            tr = trace(Wnew * Right);
        end
        compute_rmse(Xtest, Ytest, Wnew, opts) %0.1575
        save("-v6", "./data/result_addtask", "Wnew", "Xtrain", "Xtest", "Ytrain", "Ytest", "mocha_lambda", "Sigma", "opts");
    end;
end;