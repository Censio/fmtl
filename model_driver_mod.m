%% driver for prediction error of global, local, and MTL models

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

%% set hyperparameter search space
lambda_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 10]; % regularizer

%% initialize
err_constant = zeros(ntrials, 1);
err_local = zeros(ntrials, 1);
err_global = zeros(ntrials, 1);
err_mtl_batch = zeros(ntrials, 1);
err_mtl = zeros(ntrials, 1);

%dbstop("run_mocha", 42)
%dbstop("run_mocha", 70)
%dbstop("run_mocha", 91)
%dbstop("run_mocha", 113)
%dbstop("run_mocha", 128)
%dbstop("run_mocha", 135)
%dbstop("model_driver", 65)

%% compare global, local, and MTL over iters trials
for trial = 1:ntrials
    %% partition the data randomly
    %trial = 1
    rng(trial+2, 'twister'); 
    [Xtrain, Ytrain, Xtest, Ytest] = split_data(X, Y, training_percent);

    %% global model
    if true;
        opts.type = 'global';
        opts.max_sdca_iters = 500;
        opts.tol = 1e-5;
        global_lambda = cross_val_1(Xtrain, Ytrain, 'baselines', opts, lambda_range, 5, [], []); % determine via 5-fold cross val
        %global_lambda = 1.0000e-04
        %err_global = 0.2322
        [err_global(trial), Wglobal] = baselines(Xtrain, Ytrain, Xtest, Ytest, global_lambda, opts, [], []);
        save("-v6", "./data/result_global", "Wglobal", "Xtrain", "Xtest", "Ytrain", "Ytest", "global_lambda", "opts");
    end
    
    %% local model
    if true;
        opts.type = 'local';
        opts.max_sdca_iters = 500;
        opts.tol = 1e-5;
        local_lambda = cross_val_1(Xtrain, Ytrain, 'baselines', opts, lambda_range, 5, [], []); % determine via 5-fold cross val
        %local_lambda = 1.0000e-03
        %local error = 1.931182e-01
        [err_local(trial), Wlocal] = baselines(Xtrain, Ytrain, Xtest, Ytest, local_lambda, opts, [], []);
        save("-v6", "./data/result_local", "Wlocal", "Xtrain", "Xtest", "Ytrain", "Ytest", "local_lambda", "opts");
        fprintf("local error: %d\n", err_local(end));
    end
    
    %% MTL model (mocha)
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
    
    %mocha_lambda = cross_val_1(Xtrain, Ytrain, 'run_mocha', opts, lambda_range, 5, W, Sigma); % determine via 5-fold cross val
    mocha_lambda = 1.000000e-05 ;
    fprintf("lambda: %d \n", mocha_lambda);
       
    %%
    if false;
        fprintf('===\n');
 
        Right = Sigma\W';
        tr = trace(W * Right);
        [rmse_mocha_reg, primal_mocha_reg, dual_mocha_reg, W, Sigma, rho] = run_mocha_new(Xtrain, Ytrain, Xtest, Ytest, mocha_lambda, opts, W, Sigma, tr);
        err_mtl_batch(trial) = rmse_mocha_reg(end);
        %batch error(run_mocha_mod) = 1.676245e-01
        %batch error(run_mocha_new) = 1.623195e-01
        compute_rmse(Xtest, Ytest, W, opts) %0.1623
        fprintf("batch error: %d\n", rmse_mocha_reg(end));
    end
    
   
    
end

