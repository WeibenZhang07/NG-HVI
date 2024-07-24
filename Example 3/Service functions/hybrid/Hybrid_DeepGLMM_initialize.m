
function [ADA,draws,dim,lambda,M,ng_old,nn,p,Predictive,Predictive_item,priors,repeat,time,va] = Hybrid_DeepGLMM_initialize(X_train,N,ada)

    % priors
    priors.w_sig = 1/0.02;
    priors.beta_sig = 1/0.2;
    priors.Gamma_a = 0.1;2;
    priors.Gamma_b = 0.1;

    repeat = 5;

    %model set-up
    p=1; % 1 factors

    dim.sample = length(X_train);
    [dim.T,dim.x] = size(X_train{1});

    nn = [5,5];
    layer = length(nn);
    dim.input = dim.x-1;
    NN = [dim.input,nn];
    dim.w_all = zeros(layer,1);
    for i = 1:layer
        dim.w_all(i) = NN(i+1)*(NN(i)+1);
    end
    dim.beta = nn(end)+1;
    dim.Gamma = dim.beta;

    M = sum(dim.w_all) + dim.beta + dim.Gamma; % w, beta, Gamma

    %variational parameters initialization
    draws.mu = zeros(M,1);
    weights = InitializeNN_hybrid(NN);

    weights_ini = [];
    for l = 1:length(weights)
        weights_ini = [weights_ini;weights{l}(:)];
    end

    draws.mu(1:sum(dim.w_all)) = weights_ini;


    draws.B = zeros(M,p) + 0.01;
    draws.B(~tril(ones(size(draws.B)))) = 0;
    draws.z = normrnd(0,1,p,1);
    draws.d = zeros(M,1)+ 0.01;

    %latent variables initialization
    draws.ystar = cell(1,dim.sample);
    draws.alpha_i = cell(1,dim.sample);
    for i = 1:dim.sample
        draws.ystar{i} = normrnd(0,2,dim.T,1);
        draws.alpha_i{i} = normrnd(0,1,dim.Gamma,1);
    end


    if ada ==1
        % ADADELTA Initialization
        ADA.perturbation = 10^-6;
        ADA.v = 0.95;

        ADA.delta2_mu = zeros(M,1);
        ADA.delta2_B = zeros(size(draws.B(:),1),1);
        ADA.delta2_d = zeros(M,1);

        ADA.g2_mu = zeros(M,1);
        ADA.g2_B = zeros(size(draws.B(:),1),1);
        ADA.g2_d = zeros(M,1);
    else
        ADA=[];
    end
    %Natural gradient initialization
    ng_old.bar1 = zeros(size(draws.mu,1),1);
    ng_old.bar2 = zeros(size(draws.B(:),1),1);
    ng_old.bar3 = zeros(size(draws.d(:),1),1);



    %storage space
    va.w = zeros(sum(dim.w_all),N);
    va.beta = zeros(dim.beta, N);
    va.Gamma = zeros(dim.Gamma,N);

    lambda.mu = zeros(M,N);
    lambda.B = zeros(M,p,N);
    lambda.d = zeros(M,1,N);
    lambda.z = zeros(p,N);

    Predictive = cell(1,8);
    Predictive_item = {'PCE(MC)','MCR(MC)', 'PRE(MC)', 'REC(MC)',...
                    'PCE','MCR', 'PRE', 'REC'};
    time = zeros(1,N);

end