%% Initiation with random effect for intercept only
%% No y_star
function [ADA,draws,dim,lambda,M,ng_old,Predictive,Predictive_item,priors,time,va] = Hybrid_DeepLMM_vecRE_initialize(X_train,Z_train,N,nn,p,ada)

    % priors
    priors.weights_sig = 100;
    priors.beta_sig = 100;
    priors.Sigma_a = 1.01;
    priors.Sigma_b = 1.01;

    r = nn(end)+1;
    S = 0.01*eye(r);
    priors.S = S;
    priors.v =r+1;

    dim.sample = length(X_train);
    
    [~,dim.x] = size(X_train{1});
    % allow for different observations in each group
    dim.H = zeros(length(X_train),1);
    for h = 1:length(X_train)
        dim.H(h) = size(X_train,1);
    end

    layer = length(nn);
    dim.input = dim.x-1;
    NN = [dim.input,nn];
    dim.weights = zeros(layer,1);
    for i = 1:layer
        dim.weights(i) = NN(i+1)*(NN(i)+1);
    end
    dim.weights_vec = sum(dim.weights);
    dim.beta = nn(end)+1;
    dim.Sigma = 1;
    dim.w = r*(r+1)/2;

    M = dim.weights_vec + dim.beta + dim.Sigma + dim.w; % beta, vech(Omega), Sigma

    %variational parameters initialization
    draws.mu = zeros(M,1)+0.01;
    weights = InitializeNN_hybrid(NN);
    weights_ini = [];
    for l = 1:length(weights)
        weights_ini = [weights_ini;weights{l}(:)];
    end

    draws.mu(1:dim.weights_vec) = weights_ini;

    draws.B = zeros(M,p) + 0.01;
    draws.d = zeros(M,1)+ 1;
    draws.B(~tril(ones(size(draws.B)))) = 0;
    draws.z = normrnd(0,1,p,1);

    %latent variables initialization
    draws.alpha_i = cell(1,dim.sample);
    for i = 1:dim.sample
        draws.alpha_i{i} = normrnd(0,1,r,1);
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
    va.weights = zeros(dim.weights_vec,N);
    va.beta = zeros(dim.beta, N);
    va.w = zeros(dim.w,N);
    va.TSigma = zeros(dim.Sigma,N);

    lambda.mu = zeros(M,N);
    lambda.B = zeros(M,p,N);
    lambda.d = zeros(M,1,N);
    lambda.z = zeros(p,N);

    Predictive = cell(1,6);
    Predictive_item = {'elbo','log_score','RMSE','R2','R2_m','R2_c'};
    time = zeros(1,N);

end