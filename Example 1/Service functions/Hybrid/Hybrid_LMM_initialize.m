%% Initiation with random effect for intercept only
%% No y_star
function [ADA,draws,dim,lambda,M,ng_old,Predictive,Predictive_item,priors,time,va] = Hybrid_LMM_initialize(X_train,N,p,ada)

    % priors
    priors.beta_sig = 100;
    priors.Gamma_a = 1.01;3;0.01;
    priors.Gamma_b = 1.01;2;0.01;
    priors.Sigma_a = 1.01;1.001;3;
    priors.Sigma_b = 1.01;1.001;2;
    
    dim.sample = length(X_train);
    
    [~,dim.x] = size(X_train{1});
    % allow for different observations in each group
    dim.H = zeros(length(X_train),1);
    for h = 1:length(X_train)
        dim.H(h) = size(X_train,1);
    end

    dim.beta = dim.x;
    dim.Gamma = 1;
    dim.Sigma = 1;

    M = dim.beta + dim.Gamma + dim.Sigma; % w, beta, Gamma, Sigma

    %variational parameters initialization
    draws.mu = zeros(M,1)+0.01;0.01;
    draws.B = zeros(M,p) + 0.01;
    draws.d = zeros(M,1)+ 1;1;0.01;
    draws.B(~tril(ones(size(draws.B)))) = 0;
    draws.z = normrnd(0,1,p,1);

    %latent variables initialization
    draws.alpha_i = cell(1,dim.sample);
    for i = 1:dim.sample
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
    va.beta = zeros(dim.beta, N);
    va.TGamma = zeros(dim.Gamma,N);
    va.TSigma = zeros(dim.Sigma,N);

    lambda.mu = zeros(M,N);
    lambda.B = zeros(M,p,N);
    lambda.d = zeros(M,1,N);
    lambda.z = zeros(p,N);

    Predictive = cell(1,6);
    Predictive_item = {'Roos(MC)','MSE(MC)',...
                    'Roos','MSE','Roos(MC2)','MSE(MC2)'};
    time = zeros(1,N);

end