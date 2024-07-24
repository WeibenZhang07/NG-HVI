function MCMC = LMM_MCMC(X_train,Y_train,N,priors)

    rng(123);

    H = zeros(length(Y_train),1);
    for h = 1:length(Y_train)
        H(h) = length(Y_train{h});
    end

    %initial values
    m = size(X_train{1},2);
    beta = randn(m,1);
    sigma2_e = 1;
    sigma2_a = 1;
    [Alpha] = MCMC_Alpha(X_train,Y_train,beta,sigma2_e,sigma2_a);

    %storage space
    MCMC.beta = zeros(m,N);
    MCMC.sigma2_e = zeros(1,N);
    MCMC.sigma2_a = zeros(1,N);
    MCMC.Alpha = zeros(length(H),N);
    for i = 1:N
        [beta] = MCMC_beta(X_train, Y_train, H,sigma2_e,Alpha,priors);
        sigma2_a = MCMC_sigma2_a(H,Alpha,priors);
        sigma2_e = MCMC_sigma2_e(X_train,Y_train,H,beta,Alpha,priors);
        [Alpha] = MCMC_Alpha(X_train,Y_train,beta,sigma2_e,sigma2_a);
        
            MCMC.beta(:,i) = beta;
            MCMC.sigma2_e(i) = sigma2_e;
            MCMC.sigma2_a(i) = sigma2_a;
            MCMC.Alpha(:,i) = vertcat(Alpha{:});
        
    end

