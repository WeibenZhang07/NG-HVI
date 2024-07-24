function sigma2_e = MCMC_sigma2_e(X_train,Y_train,H,beta,Alpha,priors)

    x = vertcat(X_train{:});
    y = vertcat(Y_train{:});
    n = length(y);
    alpha_H_vec = repelem(vertcat(Alpha{:}),H);
    eta = y - x*beta - alpha_H_vec;
    post_a = priors.sigma2_e_a +n/2;
    post_b = priors.sigma2_e_b + eta'*eta/2;
    sigma2_e = 1/gamrnd(post_a, 1/post_b);
end