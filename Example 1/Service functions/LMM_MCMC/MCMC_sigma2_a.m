function sigma2_a = MCMC_sigma2_a(H,Alpha,priors)

    h = length(H);
    alpha_vec = vertcat(Alpha{:});
    post_a = priors.sigma2_a_a +h/2;
    post_b = priors.sigma2_a_b + alpha_vec'*alpha_vec/2;
    sigma2_a = 1/gamrnd(post_a, 1/post_b);
end