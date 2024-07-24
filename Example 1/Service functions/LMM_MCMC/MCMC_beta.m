function [beta] = MCMC_beta(X_train, Y_train, H,sigma2_e,Alpha,priors)
    
    x = vertcat(X_train{:});
    y = vertcat(Y_train{:});
    m = size(x,2);
    alpha_H_vec = repelem(vertcat(Alpha{:}),H);
    post_var = (x'*x/sigma2_e + 1/priors.beta_sig*eye(m))\eye(m);
    post_mean = post_var*x'*(y-alpha_H_vec)/sigma2_e;
    beta = mvnrnd(post_mean,post_var)';



end