function [log_joint_density] = f_log_joint(X_train,Y_train,beta,sigma2_e,sigma2_alpha,tAlpha_h,sigma2_beta,a_e,b_e,a_alpha,b_alpha)
    
    H = length(X_train);
    log_likelihood = 0;
    log_alpha_h = 0;
    log_jacobian = 0;
    for h = 1:H
        x_h = X_train{h};
        y_h = Y_train{h};
        talpha_h = tAlpha_h(h);
        alpha_h =f_alpha_h(x_h,y_h,beta, talpha_h,sigma2_e,sigma2_alpha);
        n_h = length(y_h);
        log_likelihood = log_likelihood -n_h/2*log(2*pi) - n_h/2*log(sigma2_e) - 0.5/sigma2_e*(y_h-x_h*beta-ones(n_h,1)*alpha_h)'*(y_h-x_h*beta-ones(n_h,1)*alpha_h);
        log_alpha_h = log_alpha_h + -1/2*log(2*pi) - 1/2*log(sigma2_alpha) - 0.5/sigma2_alpha*(alpha_h^2);
        log_jacobian = log_jacobian + f_log_jacobian(sigma2_e,sigma2_alpha,n_h);
    end
    d_beta = length(beta);
    log_beta_prior = -d_beta/2*log(2*pi) - d_beta/2*log(sigma2_beta) - 0.5/sigma2_beta*(beta'*beta);
    log_theta_e_prior = a_e*log(b_e)-log(gamma(a_e)) - (a_e)*log(sigma2_e) - b_e/sigma2_e;
    log_theta_alpha_prior = a_alpha*log(b_alpha)-log(gamma(a_alpha)) - (a_alpha)*log(sigma2_alpha) - b_alpha/sigma2_alpha;
    
    log_joint_density = log_likelihood+log_alpha_h+ log_jacobian + log_beta_prior+log_theta_e_prior+log_theta_alpha_prior;
    
    
end