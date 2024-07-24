function [mu_alpha_i] = VB_vec_alpha_i(X_train,Z_train,Y_train,beta,omega,sigma_e)
    % Sampler for alpha_i
    H = length(Z_train);
    mu_alpha_i = cell(1,H);
    for h = 1:H
        post_var_inv = (omega) + Z_train{h}'*Z_train{h}/sigma_e;

        [cor_inv,sigma_e_inv] = corrcov(post_var_inv);
        post_var = corr2cov(1./(sigma_e_inv),inv(cor_inv));
        post_var = (post_var + post_var')/2; % avoid non-symmetric issue caused by numerical error
        
        post_mean = post_var*Z_train{h}'*(Y_train{h} - X_train{h}*beta)/sigma_e;
        mu_alpha_i{h} = mvnrnd(post_mean,post_var)';
    end
    
end
