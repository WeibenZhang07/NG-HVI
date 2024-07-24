function [Alpha] = MCMC_Alpha(X_train,Y_train,beta,sigma2_e,sigma2_a)
    % Sampler for alpha_i
    N = length(X_train);
    Alpha = cell(N,1);
    
    for i = 1:N
        n_h = length(Y_train{i});

        post_var = 1/(n_h/sigma2_e + 1/sigma2_a);        
        post_mean = post_var/sigma2_e*sum(Y_train{i} - X_train{i}*beta);
        Alpha{i} = normrnd(post_mean,sqrt(post_var));
    end
    
end