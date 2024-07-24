 
function [mu_alpha_i] = VB_alpha_i(X_train,Y_train,beta, Gamma,Sigma)
    % Sampler for alpha_i
    N = length(X_train);
    mu_alpha_i = cell(1,N);
    
    for i = 1:N
        n_h = length(Y_train{i});
        post_var = 1/(n_h/Sigma + 1/Gamma);   
        post_mean = post_var/Sigma*sum(Y_train{i} - X_train{i}*beta);
        mu_alpha_i{i} = normrnd(post_mean,sqrt(post_var));
    end
    
end
