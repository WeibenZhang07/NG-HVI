function [X_train,Y_train,X_test,Y_test,beta,alpha,Z_out,eta,group_ind,u,R2_m,R2_c] = data_sim(rho,n,num_group,sigma_e, sigma_a)
rng(123)
% precision is the precision matrix for the correlation matrix of copula model
% n is the total number of observations in the data
% sigma_e is the variance of error term
% sigma_a is the variance of the random effect
    
    u = copularnd('Gaussian',rho,n);
    m = size(rho,2);
    x = zeros(n,m);
    for i = 1:m
        x(:,i) = norminv(u(:,i));
    end
    
    
    beta = [0.8292;
            -1.3250;
            0.9909;
            1.6823;
            -1.7564;
            0.0580];

    group_ind = 0:n/num_group:n;
    group_ind = group_ind(2:end);

    X = cell(1,num_group);
    alpha = cell(1,num_group);
    eta = cell(1,num_group);
    Z_out = cell(1,num_group);
    fixed_out = cell(1,num_group);
    re_out = cell(1,num_group);
    
    p_fit = 0.5;
    
    X_train = cell(1,num_group);
    X_test = cell(1,num_group);
    Y_train = cell(1,num_group);
    Y_test = cell(1,num_group);
    Y = cell(1,num_group);
    start_ind = 0;

    for i = 1:num_group
        ind = (start_ind + 1: group_ind(i));
        start_ind= group_ind(i);
        num_obs = length(ind);
    
        X{i} = [ones(num_obs,1) x(ind,:)];
    
        alpha{i} = normrnd(0,sqrt(sigma_a));
        eta{i} = X{i}*(beta)+alpha{i}*ones(size(X{i},1),1);
        
        fixed_out{i}=X{i}*(beta);
        Y{i} = eta{i}+ normrnd(0,sqrt(sigma_e),num_obs,1);
        
        n_train = round(num_obs*p_fit);
        X_train{i} =X{i}(1:n_train,:);
        X_test{i} = X{i}(n_train+1:end,:);
        Y_train{i} =Y{i}(1:n_train,:);
        Y_test{i} = Y{i}(n_train+1:end,:);        
    end
    
    var_RE = sigma_a;
    fixed_out_all = vertcat(fixed_out{:});
    var_fixed = var(fixed_out_all);
    R2_m = var_fixed /(var_fixed + var_RE + sigma_e);
    R2_c = (var_fixed+ var_RE) /(var_fixed + var_RE + sigma_e);

end