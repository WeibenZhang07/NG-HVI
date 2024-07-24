function [X_train,Y_train,X_val,Y_val,X_test,Y_test,weights,beta,alpha,Z_out,eta,varargout] = data_sim_large(rho,n,num_group,deepnet,sigma_e,sigma_a,weights,beta,linear,vec_RE,link)
%rng(123)
% precision is the precision matrix for the correlation matrix of copula model
% n is the total number of observations in the data
% sigma_e is the variance of error term
% sigma_re is the variance of the random effect
    
    u = copularnd('Gaussian',rho,n);
    m = size(rho,2);
    x = zeros(n,m);
    for i = 1:m
        x(:,i) = norminv(u(:,i));
    end
    NN = [m deepnet];    

    group_ind = 0:n/num_group:n;
    group_ind = group_ind(2:end);

    X = cell(1,num_group);
    alpha = cell(1,num_group);
    eta = cell(1,num_group);
    Z_out = cell(1,num_group);
    fixed_out = cell(1,num_group);
    re_out = cell(1,num_group);
    
    p_train = 0.6;
    p_val = 0.2;
    
    X_train = cell(1,num_group);
    X_val = cell(1,num_group);
    X_test = cell(1,num_group);

    Y_train = cell(1,num_group);
    Y_val = cell(1,num_group);
    Y_test = cell(1,num_group);
    Y = cell(1,num_group);
    start_ind = 0;

    for i = 1:num_group
        if linear ==1
            ind = (start_ind + 1: group_ind(i));
            start_ind= group_ind(i);
            num_obs = length(ind);
    
            X{i} = [ones(num_obs,1) x(ind,:)];
            if vec_RE==1
               
                alpha{i} = reshape(mvnrnd(zeros(NN(end)+1,1),sigma_a),[],1);
                re_out{i} = X{i}*alpha{i};                
                eta{i} = X{i}*(beta+alpha{i});
            else
                alpha{i} = normrnd(0,sqrt(sigma_a));
                re_out{i} = alpha{i}*ones(size(X{i},1),1);
                eta{i} = X{i}*(beta)+alpha{i}*ones(size(X{i},1),1);
            end
            fixed_out{i}=X{i}*(beta);
            if strcmp(link,'gaussian')
                Y{i} = eta{i}+ normrnd(0,sqrt(sigma_e),num_obs,1);

                n_train = round(num_obs*p_train);
                n_val = round(num_obs*p_val);
                X_train{i} =X{i}(1:n_train,:);
                X_val{i} = X{i}(n_train+1:n_train+n_val,:);
                X_test{i} = X{i}(n_train+n_val+1:end,:);
                Y_train{i} =Y{i}(1:n_train,:);
                Y_val{i} = Y{i}(n_train+1:n_train+n_val,:);
                Y_test{i} = Y{i}(n_train+n_val+1:end,:);
            else % logit link
                p = 1./(exp(-eta{i}));
                Y{i} = binornd(1,p);

                n_train = round(num_obs*p_train);
                n_val = round(num_obs*p_val);
                X_train{i} =X{i}(1:n_train,:);
                X_val{i} = X{i}(n_train+1:n_train+n_val,:);
                X_test{i} = X{i}(n_train+n_val+1:end,:);
                Y_train{i} =Y{i}(1:n_train,:);
                Y_val{i} = Y{i}(n_train+1:n_train+n_val,:);
                Y_test{i} = Y{i}(n_train+n_val+1:end,:);
            end
        else
            ind = (start_ind + 1: group_ind(i));
            start_ind= group_ind(i);
            num_obs = length(ind);
    
            X{i} = [ones(num_obs,1) x(ind,:)];
            Z_out{i} = features_output(X{i},weights);
            if vec_RE==1
                alpha{i} = reshape(mvnrnd(zeros(NN(end)+1,1),sigma_a),[],1);
                eta{i} = Z_out{i}*(beta+alpha{i});
                fixed_out{i}=Z_out{i}*(beta);
                re_out{i} = Z_out{i}*alpha{i};
            else
                alpha{i} = normrnd(0,sqrt(sigma_a));
                eta{i} = Z_out{i}*(beta)+alpha{i}*ones(size(Z_out,1),1);
                fixed_out{i}=Z_out{i}*(beta);
                re_out{i} = alpha{i}*ones(size(Z_out,1),1);
            end
            
            if strcmp(link,'gaussian')
                Y{i} = eta{i} + normrnd(0,sqrt(sigma_e),num_obs,1);

                n_train = round(num_obs*p_train);
                n_val = round(num_obs*p_val);
                X_train{i} =X{i}(1:n_train,:);
                X_val{i} = X{i}(n_train+1:n_train+n_val,:);
                X_test{i} = X{i}(n_train+n_val+1:end,:);
                Y_train{i} =Y{i}(1:n_train,:);
                Y_val{i} = Y{i}(n_train+1:n_train+n_val,:);
                Y_test{i} = Y{i}(n_train+n_val+1:end,:);
            else % logit link
                p = 1./(exp(-eta{i}));
                Y{i} = binornd(1,p);

                n_train = round(num_obs*p_train);
                n_val = round(num_obs*p_val);
                X_train{i} =X{i}(1:n_train,:);
                X_val{i} = X{i}(n_train+1:n_train+n_val,:);
                X_test{i} = X{i}(n_train+n_val+1:end,:);
                Y_train{i} =Y{i}(1:n_train);
                Y_val{i} = Y{i}(n_train+1:n_train+n_val);
                Y_test{i} = Y{i}(n_train+n_val+1:end);
            end
        end
    end
    if strcmp(link,'gaussian')
        fixed_out_all = vertcat(fixed_out{:});
        re_out_all = vertcat(re_out{:});
        var_fixed = var(fixed_out_all);
        var_re = var(re_out_all);
        R2_m = var_fixed /(var_fixed + var_re + sigma_e);
        R2_c = (var_fixed+ var_re) /(var_fixed + var_re + sigma_e);
        varargout{1} = R2_m;
        varargout{2} = R2_c;
    else
        pos_rate= sum(vertcat(Y{:}))/length(vertcat(Y{:}));
        varargout{1} = pos_rate;
    end


end