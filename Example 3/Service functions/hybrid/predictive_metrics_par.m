
function [predictive] = predictive_metrics_par(X,y, X_test,y_test,nn,J,lambda,results,varargin)
    %% Calculate posterior predictive metrics
    
    y_test_all = vertcat(y_test{:});
    
    dim.sample = length(X_test);
    [dim.test_T,dim.x] = size(X_test{1});
    [dim.T,~] = size(X{1});
    layer = length(nn);
    dim.input = dim.x-1;
    NN = [dim.input,nn];
    dim.w_all = zeros(layer,1);
    n_gibbs = 2;
    
    P = zeros(dim.sample*dim.test_T,J);
    for i = 1:layer
        dim.w_all(i) = NN(i+1)*(NN(i)+1);
    end
    dim.beta = nn(end)+1;
    dim.Gamma = dim.beta;
    
    if results == 1
        lambdahat_mu = mean(lambda.mu(:,end-100:end),2);
        lambdahat_B = mean(lambda.B(:,:,end-100:end),3);
        lambdahat_d = mean(lambda.d(:,:,end-100:end),3);
       
    else
        ind = varargin{1};
        lambdahat_mu = lambda.mu(:,ind);
        lambdahat_B =lambda.B(:,:,ind);
        lambdahat_d = lambda.d(:,:,ind);
 
    end
    
%% Account for parameter uncertainty by using random draws from approximation density
%% Parallel computing
if J >10
    [M,p] = size(lambdahat_B);
    ystar_insample = cell(1,dim.sample);
    alpha_i_ini = cell(1,dim.sample);

    for i = 1:dim.sample
        ystar_insample{i} = normrnd(0,2,dim.T,1);
        alpha_i_ini{i} = normrnd(0,1,dim.Gamma,1);
    end
    
    N = dim.sample;
    N_w_all = dim.w_all;
    N_input = dim.input;
    N_beta = dim.beta;
    N_Gamma = dim.Gamma;
    parfor j = 1:J
        
        eta_outsample = cell(1,N);
        z_out_insample = cell(1,N);
        z_out_outsample= cell(1,N);
        alpha_i= alpha_i_ini;
        
        epsilon = normrnd(0,1,M,1);
        z = normrnd(0,1,p,1);
        theta = lambdahat_mu + lambdahat_B * z + (lambdahat_d.*epsilon);
        
        
        w_all =  theta(1:sum(N_w_all));
        w = cell(layer,1);
        for l = 1 : layer
            if l == 1
               w{l} = w_all(1:N_w_all(l));
               w{l} = reshape(w{l},nn(l),N_input+1);
            else
               w{l} = w_all(N_w_all(l-1)+1:(N_w_all(l-1)+N_w_all(l)));
               w{l} = reshape(w{l},nn(l),nn(l-1)+1);
            end
        end
        
        beta = theta(sum(N_w_all)+1:sum(N_w_all)+N_beta);
        TGamma = theta(end - N_Gamma+1:end);

        for l = 1:N
            z_out_insample{l} =  features_output(X{l},w);
        end
        
        for minigibbs = 1:n_gibbs
           [ystar_insample] = VB_ystar(z_out_insample,y,beta,alpha_i,dim);
          
           [alpha_i] = VB_alpha_i(z_out_insample,ystar_insample,beta, exp(TGamma));
        end

        for n = 1:dim.sample
            z_out_outsample{n} =  features_output(X_test{n},w);
            eta_outsample{n} = z_out_outsample{n}*(beta+ alpha_i{n});
        end
        
        eta_all = vertcat(eta_outsample{:});
        p_i = normcdf(eta_all);
        P(:,j) = p_i;
       
    end
  
%% Single core computing
elseif J >0
    [M,p] = size(lambdahat_B);
    alpha_i = cell(1,dim.sample);
    ystar_insample = cell(1,dim.sample);
    eta_outsample = cell(1,dim.sample);
    z_out_insample = cell(1,dim.sample);
    z_out_outsample= cell(1,dim.sample);
    

    for i = 1:dim.sample
        ystar_insample{i} = normrnd(0,2,dim.T,1);
        alpha_i{i} = normrnd(0,1,dim.Gamma,1);
    end
     
    for j = 1:J
        epsilon = normrnd(0,1,M,1);
        z = normrnd(0,1,p,1);
        theta = lambdahat_mu + lambdahat_B * z + (lambdahat_d.*epsilon);
        
        
        w_all =  theta(1:sum(dim.w_all));
        w = cell(layer,1);
        for l = 1 : layer
            if l == 1
               w{l} = w_all(1:dim.w_all(l));
               w{l} = reshape(w{l},nn(l),dim.input+1);
            else
               w{l} = w_all(dim.w_all(l-1)+1:(dim.w_all(l-1)+dim.w_all(l)));
               w{l} = reshape(w{l},nn(l),nn(l-1)+1);
            end
        end
        
        beta = theta(sum(dim.w_all)+1:sum(dim.w_all)+dim.beta);
        TGamma = theta(end - dim.Gamma+1:end);

        for l = 1:dim.sample
            z_out_insample{l} =  features_output(X{l},w);
        end
        
        for minigibbs = 1:n_gibbs
           [ystar_insample] = VB_ystar(z_out_insample,y,beta,alpha_i,dim);
          
           [alpha_i] = VB_alpha_i(z_out_insample,ystar_insample,beta, exp(TGamma));
        end

        for n = 1:dim.sample
            z_out_outsample{n} =  features_output(X_test{n},w);
            eta_outsample{n} = z_out_outsample{n}*(beta+ alpha_i{n});
        end
        
        eta_all = vertcat(eta_outsample{:});
        p_i = normcdf(eta_all);
        P(:,j) = p_i;
    end
 
end   
        p_pos = mean(P,2);
        predictive.pce = CRPS(y_test_all,p_pos)/length(p_pos);
        predictive.mcr = sum(abs((p_pos>0.5)- y_test_all))/length(y_test_all);
        predictive.precision = sum((p_pos>0.5) & y_test_all==1)/sum(p_pos>0.5);
        predictive.recall = sum((p_pos>0.5) & y_test_all==1)/sum(y_test_all==1);
        predictive.p_pos = p_pos;
%% Use the mean of the approximation density as model parameters
    alpha_i_mu = cell(1,dim.sample);
    ystar_insample_mu = cell(1,dim.sample);
    eta_outsample_mu = cell(1,dim.sample);
    z_out_insample_mu = cell(1,dim.sample);
    z_out_outsample_mu= cell(1,dim.sample);
    
    w_mu = cell(1,layer);
    w_mu_all =  lambdahat_mu(1:sum(dim.w_all));
    beta_mu = lambdahat_mu(sum(dim.w_all)+1:sum(dim.w_all)+dim.beta);
    TGamma_mu = lambdahat_mu(end - dim.Gamma+1:end);

    for l = 1 : layer
       if l == 1
           w_mu{l} = w_mu_all(1:dim.w_all(l));
           w_mu{l} = reshape(w_mu{l},nn(l),dim.input+1);
       else
           w_mu{l} = w_mu_all(dim.w_all(l-1)+1:(dim.w_all(l-1)+dim.w_all(l)));
           w_mu{l} = reshape(w_mu{l},nn(l),nn(l-1)+1);
       end
    end
    
    for l = 1:dim.sample
           z_out_insample_mu{l} =  features_output(X{l},w_mu);
    end
        
    for i = 1:dim.sample
        ystar_insample_mu{i} = normrnd(0,2,dim.T,1);
        alpha_i_mu{i} = normrnd(0,1,dim.Gamma,1);
    end
    
    for minigibbs = 1:n_gibbs
      [ystar_insample_mu] = VB_ystar(z_out_insample_mu,y,beta_mu,alpha_i_mu,dim);

      [alpha_i_mu] = VB_alpha_i(z_out_insample_mu,ystar_insample_mu,beta_mu, exp(TGamma_mu));
     
    end

    for n = 1:dim.sample
        z_out_outsample_mu{n} =  features_output(X_test{n},w_mu);
        eta_outsample_mu{n} = z_out_outsample_mu{n}*(beta_mu+ alpha_i_mu{n});
    end
    
    eta_all_mu = vertcat(eta_outsample_mu{:});
    p_i_mu = normcdf(eta_all_mu);
    
    predictive.pce2 = CRPS(y_test_all,p_i_mu)/length(eta_all_mu);
    predictive.mcr2 = sum(abs((eta_all_mu>0)- y_test_all))/length(y_test_all);
    predictive.precision2 = sum((eta_all_mu>0) & y_test_all==1)/sum(eta_all_mu>0);
    predictive.recall2 = sum((eta_all_mu>0) & y_test_all==1)/sum(y_test_all==1);
    predictive.p_pos2 = p_i_mu;
 
end