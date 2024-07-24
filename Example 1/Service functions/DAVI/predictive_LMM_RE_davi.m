function [elbo,RMSE,R2] = predictive_LMM_RE_tan(X_train,Y_train,lambda,sigma2_beta,a_e,b_e,a_alpha,b_alpha,J,results,varargin)
    rng(123)
    if results == 1
        lambdahat_mu = mean(lambda.mu(:,end-500:end),2);
        lambdahat_C1_vech = mean(lambda.C1_vech(:,end-500:end),2);
        lambdahat_C2 = mean(lambda.C2(:,end-500:end),2);
        
    else
        ind = varargin{1};
        lambdahat_mu = lambda.mu(:,ind);
        lambdahat_C1_vech =lambda.C1_vech(:,ind);
        lambdahat_C2 =lambda.C2(:,ind); 
    end

    mu = lambdahat_mu;
    C1_vech  = lambdahat_C1_vech;
    C2  = lambdahat_C2;
    
    %% draw theta from q_lambda(theta) and then calculate noisy ELBO using 
   
    H = length(Y_train);
    d_beta = size(X_train{1},2);
    d_global = d_beta +1 +1;
    d = d_global +H;

    C1 = invert_vech(C1_vech);

    elbo_vec = zeros(1,J);
    Mu_y = zeros(length(vertcat(Y_train{:})),J);
    
    for j = 1:J

        %% Do not integrate out random effects
        z1 = normrnd(0,1,d_global,1);
        z2 = normrnd(0,1,H,1);
        theta = C1*z1 + mu(1:d_global);
        tAlpha_h = C2.*z2 + mu(d_global+1:end);
        beta = theta(1:d_beta);
        sigma2_e = exp(theta(d_beta+1));
        sigma2_alpha = exp(theta(d_beta+2));

        
       
        Alpha_h = cell(H,1);
        Y_hat = cell(H,1);
        for h = 1:H
        x_h = X_train{h};
        y_h = Y_train{h};
        [Alpha_h{h}] = f_alpha_h(x_h,y_h,beta, tAlpha_h(h),sigma2_e,sigma2_alpha);
        Y_hat{h} = x_h*beta + Alpha_h{h};
        end
        [log_joint_density] = f_log_joint(X_train,Y_train,beta,sigma2_e,sigma2_alpha,tAlpha_h,sigma2_beta,a_e,b_e,a_alpha,b_alpha);
        log_q_lambda = - d_global/2*log(2*pi) - 0.5*log(det(C1*C1')) - 0.5*(z1'*z1)...
         -H/2*log(2*pi)-0.5*sum(log(C2.^2)) - 0.5*sum(z2.*z2);

        elbo_vec(j) = log_joint_density - log_q_lambda;
        Mu_y(:,j) = vertcat(Y_hat{:});
    end
    elbo = mean(elbo_vec);
    mu_y_mean = mean(Mu_y,2);
    SE = sum((vertcat(Y_train{:}) - mu_y_mean).^2);
    R2 = 1-SE/sum((vertcat(Y_train{:})).^2);
    RMSE = sqrt(mean((vertcat(Y_train{:}) - mu_y_mean).^2));
   
end