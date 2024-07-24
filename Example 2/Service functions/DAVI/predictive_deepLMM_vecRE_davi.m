function [elbo,RMSE,R2,y_pred] = predictive_deepLMM_vecRE_tan(X_test,Y_test,X_train,Y_train,nn,lambda,sigma2_weights_vec,sigma2_beta,a_e,b_e,S,v,J,results,varargin)

    if results == 1
        lambdahat_mu = mean(lambda.mu(:,end-100:end),2);
        lambdahat_C1_vech = mean(lambda.C1_vech(:,end-100:end),2);
        lambdahat_C2_vech = mean(lambda.C2_vech(:,end-100:end),2);
        
    else
        ind = varargin{1};
        lambdahat_mu = lambda.mu(:,ind);
        lambdahat_C1_vech =lambda.C1_vech(:,ind);
        lambdahat_C2_vech =lambda.C2_vech(:,ind); 
    end

    mu = lambdahat_mu;
    C1_vech  = lambdahat_C1_vech;
    C2_vech  = lambdahat_C2_vech;
    
    %% draw theta from q_lambda(theta) and then calculate noisy ELBO using 
   
    H = length(X_test);
    dim.x = size(X_test{1},2);
    
    layer = length(nn);
    dim.input = dim.x-1;
    NN = [dim.input,nn];
    d_weights = zeros(layer,1);
    for i = 1:layer
        d_weights(i) = NN(i+1)*(NN(i)+1);
    end
    d_weights_vec = sum(d_weights);
    r = nn(end)+1;
    d_beta = nn(end)+1;
    d_w = r*(r+1)/2;

    d_global = d_weights_vec + d_beta +1 +d_w;
    d_local = r*H;
    d = d_global +d_local;
    
    C2_all = C2_vech;
    C2 = cell(H,1);
    for h = 1:H
        len_vec_c2 = r*(r+1)/2;
        C2{h} = C2_all(1 + (h-1)*len_vec_c2:h*len_vec_c2);
        C2{h} = invert_vech(C2{h});
    end
    C1 = invert_vech(C1_vech);

    elbo_vec = zeros(1,J);
    Mu_y = zeros(length(vertcat(Y_test{:})),J);
    Pred_y = zeros(length(vertcat(Y_test{:})),J);    

    tAlpha_h = cell(H,1);
    z2= cell(H,1);
    for j = 1:J

        %% Do not integrate out random effects
        z1 = normrnd(0,1,d_global,1);
        theta = C1*z1 + mu(1:d_global);

        for h = 1:H
            z2{h} = normrnd(0,1,r,1);
            tAlpha_h{h} = C2{h}*z2{h} + mu(d_global+1+(h-1)*r:d_global+r+(h-1)*r);
        end
        [weights,~] = w_vec2mat(theta,NN,d_weights_vec);
        beta = theta(d_weights_vec+1:d_weights_vec+d_beta);
        theta_e = theta(d_weights_vec+d_beta+1);
        w = theta(d_weights_vec+d_beta+2:end);
        sigma2_e = exp(theta_e);

        
        [omega,~] = f_gen_omega(w);
        Alpha_h = cell(H,1);
        Z_out = cell(H,1);
        Y_hat = cell(H,1);
        pred_y = cell(H,1);
        for h = 1:H
            x_h = X_test{h};
            X_h_insample = X_train{h};
            Z_out{h} = features_output(x_h,weights);
            Z_out_insample = features_output(X_h_insample,weights);
            z_out_h = Z_out{h};
            y_h = Y_test{h};
            y_h_insample = Y_train{h};
            z_h = Z_out{h};
            talpha_h = tAlpha_h{h};
            [Alpha_h{h}] = f_alpha_h(Z_out_insample,Z_out_insample,y_h_insample,beta, talpha_h,sigma2_e,omega);
            Y_hat{h} = z_out_h*beta + z_h*Alpha_h{h};
            
%            cov_y = sigma2_e*ones(length(y_h),1);
            pred_y{h} = Y_hat{h};%normrnd(Y_hat{h}, sqrt(cov_y));
        end
        [log_joint_density] = f_log_joint_deepLMM(X_test,Y_test,weights,beta,theta_e,w,tAlpha_h,sigma2_weights_vec,sigma2_beta,a_e,b_e,S,v);
        log_q_lambda = - d_global/2*log(2*pi) - 0.5*sum(log(eig(C1*C1'))) - 0.5*(z1'*z1);
        for h = 1:H
            log_q_lambda = log_q_lambda -r/2*log(2*pi)-0.5*sum(log(eig(C2{h}*C2{h}'))) - 0.5*sum(z2{h}'*z2{h});
        end

        elbo_vec(j) = log_joint_density - log_q_lambda;
        Mu_y(:,j) = vertcat(Y_hat{:});
        Pred_y(:,j) = vertcat(pred_y{:});
    end
    elbo = mean(elbo_vec);

    y_all = vertcat(Y_test{:});
    if results == 1     
        y_pred = mean(Pred_y,2);
        RMSE = sqrt(mean((y_all - y_pred).^2));
        SE = mean((y_all - y_pred).^2);
        R2 = 1-SE/var(y_all);
    else
        y_pred = mean(Mu_y,2);
        RMSE = sqrt(mean((y_all - y_pred).^2));
        SE = mean(((y_all - y_pred).^2));
        R2 = 1-SE/var(y_all);
       
    end

end