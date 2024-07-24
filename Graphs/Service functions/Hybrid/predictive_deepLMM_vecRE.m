function [elbo,log_score,RMSE,R2,R2_m,R2_c] = predictive_deepLMM_vecRE(X_test,Y_test,X_train,Y_train,nn,lambda,priors,J,results,varargin)

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

    lambdahat.mu = lambdahat_mu;
    lambdahat.B  = lambdahat_B;
    lambdahat.d = lambdahat_d;
    
    %% draw theta from q_lambda(theta) and then calculate noisy ELBO using log(p(y,theta)/q_lambda(theta))
    dim.sample = length(X_test);
    dim.x = size(X_test{1},2);
    [M,p] = size(lambdahat.B);
    
    dim.sample = length(X_test);
    dim.x = size(X_test{1},2);
    
    layer = length(nn);
    dim.input = dim.x-1;
    NN = [dim.input,nn];
    dim.weights = zeros(layer,1);
    for i = 1:layer
        dim.weights(i) = NN(i+1)*(NN(i)+1);
    end
    dim.weights_vec = sum(dim.weights);
    r = nn(end)+1;
    dim.beta = nn(end)+1;
    dim.Sigma = 1;
    dim.w = r*(r+1)/2;


    for h = 1:dim.sample
        dim.H(h,1) = size(X_test{h},1); 
    end
    
    elbo_vec = zeros(1,J);
    Pred_y = zeros(length(vertcat(Y_test{:})),J);
    P = zeros(length(vertcat(Y_test{:})),J);
    Fixed = zeros(length(vertcat(Y_test{:})),J);
    Random = zeros(length(vertcat(Y_test{:})),J);
    Sigma_e = zeros(1,J);
    for j = 1:J

        %% Do not integrate out random effects
        epsilon = normrnd(0,1,M,1);
        z = normrnd(0,1,p,1);
        theta = lambdahat.mu + lambdahat.B * z + (lambdahat.d.*epsilon);

        [weights,weights_vec] = w_vec2mat(theta,NN,dim.weights_vec);
        beta = theta(dim.weights_vec+1:dim.weights_vec+dim.beta);
        TSigma = theta(dim.weights_vec+dim.beta+1);
        w = theta(dim.weights_vec+dim.beta+2:end);
       
        sigma_e = exp(TSigma);
        [omega,W] = f_gen_omega(w);
        
        log_y = 0;
        log_alpha_h = 0;
        log_alpha_h_posterior = 0;
        z_out = cell(dim.sample,1);
        z_out_insample = cell(dim.sample,1);        
        fixed = cell(dim.sample,1);
        random = cell(dim.sample,1);
        pred_y = cell(dim.sample,1);
        p_j= cell(dim.sample,1);

        for ii = 1:dim.sample
            z_out{ii} =  features_output(X_test{ii},weights);
            z_out_insample{ii} =  features_output(X_train{ii},weights);
        end

        Alpha_h = VB_vec_alpha_i(z_out_insample,z_out_insample,Y_train,beta,omega,sigma_e);%fixed for both training sample and testing sample
        for ii = 1:dim.sample
            z_out_h = z_out{ii};
            y_h = Y_test{ii};
            z_h = z_out{ii};
            n_h = length(y_h);
            alpha_h = Alpha_h{ii};
            fixed{ii} = z_out_h*beta;
            random{ii} = z_h *alpha_h;
            mu = fixed{ii} + random{ii};
            pred_y{ii} = mu;
            eta = y_h - mu;

            log_y = log_y -0.5*n_h*log(2*pi) -0.5*n_h*log(sigma_e) - 0.5/sigma_e*(eta'*eta);
            log_alpha_h = log_alpha_h -0.5*log(2*pi) + 0.5*log(det(omega)) - 0.5*alpha_h'*omega*alpha_h;
            inv_cov_alpha_h_posterior = omega+ z_h'*z_h/sigma_e;
            mean_alpha_h_posterior = (inv_cov_alpha_h_posterior\z_h'*(y_h -z_out_h*beta))/sigma_e;
            log_alpha_h_posterior = log_alpha_h_posterior -0.5*r*log(2*pi) + 0.5*log(det(inv_cov_alpha_h_posterior)) ...
                - 0.5*(alpha_h-mean_alpha_h_posterior)'*inv_cov_alpha_h_posterior*(alpha_h-mean_alpha_h_posterior);

             p_j{ii} = normpdf(y_h,mu,sqrt(sigma_e));
        end
        
        log_weights_vec = -0.5*dim.weights_vec*log(2*pi) -0.5*dim.weights_vec*log(priors.weights_sig) - 0.5/priors.weights_sig*(weights_vec'*weights_vec);
        log_beta = -0.5*dim.beta*log(2*pi) -0.5*dim.beta*log(priors.beta_sig) - 0.5/priors.beta_sig*(beta'*beta);
        log_Tsigma = priors.Sigma_a  *log(priors.Sigma_b) -log(gamma(priors.Sigma_a)) ...
                        -(priors.Sigma_a )*TSigma - priors.Sigma_b/exp(TSigma);
        log_p_w =(logwishpdf(omega,priors.S,priors.v))+ r*log(2) + sum(((r+2)*ones(r,1)-(1:r)').*log(diag(W)));
        
        Sigma = lambdahat.B*lambdahat.B'+diag(lambdahat.d.^2);
        log_q_0_lambda = -0.5*M*log(2*pi) -0.5*sum(log(eig(Sigma))) - 0.5*(theta-lambdahat.mu)'/(Sigma)*(theta-lambdahat.mu);

        elbo_vec(j) = log_y + log_alpha_h + log_weights_vec + log_beta +log_Tsigma + log_p_w - log_q_0_lambda - log_alpha_h_posterior;
        P(:,j) = vertcat(p_j{:});
        Pred_y(:,j) = vertcat(pred_y{:});
        Fixed(:,j) = vertcat(fixed{:});
        Random(:,j) = vertcat(random{:});
        Sigma_e(j) = sigma_e;
    end
    elbo = mean(elbo_vec);
    P(any(P,'all')) = 1e-10;
    p_mean = mean(P,2);
    pred_y_mean = mean(Pred_y,2);
    log_score = mean(log(p_mean));
    SE = mean((vertcat(Y_test{:}) - pred_y_mean).^2);
    R2 = 1-SE/var(vertcat(Y_test{:}));
    Fixed_mean = mean(Fixed,2);
    Random_mean = mean(Random,2);
    Sigma_e_mean = mean(Sigma_e,2);
    var_fixed = var(Fixed_mean);
    var_random = var(Random_mean);

    R2_m = var_fixed/(var_fixed+var_random+Sigma_e_mean);
    R2_c = (var_fixed+var_random)/(var_fixed+var_random+Sigma_e_mean);
    RMSE = sqrt(mean((vertcat(Y_test{:}) - pred_y_mean).^2));
end