function [elbo,elbo2] = stochastic_elbo_LMM(X_train,Y_train,lambda,priors,J,results,varargin)
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
    dim.sample = length(X_train);
    dim.x = size(X_train{1},2);
    [M,p] = size(lambdahat.B);
    
    dim.beta = dim.x;
    dim.Gamma = 1;
    dim.Sigma = 1;
    

    for h = 1:dim.sample
        dim.H(h,1) = size(X_train{h},1); 
    end
    
    elbo_vec = zeros(1,J);
    elbo_vec2 = zeros(1,J);
    for j = 1:J
    %% Integrate out random effects
        epsilon = normrnd(0,1,M,1);
        z = normrnd(0,1,p,1);
        theta = lambdahat.mu + lambdahat.B * z + (lambdahat.d.*epsilon);

        beta = lambdahat.mu(1:dim.beta);
        TGamma = lambdahat.mu(end - dim.Gamma - dim.Sigma+1:end - dim.Sigma);
        TSigma = lambdahat.mu(end - dim.Sigma+1:end);
       
        sigma_y = exp(TSigma);
        sigma_a = exp(TGamma);

        log_ystar = 0;
        for ii = 1:dim.sample
            n_h = length(Y_train{ii});
            mu_y = X_train{ii}*beta;
            iota = ones(n_h,1);
            inv_cov_y = eye(n_h)./sigma_y - iota*iota'./((1/sigma_a + n_h/sigma_y)*sigma_y^2);
            eta = Y_train{ii} - mu_y;
            log_ystar = log_ystar -0.5*n_h*log(2*pi) +0.5*log(det(inv_cov_y)) - 0.5*eta'*inv_cov_y*eta;
        end

        log_beta = -0.5*dim.beta*log(2*pi) -0.5*dim.beta*log(priors.beta_sig) - 0.5/priors.beta_sig*(beta'*beta);
        log_sigma_y = priors.Sigma_a  *log(priors.Sigma_b) -log(gamma(priors.Sigma_a)) ...
                        -(priors.Sigma_a + 1)*log(sigma_y) - priors.Sigma_b/sigma_y;
        log_sigma_a = priors.Gamma_a    *log(priors.Gamma_b) -log(gamma(priors.Gamma_a)) ...
                        -(priors.Gamma_a + 1)*log(sigma_a) - priors.Gamma_b/sigma_a;
        
        Sigma = lambdahat.B*lambdahat.B'+diag(lambdahat.d.^2);
        log_q_0_lambda = -0.5*M*log(2*pi) -0.5*log(det(Sigma)) - 0.5*(theta-lambdahat.mu)'/(Sigma)*(theta-lambdahat.mu);

         elbo_vec(j) = log_ystar + log_beta + log_sigma_a +log_sigma_y - log_q_0_lambda;
        %% Do not integrate out random effects
        epsilon2 = normrnd(0,1,M,1);
        z2 = normrnd(0,1,p,1);
        theta2 = lambdahat.mu + lambdahat.B * z2 + (lambdahat.d.*epsilon2);

        beta2 = lambdahat.mu(1:dim.beta);
        TGamma2 = lambdahat.mu(end - dim.Gamma - dim.Sigma+1:end - dim.Sigma);
        TSigma2 = lambdahat.mu(end - dim.Sigma+1:end);
       
        sigma_y2 = exp(TSigma2);
        sigma_a2 = exp(TGamma2);

        log_y2 = 0;
        log_alpha_h = 0;
        log_alpha_h_posterior = 0;
        Alpha_h = VB_alpha_i(X_train,Y_train,beta2, sigma_a2,sigma_y2);
        for ii = 1:dim.sample
            n_h = length(Y_train{ii});
            alpha_h = Alpha_h{ii};
            iota = ones(n_h,1);
            mu_y2 = X_train{ii}*beta2 +alpha_h.*iota;
            eta2 = Y_train{ii} - mu_y2;
            log_y2 = log_y2 -0.5*n_h*log(2*pi) -0.5*n_h*log(sigma_y2) - 0.5/sigma_y2*(eta2'*eta2);
            log_alpha_h = log_alpha_h -0.5*log(2*pi) -0.5*log(sigma_a2) - 0.5/sigma_a2*alpha_h^2;
            cov_alpha_h_posterior = 1/(1/sigma_a2 + n_h/sigma_y2);
            mean_alpha_h_posterior = cov_alpha_h_posterior*iota'*(Y_train{ii} - X_train{ii}*beta2)/sigma_y2;
            log_alpha_h_posterior = log_alpha_h_posterior -0.5*log(2*pi) -0.5*log(cov_alpha_h_posterior) - 0.5/cov_alpha_h_posterior*(alpha_h-mean_alpha_h_posterior)^2;
        end

        log_beta2 = -0.5*dim.beta*log(2*pi) -0.5*dim.beta*log(priors.beta_sig) - 0.5/priors.beta_sig*(beta2'*beta2);
        log_sigma_y2 = priors.Sigma_a  *log(priors.Sigma_b) -log(gamma(priors.Sigma_a)) ...
                        -(priors.Sigma_a + 1)*log(sigma_y2) - priors.Sigma_b/sigma_y2;
        log_sigma_a2 = priors.Gamma_a    *log(priors.Gamma_b) -log(gamma(priors.Gamma_a)) ...
                        -(priors.Gamma_a + 1)*log(sigma_a2) - priors.Gamma_b/sigma_a2;
        
        Sigma2 = lambdahat.B*lambdahat.B'+diag(lambdahat.d.^2);
        log_q_0_lambda = -0.5*M*log(2*pi) -0.5*log(det(Sigma2)) - 0.5*(theta2-lambdahat.mu)'/(Sigma2)*(theta2-lambdahat.mu);

        elbo_vec2(j) = log_y2 + log_alpha_h + log_beta2 + log_sigma_a2 +log_sigma_y2 - log_q_0_lambda - log_alpha_h_posterior;
    end
    elbo = mean(elbo_vec);
    elbo2 = mean(elbo_vec2);
end