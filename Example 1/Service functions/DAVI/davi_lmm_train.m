function [results] = davi_lmm_train(X_train,Y_train,N,stopping)

rng(1234);
elbo = zeros(N,1);
elbo_mean =[];
time_davi = zeros(N,1);
%% Set parameters
sigma2_beta = 100;
a_e = 1.01;
b_e = 1.01;
a_alpha = 1.01;
b_alpha = 1.01;
m = 0;
beta_weight = 0.9;

%% Initial values
H = length(Y_train);
d_beta = size(X_train{1},2);
d_global = d_beta +1 +1;
d = d_global +H;
mu = zeros(d,1);
C1 = 1*eye(d_global); % global parameter
C2 = ones(H,1);         % local parameter
lambda = [mu;vech(C1);C2];
alpha = 0.001*sqrt(length(lambda));

%% Store results
mu_results = zeros(d,N);
C1_vech_results = zeros(d_global*(d_global+1)/2,N); % global parameter
C2_results = zeros(H,1);         % local parameter
%% Logistic regression without random effect
for i = 1:N
    tic
    z1 = normrnd(0,1,d_global,1);
    z2 = normrnd(0,1,H,1);
    theta = C1*z1 + mu(1:d_global);
    tAlpha_h = C2.*z2 + mu(d_global+1:end);
    beta = theta(1:d_beta);
    sigma2_e = exp(theta(d_beta+1));
    sigma2_alpha = exp(theta(d_beta+2));
    Alpha_h = cell(H,1);
    for h = 1:H
        x_h = X_train{h};
        y_h = Y_train{h};
        [Alpha_h{h}] = f_alpha_h(x_h,y_h,beta, tAlpha_h(h),sigma2_e,sigma2_alpha);
    end
    
    [g_log_joint_density] = f_g_log_joint_density(X_train,Y_train,beta,Alpha_h,sigma2_alpha,sigma2_e,sigma2_beta,a_e,b_e,a_alpha,b_alpha);
    g_log_q_lambda = [-C1'\z1;-1./C2.*z2];
    g_h = g_log_joint_density-g_log_q_lambda;
    g_h1= g_h(1:d_global);
    g_h2 = g_h(d_global+1:end);
    G11 = g_h1*z1';
    G11_bar = tril(G11);
    H1 = C1'*G11_bar;
    double_bar_H1 = tril(H1) - diag(diag(H1))/2;
    G12 = g_h2.*z2;
    G12_bar = (G12);
    H2 = C2.*G12_bar;
    double_bar_H2 = (H2)/2;
    Sigma1 = C1*C1';
    ng = [Sigma1 * g_h1; C2.^2.*g_h2;vech(C1 * double_bar_H1);C2.*double_bar_H2];
    norm_ng = norm(ng);
    m = beta_weight*m + (1-beta_weight)/norm_ng*ng;
    m_hat = m/(1-beta_weight^i);
    lambda = lambda + alpha*m_hat;
    mu = lambda(1:d);
    C1_vech = lambda(d+1:d+d_global*(d_global+1)/2);
    C1 = invert_vech(C1_vech);
    C2 = lambda(d+d_global*(d_global+1)/2+1:end);
    time_davi(i) = toc;
     [log_joint_density] = f_log_joint(X_train,Y_train,beta,sigma2_e,sigma2_alpha,tAlpha_h,sigma2_beta,a_e,b_e,a_alpha,b_alpha);
      log_q_lambda = - d_global/2*log(2*pi) - sum(log(abs(diag(C1)))) - 0.5*(z1'*z1)...
          -H/2*log(2*pi)-0.5*sum(log(C2.^2)) - 0.5*sum(z2.*z2);
     elbo(i) = log_joint_density - log_q_lambda;

        if stopping ==1
            if mod(i,1000)==0
                elbo_mean = [elbo_mean mean(real(elbo(i-999:i)))];
            end
            if i >3000
                mdl = fitlm(1:3,elbo_mean(end-2:end));
            
                if mdl.Coefficients{2,1}<0.01
                    break
                end
            end
        end   
    mu_results(:,i) = mu;
    C1_vech_results(:,i) = C1_vech; % global parameter
    C2_results(:,i) = C2;         % local parameter
    
end

results.mu = mu_results;
results.C1_vech = C1_vech_results;
results.C2 = C2_results;
results.time = time_davi;
results.elbo = elbo;
if stopping ==1
    results.elbo_mean = elbo_mean;
end

end