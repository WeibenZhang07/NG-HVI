function [results] = davi_deeplmm_vecRE_train(X_train,Y_train,nn,N,stopping)

rng(1234)
elbo = zeros(N,1);
rmse = zeros(N,1);
R2 = zeros(N,1);
elbo_mean =[];
time_davi = zeros(N,1);

%% Deep model parameters
dim.X = size(X_train{1},2);
dim.input = dim.X-1;
NN = [dim.input nn]; % include input layer
for i = 1:(length(NN)-1)
    dim.weights(i) = NN(i+1)*(NN(i)+1);
end
dim.weights_vec = sum(dim.weights);

%% Set parameters
sigma2_beta = 100;
sigma2_weights_vec = 100;
a_e = 1.01;
b_e = 1.01;
r = NN(end)+1;
v = r+1;

S = 0.01*eye(r);
m = 0;
beta_weight = 0.9;

%% Initial values
H = length(Y_train);
d_weights_vec = dim.weights_vec;
d_beta = NN(end)+1;
d_w = length(vech(S));
d_global = d_weights_vec + d_beta +1 +d_w;
d_local = r*H;
d = d_global +d_local;
mu = zeros(d,1);
weights = InitializeNN_hybrid(NN);
mu(1:d_weights_vec) = [reshape(weights{1,1},[],1); reshape(weights{1,2},[],1)];
% global parameter
C1 = 0.1*eye(d_global);
% local parameter
C2 = cell(H,1);
z2 = cell(H,1);
tAlpha_h = cell(H,1);
g_h2 = cell(H,1);
G12 = cell(H,1);
G12_bar = cell(H,1);
H2 = cell(H,1);
double_bar_H2 = cell(H,1);
sigma2_g_h2 = cell(H,1);
C2_double_bar_H2 = cell(H,1);
lambda2 = [];
for h = 1:H
C2{h} = 0.1*eye(r);         
z2{h} = normrnd(0,1,r,1);
tAlpha_h{h} = C2{h}*z2{h} + mu(d_global+1+(h-1)*r:d_global+r+(h-1)*r);
lambda2 = [lambda2;vech(C2{h})];
end

lambda = [mu;vech(C1);lambda2];
alpha = 0.001*sqrt(length(lambda));

%% Logistic regression without random effect
for i = 1:N
    tic
z1 = normrnd(0,1,d_global,1);
theta = C1*z1 + mu(1:d_global);
for h = 1:H
z2{h} = normrnd(0,1,r,1);
tAlpha_h{h} = C2{h}*z2{h} + mu(d_global+1+(h-1)*r:d_global+r+(h-1)*r);
lambda2 = [lambda2;vech(C2{h})];
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
for h = 1:H
    x_h = X_train{h};
    Z_out{h} = features_output(x_h,weights);
    z_out_h = Z_out{h};
    y_h = Y_train{h};
    z_h = Z_out{h};
    talpha_h = tAlpha_h{h};
    [Alpha_h{h}] = f_alpha_h(z_out_h,z_h,y_h,beta, talpha_h,sigma2_e,omega);
    Y_hat{h} = z_out_h*beta + z_h*Alpha_h{h};
end

[g_log_joint_density] = f_g_log_joint_density(X_train,Z_out,Y_train,Z_out,weights,beta,Alpha_h,omega,exp(theta_e),sigma2_weights_vec,sigma2_beta,a_e,b_e,v,S);
g_log_q_lambda = -C1'\z1;
for h = 1:H
    g_log_q_lambda = [g_log_q_lambda;-C2{h}\z2{h}];
end
g_h = g_log_joint_density-g_log_q_lambda;
g_h1= g_h(1:d_global);
for h = 1:H
g_h2{h} = g_h(d_global+1+(h-1)*r:d_global+r+(h-1)*r);
G12{h} = g_h2{h}*z2{h}';
G12_bar{h} = tril(G12{h});
H2{h} = C2{h}'*G12_bar{h};
double_bar_H2{h} = tril(H2{h}) - diag(diag(H2{h}))/2;
sigma2_g_h2{h} = C2{h}'*C2{h}*g_h2{h};
C2_double_bar_H2{h} = vech(C2{h} * double_bar_H2{h});
end
G11 = g_h1*z1';
G11_bar = tril(G11);
H1 = C1'*G11_bar;
double_bar_H1 = tril(H1) - diag(diag(H1))/2;

Sigma1 = C1*C1';
ng = [Sigma1 * g_h1; vertcat(sigma2_g_h2{:});vech(C1 * double_bar_H1);vertcat(C2_double_bar_H2{:} )];
norm_ng = norm(ng);
m = beta_weight*m + (1-beta_weight)/norm_ng*ng;
m_hat = m/(1-beta_weight^i);
lambda = lambda + alpha*m_hat;
mu = lambda(1:d);
C1_vech = lambda(d+1:d+d_global*(d_global+1)/2);
C1 = invert_vech(C1_vech);
C2_all = lambda(d+d_global*(d_global+1)/2+1:end);
for h = 1:H
    len_vec_c2 = r*(r+1)/2;
    C2{h} = C2_all(1 + (h-1)*len_vec_c2:h*len_vec_c2);
    C2{h} = invert_vech(C2{h});
end
time_davi(i) = toc;

        if stopping ==1
            if mod(i,1000)==0
                elbo_mean = [elbo_mean mean(real(elbo(i-999:i)))];
                i
            end
            if i >3000
                mdl = fitlm(1:3,elbo_mean(end-2:end));

                if mdl.Coefficients{2,1}<0.01
                    break
                end
            end
        end   

    lambda_train.mu(:,i) = mu;
    lambda_train.C1_vech(:,i) = C1_vech; % global parameter
    lambda_train.C2_vech(:,i) = C2_all;         % local parameter
    [elbo(i),rmse(i),R2(i),~] = predictive_deepLMM_vecRE_davi(X_train,Y_train,X_train,Y_train,nn,lambda_train,sigma2_weights_vec,sigma2_beta,a_e,b_e,S,v,1,0,i);    
    if rem(i,100) == 0
       disp(['Iteration: ',num2str(i),'   -  ELBO: ',num2str(elbo(i))]);
    end

end

results.lambda = lambda_train;
results.time = time_davi;
results.elbo = elbo;
results.rmse = rmse;
results.R2 = R2;
results.priors.sigma2_weights_vec = sigma2_weights_vec;
results.priors.sigma2_beta = sigma2_beta;
results.priors.a_e = a_e;
results.priors.b_e = b_e;
results.priors.S = S;
results.priors.v = v;
if stopping ==1
    results.elbo_mean = elbo_mean;
end

end