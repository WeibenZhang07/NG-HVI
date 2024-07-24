%% If run on standard computer, change the 'parfor' to 'for' 
% to avoid out-of-memory error.

clear
clc
warning('off','MATLAB:nearlySingularMatrix')
rng(1234);

folder = fileparts(which('Eg2_DeepLMM_a_robust.m')); 
addpath(genpath(folder));

plotBOX = false; % change to true for boxplots after estimation
%% Use true DeepGLMM data generating process

R = 100;    % number of repeated simulation for robustness test.
results_ng = cell(1,R);
results_sg = cell(1,R);
results_davi = cell(1,R);

predictive_ng = cell(1,R);
predictive_sg = cell(1,R);
predictive_davi = cell(1,R);

X_sim_train= cell(1,R);
X_sim_val= cell(1,R);
X_sim_test= cell(1,R);
Y_sim_train= cell(1,R);
Y_sim_val= cell(1,R);
Y_sim_test= cell(1,R);

error_ng = cell(1,R);
error_sg = cell(1,R);
error_davi = cell(1,R);
warnings_ng = zeros(1,R);
warnings_sg = zeros(1,R);
warnings_davi = zeros(1,R);

% Settings for the hybrid methods. 
% Refer to README file for details.
setting = default_settings();
setting.J = 0;      % number of MC simulation in predicting step. Temporarily set to 0.
setting.N = 3000;   % number of optimization steps.

load("parameters_small.mat")
vec_RE = 1; % generate multivariate random effects
linear = 0; % use deepnet in DGP
p=3;        % number of factors in covariance matrix of VA
parfor r = 1:R
    rng(r)
    [X_train,Y_train,X_val,Y_val,X_test,Y_test,weights,beta,alpha,Z_out,eta,R2_m,R2_c] = data_sim(V_x,n,K,deepnet,sigma_e,Sigma_alpha,linear,vec_RE,'gaussian');
    
    try
        setting_ng = setting;
        results_ng{1,r} = hybrid_Deeplmm_vecRE_train(X_train,Y_train,X_test,Y_test,nn,p,setting_ng);
        [~,~,predictive_ng{1,r}.RMSE,predictive_ng{1,r}.R2,~,~] = predictive_deepLMM_vecRE(X_train,Y_train,X_train,Y_train,nn,results_ng{1,r}.lambda,results_ng{1,r}.priors,1000,1);
        [~,~,predictive_ng{1,r}.RMSE_out,predictive_ng{1,r}.R2_out,~,~] = predictive_deepLMM_vecRE(X_test,Y_test,X_train,Y_train,nn,results_ng{1,r}.lambda,results_ng{1,r}.priors,1000,1);
    catch ME
        warning('Natural gradient method not completed!')
        warnings_ng(r) = 1;
        error_ng{1,r} = ME.message;
    end
    
    try
        setting_sg = setting;
        setting_sg.SGA = 1;
        results_sg{1,r} = hybrid_Deeplmm_vecRE_train(X_train,Y_train,X_test,Y_test,nn,p,setting_sg);
        [~,~,predictive_sg{1,r}.RMSE,predictive_sg{1,r}.R2,~,~] = predictive_deepLMM_vecRE(X_train,Y_train,X_train,Y_train,nn,results_sg{1,r}.lambda,results_sg{1,r}.priors,1000,1);
        [~,~,predictive_sg{1,r}.RMSE_out,predictive_sg{1,r}.R2_out,~,~] = predictive_deepLMM_vecRE(X_test,Y_test,X_train,Y_train,nn,results_sg{1,r}.lambda,results_sg{1,r}.priors,1000,1);

    catch ME
        warning('Gradient method not completed!')
        warnings_sg(r) = 1;
        error_sg{1,r} = ME.message;
    end    

    try
    
    [results_davi{1,r}] = davi_deeplmm_vecRE_train(X_train,Y_train,nn,setting.N,0);
    [~,predictive_davi{1,r}.RMSE,predictive_davi{1,r}.R2,~] = predictive_deepLMM_vecRE_davi(X_train,Y_train,X_train,Y_train,nn,results_davi{1,r}.lambda,results_davi{1,r}.priors.sigma2_weights_vec,results_davi{1,r}.priors.sigma2_beta,results_davi{1,r}.priors.a_e,results_davi{1,r}.priors.b_e,results_davi{1,r}.priors.S,results_davi{1,r}.priors.v,1000,1);
    [~,predictive_davi{1,r}.RMSE_out,predictive_davi{1,r}.R2_out,~] = predictive_deepLMM_vecRE_davi(X_test,Y_test,X_train,Y_train,nn,results_davi{1,r}.lambda,results_davi{1,r}.priors.sigma2_weights_vec,results_davi{1,r}.priors.sigma2_beta,results_davi{1,r}.priors.a_e,results_davi{1,r}.priors.b_e,results_davi{1,r}.priors.S,results_davi{1,r}.priors.v,1000,1);

    catch ME
        warning('DAVI not completed!');
        warnings_davi(r) = 1;
        error_davi{1,r} = ME.message;
    end
    X_sim_train{1,r}= X_train;
    X_sim_val{1,r}= X_val;
    X_sim_test{1,r}= X_test;

    Y_sim_train{1,r}= Y_train;
    Y_sim_val{1,r}= Y_val;
    Y_sim_test{1,r}= Y_test;
end
save('Results/Robust_test_deepLMM.mat','-v7.3')

R = 100;            %length(X_sim_train);
R2 = zeros(4,R);    % R^2, the higher the better
RMSE = zeros(4,R);  % RMSE, the lower the better
R2_out = zeros(4,R);    % R^2, the higher the better
RMSE_out = zeros(4,R);  % RMSE, the lower the better


for r = 1:R
    % R2
    R2(1,r) = predictive_davi{1, r}.R2;
    R2(3,r) = predictive_sg{1, r}.R2;
    R2(4,r) = predictive_ng{1, r}.R2;

    R2_out(1,r) = predictive_davi{1, r}.R2_out;
    R2_out(3,r) = predictive_sg{1, r}.R2_out;
    R2_out(4,r) = predictive_ng{1, r}.R2_out;

    
    % RMSE
    RMSE(1,r) = predictive_davi{1, r}.RMSE;
    RMSE(3,r) = predictive_sg{1, r}.RMSE;
    RMSE(4,r) = predictive_ng{1, r}.RMSE;
 
    RMSE_out(1,r) = predictive_davi{1, r}.RMSE_out;
    RMSE_out(3,r) = predictive_sg{1, r}.RMSE_out;
    RMSE_out(4,r) = predictive_ng{1, r}.RMSE_out;    
end

if plotBOX
    R2_out_ratio = vertcat(R2_out(3,:) ./R2_out(1,:),R2_out(4,:) ./R2_out(1,:)); %#ok<UNRCH>
    figure('DefaultAxesFontSize',12,'Position', [10 10 600 300])
    boxplot(R2_out_ratio','Labels',{'SG-HVI/DAVI','NG-HVI/DAVI'})
    xlabel('Methods')
    ylabel('R^2_{test} Ratio')
    set(gca,'fontsize',12);
    exportgraphics(gcf,'Graphs\R2_deepLMM_out_ratio.pdf','Resolution',300);
    saveas(gcf,'Graphs\R2_deepLMM_out_ratio.fig');
end
