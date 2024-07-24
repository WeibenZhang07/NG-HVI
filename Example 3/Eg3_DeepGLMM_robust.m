%% If run on standard computer, change the 'parfor' to 'for' 
% to avoid out-of-memory error.


clear
clc
warning('off','MATLAB:nearlySingularMatrix')

folder = fileparts(which('Eg3_DeepGLMM_robust.m')); 
addpath(genpath(folder));

plotBOX = false; % change to true for boxplots after estimation
%% Use true DeepGLMM data generating process

nn = [5,5]; % hidden layers
R = 100;    % number of repeated simulation for robustness test.
results3_hybrid_ng = cell(1,R);
results3_hybrid_sg = cell(1,R);
results3_nagvac = cell(1,R);
predictive3_hybrid_ng = cell(1,R);
predictive3_hybrid_sg = cell(1,R);
predictive3_nagvac = cell(1,R);
naive3 = cell(1,R);
predictive3_hybrid_ng_out = cell(1,R);
predictive3_hybrid_sg_out = cell(1,R);
predictive3_nagvac_out = cell(1,R);
naive3_out = cell(1,R);
error_hybrid_ng3 = cell(1,R);
error_hybrid_sg3 = cell(1,R);
error_nagvac3 = cell(1,R);
warnings_hybrid_ng3 = zeros(1,R);
warnings_hybrid_sg3 = zeros(1,R);
warnings_nagvac3 = zeros(1,R);
X_sim_train= cell(1,R);
X_sim_val= cell(1,R);
X_sim_test= cell(1,R);
Y_sim_train= cell(1,R);
Y_sim_val= cell(1,R);
Y_sim_test= cell(1,R);

% Settings for the hybrid methods. 
% Refer to README file for details.
setting = default_settings();
setting.J = 0;      % number of MC simulation in predicting step. Temporarily set to 0.
setting.N = 3000;   % number of optimization steps.
        
parfor r = 1:R
    rng(r);

    Deepnet = [5,5];    % hidden layers for DGP
    K = 1000;           % number of groups
    m = 5;              % dimension of covariates
    num_obs = 20;       % number of total observations in each group
    [X_train,y_train,X_val,y_val,X_test,y_test,alpha,eta,pos_rat] = simulate_DeepGLMM(m,K,num_obs);
    
    try
        setting_ng = setting;
        setting_ng.SGA = 0;
        results3_hybrid_ng{1,r} = hybrid_deepglmm_train(X_train,y_train,X_val,y_val,setting_ng);
        [predictive3_hybrid_ng{1,r}] = predictive_metrics_par(X_train,y_train, X_train,y_train,nn,1000,results3_hybrid_ng{1,r}.lambda,1);
        [predictive3_hybrid_ng_out{1,r}] = predictive_metrics_par(X_train,y_train, X_test,y_test,nn,1000,results3_hybrid_ng{1,r}.lambda,1);
    catch ME
        warning('Natural gradient method not completed!')
        warnings_hybrid_ng3(r) = 1;
        error_hybrid_ng3{1,r} = ME.message;
    end
    
     try
        setting_sg = setting;
        setting_sg.SGA = 1;
        results3_hybrid_sg{1,r} = hybrid_deepglmm_train(X_train,y_train,X_val,y_val,setting_sg);
        [predictive3_hybrid_sg{1,r}] = predictive_metrics_par(X_train,y_train, X_train,y_train,nn,1000,results3_hybrid_sg{1,r}.lambda,1);
        [predictive3_hybrid_sg_out{1,r}] = predictive_metrics_par(X_train,y_train, X_test,y_test,nn,1000,results3_hybrid_sg{1,r}.lambda,1);
    catch ME
        warning('Gradient method not completed!');
        warnings_hybrid_sg3(r) = 1;
        error_hybrid_sg3{1,r} = ME.message;
     end
    
    try
        results3_nagvac{1,r} =deepGLMMfit(X_train,y_train,...  
                      X_val,y_val,...
                      'Distribution','binomial',...
                      'Network',nn,... 
                      'Lrate',0.1,...           
                      'Verbose',1,...             % Display training result each iteration
                      'MaxIter',setting.N,...
                      'Patience',10,...          % Higher patience values could lead to overfitting
                      'S',10,...
                      'Seed',100);
        [predictive3_nagvac{1,r}]  = predictive_deepGLMM_vecRE_nagvac(X_train,y_train,X_train,y_train,nn,results3_nagvac{1,r},1000);

        [predictive3_nagvac_out{1,r}]  = predictive_deepGLMM_vecRE_nagvac(X_test,y_test,X_train,y_train,nn,results3_nagvac{1,r},1000);

    catch ME
        warning('NAGVAC not completed!');
        warnings_nagvac3(r) = 1;
        error_nagvac3{1,r} = ME.message;
    end
    [naive3{1,r}] = naive_prediction(y_train,y_train);
    [naive3_out{1,r}] = naive_prediction(y_train,y_test);
    
    X_sim_train{1,r}= X_train;
    X_sim_val{1,r}= X_val;
    X_sim_test{1,r}= X_test;

    Y_sim_train{1,r}= y_train;
    Y_sim_val{1,r}= y_val;
    Y_sim_test{1,r}= y_test;
end
save('Results/Robust_test_polynomialDGP.mat')

R = length(naive3);

%Naive, NAGVAC, NG-HVI, SG-HVI
pce = zeros(4,R); % predictive cross entropy, the lower the better
mcr = zeros(4,R); % misclassification rate, the lower the better
precision = pce;  % precision, the higher the better
recall = pce;     % recall, the higher the better
f1 = pce;         % F1 score, the higher the better

pce_test = zeros(4,R); % predictive cross entropy, the lower the better
mcr_test = zeros(4,R); % misclassification rate, the lower the better
precision_test = pce;  % precision, the higher the better
recall_test = pce;     % recall, the higher the better
f1_test = pce;         % F1 score, the higher the better


for r = 1:R
    % predictive cross entropy
    pce(1,r) = naive3{1, r}.pce;
    pce(2,r) = predictive3_nagvac{1, r}.pce;
    pce(3,r) = predictive3_hybrid_ng{1, r}.pce;
    pce(4,r) = predictive3_hybrid_sg{1, r}.pce;

    pce_test(1,r) = naive3_out{1, r}.pce;
    pce_test(2,r) = predictive3_nagvac_out{1, r}.pce;
    pce_test(3,r) = predictive3_hybrid_ng_out{1, r}.pce;
    pce_test(4,r) = predictive3_hybrid_sg_out{1, r}.pce;
    
    % misclassification rate
    mcr(1,r) = naive3{1, r}.mcr;
    mcr(2,r) = predictive3_nagvac{1, r}.mcr;
    mcr(3,r) = predictive3_hybrid_ng{1, r}.mcr;
    mcr(4,r) = predictive3_hybrid_sg{1, r}.mcr;
    
    mcr_test(1,r) = naive3_out{1, r}.mcr;
    mcr_test(2,r) = predictive3_nagvac_out{1, r}.mcr;
    mcr_test(3,r) = predictive3_hybrid_ng_out{1, r}.mcr;
    mcr_test(4,r) = predictive3_hybrid_sg_out{1, r}.mcr;

    % precision
    precision(1,r) = naive3{1, r}.precision;
    precision(2,r) = predictive3_nagvac{1, r}.precision;
    precision(3,r) = predictive3_hybrid_ng{1, r}.precision;
    precision(4,r) = predictive3_hybrid_sg{1, r}.precision;
    
    precision_test(1,r) = naive3_out{1, r}.precision;
    precision_test(2,r) = predictive3_nagvac_out{1, r}.precision;
    precision_test(3,r) = predictive3_hybrid_ng_out{1, r}.precision;
    precision_test(4,r) = predictive3_hybrid_sg_out{1, r}.precision;

    % recall
    recall(1,r) = naive3{1, r}.recall;
    recall(2,r) = predictive3_nagvac{1, r}.recall;
    recall(3,r) = predictive3_hybrid_ng{1, r}.recall;
    recall(4,r) = predictive3_hybrid_sg{1, r}.recall;
    
    recall_test(1,r) = naive3_out{1, r}.recall;
    recall_test(2,r) = predictive3_nagvac_out{1, r}.recall;
    recall_test(3,r) = predictive3_hybrid_ng_out{1, r}.recall;
    recall_test(4,r) = predictive3_hybrid_sg_out{1, r}.recall;

    % F1 score
    f1(1,r) = 2*(precision(1,r) * recall(1,r))/(precision(1,r) + recall(1,r));
    f1(2,r) = 2*(precision(2,r) * recall(2,r))/(precision(2,r) + recall(2,r));
    f1(3,r) = 2*(precision(3,r) * recall(3,r))/(precision(3,r) + recall(3,r));
    f1(4,r) = 2*(precision(4,r) * recall(4,r))/(precision(4,r) + recall(4,r));

    f1_test(1,r) = 2*(precision_test(1,r) * recall_test(1,r))/(precision_test(1,r) + recall_test(1,r));
    f1_test(2,r) = 2*(precision_test(2,r) * recall_test(2,r))/(precision_test(2,r) + recall_test(2,r));
    f1_test(3,r) = 2*(precision_test(3,r) * recall_test(3,r))/(precision_test(3,r) + recall_test(3,r));
    f1_test(4,r) = 2*(precision_test(4,r) * recall_test(4,r))/(precision_test(4,r) + recall_test(4,r));

end
if plotBOX 
    figure('DefaultAxesFontSize',12,'Position', [10 10 600 300]) %#ok<UNRCH>
    boxplot(pce_test([1:2,4,3],:)','Labels',{'Naive','NAGVAC','SG-HVI','NG-HVI'})
    xlabel('Methods')
    ylabel('PCE_{test}')
    set(gca,'fontsize',12);
    exportgraphics(gcf,'Graphs\compare_pce_test.pdf','Resolution',300);
    saveas(gcf,'Graphs\compare_pce_test.fig');
    
    
    figure('DefaultAxesFontSize',12,'Position', [10 10 600 300])
    boxplot(f1_test([1:2,4,3],:)','Labels',{'Naive','NAGVAC','SG-HVI','NG-HVI'})
    xlabel('Methods')
    ylabel('F1_{test}')
    set(gca,'fontsize',12);
    exportgraphics(gcf,'Graphs\compare_f1_test.pdf','Resolution',300);
    saveas(gcf,'Graphs\compare_f1_test.fig');
end