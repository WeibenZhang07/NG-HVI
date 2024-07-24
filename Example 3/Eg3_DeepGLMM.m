clear
clc
warning('off','MATLAB:nearlySingularMatrix')
rng(1);

folder = fileparts(which('Eg3_DeepGLMM.m')); 
addpath(genpath(folder));

plotPCE = false; % change to true for PCE trace plot after estimation
%% Use true DeepGLMM data generating process

nn = [5,5]; % hidden layers
R = 100;    % number of repeated simulation for robustness test.

% Settings for the hybrid methods. 
% Refer to README file for details.
setting = default_settings();
setting.J = 0;      % number of MC simulation in predicting step. Temporarily set to 0.
setting.N = 3000;   % number of optimization steps.

Deepnet = [5,5];    % hidden layers for DGP
K = 1000;           % number of groups
m = 5;              % dimension of covariates
num_obs = 20;       % number of total observations in each group
[X_train,y_train,X_val,y_val,X_test,y_test,alpha,eta,pos_rat] = simulate_DeepGLMM(m,K,num_obs);

try
    setting_ng = setting;
    setting_ng.SGA = 0;
    results3_hybrid_ng = hybrid_deepglmm_train(X_train,y_train,X_val,y_val,setting_ng);
    [predictive3_hybrid_ng] = predictive_metrics_par(X_train,y_train, X_train,y_train,nn,1000,results3_hybrid_ng.lambda,1);
    [predictive3_hybrid_ng_out] = predictive_metrics_par(X_train,y_train, X_test,y_test,nn,1000,results3_hybrid_ng.lambda,1);
catch ME
    warning('Natural gradient method not completed!')
    error_hybrid_ng3 = ME.message;
end

 try
    setting_sg = setting;
    setting_sg.SGA = 1;
    results3_hybrid_sg = hybrid_deepglmm_train(X_train,y_train,X_val,y_val,setting_sg);
    [predictive3_hybrid_sg] = predictive_metrics_par(X_train,y_train, X_train,y_train,nn,1000,results3_hybrid_sg.lambda,1);
    [predictive3_hybrid_sg_out] = predictive_metrics_par(X_train,y_train, X_test,y_test,nn,1000,results3_hybrid_sg.lambda,1);
catch ME
    warning('Gradient method not completed!');
    error_hybrid_sg3 = ME.message;
 end

try
    results3_nagvac =deepGLMMfit(X_train,y_train,...  
                  X_val,y_val,...
                  'Distribution','binomial',...
                  'Network',nn,... 
                  'Lrate',0.1,...           
                  'Verbose',1,...             % Display training result each iteration
                  'MaxIter',setting.N,...
                  'Patience',10,...          % Higher patience values could lead to overfitting
                  'S',10,...
                  'Seed',100);
    [predictive3_nagvac]  = predictive_deepGLMM_vecRE_nagvac(X_train,y_train,X_train,y_train,nn,results3_nagvac,1000);
    [predictive3_nagvac_out]  = predictive_deepGLMM_vecRE_nagvac(X_test,y_test,X_train,y_train,nn,results3_nagvac,1000);

catch ME
    warning('NAGVAC not completed!');
    error_nagvac3 = ME.message;
end
[naive3] = naive_prediction(y_train,y_train);
[naive3_out] = naive_prediction(y_train,y_test);

save('Results/deepGLMM.mat')


t_nagvac = 33.5891*(1:results3_nagvac.out.iteration)/60;
t_hybrid_sg = 0.4845*(1:setting.N)/60;
t_hybrid_ng = 0.4976*(1:setting.N)/60;
t_max = max([t_nagvac,t_hybrid_sg,t_hybrid_ng]);

if plotPCE 
    figure('DefaultAxesFontSize',8,'Position', [10 10 600 300]) %#ok<UNRCH>
    plot(t_hybrid_sg,results3_hybrid_sg.Predictive{1, 5},'-.','color',[0.30,0.75,0.93],'LineWidth',0.8)
    hold on
    plot(t_hybrid_ng,results3_hybrid_ng.Predictive{1, 5}, 'k','LineWidth',0.8)
    plot(t_nagvac,results3_nagvac.out.loss,':r','LineWidth',1.2)
    xlim([0,t_max+2])
    ylabel('PCE')
    xlabel('Clock time (min)') 
    legend('SG-HVI','NG-HVI','NAGVAC','location','northeast','FontSize',8)
    legend boxoff;
    exportgraphics(gcf,'Graphs\pce_time_polynomial.pdf','Resolution',300);
    saveas(gcf,'Graphs\pce_time_polynomial.fig');
end

%% rerun NAGVAC without using stopping rule.
try
    results3_nagvac_nostop =deepGLMMfit(X_train,y_train,...
        X_val,y_val,...
        'Distribution','binomial',...
        'Network',nn,...
        'Lrate',0.1,...
        'Verbose',1,...             % Display training result each iteration
        'MaxIter',setting.N,...
        'Patience',setting.N,...          % Higher patience values could lead to overfitting
        'S',10,...
        'Seed',100);
    [predictive3_nagvac_nostop]  = predictive_deepGLMM_vecRE_nagvac(X_train,y_train,X_train,y_train,nn,results3_nagvac_nostop,1000);

    [predictive3_nagvac_out_nostop]  = predictive_deepGLMM_vecRE_nagvac(X_test,y_test,X_train,y_train,nn,results3_nagvac_nostop,1000);
catch ME
    warning('NAGVAC not completed!');
    error_nagvac3_nostop = ME.message;
end

save('Results/nagvac_nostop.mat','results3_nagvac_nostop','predictive3_nagvac_nostop','predictive3_nagvac_out_nostop')

pce = zeros(1,5); % predictive cross entropy, the lower the better
precision = pce;  % precision, the higher the better
recall = pce;     % recall, the higher the better
f1 = pce;         % F1 score, the higher the better

pce_test = pce; % predictive cross entropy, the lower the better
precision_test = pce;  % precision, the higher the better
recall_test = pce;     % recall, the higher the better
f1_test = pce;         % F1 score, the higher the better

pce(1) = naive3.pce;
pce(2) = predictive3_nagvac.pce;
pce(3) = predictive3_nagvac_nostop.pce;
pce(4) = predictive3_hybrid_sg.pce;
pce(5) = predictive3_hybrid_ng.pce;

pce_test(1) = naive3_out.pce;
pce_test(2) = predictive3_nagvac_out.pce;
pce_test(3) = predictive3_nagvac_out_nostop.pce;
pce_test(4) = predictive3_hybrid_sg_out.pce;
pce_test(5) = predictive3_hybrid_ng_out.pce;

% precision
precision(1) = naive3.precision;
precision(2) = predictive3_nagvac.precision;
precision(3) = predictive3_nagvac_nostop.precision;
precision(4) = predictive3_hybrid_sg.precision;
precision(5) = predictive3_hybrid_ng.precision;


precision_test(1) = naive3_out.precision;
precision_test(2) = predictive3_nagvac_out.precision;
precision_test(3) = predictive3_nagvac_out_nostop.precision;
precision_test(4) = predictive3_hybrid_sg_out.precision;
precision_test(5) = predictive3_hybrid_ng_out.precision;

% recall
recall(1) = naive3.recall;
recall(2) = predictive3_nagvac.recall;
recall(3) = predictive3_nagvac_nostop.recall;
recall(4) = predictive3_hybrid_sg.recall;
recall(5) = predictive3_hybrid_ng.recall;

recall_test(1) = naive3_out.recall;
recall_test(2) = predictive3_nagvac_out.recall;
recall_test(3) = predictive3_nagvac_out_nostop.recall;
recall_test(4) = predictive3_hybrid_sg_out.recall;
recall_test(5) = predictive3_hybrid_ng_out.recall;

% F1 score
f1(1) = 2*(precision(1) * recall(1))/(precision(1) + recall(1));
f1(2) = 2*(precision(2) * recall(2))/(precision(2) + recall(2));
f1(3) = 2*(precision(3) * recall(3))/(precision(3) + recall(3));
f1(4) = 2*(precision(4) * recall(4))/(precision(4) + recall(4));
f1(5) = 2*(precision(5) * recall(5))/(precision(5) + recall(5));

f1_test(1) = 2*(precision_test(1) * recall_test(1))/(precision_test(1) + recall_test(1));
f1_test(2) = 2*(precision_test(2) * recall_test(2))/(precision_test(2) + recall_test(2));
f1_test(3) = 2*(precision_test(3) * recall_test(3))/(precision_test(3) + recall_test(3));
f1_test(4) = 2*(precision_test(4) * recall_test(4))/(precision_test(4) + recall_test(4));
f1_test(5) = 2*(precision_test(5) * recall_test(5))/(precision_test(5) + recall_test(5));

predictive_results = round([pce;pce_test;f1;f1_test],4);
T = table(predictive_results(:, 1), predictive_results(:, 2), predictive_results(:, 3),predictive_results(:, 4),predictive_results(:, 5), ...
          'VariableNames', {'Naive','NAGVAC','NAGVAC(no rule)','SG-HVI','NG-HVI'},...
          'RowNames', {'PCE_train', 'PCE_test','F1_train', 'F1_test'});

disp(T)