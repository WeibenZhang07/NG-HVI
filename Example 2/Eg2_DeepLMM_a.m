clear
clc
warning('off','MATLAB:nearlySingularMatrix')
rng(15);

folder = fileparts(which('Eg2_DeepLMM_a.m')); 
addpath(genpath(folder));

plotELBO = false; % change to true for ELBO plot after estimation
%% Use true DeepGLMM data generating process

% Settings for the hybrid methods. 
% Refer to README file for details.
setting = default_settings();
setting.J = 0;      % number of MC simulation in predicting step. Temporarily set to 0.
setting.N = 3000;   % number of optimization steps.

load("parameters_small.mat")

vec_RE = 1; % generate multivariate random effects
linear = 0; % use deepnet in DGP
p=3;        % number of factors in covariance matrix of VA

[X_train,Y_train,X_val,Y_val,X_test,Y_test,weights,beta,alpha,Z_out,eta,R2_m,R2_c] = data_sim(V_x,n,K,nn,sigma_e,Sigma_alpha,linear,vec_RE,'gaussian');

try
    setting_ng = setting;
    results_ng = hybrid_Deeplmm_vecRE_train(X_train,Y_train,X_test,Y_test,nn,p,setting_ng);
    [~,~,predictive_ng.RMSE,predictive_ng.R2,~,~] = predictive_deepLMM_vecRE(X_train,Y_train,X_train,Y_train,nn,results_ng.lambda,results_ng.priors,1000,1);
    [~,~,predictive_ng.RMSE_out,predictive_ng.R2_out,~,~] = predictive_deepLMM_vecRE(X_test,Y_test,X_train,Y_train,nn,results_ng.lambda,results_ng.priors,1000,1);
catch ME
    warning('Natural gradient method not completed!')
    error_ng = ME.message;
end

try
    setting_sg = setting;
    setting_sg.SGA = 1;
    results_sg = hybrid_Deeplmm_vecRE_train(X_train,Y_train,X_test,Y_test,nn,p,setting_sg);
    [~,~,predictive_sg.RMSE,predictive_sg.R2,~,~] = predictive_deepLMM_vecRE(X_train,Y_train,X_train,Y_train,nn,results_sg.lambda,results_sg.priors,1000,1);
    [~,~,predictive_sg.RMSE_out,predictive_sg.R2_out,~,~] = predictive_deepLMM_vecRE(X_test,Y_test,X_train,Y_train,nn,results_sg.lambda,results_sg.priors,1000,1);

catch ME
    warning('Gradient method not completed!')
    error_sg = ME.message;
end    

try

[results_davi] = davi_deeplmm_vecRE_train(X_train,Y_train,nn,setting.N,0);
[~,predictive_davi.RMSE,predictive_davi.R2,~] = predictive_deepLMM_vecRE_davi(X_train,Y_train,X_train,Y_train,nn,results_davi.lambda,results_davi.priors.sigma2_weights_vec,results_davi.priors.sigma2_beta,results_davi.priors.a_e,results_davi.priors.b_e,results_davi.priors.S,results_davi.priors.v,1000,1);
[~,predictive_davi.RMSE_out,predictive_davi.R2_out,~] = predictive_deepLMM_vecRE_davi(X_test,Y_test,X_train,Y_train,nn,results_davi.lambda,results_davi.priors.sigma2_weights_vec,results_davi.priors.sigma2_beta,results_davi.priors.a_e,results_davi.priors.b_e,results_davi.priors.S,results_davi.priors.v,1000,1);

catch ME
    warning('DAVI not completed!');
    error_davi = ME.message;
end


if plotELBO
    figure('DefaultAxesFontSize',12,'Position', [10 10 600 295]) %#ok<UNRCH>
    plot(results_ng.predictive{1,1},  'k','LineWidth',0.8)
    hold on
    plot(results_sg.predictive{1,1}, '-.','color',[0.30,0.75,0.93],'LineWidth',0.8)
    plot(results_davi.elbo,':r','LineWidth',1.2)
    ylim([-150000 -9000])
    xlabel("Step")
    ylabel("ELBO")
    legend(sprintf('NG-HVI: %.0f',mean(results_ng.predictive{1,1}(end - 100:end))),sprintf('SG-HVI: %.0f',mean(results_sg.predictive{1,1}(end - 100:end))),sprintf('DAVI: %.0f',mean(results_davi.elbo(end - 100:end))),'location','southeast','FontSize',12)
    legend boxoff
    str = strcat("compare_ELBO_deepLMM_vecRE_sigma_a_",num2str(1),"_P_",num2str(3),".pdf");
    str1 = strcat("compare_ELBO_deepLMM_vecRE_sigma_a_",num2str(1),"_P_",num2str(3),".fig");
    exportgraphics(gcf,str,'Resolution',300);
    saveas(gcf,str1)
end

try
    setting_sg = default_settings();
    setting_sg.SGA = 1;
    setting_sg.J = 0;
    setting_sg.N = 10000;

    results_sg_10000 = hybrid_Deeplmm_vecRE_train(X_train,Y_train,X_test,Y_test,nn,p,setting_sg);
    [~,~,predictive_sg_10000.RMSE,predictive_sg_10000.R2,~,~] = predictive_deepLMM_vecRE(X_train,Y_train,X_train,Y_train,nn,results_sg_10000.lambda,results_sg_10000.priors,1000,1);
    [~,~,predictive_sg_10000.RMSE_out,predictive_sg_10000.R2_out,~,~] = predictive_deepLMM_vecRE(X_test,Y_test,X_train,Y_train,nn,results_sg_10000.lambda,results_sg_10000.priors,1000,1);

catch ME
    warning('Gradient method not completed!')
    error_sg = ME.message;
end   

save('Results/deepLMM_a.mat','-v7.3')

T = table(round([predictive_davi.R2;predictive_davi.R2_out],4), ...
          round([predictive_sg.R2;predictive_sg.R2_out],4),...
          round([predictive_ng.R2;predictive_ng.R2_out],4),...
          round([predictive_sg_10000.R2;predictive_sg_10000.R2_out],4), ...
          'VariableNames', {'DAVI','SG-HVI-1','NG-HVI','SG-HVI-2'},...
          'RowNames', {'R2_train', 'R2_test'});

disp(T)
