clear
clc
warning('off','MATLAB:nearlySingularMatrix')
rng(1)

folder = fileparts(which('Eg2_DeepLMM_b.m')); 
addpath(genpath(folder));
plotELBO = false; % change to true for ELBO plot after estimation

% Settings for the hybrid methods. 
% Refer to README file for details.
setting = default_settings();
setting.J = 0;
setting.N = 3000;

load("parameters_large.mat")
vec_RE = 1; % generate multivariate random effects
linear = 0; % use deepnet in DGP
p=3;        % number of factors in covariance matrix of VA

[X_train,Y_train,X_val,Y_val,X_test,Y_test,~,~,alpha,Z_out,eta,R2_m,R2_c] = data_sim_large(V_x,n,K,nn,Sigma_e,Sigma_alpha,weights,beta,linear,vec_RE,'gaussian');

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


if plotELBO
    figure('DefaultAxesFontSize',12,'Position', [10 10 600 295]) %#ok<UNRCH> 
    plot(results_ng.predictive{1,1},  'k','LineWidth',0.8)
    hold on
    plot(results_sg.predictive{1,1}, '-.','color',[0.30,0.75,0.93],'LineWidth',0.8)
    xlabel("Step")
    ylabel("ELBO")
    legend(sprintf('NG-HVI: %.0f',mean(results_ng.predictive{1,1}(end - 100:end))),sprintf('SG-HVI: %.0f',mean(results_sg.predictive{1,1}(end - 100:end))),'location','southeast','FontSize',12)
    legend boxoff
    axes('position',[0.58 0.47 0.29 0.28])
    plot(results_ng.predictive{1,1}(end-100:end),'k','LineWidth',0.8)
    hold on
    plot(results_sg.predictive{1,1}(end-100:end),'-.','color',[0.30,0.75,0.93],'LineWidth',0.8)    
    
    str = strcat("compare_ELBO_large_deepLMM_vecRE_sigma_a_",num2str(1),"_P_",num2str(3),".pdf");
    str1 = strcat("compare_ELBO_large_deepLMM_vecRE_sigma_a_",num2str(1),"_P_",num2str(3),".fig");
    exportgraphics(gcf,str,'Resolution',300);
    saveas(gcf,str1)
end

setting_sg = default_settings();
setting_sg.SGA = 1;
setting_sg.J = 0;
setting_sg.N = 10000;

results_sg_10000 = hybrid_Deeplmm_vecRE_train(X_train,Y_train,X_test,Y_test,nn,p,setting_sg);
[~,~,predictive_sg_10000.RMSE,predictive_sg_10000.R2,~,~] = predictive_deepLMM_vecRE(X_train,Y_train,X_train,Y_train,nn,results_sg_10000.lambda,results_sg_10000.priors,1000,1);
[~,~,predictive_sg_10000.RMSE_out,predictive_sg_10000.R2_out,~,~] = predictive_deepLMM_vecRE(X_test,Y_test,X_train,Y_train,nn,results_sg_10000.lambda,results_sg_10000.priors,1000,1);

save('Results/deepLMM_b.mat','-v7.3')


T = table(round([predictive_sg.R2;predictive_sg.R2_out],4),...
          round([predictive_ng.R2;predictive_ng.R2_out],4),...
          round([predictive_sg_10000.R2;predictive_sg_10000.R2_out],4), ...
          'VariableNames', {'SG-HVI-1','NG-HVI','SG-HVI-2'},...
          'RowNames', {'R2_train', 'R2_test'});

disp(T)
