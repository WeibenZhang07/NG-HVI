%% Data simulation
clear
rng(1234)
folder = fileparts(which('VI_Hybrid_LMM.m'));
addpath(genpath(folder));
warning('off','MATLAB:nearlySingularMatrix')
PlotPDF = false; % change to true for posterior plot after estimation 
PlotELBO = false;% change to true for ELBO plot after estimation 
%% change both m and rho to change number of predictors
rho = [1    0 -0.5   0.2   0;
       0    1    0  -0.5 0.2;
    -0.5    0    1    0 -0.5;
     0.2 -0.5    0    1    0;
       0  0.2 -0.5    0    1];
n = 10000;
m = 5; % number of non-intercept covariates
num_input = m ;
S_input = cov(randn(20,num_input));
   
num_group = 1000; 
sigma_e_input = 1;

variableNames = {'K', 'ratio', 'Elbo_davi', 'p=0', 'p=1', 'p=2', 'p=3'};
variableTypes = {'double', 'double', 'double', 'double', 'double', 'double', 'double'};
T_summary = table('Size', [0, length(variableNames)], 'VariableTypes', variableTypes, 'VariableNames', variableNames);


for sigma_a =[0.01,1,10]
[X_train,Y_train,X_test,Y_test,beta_true,alpha_true,Z_out,eta,group_ind,u,R2_m,R2_c] = data_sim(rho,n,num_group,sigma_e_input, sigma_a);

x = vertcat(X_train{:});
y = vertcat(Y_train{:});

%% MCMC
    priors.beta_sig = 100;
    priors.sigma2_a_a = 1.01;
    priors.sigma2_a_b = 1.01;
    priors.sigma2_e_a = 1.01;
    priors.sigma2_e_b = 1.01;

    N = 10000;

    MCMC = LMM_MCMC(X_train,Y_train,N,priors);
    mcmc_mat = [MCMC.beta(:,N/2+1:end);MCMC.sigma2_a(N/2+1:end);MCMC.sigma2_e(N/2+1:end)];
%% DAVI

[results_davi] = davi_lmm_train(X_train,Y_train,N,0);

%% Hybrid
results = cell(3,1);
results_sga = cell(3,1);
p_num=[1,2,3,0];
for p_ind = 1:4
    p = p_num(p_ind);
    setting = default_settings();
    setting.J = 0;
    setting.N = N;
    setting.damping=10;
    results{p_ind,1} = hybrid_lmm_train(X_train,Y_train,X_test,Y_test,p,setting);

    setting = default_settings();
    setting.SGA = 1;
    setting.J = 0;
    setting.N = N;
    results_sga{p_ind,1} = hybrid_lmm_train(X_train,Y_train,X_test,Y_test,p,setting);
     %% Convergence by ELBO
    if PlotELBO
        figure('DefaultAxesFontSize',14,'Position', [10 10 500 400]) %#ok<UNRCH>
        plot(results_davi.elbo(1:2000),':r','LineWidth',1.2)
        hold on 
        plot(results_sga{p_ind,1}.elbo2(1:2000),'-.','color',[0.30,0.75,0.93],'LineWidth',1.6)
        plot(results{p_ind,1}.elbo2(1:2000), 'k','LineWidth',0.8)
        xlabel("Step")
        ylabel("ELBO")
        ylim([-100000,-5000])
        legend(sprintf('DAVI: %.0f',mean(results_davi.elbo(1900:2000))),sprintf('SG-HVI: %.0f',mean(results_sga{p_ind,1}.elbo2(1900:2000))),sprintf('NG-HVI: %.0f',mean(results{p_ind,1}.elbo2(1900:2000))),'location','southeast','FontSize',12)
        axes('position',[0.58 0.47 0.29 0.28])
        plot(results_davi.elbo(2000-100:2000),':r','LineWidth',1.2)
        hold on
        plot(results_sga{p_ind,1}.elbo2(2000-100:2000),'-.','color',[0.30,0.75,0.93],'LineWidth',2)    
        plot(results{p_ind,1}.elbo2(2000-100:2000),'k','LineWidth',0.1)
        xticks([0 50 100])
        xticklabels({num2str(2000-100),num2str(2000-50),num2str(2000)})
        str = strcat("compare_ELBO_LMM_sigma_a_",num2str(sigma_a),"_P_",num2str(p),".pdf");
        str1 = strcat("compare_ELBO_LMM_sigma_a_",num2str(sigma_a),"_P_",num2str(p),".fig");    
        exportgraphics(gcf,str,'Resolution',300);
        saveas(gcf,str1)
        
        t1 = mean(results_davi.time(N-500:N));
        t2 = mean(results{p_ind,1}.time(N-500:N));
        t3 = mean(results_sga{p_ind,1}.time(N-500:N));
        T1 = (1:2000).*t1;
        T2 = (1:2000).*t2;
        T3 = (1:2000).*t3;
        figure('DefaultAxesFontSize',14,'Position', [10 10 500 400])
        plot(T1,results_davi.elbo(1:2000),':r','LineWidth',1.2)
        hold on 
        plot(T3,results_sga{p_ind,1}.elbo2(1:2000),'-.','color',[0.30,0.75,0.93],'LineWidth',1.6)
        plot(T2,results{p_ind,1}.elbo2(1:2000),'k','LineWidth',0.8)
        ylim([-36000,-5000])
        xlabel("Clock time (s)")
        ylabel("ELBO")
        legend(sprintf('DAVI: %.0f',mean(results_davi.elbo(1900:2000))),sprintf('SG-HVI: %.0f',mean(results_sga{p_ind,1}.elbo2(1900:2000))),sprintf('NG-HVI: %.0f',mean(results{p_ind,1}.elbo2(1900:2000))),'location','southeast','FontSize',12)
        str = strcat("compare_time_ELBO_LMM_sigma_a_",num2str(sigma_a),"_P_",num2str(p),".pdf");
        str1 = strcat("compare_time_ELBO_LMM_sigma_a_",num2str(sigma_a),"_P_",num2str(p),".fig");
        exportgraphics(gcf,str,'Resolution',300);
        saveas(gcf,str1)
    end
end
close all
str = strcat("Results_LMM_sigma_a_",num2str(sigma_a),".mat");
save(str)
if PlotPDF
    [posterior_beta, posterior_sigma2_e, posterior_sigma2_a] = results_compare_LMM_2(beta_true,MCMC, results{1,1},results_davi,0.5*N,N);
    
    for i = 1:m+1
    str = strcat("LMM_sigma_a_",num2str(sigma_a),"posterior_beta_",num2str(i),".pdf");
    str1 = strcat("LMM_sigma_a_",num2str(sigma_a),"posterior_beta_",num2str(i),".fig");
    name = strcat( 'beta_',num2str(i));
    exportgraphics(posterior_beta.(name),str,'Resolution',300);
    saveas(posterior_beta.(name),str1)
    end
    str = strcat("LMM_sigma_a_",num2str(sigma_a),"posterior_sigma2_e",".pdf");
    str1 = strcat("LMM_sigma_a_",num2str(sigma_a),"posterior_sigma2_e",".fig");
    exportgraphics(posterior_sigma2_e,str,'Resolution',300);
    saveas(gcf,str1)
    
    str = strcat("LMM_sigma_a_",num2str(sigma_a),"posterior_sigma2_a",".pdf");
    str1 = strcat("LMM_sigma_a_",num2str(sigma_a),"posterior_sigma2_a",".fig");
    exportgraphics(posterior_sigma2_a,str,'Resolution',300);
    saveas(gcf,str1)
end

close all

elbo_davi = round(mean(results_davi.elbo(end - 100:end)),1);
fac = [4,1,2,3];
elbo_diff = zeros(size(fac));
for i = 1:4
    elbo_diff(i) = round(mean(results{fac(i), 1}.elbo2(end - 100:end))-elbo_davi,1);

end
T = table(num_group,sigma_a/sigma_e_input,elbo_davi, elbo_diff(1), elbo_diff(2), elbo_diff(3), elbo_diff(4), ...
          'VariableNames', {'K','ratio','Elbo_davi', 'p=0', 'p=1', 'p=2', 'p=3'});

T_summary = [T_summary; T];
end
disp(T_summary)