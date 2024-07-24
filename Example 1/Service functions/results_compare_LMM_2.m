function [posterior_beta, posterior_sigma2_e, posterior_sigma2_a] = results_compare_LMM_2(beta,mcmc, results1,results_tan,burn_mcmc,N)

n_para = length(beta) +1;
var_name = cell(n_para,1);
for i = 1:n_para
    if i == n_para
        var_name{i,1} = '\sigma_\alpha^2';
    else
        var_name{i,1} = strcat( 'beta_',num2str(i));
    end
end

% Beta
    mcmc_beta_mean = mean(mcmc.beta(:,(burn_mcmc+1):N),2);
    mcmc_beta_cov = cov(mcmc.beta(:,(burn_mcmc+1):N)');
    
    mu1_1 = mean(results1.lambda.mu(:,(end-100):end),2);
    B1_1 = mean(results1.lambda.B(:,:,(end-100):end),3);
    d1_1 = mean(results1.lambda.d(:,:,(end-100):end),3);
    mean_va1_1 = mu1_1;
    cov_va1_1 = B1_1*B1_1'+ diag(d1_1)^2;
    sd_va1_1 = sqrt(diag(B1_1*B1_1'+ diag(d1_1)^2));

    mu_tan = mean(results_tan.mu(:,(end-100):end),2);
    C1_vec = mean(results_tan.C1_vech(:,(end-100):end),2);
    C1 = invert_vech(C1_vec);
    mean_va_tan = mu_tan(1:size(C1,1));
    cov_va_tan = C1*C1';
    sd_va_tan = sqrt(diag(cov_va_tan));
    
    lim_low1_1 = mean_va1_1 - 3.*sd_va1_1;
    lim_high1_1 = mean_va1_1 + 3.*sd_va1_1;
    lim_low_tan = mean_va_tan - 3.*sd_va_tan;
    lim_high_tan = mean_va_tan + 3.*sd_va_tan;

    m = length(beta);
    xl = zeros(m,2);
for i = 1:m
    posterior_beta.(var_name{i,1})=figure('DefaultAxesFontSize',14,'Position', [10 10 500 400]);
    [density, xi] = ksdensity(mcmc.beta(i,(burn_mcmc+1):N));
    
   nexttile 

   plot(xi, density,'-b')
   hold on
   plot(xi,normpdf(xi,mean_va1_1(i),sd_va1_1(i)), '--k','LineWidth',1)
   if sd_va_tan(i)<1e-4
        xline(mean_va_tan(i),':r','LineWidth',1.2)
   else 
        plot(xi,normpdf(xi,mean_va_tan(i),sd_va_tan(i)),':r','LineWidth',1.2);
   end
   legend('MCMC','NG-HVI','DAVI')
   legend boxoff
   hold off
   xl(i,:) = xlim;
end
posterior_sigma2_a= figure('DefaultAxesFontSize',14,'Position', [10 10 500 400]);

[density, xi] = ksdensity(log(mcmc.sigma2_a((burn_mcmc+1):N)));
   n = length(xi);
   ind1_1 = [lim_low1_1(m+1):((lim_high1_1(m+1) - lim_low1_1(m+1))/(n-1)):lim_high1_1(m+1)];
   ind_tan = [lim_low_tan(m+2):((lim_high_tan(m+2) - lim_low_tan(m+2))/(n-1)):lim_high_tan(m+2)];

   plot(xi, density,'-b')
   hold on
   plot(ind1_1,normpdf(ind1_1,mean_va1_1(m+1),sd_va1_1(m+1)), '--k','LineWidth',1)
   plot(ind_tan,normpdf(ind_tan,mean_va_tan(m+2),sd_va_tan(m+2)), ':r','LineWidth',1.2)
   legend('MCMC','NG-HVI','DAVI')
   legend boxoff
hold off

posterior_sigma2_e= figure('DefaultAxesFontSize',14,'Position', [10 10 500 400]);

[density, xi] = ksdensity(log(mcmc.sigma2_e((burn_mcmc+1):N)));
   n = length(xi);
   ind1_1 = [lim_low1_1(m+2):((lim_high1_1(m+2) - lim_low1_1(m+2))/(n-1)):lim_high1_1(m+2)];
   ind_tan = [lim_low_tan(m+1):((lim_high_tan(m+1) - lim_low_tan(m+1))/(n-1)):lim_high_tan(m+1)];

   plot(xi, density,'-b')
   hold on
   plot(ind1_1,normpdf(ind1_1,mean_va1_1(m+2),sd_va1_1(m+2)), '--k','LineWidth',1)
   plot(ind_tan,normpdf(ind_tan,mean_va_tan(m+1),sd_va_tan(m+1)),':r','LineWidth',1.2)
   legend('MCMC','NG-HVI','DAVI')
   legend boxoff
hold off
end