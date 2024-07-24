function [g_alpha_h_beta] = f_g_alpha_h_beta(x_h, sigma2_alpha,sigma2_e)

    n_h = size(x_h,1);
    g_alpha_h_beta = -sum(x_h,1)'/(sigma2_e/sigma2_alpha + n_h);


end