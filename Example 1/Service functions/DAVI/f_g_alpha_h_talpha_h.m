function [g_alpha_h_talpha_h] = f_g_alpha_h_talpha_h(x_h,sigma2_alpha,sigma2_e)

    n_h = size(x_h,1);

    g_alpha_h_talpha_h = (1/sigma2_alpha + n_h/sigma2_e)^(-1/2);


end