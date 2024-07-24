function [g_alpha_h_theta_e] = f_g_alpha_h_theta_e(x_h, y_h, beta, talpha_h,sigma2_alpha,sigma2_e)

    n_h = size(x_h,1);

    g_alpha_h_theta_e = 0.5*n_h*(1/sigma2_alpha + n_h/sigma2_e)^(-3/2)/sigma2_e*talpha_h...
        - (sum(y_h - x_h*beta)/(sigma2_e/sigma2_alpha + n_h)^2)*sigma2_e/sigma2_alpha;


end