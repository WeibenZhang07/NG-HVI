function [talpha_h] = f_talpha_h(x_h,y_h,beta, alpha_h,sigma2_e,sigma2_alpha)
    n_h = length(y_h);
    talpha_h = sqrt(1/sigma2_alpha + n_h/sigma2_e)*(alpha_h - sum(y_h - x_h*beta)/(sigma2_e/sigma2_alpha + n_h));

end