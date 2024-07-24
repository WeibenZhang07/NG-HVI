function [alpha_h] = f_alpha_h(x_h,y_h,beta, talpha_h,sigma2_e,sigma2_alpha)
    n_h = length(y_h);
    alpha_h = 1/sqrt(1/sigma2_alpha + n_h/sigma2_e)*(talpha_h) + sum(y_h - x_h*beta)/(sigma2_e/sigma2_alpha + n_h);

end