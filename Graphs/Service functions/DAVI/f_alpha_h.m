function [alpha_h] = f_alpha_h(x_h,z_h,y_h,beta, talpha_h,sigma2_e,omega)
    inv_Sigma_h = omega + z_h'*z_h/sigma2_e;
    L_h = chol(inv(inv_Sigma_h))';
    %alpha_h = L_h*(talpha_h) + 1/sigma2_e*(inv_Sigma_h)\z_h'*(y_h - x_h*beta);
    alpha_h = L_h*(talpha_h) + (inv_Sigma_h)\z_h'*(y_h - x_h*beta)/sigma2_e;
end