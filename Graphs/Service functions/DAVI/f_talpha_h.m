function [talpha_h] = f_talpha_h(x_h,z_h,y_h,beta, alpha_h,sigma2_e,omega)

    inv_Sigma_h = omega + z_h'*z_h/sigma2_e;
    L_h = chol(inv(inv_Sigma_h))';
    talpha_h = L_h\(alpha_h - 1/sigma2_e*((inv_Sigma_h)\z_h'*(y_h - x_h*beta)));

end