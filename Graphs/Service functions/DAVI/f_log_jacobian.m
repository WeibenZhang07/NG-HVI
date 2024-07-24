function [log_jacobian] = f_log_jacobian(sigma2_e,omega,z_h)

    sigma_h = inv(omega+z_h'*z_h/sigma2_e);
    L_h = chol(sigma_h)';
    log_jacobian = log(det(L_h));




end