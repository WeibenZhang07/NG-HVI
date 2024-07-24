function [log_jacobian] = f_log_jacobian(sigma2_e,sigma2_alpha,n_h)

    log_jacobian = -0.5*log(1/sigma2_alpha + n_h / sigma2_e);




end