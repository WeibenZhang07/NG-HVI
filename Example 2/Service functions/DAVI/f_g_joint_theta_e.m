function g_joint_theta_e = f_g_joint_theta_e(X,Z,Y,beta,Alpha_h,omega,sigma2_e,a_e,b_e)
    
    
    H = length(X);
    g_joint_theta_e = 0;
    theta_e = log(sigma2_e);
    for h = 1:H
        x_h = X{h};
        z_h = Z{h};        
        y_h = Y{h};
        n_h = length(y_h);
        alpha_h= Alpha_h{h};
        inv_sigma_h = omega + z_h'*z_h/sigma2_e;
        sigma_h = (inv(inv_sigma_h)+inv(inv_sigma_h)')/2;
        L_h = chol(sigma_h)';
        %mu_h = inv_sigma_h\z_h'*(y_h - x_h*beta)/sigma2_e;
        [g_joint_alpha_h] = f_g_joint_alpha_h(x_h,z_h,y_h,beta,alpha_h,omega,sigma2_e);
        zz = z_h'*z_h;
        g_u_h_theta_e = (1/sigma2_e)^2*sigma_h*zz*sigma_h*z_h'*(y_h - x_h*beta)- sigma_h*z_h'*(y_h - x_h*beta)/sigma2_e;
        g_vec_sigma_h_theta_e = (kron(sigma_h,sigma_h)*zz(:)/sigma2_e);
        [talpha_h] = f_talpha_h(x_h,z_h,y_h,beta, alpha_h,sigma2_e,omega);
        tB_h = f_tB_h(L_h, g_joint_alpha_h,talpha_h);
        vec_lbl = f_vec_lbl(L_h,tB_h);
        vec_inv_sigma = f_vec_inv_sigma(sigma_h);
        g_joint_alpha_h_theta_e = g_u_h_theta_e' * g_joint_alpha_h + 0.5*g_vec_sigma_h_theta_e'*vec_lbl;
        g_jacobian_theta_e = 0.5*g_vec_sigma_h_theta_e'*vec_inv_sigma;
        g_joint_theta_e = g_joint_theta_e + g_joint_alpha_h_theta_e +  g_jacobian_theta_e...
            -0.5*n_h+(0.5/sigma2_e)*(y_h - x_h*beta - z_h* alpha_h)'*(y_h - x_h*beta - z_h* alpha_h);

    end
    g_joint_theta_e =g_joint_theta_e - (a_e ) + b_e * exp(-theta_e);
end