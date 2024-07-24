function g_joint_w = f_g_joint_w(X,Z,Y,beta,Alpha_h,omega,sigma2_e,v,S)
    
    
    H = length(X);
    g_joint_w = 0;
    [W,~,w] = f_gen_W(omega);
    DW = f_DW(W);
    r = length(Alpha_h{1});
    E_r = f_E_r(r);
    N_r = f_N_r(r);
    u = r.*ones(r,1) - reshape(1:r,r,1) +2;
    for h = 1:H
        x_h = X{h};
        z_h = Z{h};        
        y_h = Y{h};
        alpha_h= Alpha_h{h};
        inv_sigma_h = omega + z_h'*z_h/sigma2_e;
        sigma_h = inv(inv_sigma_h);
        L_h = chol(sigma_h)';
        mu_h = inv_sigma_h\z_h'*(y_h - x_h*beta)/sigma2_e;
        [g_joint_alpha_h] = f_g_joint_alpha_h(x_h,z_h,y_h,beta,alpha_h,omega,sigma2_e);
        g_u_h_w = -DW*E_r*(kron(W'/inv_sigma_h,mu_h) + kron(W'*mu_h,sigma_h));
        g_vec_sigma_h_w = -2*DW*E_r*(kron(W'/inv_sigma_h,sigma_h))*N_r;
        [talpha_h] = f_talpha_h(x_h,z_h,y_h,beta, alpha_h,sigma2_e,omega);
        tB_h = f_tB_h(L_h, g_joint_alpha_h,talpha_h);
        vec_lbl = f_vec_lbl(L_h,tB_h);
        vec_inv_sigma = f_vec_inv_sigma(sigma_h);
        g_joint_alpha_h_w = g_u_h_w * g_joint_alpha_h + 0.5*g_vec_sigma_h_w*vec_lbl;
        g_jacobian_w = 0.5*g_vec_sigma_h_w*vec_inv_sigma;
        g_joint_w = g_joint_w + g_joint_alpha_h_w + g_jacobian_w+ DW*vech(inv(W') - alpha_h*alpha_h'*W);

    end
    g_joint_w =g_joint_w + DW*vech((v - r - 1)*inv(W)' - S\W) + vech(diag(u));
end