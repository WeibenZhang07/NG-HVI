function    g_joint_weights_vec = f_g_joint_weights_vec(X,Z_out,Z,Y,weights,beta,Alpha_h,omega,sigma2_e,sigma2_weights_vec)
%X=X_train;Y=Y_train;
    H = length(X);
    r = size(Z_out{1},2);
    [K_r] = f_K_r(r);
    weights_vec=[];
    for l = 1:length(weights)
        weights_vec = [weights_vec;weights{l}(:)];
    end
    g_joint_weights_vec = 0;
    for h = 1:H
        x_h = X{h};
        z_h = Z{h};
        z_out_h = Z_out{h};
        y_h = Y{h};
        n_h = length(y_h);
        K_nr =  commutation(n_h,r);
        alpha_h= Alpha_h{h};
        inv_sigma_h = omega + z_h'*z_h/sigma2_e;
        sigma_h = inv(inv_sigma_h);
        L_h = chol(sigma_h)';
        mu_h = inv_sigma_h\z_h'*(y_h - z_out_h*beta)/sigma2_e;
        [g_joint_alpha_h] = f_g_joint_alpha_h(z_out_h,z_h,y_h,beta,alpha_h,omega,sigma2_e);
        g_Z_out_h_weights_vec = f_g_Z_out_h_weights_vec(x_h,weights);
        d_weights_vec = size(g_Z_out_h_weights_vec,2);

        g_vec_sigma_h_weights_vec = -(1/sigma2_e)*(kron(sigma_h,sigma_h)*(K_r + eye(r^2))*kron(eye(r),z_h'))*g_Z_out_h_weights_vec;
        g_u_h_weights_vec = 1/sigma2_e*(kron((y_h - z_out_h*beta)'*z_h,eye(r))*g_vec_sigma_h_weights_vec ...
            + kron((y_h - z_out_h*beta)',sigma_h)*K_nr*g_Z_out_h_weights_vec...
            - kron(beta',sigma_h*z_h')*g_Z_out_h_weights_vec);
        [talpha_h] = f_talpha_h(z_out_h,z_h,y_h,beta, alpha_h,sigma2_e,omega);
        tB_h = f_tB_h(L_h, g_joint_alpha_h,talpha_h);
        vec_lbl = f_vec_lbl(L_h,tB_h);
        vec_inv_sigma = f_vec_inv_sigma(sigma_h);
        g_weights_vec = (1/sigma2_e)*nn_backpropagation_deepLMM_vector(x_h,y_h,weights,beta,alpha_h);
        g_weights_vec = g_weights_vec(1:d_weights_vec);
        g_joint_alpha_h_weights_vec = g_u_h_weights_vec' * g_joint_alpha_h + 0.5*g_vec_sigma_h_weights_vec'*vec_lbl;
        g_jacobian_weights_vec = 0.5*g_vec_sigma_h_weights_vec'*vec_inv_sigma;
        g_joint_weights_vec = g_joint_weights_vec + g_joint_alpha_h_weights_vec + g_jacobian_weights_vec+ g_weights_vec;
    end
    g_joint_weights_vec =g_joint_weights_vec - 1/sigma2_weights_vec*weights_vec;
end

 