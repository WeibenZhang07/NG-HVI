function g_joint_beta = f_g_joint_beta(X,Z,Y,beta,Alpha_h,omega,sigma2_e,sigma2_beta)
    %X = X_train;Z = X_train;Y = Y_train;
    
    H = length(X);
    g_joint_beta = 0;
    for h = 1:H
        x_h = X{h};
        z_h = Z{h};        
        y_h = Y{h};
        alpha_h= Alpha_h{h};
        inv_sigma_h = omega + z_h'*z_h/sigma2_e;

        [g_joint_alpha_h] = f_g_joint_alpha_h(x_h,z_h,y_h,beta,alpha_h,omega,sigma2_e);
        g_u_h_beta = -(x_h'*z_h/inv_sigma_h)/sigma2_e;
        g_joint_beta = g_joint_beta + g_u_h_beta * g_joint_alpha_h + 1/sigma2_e*(x_h'*(y_h - x_h*beta- z_h*alpha_h));

    end
    g_joint_beta =g_joint_beta - 1/sigma2_beta*beta;
end