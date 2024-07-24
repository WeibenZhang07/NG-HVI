function [g_joint_talpha_h] = f_g_joint_talpha_h(X,Z,Y,beta,Alpha_h,omega,sigma2_e)
    H = length(X);
    g_joint_talpha_h = cell(H,1);
    for h = 1:H
        x_h = X{h};
        z_h = Z{h};        
        y_h = Y{h};
        inv_sigma_h = omega + z_h'*z_h/sigma2_e;
        sigma_h = inv(inv_sigma_h);
        L_h = chol(sigma_h)';
        alpha_h= Alpha_h{h};        
        [g_joint_alpha_h] = f_g_joint_alpha_h(x_h,z_h,y_h,beta,alpha_h,omega,sigma2_e);
        g_joint_talpha_h{h} = L_h'*g_joint_alpha_h;
    end
end