function [g_joint_talpha_h] = f_g_joint_talpha_h(X,Y,beta,Alpha_h,sigma2_alpha,sigma2_e)

    
    H = length(X);
    g_joint_talpha_h = zeros(H,1);
    for h = 1:H
        y_h = Y{h};
        x_h = X{h};
        alpha_h = Alpha_h{h};
        [g_joint_alpha_h] = f_g_joint_alpha_h(x_h,y_h,beta,alpha_h,sigma2_alpha,sigma2_e);
        [g_alpha_h_talpha_h] = f_g_alpha_h_talpha_h(x_h,sigma2_alpha,sigma2_e);
        g_joint_talpha_h(h) = g_joint_alpha_h*g_alpha_h_talpha_h;
        
    end


end