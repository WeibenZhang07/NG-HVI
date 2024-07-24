function [g_joint_theta_e] = f_g_joint_theta_e(X,Y,beta,Alpha_h,sigma2_alpha,sigma2_e,a_e,b_e)

    g_joint_theta_e = 0;
    H = length(X);
    for h = 1:H
        y_h = Y{h};
        x_h = X{h};
        alpha_h = Alpha_h{h};
        n_h = length(y_h);
        [talpha_h] = f_talpha_h(x_h,y_h,beta, alpha_h,sigma2_e,sigma2_alpha);
        [g_joint_alpha_h] = f_g_joint_alpha_h(x_h,y_h,beta,alpha_h,sigma2_alpha,sigma2_e);
        [g_alpha_h_theta_e] = f_g_alpha_h_theta_e(x_h, y_h, beta, talpha_h,sigma2_alpha,sigma2_e);
        g_joint_theta_e = g_joint_theta_e -0.5*n_h + 0.5/sigma2_e*(y_h - x_h*beta - ones(n_h,1)*alpha_h)'*(y_h - x_h*beta - ones(n_h,1)*alpha_h)...
            + 0.5*n_h/((1/sigma2_alpha + n_h/sigma2_e)*sigma2_e)...
            + g_joint_alpha_h*g_alpha_h_theta_e;
    end
    g_joint_theta_e = g_joint_theta_e - a_e + b_e/sigma2_e;


end