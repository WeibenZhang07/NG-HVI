function [g_joint_theta_alpha] = f_g_joint_theta_alpha(X,Y,beta,Alpha_h,sigma2_alpha,sigma2_e,a_alpha,b_alpha)

    g_joint_theta_alpha = 0;
    H = length(X);
    for h = 1:H
        y_h = Y{h};
        x_h = X{h};
        alpha_h = Alpha_h{h};
        n_h = length(y_h);
        [talpha_h] = f_talpha_h(x_h,y_h,beta, alpha_h,sigma2_e,sigma2_alpha);
        [g_joint_alpha_h] = f_g_joint_alpha_h(x_h,y_h,beta,alpha_h,sigma2_alpha,sigma2_e);
        [g_alpha_h_theta_alpha] = f_g_alpha_h_theta_alpha(x_h, y_h, beta, talpha_h,sigma2_alpha,sigma2_e);
        g_joint_theta_alpha = g_joint_theta_alpha -0.5 + 0.5*alpha_h^2/sigma2_alpha + 0.5/((1/sigma2_alpha + n_h/sigma2_e)*sigma2_alpha)...
            + g_joint_alpha_h*g_alpha_h_theta_alpha;
    end
    g_joint_theta_alpha = g_joint_theta_alpha - a_alpha + b_alpha/sigma2_alpha;


end