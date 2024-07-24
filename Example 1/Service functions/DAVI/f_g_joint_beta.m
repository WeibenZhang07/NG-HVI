function [g_joint_beta] = f_g_joint_beta(X_train,Y_train,beta,Alpha_h,sigma2_alpha,sigma2_e,sigma2_beta)

    g_joint_beta = 0;
    H = length(X_train);
    for h = 1:H
        y_h = Y_train{h};
        x_h = X_train{h};
        alpha_h = Alpha_h{h};
        n_h = length(y_h);
        [g_joint_alpha_h] = f_g_joint_alpha_h(x_h,y_h,beta,alpha_h,sigma2_alpha,sigma2_e);
        [g_alpha_h_beta] = f_g_alpha_h_beta(x_h, sigma2_alpha,sigma2_e);
        g_joint_beta = g_joint_beta + (1/sigma2_e)*x_h'*(y_h - x_h*beta - ones(n_h,1)*alpha_h) + g_joint_alpha_h*g_alpha_h_beta;
    end
    g_joint_beta = g_joint_beta - beta/sigma2_beta;


end