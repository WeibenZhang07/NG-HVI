function[g_log_joint_density] = f_g_log_joint_density(X_train,Y_train,beta,Alpha_h,sigma2_alpha,sigma2_e,sigma2_beta,a_e,b_e,a_alpha,b_alpha)

    [g_joint_beta] = f_g_joint_beta(X_train,Y_train,beta,Alpha_h,sigma2_alpha,sigma2_e,sigma2_beta);
    [g_joint_theta_e] = f_g_joint_theta_e(X_train,Y_train,beta,Alpha_h,sigma2_alpha,sigma2_e,a_e,b_e);
    [g_joint_theta_alpha] = f_g_joint_theta_alpha(X_train,Y_train,beta,Alpha_h,sigma2_alpha,sigma2_e,a_alpha,b_alpha);
    [g_joint_talpha_h] = f_g_joint_talpha_h(X_train,Y_train,beta,Alpha_h,sigma2_alpha,sigma2_e);
    g_log_joint_density = [g_joint_beta;g_joint_theta_e;g_joint_theta_alpha;g_joint_talpha_h];
  
end