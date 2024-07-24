function[g_log_joint_density] = f_g_log_joint_density(X_train,Z_out,Y_train,Z,weights,beta,Alpha_h,omega,sigma2_e,sigma2_weights_vec,sigma2_beta,a_e,b_e,v,S)
% X= X_train;Y=Y_train;Z= Z_out;
    g_joint_weights_vec = f_g_joint_weights_vec(X_train,Z_out,Z,Y_train,weights,beta,Alpha_h,omega,sigma2_e,sigma2_weights_vec);
    g_joint_beta = f_g_joint_beta(Z_out,Z,Y_train,beta,Alpha_h,omega,sigma2_e,sigma2_beta);
    g_joint_theta_e = f_g_joint_theta_e(Z_out,Z,Y_train,beta,Alpha_h,omega,sigma2_e,a_e,b_e);
    g_joint_w = f_g_joint_w(Z_out,Z,Y_train,beta,Alpha_h,omega,sigma2_e,v,S);
    [g_joint_talpha_h] = f_g_joint_talpha_h(Z_out,Z,Y_train,beta,Alpha_h,omega,sigma2_e);
    g_log_joint_density = [g_joint_weights_vec;g_joint_beta;g_joint_theta_e;g_joint_w;vertcat(g_joint_talpha_h{:})];
  
end