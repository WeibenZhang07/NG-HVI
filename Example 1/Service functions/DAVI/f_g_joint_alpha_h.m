function [g_joint_alpha_h] = f_g_joint_alpha_h(x_h,y_h,beta,alpha_h,sigma2_alpha,sigma2_e)

    n_h = length(y_h);
    g_joint_alpha_h = 1/sigma2_e*ones(n_h,1)'*(y_h - x_h*beta - ones(n_h,1)*alpha_h) - 1/sigma2_alpha*alpha_h;


end