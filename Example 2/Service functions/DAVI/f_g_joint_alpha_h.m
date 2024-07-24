function [g_joint_alpha_h] = f_g_joint_alpha_h(x_h,z_h,y_h,beta,alpha_h,omega,sigma2_e)

    g_joint_alpha_h = 1/sigma2_e*(z_h'*(y_h - x_h*beta - z_h*alpha_h)) - omega*alpha_h;


end