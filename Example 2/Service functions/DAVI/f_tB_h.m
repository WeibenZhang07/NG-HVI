function tB_h = f_tB_h(L_h, g_joint_alpha_h,talpha_h)
    B_h = L_h'*g_joint_alpha_h*talpha_h';
    tB_h = tril(B_h)+ tril(B_h)'-diag(diag(B_h));

end