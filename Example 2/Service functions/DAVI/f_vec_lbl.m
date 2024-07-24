function vec_lbl = f_vec_lbl(L_h,tB_h)
    lbl = L_h'\tB_h/(L_h);
    vec_lbl = lbl(:);
end