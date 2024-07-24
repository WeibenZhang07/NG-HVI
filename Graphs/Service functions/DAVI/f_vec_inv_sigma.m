function vec_inv_sigma = f_vec_inv_sigma(sigma_h)
    inv_sigma = inv(sigma_h);
    vec_inv_sigma = inv_sigma(:);
end