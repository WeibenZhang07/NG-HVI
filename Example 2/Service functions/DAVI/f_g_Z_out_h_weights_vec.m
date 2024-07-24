function g_Z_out_h_weights_vec = f_g_Z_out_h_weights_vec(x_h,weights)
    n_train = size(x_h,1);
    r = size(weights{end},1)+1;
    L = length(weights);
    d_weights_vec = 0;
    for l = 1:L
        [d1,d2] = size(weights{l,1});
        d_weights_vec = d_weights_vec+ d1*d2;
    end
    gradient_h = zeros(n_train*r,d_weights_vec);
    ind = [1:n_train:n_train*r];
    for n = 1:n_train
        gradient_n = nn_backpropagation_z_out_weights(x_h(n,:),weights,d_weights_vec);
        gradient_h(ind,:) = gradient_n';
        ind = ind+1;
    end

g_Z_out_h_weights_vec= gradient_h;


end