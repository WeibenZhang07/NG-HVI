function [gen_omega,W] = f_gen_omega(w)

    W_star = invert_vech(w);
    W = W_star + diag(exp(diag(W_star)) - diag(W_star));
    gen_omega = W*W';

end