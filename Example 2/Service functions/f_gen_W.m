function [W,W_star,w] = f_gen_W(omega)

    W = chol(omega)';
    W_star = W + diag(log(diag(W)) - diag(W));
    w = vech(W_star);


end