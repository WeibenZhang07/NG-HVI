function DW = f_DW(W)
    %%D^W is a diagonal matrix of order r(r+1)/2, where the diagonal is
    %%given by vech(J^W), and J^W is a r by r matrix with J^W_ii = W_ii and
    %%J^W_ij = 1 if i ~= j.
    r = size(W,2);
    JW = ones(r);
    JW = JW + diag(diag(W) - diag(JW));
    DW = diag(vech(JW));





end