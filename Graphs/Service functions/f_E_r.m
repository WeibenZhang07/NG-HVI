function E_r = f_E_r(r)
%% elimination matrix E_r vec(A) = vech(A)
    A = reshape(1:r^2,r,r);
    vech_A = vech(A);
    L = length(vech_A);
    E_r = zeros(L,r^2);
    for i = 1:L
        E_r(i,vech_A(i)) = 1;
    end

end