function [K_r] = f_K_r(r)
% K_r*vec(A) = vec(A')
I = reshape(1:r*r,r,r);
I = I';
I = I(:);
K_r = speye(r*r);
K_r = K_r(I,:);


end