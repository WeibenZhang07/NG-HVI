function [Y] = commutation(m,p)
% Y*vec(B) = vec(B')
I = reshape(1:m*p,m,p);
I = I';
I = I(:);
Y = speye(m*p);
Y = Y(I,:);


end