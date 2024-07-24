function [bbd] = woodbury(B,d)
% Using Woodbury formula
% d is a vector representing diagnal element of a diagonal matrix
d = reshape(d,[],1);
p = size(B,2);

invd2 = diag(1./d.^2);
invd2b = invd2*B;

bbd = invd2 - invd2b/(eye(p)+ B'*invd2b)*(invd2b)';
end