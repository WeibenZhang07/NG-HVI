function prod = inverse_fisher_times_grad(b,c,gradient_bar)
% compute the product inverse_fisher x grad

d = length(b);
grad1 = gradient_bar(1:d);
grad2 = gradient_bar(d+1:2*d);
grad3 = gradient_bar(2*d+1:end);

c2 = c.^2;
b2 = b.^2;

prod1 = (b'*grad1)*b+(grad1.*c2);

alpha = 1/(1+sum(b2./c2));
Cminus = diag(1./c2);
Cminus_b = b./c2;
Sigma_inv = Cminus-alpha*(Cminus_b*Cminus_b');

A11_inv = (1/(1-alpha))*((1-1/(sum(b2)+1-alpha))*(b*b')+diag(c2));

C = diag(c);
A12 = 2*(C*Sigma_inv*b*ones(1,d)).*Sigma_inv;
A21 = A12';
A22 = 2*C*(Sigma_inv.*Sigma_inv)*C;

D = A22-A21*A11_inv*A12;
prod2 = A11_inv*grad2+(A11_inv*A12)*(D\A21)*(A11_inv*grad2)-(A11_inv*A12)*(D\grad3);
prod3 = -(D\A21)*(A11_inv*grad2)+D\grad3;

prod = [prod1;prod2;prod3];
end

