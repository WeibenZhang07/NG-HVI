function [mu,B,d] = NGA_full_fisher(ng,mu,B,d,lrate,tau,iter)

if iter > tau
   stepsize = lrate*tau/iter;
else
   stepsize = lrate;
end

mu = mu + stepsize*ng.bar1;

B_tilde = B;
[M,p] = size (B);


step_B_matrix = zeros(M,p);
c = 0;
for i =1:p
    
    step_B_matrix((c+1):(M),i) = ng.bar2(1:(M-c));
    ng.bar2(1:(M-c))=[];
    c = c+1;
end

B_tilde = B_tilde + stepsize*(step_B_matrix);
B_tilde(~tril(ones(size(B)))) = 0;
B = B_tilde;


 d_tilde = (d);
d_tilde = d_tilde + stepsize*ng.bar3;
d = (d_tilde);

end




