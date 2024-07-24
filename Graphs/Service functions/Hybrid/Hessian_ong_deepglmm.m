function [ng]= Hessian_ong_deepglmm(B,d,gradients,ng_old,grad_weight,damping)
%B = draws.B;d =  draws.d;
% Ong et al CSDA2018

[M,p] = size(B);

gradient1 = gradients.l_mu;

C = (eye(p) +B'*(1./d.^2.*B));
C_tilde = chol(inv(C))';
E = B/C;
E_tilde = damping*sum(E.*B,2)./(d.^4);
G = (1./d.^2).*B*C_tilde;
H = E_tilde - (1+damping).*(1./d.^2);
ng_lambda1 = -(1./H).*gradient1 + (1./H).*(G/(eye(p)+G'*diag(1./H)*G)*G')*((1./H).*gradient1);

invbbd = woodbury(B,d);
BtiD2B = (B'*(B./d.^2));

G22 = 2*kron(BtiD2B - BtiD2B/C*BtiD2B,invbbd);
G33 = 2*(d.*invbbd).*(invbbd.*d);

BtiSigD = (B./d)'-BtiD2B/C*(B./d)';
G23 = zeros(M*p,M);
for i = 1:M
G23(:,i) = 2*kron(BtiSigD(:,i),invbbd(:,i));
end

%{%
F22 = [G22 G23;
      G23' G33];
%{%
if p > 1
    b0 = reshape(1:M*p,M,p);
    b0 = b0(~tril(ones(M,p)));
    F22(b0,:) = [];
    F22(:,b0) = [];
    gradients.l_b(b0) = [];
end
gradient2 = [gradients.l_b(:);gradients.l_d];


%{%
F2 = F22 + damping*diag(diag(F22));
%}


[ng_lambda2,~] = pcg(F2,gradient2);

ng_lambda = [ng_lambda1;ng_lambda2];

ng.prod1 = ng_lambda(1:M);
ng.prod2 = ng_lambda((M+1):(end - M));
%{%
prod2_matrix = zeros(M,p);
c = 0;
for i =1:p
    
    prod2_matrix((c+1):(M),i) = ng.prod2(1:(M-c));
    ng.prod2(1:(M-c))=[];
    c = c+1;
end
ng.prod2 = prod2_matrix(:);
%}
ng.prod3 = ng_lambda((end - M+1):end);
%}


 %{%
        norm_ng = norm(ng_lambda,'fro');
        shreshold = 50;
        if  norm_ng> shreshold
            ng.prod1 = shreshold/norm_ng*ng.prod1;
            ng.prod2 = shreshold/norm_ng*ng.prod2;
            ng.prod3 = shreshold/norm_ng*ng.prod3;
        end
 %}
 
%{%
ng.bar1 = grad_weight*ng_old.bar1 + (1 - grad_weight)*ng.prod1;      
ng.bar2 = grad_weight*ng_old.bar2 + (1 - grad_weight)*ng.prod2;      
ng.bar3 = grad_weight*ng_old.bar3 + (1 - grad_weight)*ng.prod3; 
%}


end