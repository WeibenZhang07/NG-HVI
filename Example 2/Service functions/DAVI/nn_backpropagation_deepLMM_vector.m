function gradient = nn_backpropagation_deepLMM_vector(Xi,yi,W_seq,beta,alpha_i)
%Xi = X_train{1};yi=Y_train{1};W_seq = draws.w;beta= draws.beta; alpha_i = draws.alpha_i{1};
% compute the gradient in a L-layer neural net using backpropagation

n_train = size(Xi,1);
L = length(W_seq);
a_seq = cell(1,L);
Z_seq = cell(1,L);
aux = (beta+alpha_i);
% aux = beta;
% aux(1) = aux(1) + alpha_i;

a_seq{1} = W_seq{1}*Xi';
Z_seq{1} = [ones(1,n_train);activation(a_seq{1},'ReLU')];
for j=2:L
    a_seq{j} = W_seq{j}*Z_seq{j-1};
    Z_seq{j} = [ones(1,n_train);activation(a_seq{j},'ReLU')];
end

delta_seq = cell(1,L+1);
eta = aux'*Z_seq{L};

%pi = normpdf(ystar);
%{
pi = zeros(1,n_train);
for ii = 1:n_train
    if yi(ii)==0
    pi(1,ii)  = cnormpdf( ystar(ii), 0,1,-Inf,0 ) ;
    else
    pi(1,ii)  = cnormpdf( ystar(ii), 0,1,0,Inf ) ;    
    end
end
%}
    delta_seq{L+1} = (yi - eta')';
    %delta_seq{L+1} = (yi - pi')';
 
 
delta_seq{L} = (aux(2:end)*delta_seq{L+1}).*activation_derivative(a_seq{L},'ReLU'); % effect of each neuron z_L
for j=L-1:-1:1
    Wj_tilde = W_seq{j+1};
    Wj_tilde = Wj_tilde(:,2:end);
    delta_seq{j} = (activation_derivative(a_seq{j},'ReLU')).*(Wj_tilde'*delta_seq{j+1});% effect of each neuron z_{L-1}
end
gradient_W1 = delta_seq{1}*Xi;
gradient = gradient_W1(:);
for j = 2:L
    gradient_Wj = delta_seq{j}*(Z_seq{j-1})';
    gradient = [gradient;gradient_Wj(:)];
end
gradient = [gradient;Z_seq{L}*delta_seq{L+1}'];


end
