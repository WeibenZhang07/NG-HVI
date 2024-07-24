function gradient_i = nn_backpropagation_z_out_weights(Xi,W_seq,d_weights_vec)
%Xi = x_h;yi=y_h;W_seq = weights;beta= beta; alpha_i = alpha_h;
% compute the gradient in a L-layer neural net using backpropagation

[n_train,~] = size(Xi);
L = length(W_seq);

a_seq = cell(1,L);
Z_seq = cell(1,L);

a_seq{1} = W_seq{1}*Xi';
Z_seq{1} = [ones(1,n_train);activation(a_seq{1},'ReLU')];
for j=2:L
    a_seq{j} = W_seq{j}*Z_seq{j-1};
    Z_seq{j} = [ones(1,n_train);activation(a_seq{j},'ReLU')];
end
R = length(Z_seq{L});
delta_seq = cell(1,L);
gradient_i = zeros(d_weights_vec,1);
for r = 1:(R-1)
    %delta_seq{L} = (aux(2:end)*delta_seq{L+1}).*activation_derivative(a_seq{L},'ReLU'); % effect of each neuron z_L
    ind = zeros(R-1,1);
    ind(r) = 1;
    delta_seq{L} =  ind.* activation_derivative(a_seq{L},'ReLU'); % effect of each neuron z_L
    for j=L-1:-1:1
        Wj_tilde = W_seq{j+1};
        Wj_tilde = Wj_tilde(:,2:end);
        delta_seq{j} = (activation_derivative(a_seq{j},'ReLU')).*(Wj_tilde'*delta_seq{j+1});% effect of each neuron z_{L-1}
    end
    gradient_W1 = delta_seq{1}*Xi;
    gradient_r = gradient_W1(:);
    for j = 2:L
        gradient_Wj = delta_seq{j}*(Z_seq{j-1})';
        gradient_r = [gradient_r;gradient_Wj(:)];
    end
    gradient_i = [gradient_i gradient_r];
end
end