function [weights,weights_vec] = w_vec2mat(theta,NN,dim_weights_vec)
    layer = length(NN);
    weights_vec = theta(1:dim_weights_vec);
    dim_weights = zeros(layer-1,1);

    for i = 1:(length(NN)-1)
    dim_weights(i) = NN(i+1)*(NN(i)+1);
    end

    weights = cell(layer-1,1);
    for l = 1 : (layer-1)
        if l ==1
            weights{l} = theta(1:dim_weights(l));
        else
            start = sum(dim_weights(1:l-1));
            weights{l} = theta((start +1):(start+dim_weights(l)));
        end
        weights{l} = reshape(weights{l},NN(l+1),NN(l)+1);
    end

end