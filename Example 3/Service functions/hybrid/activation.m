function output = activation(z,text)

% Calculate activation output of hidden units
% z: pre-activation of current hidden unit -> can be a scalar or array
% vector (all units of a single hidden layer)
% text: specified activation function
% text = {Sigmoid, Tanh, ReLU, LeakyReLU, Maxout}


    switch text
        case 'Linear'
            output = z;
        case 'Sigmoid'
            output = 1.0 ./ (1.0 + exp(-z));
        case 'Tanh'
            output = tanh(z);
        case 'ReLU'
            output = max(0,z);            
    end
end
