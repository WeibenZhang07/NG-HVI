function weights = InitializeNN_hybrid(NN)

%  layers: vector of doubles, each number specifing the amount of
%  nodes in a layer of the network.
%
%  weights: cell array of weight matrices specifing the
%  translation from one layer of the network to the next.
  weights = cell(1, length(NN)-1);

  for i = 1:length(NN)-1
      % Using random weights from -b to b 
      b = sqrt(6)/(NN(i)+NN(i+1));
      
      weights{i} = rand(NN(i+1),NN(i)+1)*2*b - b;  % 1 bias in input layer
      weights{i}(:,1) = 0;
  end

end

