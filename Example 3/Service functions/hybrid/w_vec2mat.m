function [w,w_all] = w_vec2mat(theta,nn,dim)
    layer = length(nn);
    w_all = theta(1:sum(dim.w_all));
    
    w = cell(layer,1);
    for l = 1 : layer
        if l == 1
           w{l} = w_all(1:dim.w_all(l));
           w{l} = reshape(w{l},nn(l),dim.input+1);
        else
           w{l} = w_all(dim.w_all(l-1)+1:(dim.w_all(l-1)+dim.w_all(l)));
           w{l} = reshape(w{l},nn(l),nn(l-1)+1);
        end
    end

end