%% convert vech(A) to A
function [A] = invert_vech(vech_A)
    l = length(vech_A);
    m = (-1+sqrt(1+8*l))/2;
    
    A = zeros(m,m);
    for i = 1:m
        A(i:m,i) = vech_A(1:m-i+1);
        vech_A = vech_A(length(1:m-i+1)+1:end);
    end

end