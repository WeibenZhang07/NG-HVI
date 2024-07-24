%% return the lower triangular of matrix as a vector
function [vech_A] = vech(A)
    [m,n] = size(A);
    if m ~= n
        error("Not square matrix");
    else
        ind = reshape(1:m*n,m,n);
        ind_vech = ind(tril(ind)>0);
        vech_A = A(:);
        vech_A = vech_A(ind_vech);
%         if find(~x) == 0
%             vech_A = A(tril(A)>0);
%         else
%             vech_A=[];
%             ind = 1;
%             for i = 1:n
%                 vech_A = [vech_A;A(ind:m,i)];
%                 ind = ind+1;
%             end
%         end

    end
end