function [y_star] = VB_ystar(X,y,beta,mu_alpha_i,dim)
%Gibbs sampler for ystar

    y_star=cell(1,dim.sample);
    parfor ii = 1:dim.sample
        eta = X{ii}*(beta+mu_alpha_i{ii});
        for i = 1:dim.T
            if y{ii}(i) ==0
               y_star{ii}(i) = cnormrnd(eta(i),1,-Inf,0);
            else
                y_star{ii}(i) = cnormrnd(eta(i),1,0,+Inf);
            end
        end
        y_star{ii} = reshape(y_star{ii},[],1);
    end


end