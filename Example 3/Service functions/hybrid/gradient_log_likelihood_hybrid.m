function g_loglike = gradient_log_likelihood_hybrid(W_seq,beta,y,X,Y_star,Alpha_i)
% compute estimate of the gradient of the log-likelihood

    n = length(y);
   
    
    gradient = 0;
    for i = 1:n
        yi = y{i};
        Xi = X{i};
        y_star = Y_star{i};
        alpha_i= Alpha_i{i};
        gradient_i = nn_backpropagation_hybrid(Xi,y_star,W_seq,beta,alpha_i);
        gradient = gradient+gradient_i;
    end
    g_loglike = gradient;

end



    
