function g_loglike = gradient_log_likelihood_hybrid_LMM(beta,Y_train,X_train,Alpha_i,TSigma)
% compute estimate of the gradient of the log-likelihood

    n = length(Y_train);
   
    
    gradient = 0;
    for i = 1:n
        yi = Y_train{i};
        Xi = X_train{i};
        alpha_i= Alpha_i{i};
        n_h = length(yi);
        gradient_i = exp(-TSigma)*Xi'*(yi-Xi*beta-ones(n_h,1)*alpha_i);
        gradient = gradient+gradient_i;
    end
    g_loglike = gradient;

end



    
