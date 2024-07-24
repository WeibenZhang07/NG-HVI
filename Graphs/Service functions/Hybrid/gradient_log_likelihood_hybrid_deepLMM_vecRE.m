function g_loglike = gradient_log_likelihood_hybrid_deepLMM_vecRE(X_train,Y_train,weights,beta,Alpha_i,TSigma)
% compute estimate of the gradient of the log-likelihood w.r.t beta

    n = length(Y_train);
   
    
    gradient = 0;
    for h = 1:n
        y_h = Y_train{h};
        x_h = X_train{h};
        alpha_i= Alpha_i{h};
        gradient_i = exp(-TSigma)*nn_backpropagation_hybrid_vec_RE(x_h,y_h,weights,beta,alpha_i);
        gradient = gradient+gradient_i;
    end
    g_loglike = gradient;

end



    
