function [gradients]= gradient_estimates(X,y,priors,dim,B,d,z,epsilon,W_seq,beta,Y_star,Alpha_i,TGamma)

    g_loglike = gradient_log_likelihood_hybrid(W_seq,beta,y,X,Y_star,Alpha_i);
    g_prior_beta = -1/priors.beta_sig.*beta;
    vec_w=[];
    for i = 1:length(W_seq)
        vec_w = [vec_w; W_seq{i}(:)];
    end
    g_prior_w = -1/priors.w_sig.*vec_w;
    
    
    alpha_i = horzcat(Alpha_i{:})';
    n_sample = length(y);
    g_TGamma = (priors.Gamma_a) .*ones(dim.Gamma,1) - priors.Gamma_b.*exp(TGamma) -0.5.* n_sample.*ones(dim.Gamma,1) + 0.5.*diag(alpha_i'*alpha_i).*exp(-TGamma);

    g_h = [g_loglike + [g_prior_w;g_prior_beta]; g_TGamma];

    
    invDB = 1./d.*B;
    invD2dep = (1./d.^2).*(d.*epsilon);
    A = eye(size(B,2)) + invDB'*invDB;
    dqdmu = (1./d).*invDB*z + invD2dep - ((1./d).*invDB/A)*((1./d).*invDB)'*B*z - ((1./d).*invDB/A)*((1./d).*invDB)'*(d.*epsilon);
    gradients.l_mu = g_h +dqdmu;
    
    gradients.l_b = (g_h * z')+ dqdmu*z';
    
    gradients.l_d = g_h .* epsilon+ dqdmu.*epsilon;

end