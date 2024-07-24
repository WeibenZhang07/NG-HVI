function [gradients,g_h]= gradient_estimates_LMM(X_train,Y_train,priors,B,d,z,epsilon,beta,Alpha_i,TGamma,TSigma)
    g_loglike = gradient_log_likelihood_hybrid_LMM(beta,Y_train,X_train,Alpha_i,TSigma);
    g_prior_beta = -1/priors.beta_sig.*beta;
    
    alpha_i = horzcat(Alpha_i{:})';
    H = zeros(length(Y_train),1);
    for i = 1:length(Y_train)
        H(i) = length(Y_train{i});
    end
    g_TGamma = -0.5*length(H) + 0.5/exp(TGamma)*(alpha_i'*alpha_i)-priors.Gamma_a+priors.Gamma_b/exp(TGamma);%-1
    g_TSigma = 0;
    for h = 1:length(H)
        for t = 1:length(Y_train{h})
            g_TSigma = g_TSigma  -0.5+0.5/exp(TSigma)*(Y_train{h}(t) - X_train{h}(t,:)*beta - Alpha_i{h})^2;
            
        end
        
    end
    g_TSigma = g_TSigma - priors.Sigma_a + priors.Sigma_b/exp(TSigma); %- 1 
    
    g_h = [g_loglike + g_prior_beta; g_TGamma;g_TSigma];

invDB = 1./d.*B;
invD2dep = (1./d.^2).*(d.*epsilon);
A = eye(size(B,2)) + invDB'*invDB;
dqdmu = (1./d).*invDB*z + invD2dep - ((1./d).*invDB/A)*((1./d).*invDB)'*B*z - ((1./d).*invDB/A)*((1./d).*invDB)'*(d.*epsilon);
gradients.l_mu = g_h +dqdmu;

gradients.l_b = (g_h * z')+ dqdmu*z';

gradients.l_d = g_h .* epsilon+ dqdmu.*epsilon;

end