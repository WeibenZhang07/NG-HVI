function [gradients,g_h]= gradient_estimates_deepLMM_vecRE(X_train,Z_out,Z_train,Y_train,priors,B,d,z,epsilon,weights,beta,Alpha_i,w,TSigma)
%X=X_train;y = Y_train;B=draws.B;d = draws.d;z = draws.z;epsilon= draws.epsilon;invbbd= draws.invbbd;beta = draws.beta;Alpha_i= draws.alpha_i;w= draws.w;TSigma =draws.TSigma;
   
    H = length(X_train);
    %% w.r.t beta, weights_vec
    g_loglike = gradient_log_likelihood_hybrid_deepLMM_vecRE(X_train,Y_train,weights,beta,Alpha_i,TSigma);
    g_prior_beta = -1/priors.beta_sig.*beta;
    weights_vec = [];
    for l = 1:length(weights)
        weights_vec=[weights_vec; reshape(weights{l},[],1)];
    end
    g_prior_weights_vec =  -1/priors.weights_sig.*weights_vec;
    %% w.r.t TSigma
    g_TSigma = 0;
    for h = 1:H
        z_out_h = Z_out{h};
        z_h = Z_train{h};
        y_h = Y_train{h};
        alpha_h= Alpha_i{h};
        n_h = length(Y_train{h});
        g_TSigma = g_TSigma -0.5*n_h + 0.5*exp(-TSigma)*(y_h - z_out_h*beta - z_h * alpha_h)'*(y_h - z_out_h*beta - z_h * alpha_h);
    end
    g_TSigma = g_TSigma - priors.Sigma_a + priors.Sigma_b/exp(TSigma);
    %% w.r.t w
    
    g_joint_w = 0;
    [~,W] = f_gen_omega(w);
    DW = f_DW(W);
    r = length(Alpha_i{1});
    u = r.*ones(r,1) - reshape(1:r,r,1) +2;
    for h = 1:H
        alpha_h= Alpha_i{h};
        g_joint_w = g_joint_w + DW*vech(inv(W') - alpha_h*alpha_h'*W);
    end
    g_joint_w =g_joint_w + DW*vech((priors.v - r - 1)*inv(W)' - priors.S\W) + vech(diag(u));
    
    g_h = [g_loglike + [g_prior_weights_vec;g_prior_beta];g_TSigma;g_joint_w];


invDB = 1./d.*B;
invD2dep = (1./d.^2).*(d.*epsilon);
A = eye(size(B,2)) + invDB'*invDB;
dqdmu = (1./d).*invDB*z + invD2dep - ((1./d).*invDB/A)*((1./d).*invDB)'*B*z - ((1./d).*invDB/A)*((1./d).*invDB)'*(d.*epsilon);

gradients.l_mu = g_h +dqdmu;

gradients.l_b = (g_h * z')+ dqdmu*z';

gradients.l_d = g_h .* epsilon+ dqdmu.*epsilon;

end