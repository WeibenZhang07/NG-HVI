function [log_joint_density,log_likelihood,log_alpha_h,log_jacobian,log_weights_vec_prior, log_beta_prior,log_theta_e_prior,log_w] = f_log_joint_deepLMM(X_train,Y_train,weights,beta,theta_e,w,tAlpha_h,sigma2_weights_vec,sigma2_beta,a_e,b_e,S,v)
    [omega,W] = f_gen_omega(w);
    sigma2_e = exp(theta_e);
    weights_vec=[];

    for l = 1:length(weights)
        weights_vec = [weights_vec;weights{l}(:)];
    end

    H = length(X_train);
    log_likelihood = 0;
    log_alpha_h = 0;
    log_jacobian = 0;
    Z_out = cell(H,1);
    for h = 1:H
        x_h = X_train{h};
        Z_out{h} = features_output(x_h,weights);
        z_out_h = Z_out{h};
        y_h = Y_train{h};
        z_h = z_out_h;
        %alpha_h = Alpha_h(h);
        %alpha_h = Alpha_h{h};
        talpha_h = tAlpha_h{h};
        [alpha_h] = f_alpha_h(z_out_h,z_h,y_h,beta,talpha_h,sigma2_e,omega);
        n_h = length(y_h);
        r = length(alpha_h);
        log_likelihood = log_likelihood -n_h/2*log(2*pi) - n_h/2*log(sigma2_e) - 0.5/sigma2_e*(y_h-z_out_h*beta-z_h*alpha_h)'*(y_h-z_out_h*beta-z_h*alpha_h);
        log_alpha_h = log_alpha_h  -r/2*log(2*pi) + 1/2*sum(log(eig(omega))) - 0.5*alpha_h'*omega*alpha_h;
        log_jacobian = log_jacobian + f_log_jacobian(sigma2_e,omega,z_h);
    end
    d_beta = length(beta);
    log_beta_prior = -d_beta/2*log(2*pi) - d_beta/2*log(sigma2_beta) - 0.5/sigma2_beta*(beta'*beta);
    d_weights_vec = length(weights_vec);
    log_weights_vec_prior = -d_weights_vec/2*log(2*pi) - d_weights_vec/2*log(sigma2_weights_vec) - 0.5/sigma2_weights_vec*(weights_vec'*weights_vec);
    log_theta_e_prior = a_e*log(b_e)-log(gamma(a_e)) - (a_e)*theta_e - b_e*exp(-theta_e);
    %[W,~,~] = f_gen_W(omega);
    log_w = logwishpdf(omega,S,v)+ r*log(2) + sum(((r+2)*ones(r,1)-(1:r)').*log(diag(W)));
    
    log_joint_density = log_likelihood+log_alpha_h+ log_jacobian + log_weights_vec_prior + log_beta_prior+log_theta_e_prior+log_w;
    
    
end