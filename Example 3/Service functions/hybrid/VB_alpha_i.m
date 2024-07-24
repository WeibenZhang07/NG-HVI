function [mu_alpha_i] = VB_alpha_i(z_out,y_star,beta, Gammaj)
    % Sampler for alpha_i
    m = length(beta);
    N = length(z_out);
    mu_alpha_i = cell(1,N);
    for i = 1:N
        
        post_var_inv = diag(1./Gammaj) + z_out{i}'*z_out{i};

        post_var = (post_var_inv)\eye(m);
        
        post_var = (post_var + post_var')/2; % avoid non-symmetric issue caused by numerical error
        
        post_mean = post_var*z_out{i}'*(y_star{i} - z_out{i}*beta);

        mu_alpha_i{i} = reshape(mvnrnd(post_mean,post_var),[],1);
    end
    
end
