function [result] = hybrid_lmm_train(X_train,Y_train,X_test,Y_test,p,setting)

rng(1234);

SGA = setting.SGA;
N = setting.N;
elbo = zeros(N,1);
elbo2 = zeros(N,1);

%set up momentum and learning rate parameter
damping = setting.damping;
grad_weight = setting.grad_weight;

[ADA,draws,dim,lambda,M,ng_old,~,~,priors,time,va] = Hybrid_LMM_initialize(X_train,N,p,1);

%start iteration
for i = 1:N
    tic
    
    draws.epsilon = normrnd(0,1,M,1);
    draws.z = normrnd(0,1,p,1);
    draws.theta = draws.mu + draws.B * draws.z + (draws.d.*draws.epsilon); 
    
    draws.beta = draws.theta(1:dim.beta);
    draws.TGamma = draws.theta(end - dim.Gamma - dim.Sigma+1:end - dim.Sigma);
    draws.TSigma = draws.theta(end - dim.Sigma+1:end);
 
    
    [draws.alpha_i] = VB_alpha_i(X_train,Y_train,draws.beta, exp(draws.TGamma),exp(draws.TSigma));
   
  
    [gradients,~]= gradient_estimates_LMM(X_train,Y_train,priors,draws.B,draws.d,draws.z,draws.epsilon,draws.beta,draws.alpha_i,draws.TGamma,draws.TSigma);


   if SGA == 1
        [draws.mu,draws.B,draws.d,ADA] = adadelta_norestrict(draws.mu,draws.B,draws.d,gradients,ADA);
   else
        [ng]= Hessian_ong_deepglmm(draws.B,draws.d,gradients,ng_old, grad_weight,damping);

        gradients_ng.l_mu = ng.bar1;
        gradients_ng.l_b = ng.bar2;
        gradients_ng.l_d = ng.bar3;
        [draws.mu,draws.B,draws.d,ADA] = adadelta_norestrict(draws.mu,draws.B,draws.d,gradients_ng,ADA);   

        ng_old.bar1 = ng.bar1;
        ng_old.bar2 = ng.bar2;
        ng_old.bar3 = ng.bar3;
        ng_old = ng;
   end
    
      
    time(i) = toc;
   
    %updating variational parameters

    lambda.mu(:,i) = draws.mu;
    lambda.B(:,:,i) = draws.B;
    lambda.d(:,i) = draws.d;
    lambda.z(:,i) = draws.z;
    lambda.epsilon(:,i) = draws.epsilon;
    %updating weighting matrix
    draws.beta = draws.theta(1:dim.beta);
    draws.TGamma = draws.theta(end - dim.Gamma - dim.Sigma+1:end - dim.Sigma);
    draws.TSigma = draws.theta(end - dim.Sigma+1:end);
    
    va.beta(:,i) = draws.beta;
    va.TGamma(:,i) = draws.TGamma;
    va.TSigma(:,i) = draws.TSigma;
    [elbo(i),elbo2(i)] = stochastic_elbo_LMM(X_train,Y_train,lambda,priors,1,0,i);

    if rem(i,100) == 0
        disp(['Iteration: ',num2str(i),'   -  ELBO: ',num2str(elbo(i))]);
    end
end

result.lambda = lambda;
result.time = time;
result.va = va;
result.elbo = elbo;
result.elbo2= elbo2;
[result.ELBO,result.ELBO2]  = stochastic_elbo_LMM(X_train,Y_train,lambda,priors,1000,1);
end