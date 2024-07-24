function [result] = hybrid_deepglmm_train(X_train,y_train,X_test,y_test,setting)

rng(1234);

SGA = setting.SGA;
N = setting.N;

%set up momentum and damping parameter
damping = setting.damping;
grad_weight = setting.grad_weight;

% Predictive MC
J = setting.J;

[ADA,draws,dim,lambda,M,ng_old,nn,p,Predictive,Predictive_item,priors,repeat,time,va] = Hybrid_DeepGLMM_initialize(X_train,N,1);
%start iteration
for i = 1:N
    tic
    
    draws.epsilon = normrnd(0,1,M,1);
    draws.z = normrnd(0,1,p,1);
    draws.theta = draws.mu + draws.B * draws.z + (draws.d.*draws.epsilon);
    draws.invbbd = woodbury(draws.B,draws.d);
    
    [draws.w,draws.w_all] = w_vec2mat(draws.theta,nn,dim);
    draws.beta = draws.theta(sum(dim.w_all)+1:sum(dim.w_all)+dim.beta);
    draws.TGamma = draws.theta(end - dim.Gamma+1:end);
    for l = 1:dim.sample
        draws.z_out{l} =  features_output(X_train{l},draws.w);
    end
 
   for j = 1:repeat
       [draws.ystar] = VB_ystar(draws.z_out,y_train,draws.beta,draws.alpha_i,dim);
       
       [draws.alpha_i] = VB_alpha_i(draws.z_out,draws.ystar,draws.beta, exp(draws.TGamma));
   end
  
   [gradients]= gradient_estimates(X_train,y_train,priors,dim,draws.B,draws.d,draws.z,draws.epsilon,draws.w,draws.beta,draws.ystar,draws.alpha_i,draws.TGamma);
    
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
    
    lambda.mu(:,i) = draws.mu;
    lambda.B(:,:,i) = draws.B;
    lambda.d(:,i) = draws.d;
    lambda.z(:,i) = draws.z;
    lambda.epsilon(:,i) = draws.epsilon;
      
    time(i) = toc;
   
    %updating weighting matrix
    [draws.w,draws.w_all] = w_vec2mat(draws.theta,nn,dim);
    draws.beta = draws.theta(sum(dim.w_all)+1:sum(dim.w_all)+dim.beta);
    draws.TGamma = draws.theta(end - dim.Gamma+1:end);
    
    va.w(:,i) = draws.w_all;
    va.beta(:,i) = draws.beta;
    va.TGamma(:,i) = draws.TGamma;
    
    
    [predictive] = predictive_metrics_par(X_train,y_train, X_test,y_test,nn,J,lambda,0,i);
    
    [Predictive] = record_loss(predictive,Predictive,J,i);
    
    if rem(i,100) == 0
        disp(['Iteration: ',num2str(i),'   -  Predictive Cross Entropy: ',num2str(predictive.pce2)]);
    end

end

result.lambda = lambda;
result.Predictive = Predictive;
result.Predictive_item = Predictive_item;
result.time = time;
result.va = va;

end