function [result] = hybrid_Deeplmm_vecRE_train(X_train,Y_train,X_test,Y_test,nn,p,setting)

rng(1234);

SGA = setting.SGA;
N = setting.N;

%set up momentum and learning rate parameter
damping = setting.damping;
grad_weight = setting.grad_weight;


[ADA,draws,dim,lambda,M,ng_old,Predictive,~,priors,time,va] = Hybrid_DeepLMM_vecRE_initialize(X_train,X_train,N,nn,p,1);
NN = [dim.input nn];
%start iteration
for i = 1:N
    tic
    
    draws.epsilon = normrnd(0,1,M,1);
    draws.z = normrnd(0,1,p,1);
    draws.theta = draws.mu + draws.B * draws.z + (draws.d.*draws.epsilon);
    draws.invbbd = woodbury(draws.B,draws.d);
    
    [draws.weights,draws.weights_vec] = w_vec2mat(draws.theta,NN,dim.weights_vec);
    draws.beta = draws.theta(dim.weights_vec+1:dim.weights_vec+dim.beta);
    draws.TSigma = draws.theta(dim.weights_vec+dim.beta+1);
    draws.w = draws.theta(dim.weights_vec+dim.beta+2:end);
    [draws.omega,draws.W] = f_gen_omega(draws.w);
    
    for l = 1:dim.sample
        draws.Z_out{l} =  features_output(X_train{l},draws.weights);
    end
 
    
    [draws.alpha_i] = VB_vec_alpha_i(draws.Z_out,draws.Z_out,Y_train,draws.beta,draws.omega,exp(draws.TSigma));
   
  
    [gradients,~]= gradient_estimates_deepLMM_vecRE(X_train,draws.Z_out,draws.Z_out,Y_train,priors,draws.B,draws.d,draws.z,draws.epsilon,draws.weights,draws.beta,draws.alpha_i,draws.w,draws.TSigma);


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
    [draws.weights,draws.weights_vec] = w_vec2mat(draws.theta,NN,dim.weights_vec);
    draws.beta = draws.theta(dim.weights_vec+1:dim.weights_vec+dim.beta);
    draws.TSigma = draws.theta(dim.weights_vec+dim.beta+1);
    draws.w = draws.theta(dim.weights_vec+dim.beta+2:end);
    [draws.omega,draws.W] = f_gen_omega(draws.w);

    va.weights_vec(:,i) = draws.weights_vec;
    va.beta(:,i) = draws.beta;
    va.TSigma(:,i) = draws.TSigma;
    va.w(:,i) = draws.w;
    [elbo,log_score,RMSE,R2,R2_m,R2_c] = predictive_deepLMM_vecRE(X_train,Y_train,X_train,Y_train,nn,lambda,priors,1,0,i);
    Predictive{1,1}(i) = elbo;
    Predictive{1,2}(i) = log_score;
    Predictive{1,3}(i) = RMSE;
    Predictive{1,4}(i) = R2;
    Predictive{1,5}(i) = R2_m;
    Predictive{1,6}(i) = R2_c;
    if rem(i,100) == 0
        disp(['Iteration: ',num2str(i),'   -  ELBO: ',num2str(elbo)]);
    end
end

result.lambda = lambda;
result.time = time;
result.va = va;
result.predictive = Predictive;
result.priors = priors;
end