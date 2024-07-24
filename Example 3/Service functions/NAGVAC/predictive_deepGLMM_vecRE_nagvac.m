function [predictive] = predictive_deepGLMM_vecRE_nagvac(X_test,Y_test,X_train,Y_train,nn,results,J)

    
    mu = results.out.vbMU;
    b  = results.out.b;
    c  = results.out.c;
    
    dim.sample = length(X_test);
    dim.x = size(X_test{1},2);
    [M,p] = size(b);
    
    dim.sample = length(X_test);
    dim.x = size(X_test{1},2);
    
    layer = length(nn);
    dim.input = dim.x-1;
    NN = [dim.input,nn];
    dim.w_all = zeros(layer,1);
    for i = 1:layer
        dim.w_all(i) = NN(i+1)*(NN(i)+1);
    end
    dim.w_all_vec = sum(dim.w_all);
    dim.beta = nn(end)+1;
    dim.error = 1;
    dim.alpha = nn(end)+1;

    y_all = vertcat(Y_test{:});
    Mu_y = zeros(length(y_all),J);
    mu_y = cell(dim.sample,1);
    P = zeros(length(y_all),J);

    for j = 1:J
        epsilon = normrnd(0,1,M,1);
        z = normrnd(0,1,p,1);
        theta = mu + b * z + (c.*epsilon);
        
        [W_seq,~] = w_vec2mat(theta,nn,dim);
        beta = theta(dim.w_all_vec+1:dim.w_all_vec+dim.beta);
        theta_gammaj = theta(dim.w_all_vec+dim.beta+1:end);
       
     for i = 1:dim.sample
        yi = Y_test{i};
        Xi = X_test{i};
        x_insample = X_train{i};
        y_insample = Y_train{i};
        z_i =  features_output(Xi,W_seq);
        mu_alpha = zeros(length(theta_gammaj),1);
        alpha_h = mvnrnd(mu_alpha,diag(exp(theta_gammaj)))';
        mu_y{i} = z_i*(beta+alpha_h);
        
     end
    
     Mu_y(:,j) = vertcat(mu_y{:});
     P(:,j) = 1./(1+exp(-Mu_y(:,j)));

    end

    p_pos = mean(P,2);
    predictive.y_pred = p_pos > 0.5;
    y_pred = predictive.y_pred;
    predictive.pce = CRPS(y_all,p_pos)/length(p_pos);
    predictive.mcr = sum(abs(y_pred- y_all))/length(y_all);
    predictive.precision = sum(y_pred & y_all==1)/sum(y_pred);
    predictive.recall = sum(y_pred & y_all==1)/sum(y_all==1);
    predictive.f1 = 2*(predictive.precision * predictive.recall)/(predictive.precision + predictive.recall);


end
