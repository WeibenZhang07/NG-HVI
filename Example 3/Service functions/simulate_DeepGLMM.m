function [X_fit,y_fit,X_val,y_val,X_test,y_test,alpha,eta,pos_rat] = simulate_DeepGLMM(num_input,num_group,num_obs)
   %Simulate data using the same polinomial function in section 6.1.4 of
   %Tran et al (2020) with logit link function
   

    X = cell(1,num_group);
    y = cell(1,num_group);
    alpha = cell(1,num_group);
    eta = cell(1,num_group);
    
    
    n_fit = 14;
    X_fit = cell(1,num_group);
    X_test = cell(1,num_group);
    X_val = cell(1,num_group);
    y_fit = cell(1,num_group);
    y_test = cell(1,num_group);
    y_val = cell(1,num_group);
   
    for n = 1:num_group
        X{n} = unifrnd(-1,1,num_obs,num_input);
        alpha{n} = normrnd(0,0.1);
        eta{n} = 2 + 3.*(X{n}(:,1) - 2.*X{n}(:,2)).^2 - 5.*X{n}(:,3)./(1+X{n}(:,4)).^2 ...
                - 5.*X{n}(:,5) + alpha{n}.*ones(num_obs,1);% + epsilon;
        p = 1./(1+exp(-eta{n}));
        y{n} = binornd(1,p);
        
        X{n}= [ones(num_obs,1) X{n}];
        X_fit{n} =X{n}(1:n_fit,:);
        X_val{n} = X{n}(n_fit+1:n_fit+3,:);
        X_test{n} = X{n}(n_fit+4:end,:);
        y_fit{n} =y{n}(1:n_fit,:);
        y_val{n} = y{n}(n_fit+1:n_fit+3,:);
        y_test{n} = y{n}(n_fit+4:end,:);
    end
    y_all = vertcat(y{:} );
    pos_rat = sum(y_all)/length(y_all);
    end
