function [naive] = naive_prediction(y_train,y_test)
    H = length(y_test);
    n_train = length(y_train{1,1});
    n_test = length(y_test{1,1});
    p_test = y_test;
    prediction = y_test;
    
    for h = 1:H
       p_test{1,h} = ones(n_test,1)*(sum(y_train{1,h})/n_train);
       prediction{1,h} = p_test{1,h}>0.5;
    end
    y_test_all = vertcat(y_test{:});
    p_all = vertcat(p_test{:});
    prediction_all = vertcat(prediction{:});

    pce = CRPS(y_test_all,p_all)/length(y_test_all);
    mcr = sum(abs(prediction_all- y_test_all))/length(y_test_all);
    precision = sum((prediction_all==1) & y_test_all==1)/sum(prediction_all);
    recall = sum((prediction_all==1) & y_test_all==1)/sum(y_test_all==1);
    
    naive.prediction  = prediction;
    naive.pce  = pce;
    naive.mcr  = mcr;
    naive.precision  = precision;
    naive.recall  = recall;

    y_all = vertcat(y_train{:});
    p_naive = (sum(y_all)/length(y_all));

    prediction2 = ones(length(y_test_all),1)*(sum(y_all)/length(y_all)>0.5); % Naive prediction is the majority of training dataset
    pce2 = CRPS(y_test_all,p_naive)/length(prediction2);
    mcr2 = sum(abs(prediction2- y_test_all))/length(y_test_all);
    precision2 = sum((prediction2==1) & y_test_all==1)/sum(prediction2);
    recall2 = sum((prediction2==1) & y_test_all==1)/sum(y_test_all==1);
    
    naive.prediction2  = prediction2;
    naive.pce2  = pce2;
    naive.mcr2  = mcr2;
    naive.precision2  = precision2;
    naive.recall2  = recall2;

end