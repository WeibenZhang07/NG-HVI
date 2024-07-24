function [Predictive] = record_loss(predictive,Predictive,J,ind)
    % Loss_item = {'PCE(MC)','MCR(MC)', 'PRE(MC)', 'REC(MC)','P(MC)',...
    %                'PCE','MCR', 'PRE', 'REC', 'P'};
    if J >0
        Predictive{1,1}(ind) = predictive.pce;
        Predictive{1,2}(ind) = predictive.mcr;
        Predictive{1,3}(ind) = predictive.precision;
        Predictive{1,4}(ind) = predictive.recall;
    end 
    Predictive{1,5}(ind) = predictive.pce2;
    Predictive{1,6}(ind) = predictive.mcr2;
    Predictive{1,7}(ind) = predictive.precision2;
    Predictive{1,8}(ind) = predictive.recall2;
     

end