% The NG-HVI results are stored in struct array 'results_ng'.
% Training data stored in X_train and Y_train.
% nn = [32,16,8]
% Change the variable name accordingly.
folder = fileparts(which('lek_profile.m')); 
addpath(genpath(folder));

load('sic_table.mat')
load('ind_sic.mat')
sic_name = SIC_table1.Properties.RowNames(1:10);
X_names = {'ERM','SMB','HML'};

rng(1234)
lambdahat_mu = mean(results_ng.lambda.mu(:,end-100:end),2);
lambdahat_B = mean(results_ng.lambda.B(:,:,end-100:end),3);
lambdahat_d = mean(results_ng.lambda.d(:,:,end-100:end),3);
quant = 0.05:0.05:0.95;

J = 1000;
lambdahat.mu = lambdahat_mu;
lambdahat.B  = lambdahat_B;
lambdahat.d = lambdahat_d;


dim.sample = length(X_train);
dim.x = size(X_train{1},2);
[M,p] = size(lambdahat.B);

dim.sample = length(X_train);
dim.x = size(X_train{1},2);

layer = length(nn);
dim.input = dim.x-1;
NN = [dim.input,nn];
dim.weights = zeros(layer,1);
for i = 1:layer
    dim.weights(i) = NN(i+1)*(NN(i)+1);
end
dim.weights_vec = sum(dim.weights);
r = nn(end)+1;
dim.beta = nn(end)+1;
dim.Sigma = 1;
dim.w = r*(r+1)/2;


for h = 1:dim.sample
    dim.H(h,1) = size(X_train{h},1);
end

%% 2005 Jul, 7th observation
%% 2008 Oct, 46th observation
%% 2012 May, 89th observation
x = X_train{6};
% mean and std of return (y) and raw data (factors).
mean_train = [0.962989859659311	0.628833333333333	0.128500000000000	-0.0178333333333336	0.308250000000000	0.118666666666667];
sd_train = [12.3080900980544	4.34462171765931	2.32322050102824	2.53062396122746	1.58559427825752	1.29855958500033];
x_origin = x;

x_low = X_train{6}(7,:);
x_med = X_train{6}(89,:);
x_high = X_train{6}(46,:);

X_lek = cell(1,dim.input);
X_lek2 = cell(1,dim.input);
X_lek3 = cell(1,dim.input);
n_lek = length(quant);
Mu_y_lek = cell(dim.input,length(ind_sic));
mu_y_lek_mean= cell(dim.input,length(ind_sic));
Mu_y_lek2 = cell(dim.input,length(ind_sic));
mu_y_lek_mean2= cell(dim.input,length(ind_sic));
Mu_y_lek3 = cell(dim.input,length(ind_sic));
mu_y_lek_mean3= cell(dim.input,length(ind_sic));
for i = 1:dim.input
    X_lek{i} = ones(n_lek,1).*x_low;
    X_lek{i}(:,i+1) = linspace(quantile(x(:,i+1),quant(1)), quantile(x(:,i+1),quant(end)), n_lek)';
    X_lek2{i} = ones(n_lek,1).*x_med;
    X_lek2{i}(:,i+1) = linspace(quantile(x(:,i+1),quant(1)), quantile(x(:,i+1),quant(end)), n_lek)';
    X_lek3{i} = ones(n_lek,1).*x_high;
    X_lek3{i}(:,i+1) = linspace(quantile(x(:,i+1),quant(1)), quantile(x(:,i+1),quant(end)), n_lek)';
end

for j = 1:J

    %% Not integrate out random effects
    epsilon = normrnd(0,1,M,1);
    z = normrnd(0,1,p,1);
    theta = lambdahat.mu + lambdahat.B * z + (lambdahat.d.*epsilon);

    [weights,weights_vec] = w_vec2mat(theta,NN,dim.weights_vec);
    beta = theta(dim.weights_vec+1:dim.weights_vec+dim.beta);
    TSigma = theta(dim.weights_vec+dim.beta+1);
    w = theta(dim.weights_vec+dim.beta+2:end);

    sigma_e = exp(TSigma);
    [omega,W] = f_gen_omega(w);

    z_out = cell(dim.sample,1);

    for ii = 1:dim.sample
        z_out{ii} =  features_output(X_train{ii},weights);
    end

    Alpha_h = VB_vec_alpha_i(z_out,z_out,Y_train,beta,omega,sigma_e);%fixed for both training sample and testing sample

    % Lek profile

    for ii = 1:dim.input
        z_out_lek =  features_output(X_lek{ii},weights);
        z_out_lek2 =  features_output(X_lek2{ii},weights);
        z_out_lek3 =  features_output(X_lek3{ii},weights);
        for i = 1:length(ind_sic)
            alpha_h = Alpha_h{ind_sic(i)};
            Mu_y_lek{ii,i}(:,j) = 12.3081.*(z_out_lek*(beta+alpha_h))+0.9630;
            Mu_y_lek2{ii,i}(:,j) = 12.3081.*(z_out_lek2*(beta+alpha_h))+0.9630;
            Mu_y_lek3{ii,i}(:,j) = 12.3081.*(z_out_lek3*(beta+alpha_h))+0.9630;
        end
    end



end

%% plot lek profile

for ii = 1:dim.input
    for i = 1: length(ind_sic)
        mu_y_lek_mean{ii,i} = mean(Mu_y_lek{ii,i},2);
        mu_y_lek_std{ii,i} = std(Mu_y_lek{ii,i},0,2);
        mu_y_lek_mean2{ii,i} = mean(Mu_y_lek2{ii,i},2);
        mu_y_lek_std2{ii,i} = std(Mu_y_lek2{ii,i},0,2);
        mu_y_lek_mean3{ii,i} = mean(Mu_y_lek3{ii,i},2);
        mu_y_lek_std3{ii,i} = std(Mu_y_lek3{ii,i},0,2);
    end
end
mak = {'none','o','+','*','.','x','_','|','s','d'};
for ii = 1:dim.input

    lek.(X_names{ii})=  figure('DefaultAxesFontSize',12,'Position', [10 10 500 400]);
    for i = 1:length(ind_sic)
    plot(sd_train(ii+1)*X_lek{ii}(:,ii+1)+mean_train(ii+1),mu_y_lek_mean{ii,i},'Marker',mak{i})
        hold on
    end
    if ii ==1
        legend(sic_name,'location','southeast','FontSize',12,'NumColumns',2);
        legend boxoff
    end
    xlim([min(sd_train(ii+1)*X_lek{ii}(:,ii+1)+mean_train(ii+1))-0.5 max(sd_train(ii+1)*X_lek{ii}(:,ii+1)+mean_train(ii+1))+0.5]);
    xlabel(X_names{ii})
    ylabel('Profile predicted return')
end

for ii = 1:dim.input

    lek2.(X_names{ii})=  figure('DefaultAxesFontSize',12,'Position', [10 10 500 400]);
    for i = 1:length(ind_sic)
        plot(sd_train(ii+1)*X_lek2{ii}(:,ii+1)+mean_train(ii+1),mu_y_lek_mean2{ii,i},'Marker',mak{i})        
        hold on
    end
    if ii ==1
        legend(sic_name,'location','southeast','FontSize',12,'NumColumns',2);
        legend boxoff
    end
    xlim([min(sd_train(ii+1)*X_lek2{ii}(:,ii+1)+mean_train(ii+1))-0.5 max(sd_train(ii+1)*X_lek2{ii}(:,ii+1)+mean_train(ii+1))+0.5]);
    xlabel(X_names{ii})
    ylabel('Profile predicted return')
end

for ii = 1:dim.input

    lek3.(X_names{ii})=  figure('DefaultAxesFontSize',12,'Position', [10 10 500 400]);
    for i = 1:length(ind_sic)
        plot(sd_train(ii+1)*X_lek3{ii}(:,ii+1)+mean_train(ii+1),mu_y_lek_mean3{ii,i},'Marker',mak{i})
        hold on
    end
    if ii ==1
        legend(sic_name,'location','southeast','FontSize',12,'NumColumns',2);
        legend boxoff
    end
    xlim([min(sd_train(ii+1)*X_lek3{ii}(:,ii+1)+mean_train(ii+1))-0.5 max(sd_train(ii+1)*X_lek3{ii}(:,ii+1)+mean_train(ii+1))+0.5]);
    xlabel(X_names{ii})
    ylabel('Profile predicted return')
end
for ii = 1:3
    exportgraphics(lek.(X_names{ii}),strcat('ff3_low_vol_',X_names{ii},"_layers",str_nn,".pdf"),'Resolution',300);
    saveas(lek.(X_names{ii}),strcat('ff3_low_vol_',X_names{ii},"_layers",str_nn,".fig"))
    exportgraphics(lek2.(X_names{ii}),strcat('ff3_med_vol_',X_names{ii},"_layers",str_nn,".pdf"),'Resolution',300);
    saveas(lek2.(X_names{ii}),strcat('ff3_med_vol_',X_names{ii},"_layers",str_nn,".fig"))
    exportgraphics(lek3.(X_names{ii}),strcat('ff3_high_vol_',X_names{ii},"_layers",str_nn,".pdf"),'Resolution',300);
    saveas(lek3.(X_names{ii}),strcat('ff3_high_vol_',X_names{ii},"_layers",str_nn,".fig"))

end