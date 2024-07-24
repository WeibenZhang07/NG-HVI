function [ x ] = cnormrnd( mu,sig,llim,ulim )
%Generate from a constrained normal

if(ulim<llim)
 error('Fatal Error are cigrnd: upper limits <= lower limits');
end

if(isinf(llim))
    fa=0; %lower limit set to -inf
else
    fa=normcdf(llim,mu,sig); %lower limit set to F(llim), F(.) cdf of N(mu,sig)
end

if(isinf(ulim))
    fb=1; %upper limit set to +inf
else
    fb=normcdf(ulim,mu,sig);  %upper limit set to F(ulim), F(.) cdf of N(mu,sig)
end

if(fa==fb && fb==1)
        fa=fb-0.01; %trap 
end
if(fa==fb && fb==0)
        fb=fa+0.01;
end

uni=rand*(fb-fa)+fa;
x=norminv(uni,mu,sig);

while isinf(x)
    uni=rand*(fb-fa)+fa;
    x=norminv(uni,mu,sig);
end
end

