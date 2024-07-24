function logpdf = logwishpdf(X,Sigma,v)
    d = size(X,2);
    multi_gamma = 1;
    for i = 1:d
        multi_gamma = multi_gamma*gamma(v-(d-1)/2);
    end
    logpdf = ((v-d-1)/2)*sum(log(eig(X)))+(-.5*trace(Sigma\X))-log(2^(v*d/2)*pi^(d*(d-1)/4)*multi_gamma   ) + (v/2)*sum(log(eig(Sigma)));

end