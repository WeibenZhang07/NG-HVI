function f = CRPS(y,p)
% compute the continuous ranked probability score CRPS: the smaller the
% better
p(p==0) = 10e-10;
p(p==1) = 1-(10e-10);
f = -sum(y.*log(p)+(1-y).*log(1-p));
end