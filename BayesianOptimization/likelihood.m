% @author: Maziar Raissi & Paris Perdikaris

function [NLML, D_NLML]=likelihood(hyp)

global ModelInfo

x = ModelInfo.x;
y = ModelInfo.y;

jitter = ModelInfo.jitter;

sigma_n = exp(hyp(end));

[n,D] = size(x);

K = k(x, x, hyp(1:D+1),0) + eye(n).*sigma_n;

K = K + eye(n).*jitter;

% Cholesky factorisation
[L,p]=chol(K,'lower');

ModelInfo.L = L;

if p > 0
    fprintf(1,'Covariance is ill-conditioned\n');
end

alpha = L'\(L\y);
NLML = 0.5*y'*alpha + sum(log(diag(L))) + log(2*pi)*n/2;

D_NLML = 0*hyp;
Q =  L'\(L\eye(n)) - alpha*alpha';
for i=1:D+1
    DK = k(x, x, hyp(1:D+1),i);
    D_NLML(i) = sum(sum(Q.*DK))/2;
end

D_NLML(end) = sigma_n*trace(Q)/2;