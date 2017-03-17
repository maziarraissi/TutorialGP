% @author: Maziar Raissi

function [NLML,D_NLML]=likelihood(hyp)

global ModelInfo
X = ModelInfo.X;
y = ModelInfo.y;

jitter = ModelInfo.jitter;

sigma_eps = exp(hyp(end));

[n,D] = size(X);

K_xx = k(X, X, hyp(1:D+1),0) + eye(n)*sigma_eps;

K = K_xx + eye(n).*jitter;

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
    DK = k(X, X, hyp(1:D+1),i);

    D_NLML(i) = trace(Q*DK)/2;
end

D_NLML(end) = sigma_eps*trace(Q)/2;