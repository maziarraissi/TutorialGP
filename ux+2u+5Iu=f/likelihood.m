% @author: Maziar Raissi

function [NLML,D_NLML]=likelihood(hyp)

global ModelInfo
x_u = ModelInfo.x_u;
x_f = ModelInfo.x_f;

y_u = ModelInfo.y_u;
y_f = ModelInfo.y_f;

y=[y_u;y_f];

jitter = ModelInfo.jitter;

sigma_n_f = exp(hyp(end));
sigma_n_u = exp(hyp(end-1));

n_u = size(x_u,1);
[n_f,D] = size(x_f);
n = n_u+n_f;

K_uu = k_uu(x_u, x_u, hyp(1:D+1),0) + eye(n_u).*sigma_n_u;
K_uf = k_uf(x_u, x_f, hyp(1:end-2),0);
K_fu = K_uf';
K_ff = k_ff(x_f, x_f, hyp(1:end-2),0) + eye(n_f).*sigma_n_f;

K = [K_uu K_uf;
    K_fu K_ff];

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
    DK_uu = k_uu(x_u, x_u, hyp(1:D+1),i);
    DK_uf = k_uf(x_u, x_f, hyp(1:end-2),i);
    DK_fu = DK_uf';
    DK_ff = k_ff(x_f, x_f, hyp(1:end-2),i);

    DK = [DK_uu DK_uf;
        DK_fu DK_ff];

    D_NLML(i) = sum(sum(Q.*DK))/2;
end

for i=D+2:length(hyp)-2
    DK_uu = zeros(n_u,n_u);
    DK_uf = k_uf(x_u, x_f, hyp(1:end-2),i);
    DK_fu = DK_uf';
    DK_ff = k_ff(x_f, x_f, hyp(1:end-2),i);

    DK = [DK_uu DK_uf;
        DK_fu DK_ff];

    D_NLML(i) = sum(sum(Q.*DK))/2;
end

D_NLML(end-1) = sigma_n_u*trace(Q(1:n_u,1:n_u))/2;
D_NLML(end) = sigma_n_f*trace(Q(n_u+1:end,n_u+1:end))/2;