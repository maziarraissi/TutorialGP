% @author: Maziar Raissi

function [NLML,D_NLML]=likelihood(hyp)

global ModelInfo
X_L = ModelInfo.X_L;
X_H = ModelInfo.X_H;

y_L = ModelInfo.y_L;
y_H = ModelInfo.y_H;
y=[y_L;y_H];

jitter = ModelInfo.jitter;

sigma_eps_H = exp(hyp(end));
sigma_eps_L = exp(hyp(end-1));
rho = hyp(end-2);

N_L = size(X_L,1);
[N_H,D] = size(X_H);
N = N_L+N_H;

K_LL = k(X_L, X_L, hyp(1:D+1),0);
K_LH = rho*k(X_L, X_H, hyp(1:D+1),0);
K_HL = rho*k(X_H, X_L, hyp(1:D+1),0);
K_HH = (rho^2)*k(X_H, X_H, hyp(1:D+1),0) + k(X_H, X_H, hyp(D+2:2*D+2),0);

K_LL = K_LL + eye(N_L).*sigma_eps_L;
K_HH = K_HH + eye(N_H).*sigma_eps_H;

K = [K_LL K_LH;
     K_HL K_HH];

K = K + eye(N).*jitter;

% Cholesky factorisation
[L,p]=chol(K,'lower');

ModelInfo.L = L;

if p > 0
    fprintf(1,'Covariance is ill-conditioned\n');
end

alpha = L'\(L\y);
NLML = 0.5*y'*alpha + sum(log(diag(L))) + log(2*pi)*N/2;

%% Derivatives
D_NLML = 0*hyp;
Q =  L'\(L\eye(N)) - alpha*alpha';
for i=1:D+1
    DK_LL = k(X_L, X_L, hyp(1:D+1),i);
    DK_LH = rho*k(X_L, X_H, hyp(1:D+1),i);
    DK_HL = rho*k(X_H, X_L, hyp(1:D+1),i);
    DK_HH = rho^2*k(X_H, X_H, hyp(1:D+1),i);
    
    DK = [DK_LL DK_LH;
          DK_HL DK_HH];
    D_NLML(i) = sum(sum(Q.*DK))/2;
end

for i=D+2:2*D+2  
    DK_LL = zeros(N_L,N_L);
    DK_LH = zeros(N_L,N_H);
    DK_HL = zeros(N_H,N_L);
    DK_HH = k(X_H, X_H, hyp(D+2:2*D+2),i-(D+1));
    
    DK = [DK_LL DK_LH;
          DK_HL DK_HH];
    D_NLML(i) = sum(sum(Q.*DK))/2;
end

DK_LL = zeros(N_L,N_L);
DK_LH = k(X_L, X_H, hyp(1:D+1),0);
DK_HL = k(X_H, X_L, hyp(1:D+1),0);
DK_HH = (2*rho)*k(X_H, X_H, hyp(1:D+1),0);
DK = [DK_LL DK_LH;
      DK_HL DK_HH];

D_NLML(end-2) = sum(sum(Q.*DK))/2;

D_NLML(end-1) = sigma_eps_L*trace(Q(1:N_L,1:N_L))/2;
D_NLML(end) = sigma_eps_H*trace(Q(N_L+1:end,N_L+1:end))/2;