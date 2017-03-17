% @author: Maziar Raissi

function [mean_star, var_star] = predictor_f_H(x_star)

global ModelInfo

X_L = ModelInfo.X_L;
X_H = ModelInfo.X_H;
y_L = ModelInfo.y_L;
y_H = ModelInfo.y_H;
hyp = ModelInfo.hyp;
rho = hyp(end-2);

D = size(X_H,2);

y = [y_L; y_H];

L=ModelInfo.L;

psi1 = rho*k(x_star, X_L, hyp(1:D+1),0);
psi2 = rho^2*k(x_star, X_H, hyp(1:D+1),0) + k(x_star, X_H, hyp(D+2:2*D+2),0);
psi = [psi1 psi2];

% calculate prediction
mean_star = psi*(L'\(L\y));

var_star = rho^2*k(x_star, x_star, hyp(1:D+1),0) ...
  + k(x_star, x_star, hyp(D+2:2*D+2),0) ...
  - psi*(L'\(L\psi'));

var_star = abs(diag(var_star));