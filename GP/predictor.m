function [mean_star, var_star] = predictor(x_star)

global ModelInfo

X = ModelInfo.X;
y = ModelInfo.y;
hyp = ModelInfo.hyp;

D = size(X,2);

L = ModelInfo.L;

psi = k(x_star, X, hyp(1:D+1),0);

% calculate prediction
mean_star = psi*(L'\(L\y));

var_star = k(x_star, x_star, hyp(1:D+1),0) ...
    - psi*(L'\(L\psi'));

var_star = abs(diag(var_star));