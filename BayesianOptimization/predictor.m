% @author: Maziar Raissi & Paris Perdikaris

function [f, v] = predictor(x_star)

global ModelInfo

x = ModelInfo.x;
y = ModelInfo.y;

hyp = ModelInfo.hyp;
L=ModelInfo.L;

D = size(x,2);

psi = k(x_star, x, hyp(1:D+1),0);

% calculate prediction
f = psi*(L'\(L\y));

v = k(x_star, x_star, hyp(1:D+1),0) ...
  - psi*(L'\(L\psi'));

v = abs(diag(v));