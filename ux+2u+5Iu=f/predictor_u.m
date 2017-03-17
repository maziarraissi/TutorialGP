% @author: Maziar Raissi

function [pred_u_star, var_u_star] = predictor_u(x_star)

global ModelInfo

x_u = ModelInfo.x_u;
x_f = ModelInfo.x_f;
y_u = ModelInfo.y_u;
y_f = ModelInfo.y_f;
hyp = ModelInfo.hyp;

D = size(x_f,2);

y = [y_u; y_f];

L = ModelInfo.L;

psi1 = k_uu(x_star, x_u, hyp(1:D+1),0);
psi2 = k_uf(x_star, x_f, hyp(1:end-2),0);
psi = [psi1 psi2];

% calculate prediction
pred_u_star = psi*(L'\(L\y));

var_u_star = k_uu(x_star, x_star, hyp(1:D+1),0) ...
    - psi*(L'\(L\psi'));

var_u_star = abs(diag(var_u_star));