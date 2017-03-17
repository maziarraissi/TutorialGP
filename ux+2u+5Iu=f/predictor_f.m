% @author: Maziar Raissi

function [pred_f_star, var_f_star] = predictor_f(x_star)

global ModelInfo

x_u = ModelInfo.x_u;
x_f = ModelInfo.x_f;
y_u = ModelInfo.y_u;
y_f = ModelInfo.y_f;
hyp = ModelInfo.hyp;

D = size(x_f,2);

y = [y_u; y_f];

L=ModelInfo.L;

psi1 = k_uf(x_u, x_star, hyp(1:end-2),0)';
psi2 = k_ff(x_star, x_f, hyp(1:end-2),0);
psi = [psi1 psi2];

% calculate prediction
pred_f_star = psi*(L'\(L\y));

var_f_star = k_ff(x_star, x_star, hyp(1:end-2),0) ...
    - psi*(L'\(L\psi'));

var_f_star = abs(diag(var_f_star));