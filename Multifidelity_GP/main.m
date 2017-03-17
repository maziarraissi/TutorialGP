function main()
%% Pre-processing
clc; close all;
rng('default')

addpath ./Utilities

global ModelInfo

%% Setup
N_L = 12;
N_H = 3;
D = 1;
lb = zeros(1,D);
ub = ones(1,D);
jitter = eps;
noise_L = 1.0;
noise_H = 0.0;
ModelInfo.jitter=jitter;

%% Generate Data
ModelInfo.X_H = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(N_H,D)    ,(ub-lb)));
ModelInfo.y_H = f_H(ModelInfo.X_H);
ModelInfo.y_H = ModelInfo.y_H + noise_H*randn(N_H,1);

ModelInfo.X_L = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(N_L,D)    ,(ub-lb)));
ModelInfo.y_L = f_L(ModelInfo.X_L);
ModelInfo.y_L = ModelInfo.y_L + noise_L*randn(N_L,1);

%% Optimize model
% hyp = [logsigma1 logtheta1 logsigma2 logtheta2 rho logsigma_eps_L logsigma_eps_H]
hyp = [log([1 1 1 1]) 1 -4 -4];
options = optimoptions('fminunc','GradObj','on','Display','iter',...
    'Algorithm','trust-region','Diagnostics','on','DerivativeCheck','on',...
    'FinDiffType','central');
[ModelInfo.hyp,~,~,~,~,~] = fminunc(@likelihood,hyp,options);

%% Make Predictions
n_star = 200;
x_star = linspace(lb(1), ub(1), n_star)';

[mean_f_H_star, var_f_H_star] = predictor_f_H(x_star);

f_H_star = f_H(x_star);
f_L_star = f_L(x_star);

fprintf(1,'Relative L2 error f_H: %e\n', (norm(mean_f_H_star-f_H_star,2)/norm(f_H_star,2)));

%% Plot results
color = [55,126,184]/255;

fig = figure(1);
set(fig,'units','normalized','outerposition',[0 0 1 1])

clear h;
clear leg;
hold
h(1) = plot(x_star, f_H_star,'k','LineWidth',2);
h(2) = plot(ModelInfo.X_H, ModelInfo.y_H,'kx','MarkerSize',14, 'LineWidth',2);
h(3) = plot(x_star,mean_f_H_star,'b--','LineWidth',3);
[l,h(4)] = boundedline(x_star, mean_f_H_star, 2.0*sqrt(var_f_H_star), ':', 'alpha','cmap', color);
outlinebounds(l,h(4));
h(5) = plot(x_star, f_L_star,'k:','LineWidth',2);
h(6) = plot(ModelInfo.X_L, ModelInfo.y_L,'k+','MarkerSize',14, 'LineWidth',2);


leg{1} = '$f_H(x)$';
leg{2} = sprintf('%d high-fidelity training data', N_H);
leg{3} = '$\overline{f}_H(x)$'; leg{4} = 'Two standard deviations';
leg{5} = '$f_L(x)$';
leg{6} = sprintf('%d low-fidelity training data', N_L);

hl = legend(h,leg,'Location','northwestoutside');
legend boxoff
set(hl,'Interpreter','latex')
xlabel('$x$')
ylabel('$f_L(x), f_H(x)$')
title('A');

axis square
ylim(ylim + [-diff(ylim)/10 0]);
xlim(xlim + [-diff(xlim)/10 0]);
set(gca,'FontSize',16);
set(gcf, 'Color', 'w');


axes('Position',[0.05 0.35 .3 .3])
box on
hold
h(1) = plot(f_L_star, f_H_star,'b','LineWidth',3);
xlabel('$f_{L}(x)$');
ylabel('$f_{H}(x)$');
title('B -- Cross-correlation');
axis square
set(gca,'FontSize',16);
set(gca,'Xtick',[]);
set(gca,'Ytick',[]);
set(gcf, 'Color', 'w');

%% Post-processing
rmpath ./Utilities