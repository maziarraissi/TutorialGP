% @author: Maziar Raissi

function main()
%% Pre-processing
clc; close all;
rng('default')
addpath ./Utilities

set(0,'defaulttextinterpreter','latex')

global ModelInfo

%% Setup
N = 5;
D = 1;
lb = zeros(1,D);
ub = ones(1,D);
ModelInfo.jitter = eps;
noise = 0.0;

%% Generate Data
% Data on f(x)
ModelInfo.X = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(N,D)    ,(ub-lb)));
ModelInfo.y = f(ModelInfo.X) + noise*randn(N,1);

%% Optimize model
% hyp = [logsigma logtheta logsigma_eps]
hyp = [log([1 1]) -5];

options = optimoptions('fminunc','GradObj','on','Display','iter',...
    'Algorithm','trust-region','Diagnostics','on','DerivativeCheck','on',...
    'FinDiffType','central');
ModelInfo.hyp = fminunc(@likelihood,hyp,options);

%% Make Predictions
n_star = 200;
x_star = linspace(lb(1), ub(1), n_star)';

[mean_f_star, var_f_star] = predictor(x_star);

f_star = f(x_star);

fprintf(1,'Relative L2 error f: %e\n', (norm(mean_f_star-f_star,2)/norm(f_star,2)));

%% Plot Results
color = [55,126,184]/255;

fig = figure(1);
set(fig,'units','normalized','outerposition',[0 0 1 1])

clear h;
clear leg;
hold
h(1) = plot(x_star, f_star,'k','LineWidth',2);
h(2) = plot(ModelInfo.X, ModelInfo.y,'kx','MarkerSize',14, 'LineWidth',2);
h(3) = plot(x_star,mean_f_star,'b--','LineWidth',3);
[l,h(4)] = boundedline(x_star, mean_f_star, 2.0*sqrt(var_f_star), ':', 'alpha','cmap', color);
outlinebounds(l,h(4));


leg{1} = '$f(x)$';
leg{2} = sprintf('%d training data', N);
leg{3} = '$\overline{f}(x)$'; leg{4} = 'Two standard deviations';

hl = legend(h,leg,'Location','southwest');
legend boxoff
set(hl,'Interpreter','latex')
xlabel('$x$')
ylabel('$f(x), \overline{f}(x)$')


axis square
ylim(ylim + [-diff(ylim)/10 0]);
xlim(xlim + [-diff(xlim)/10 0]);
set(gca,'FontSize',16);
set(gcf, 'Color', 'w');

%% Post-processing
rmpath ./Utilities