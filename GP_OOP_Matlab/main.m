% @author: Maziar Raissi

function main()
%% Pre-processing
clc; close all;
rng('default')

save_plt = 1;

addpath ./Utilities
addpath ./Utilities/export_fig

    function CleanupFun()
        rmpath ./Utilities
        rmpath ./Utilities/export_fig
    end

finishup = onCleanup(@() CleanupFun());

%% Setup
N = 20;
D = 1;
lb = 0.0*ones(1,D);
ub = 1.0*ones(1,D);
noise = 0.1;

%% Generate Data
% Traning Data on f(x)
X = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(N,D)    ,(ub-lb)));
f = @(x) x.*cos(4*pi*x);
y = f(X);
y = y + noise*std(y)*randn(N,1);

% Test Data on f(x)
N_star = 200;
X_star = linspace(lb(1), ub(1), N_star)';
f_star = f(X_star);

%% Model Definition
hyp = [log([1 0.01]) -4];
model = GP(X,y,hyp);

%% Model Training
model = model.train(200);

%% Make Predictions
[mean_f_star, var_f_star] = model.predict(X_star);

fprintf(1,'Relative L2 error f: %e\n', (norm(mean_f_star-f_star,2)/norm(f_star,2)));

%% Plot Results
set(0,'defaulttextinterpreter','latex')

color = [217,95,2]/255;

fig = figure(1);
set(fig,'units','normalized','outerposition',[0 0 1 0.5])

clear h;
clear leg;
hold
h(1) = plot(X_star, f_star,'b','LineWidth',3);
h(2) = plot(X, y,'ro','MarkerSize',14, 'LineWidth',2);
h(3) = plot(X_star,mean_f_star,'r--','LineWidth',3);
[l,h(4)] = boundedline(X_star, mean_f_star, 2.0*sqrt(var_f_star), ':', 'alpha','cmap', color);
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
set(gca,'FontSize',20);
set(gcf, 'Color', 'w');

if save_plt == 1
    export_fig ./Figures/GP_1D_Example.png -r300
end

end