% @author: Maziar Raissi

function main()
%% Pre-Processing
clc; close all;

addpath ./Utilities

set(0,'defaulttextinterpreter','latex')

global ModelInfo

%% Setup
n_f = 3;
n_u = 4;
D = 1;
lb = zeros(1,D);
ub = ones(1,D);
ModelInfo.jitter = eps;
noise_u = 0.00;
noise_f = 0.00;

%% Generate Data
rng(1111)
% Data on u(x)
ModelInfo.x_u = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(n_u,D)    ,(ub-lb)));
ModelInfo.y_u = u(ModelInfo.x_u) + noise_u*randn(n_u,1);

rng(2222)
% Data on f(x)
ModelInfo.x_f = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(n_f,D)    ,(ub-lb)));
ModelInfo.y_f = f(ModelInfo.x_f) + noise_f*randn(n_f,1);

%% Optimize model
% hyp = [logsigma logtheta alpha beta logsigma_n_u logsigma_n_f]
hyp = log([1 1 exp(1) exp(1) 10^-3 10^-3]);
[ModelInfo.hyp,~,~] = minimize(hyp, @likelihood, -5000);

fprintf(1,'alpha = %f\nbeta = %f\n\n', ModelInfo.hyp(3), ModelInfo.hyp(4));

%% Make Predictions
n_star = 200;
x_star = linspace(lb(1), ub(1), n_star)';

[pred_u_star, var_u_star] = predictor_u(x_star);
[pred_f_star, var_f_star] = predictor_f(x_star);

u_star = u(x_star);
f_star = f(x_star);

fprintf(1,'Relative L2 error u: %e\n', (norm(pred_u_star-u_star,2)/norm(u_star,2)));
fprintf(1,'Relative L2 error f: %e\n', (norm(pred_f_star-f_star,2)/norm(f_star,2)));

%% Plot results
color = [55,126,184]/255;

fig = figure(1);
set(fig,'units','normalized','outerposition',[0 0 1 1])

if n_u > 0
    clear h;
    clear leg;
    if n_f > 0
        subplot(1,2,1)
    end
    hold
    h(1) = plot(x_star, u_star,'k','LineWidth',2);
    h(2) = plot(ModelInfo.x_u, ModelInfo.y_u,'kx','MarkerSize',14, 'LineWidth',2);
    h(3) = plot(x_star,pred_u_star,'b--','LineWidth',3);
    [l,h(4)] = boundedline(x_star, pred_u_star, 2.0*sqrt(var_u_star), ':','alpha','cmap', color);
    outlinebounds(l,h(4));
    
    leg{1} = '$u(x)$';
    leg{2} = sprintf('%d training data', n_u);
    leg{3} = '$\overline{u}(x)$'; leg{4} = 'Two standard deviations';
    
    hl = legend(h,leg,'Location','southwest');
    legend boxoff
    set(hl,'Interpreter','latex')
    xlabel('$x$')
    ylabel('$u(x)$, $\overline{u}(x)$')
    if n_f > 0
        if noise_u == 0.00 && noise_f == 0.00
            title('(A)')
        else
            title('(C)')
        end
    end
    
    axis square
    ylim(ylim + [-diff(ylim)/10 0]);
    xlim(xlim + [-diff(xlim)/10 0]);
    set(gca,'FontSize',16);
    set(gcf, 'Color', 'w');
    
end

if n_f > 0
    
    clear h;
    clear leg;
    if n_u > 0
        subplot(1,2,2)
    end
    hold
    h(1) = plot(x_star, f_star,'k','LineWidth',2);
    h(2) = plot(ModelInfo.x_f, ModelInfo.y_f,'kx','MarkerSize',14, 'LineWidth',2);
    h(3) = plot(x_star,pred_f_star,'b--','LineWidth',3);
    [l,h(4)] = boundedline(x_star, pred_f_star, 2.0*sqrt(var_f_star), ':', 'alpha','cmap', color);
    outlinebounds(l,h(4));
    
    
    leg{1} = '$f(x)$';
    leg{2} = sprintf('%d training data', n_f);
    leg{3} = '$\overline{f}(x)$'; leg{4} = 'Two standard deviations';
    
    hl = legend(h,leg,'Location','southwest');
    legend boxoff
    set(hl,'Interpreter','latex')
    xlabel('$x$')
    ylabel('$f(x), \overline{f}(x)$')
    if n_u > 0
        if noise_u == 0.00 && noise_f == 0.00
            title('(B)')
        else
            title('(D)')
        end
    end
    
    axis square
    ylim(ylim + [-diff(ylim)/10 0]);
    xlim(xlim + [-diff(xlim)/10 0]);
    set(gca,'FontSize',16);
    set(gcf, 'Color', 'w');
    
end

%% Post-processing
rmpath ./Utilities