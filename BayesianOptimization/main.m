% @author: Maziar Raissi & Paris Perdikaris

function main()
%% Pre-processing
clc; close all

addpath ./Utilities

rng('default')

set(0,'defaulttextinterpreter','latex')

global ModelInfo

%% Setup
Ntr = 2;
Nts = 2000;
dim = 1;
lb = zeros(1,dim);
ub = ones(1,dim);
jitter = 1e-8;
noise = 0.00;
ModelInfo.jitter=jitter;

max_iter = 20;
plt = 1;

%% Generate Data
% Training data
ModelInfo.x = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(Ntr,dim)    ,(ub-lb)));
ModelInfo.y = f(ModelInfo.x);
ModelInfo.y = ModelInfo.y + noise*randn(size(ModelInfo.y));

% Test points
x_star = linspace(lb,ub,Nts)';
f_star = f(x_star);

% Initialize hyp := [logsigma logtheta log_sigma_eps]
ModelInfo.hyp = [log([1 1]) -5];

%% Bayesian Optimization Loop
color = [55,126,184]/255;

fig = figure(1);
set(fig,'units','normalized','outerposition',[0 0 1 1])

for iter = 1:max_iter    
    clf
    % Training
    [ModelInfo.hyp,~,~] = minimize(ModelInfo.hyp, @likelihood, -2000);
    
    % Predict Objective
    [mean_star, var_star] = predictor(x_star);
    
    % Acquisition Function: Upper Confidence Bound (UCB)
    UCB = mean_star + 2*sqrt(var_star);
    
    % Maximize UCB using grid search
    [opt_val, opt_idx] = max(UCB);
    
    % New sampling point
    new_x = x_star(opt_idx,:);
    
    dlx = norm(ModelInfo.x(end,:) - new_x,2);
        
    fprintf(1,'Iteration: %d, Best_y = %e, dlx = %e, new_X = %f\n', iter, max(ModelInfo.y), dlx, new_x);
    
    if plt == 1
        subplot(2,1,1);
        clear h;
        clear leg;
        hold
        h(1) = plot(x_star, f_star, 'k','LineWidth',2);
        h(2) = plot(ModelInfo.x,ModelInfo.y,'o', 'MarkerSize',14, 'LineWidth',2);
        h(3) = plot(x_star,mean_star,'b--','LineWidth',3);
        [l,h(4)] = boundedline(x_star, mean_star, 2.0*sqrt(var_star), ':', 'alpha','cmap', color);
        outlinebounds(l,h(4));
        leg{1} = '$f(x)$';
        leg{2} = sprintf('%d training data', size(ModelInfo.x,1));
        leg{3} = '$\overline{f}(x)$'; leg{4} = 'Two standard deviations';
        hl = legend(h,leg,'Location','southwest');
        legend boxoff
        set(hl,'Interpreter','latex')
        xlabel('$x$')
        ylabel('$f(x), \overline{f}(x)$')

        
        plot(new_x*ones(10,1), linspace(-2,2,10),'r--', 'LineWidth', 2);
        set(gca,'FontSize',16);
        set(gcf, 'Color', 'w');
        
        subplot(2,1,2);
        hold
        clear h;
        clear leg;
        h(1) = plot(x_star, UCB, 'b-','LineWidth',3);
        plot(new_x,opt_val,'o','MarkerSize',14, 'LineWidth',2);
        h(2) = plot(new_x*ones(10,1), linspace(-2,2,10),'r--', 'LineWidth', 2);
        xlabel('$x$')
        ylabel('$\alpha(x;\mathcal{D})$')
        hl = legend(h, {'UCB: $\overline{f}(x) + 2\sigma(x)$', 'Next sampling point'},'Location','southwest');
        set(hl,'Interpreter','latex')
        legend boxoff        
        set(gca,'FontSize',16);
        set(gcf, 'Color', 'w');
        
        % w = waitforbuttonpress;
        drawnow;
        pause(2);
    end
    
    if (dlx < eps)
        break;
    else
        ModelInfo.x = [ModelInfo.x; new_x];
        ModelInfo.y = [ModelInfo.y; f(new_x)];
    end
end

%% Post-processing
rmpath ./Utilities
