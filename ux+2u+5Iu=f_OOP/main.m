% @author: Maziar Raissi

function main()
%% Pre-Processing
clc; close all;

addpath ./Utilities

set(0,'defaulttextinterpreter','latex')

%% Setup
N_f = 3;
N_u = 4;
D = 1;
lb = zeros(1,D);
ub = ones(1,D);
noise_u = 0.00;
noise_f = 0.00;

%% Generate Data
% Data on u(x)
function u = u(x)
    u = sin(2*pi*x);
end
rng(1111)
x_u = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(N_u,D)    ,(ub-lb)));
y_u = u(x_u);
y_u = y_u + noise_u*std(y_u)*randn(N_u,1);

% Data on f(x)
function f=f(x)
    f = 2.*pi.*cos(2.*pi.*x)+5.*pi.^(-1).*sin(pi.*x).^2+2.*sin(2.*pi.*x);
end
rng(2222)
x_f = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(N_f,D)    ,(ub-lb)));
y_f = f(x_f);
y_f = y_f + noise_f*std(y_f)*randn(N_f,1);

N_star = 200;
X_star = linspace(lb(1), ub(1), N_star)';
u_star = u(X_star);
f_star = f(X_star);

%% Model Definition
model = DE_GP(x_u, y_u, x_f, y_f);

%% Model Training
model = model.train();
fprintf(1,'alpha = %f\nbeta = %f\n\n', model.hyp(3), model.hyp(4));

%% Make Predictions
[pred_u_star, var_u_star] = model.predict_u(X_star);
[pred_f_star, var_f_star] = model.predict_f(X_star);

fprintf(1,'Relative L2 error u: %e\n', (norm(pred_u_star-u_star,2)/norm(u_star,2)));
fprintf(1,'Relative L2 error f: %e\n', (norm(pred_f_star-f_star,2)/norm(f_star,2)));

%% Plot results
color = [55,126,184]/255;

fig = figure(1);
set(fig,'units','normalized','outerposition',[0 0 1 1])

if N_u > 0
    clear h;
    clear leg;
    if N_f > 0
        subplot(1,2,1)
    end
    hold
    h(1) = plot(X_star, u_star,'k','LineWidth',2);
    h(2) = plot(x_u, y_u,'kx','MarkerSize',14, 'LineWidth',2);
    h(3) = plot(X_star,pred_u_star,'b--','LineWidth',3);
    [l,h(4)] = boundedline(X_star, pred_u_star, 2.0*sqrt(var_u_star), ':','alpha','cmap', color);
    outlinebounds(l,h(4));
    
    leg{1} = '$u(x)$';
    leg{2} = sprintf('%d training data', N_u);
    leg{3} = '$\overline{u}(x)$'; leg{4} = 'Two standard deviations';
    
    hl = legend(h,leg,'Location','southwest');
    legend boxoff
    set(hl,'Interpreter','latex')
    xlabel('$x$')
    ylabel('$u(x)$, $\overline{u}(x)$')
    if N_f > 0
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

if N_f > 0
    
    clear h;
    clear leg;
    if N_u > 0
        subplot(1,2,2)
    end
    hold
    h(1) = plot(X_star, f_star,'k','LineWidth',2);
    h(2) = plot(x_f, y_f,'kx','MarkerSize',14, 'LineWidth',2);
    h(3) = plot(X_star,pred_f_star,'b--','LineWidth',3);
    [l,h(4)] = boundedline(X_star, pred_f_star, 2.0*sqrt(var_f_star), ':', 'alpha','cmap', color);
    outlinebounds(l,h(4));
    
    
    leg{1} = '$f(x)$';
    leg{2} = sprintf('%d training data', N_f);
    leg{3} = '$\overline{f}(x)$'; leg{4} = 'Two standard deviations';
    
    hl = legend(h,leg,'Location','southwest');
    legend boxoff
    set(hl,'Interpreter','latex')
    xlabel('$x$')
    ylabel('$f(x), \overline{f}(x)$')
    if N_u > 0
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

end