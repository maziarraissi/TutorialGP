addpath /users/mraissi/export_fig/
set(0,'defaulttextinterpreter','latex')
close all

if D == 1
    n_star = 200;
    x_star = linspace(lb(1), ub(1), n_star)';
    
    [mean_f2_star, var_f2_star] = predictor_f2(x_star);
    
    f2_star = f2(x_star);
    f1_star = f1(x_star);
    
    fprintf(1,'Relative L2 error f: %e\n', (norm(mean_f2_star-f2_star,2)/norm(f2_star,2)));
    
    color = [55,126,184]/255;
    
    fig = figure(1);
    set(fig,'units','normalized','outerposition',[0 0 1 1])
    
    clear h;
    clear leg;
    hold
    h(1) = plot(x_star, f2_star,'k','LineWidth',2);
    h(2) = plot(ModelInfo.X2, ModelInfo.y2,'kx','MarkerSize',14, 'LineWidth',2);
    h(3) = plot(x_star,mean_f2_star,'b--','LineWidth',3);
    [l,h(4)] = boundedline(x_star, mean_f2_star, 2.0*sqrt(var_f2_star), ':', 'alpha','cmap', color);
    outlinebounds(l,h(4));
    h(5) = plot(x_star, f1_star,'k:','LineWidth',2);
    h(6) = plot(ModelInfo.X1, ModelInfo.y1,'k+','MarkerSize',14, 'LineWidth',2);
    
    
    leg{1} = '$f_2(x)$';
    leg{2} = sprintf('%d high-fidelity training data', n2);
    leg{3} = '$\overline{f}_2(x)$'; leg{4} = 'Two standard deviations';
    leg{5} = '$f_1(x)$';
    leg{6} = sprintf('%d low-fidelity training data', n1);
    
    hl = legend(h,leg,'Location','northwestoutside');
    legend boxoff
    set(hl,'Interpreter','latex')
    xlabel('$x$')
    ylabel('$f_1(x), f_2(x)$')
    title('A');
    
    axis square
    ylim(ylim + [-diff(ylim)/10 0]);
    xlim(xlim + [-diff(xlim)/10 0]);
    set(gca,'FontSize',16);
    set(gcf, 'Color', 'w');
    
    
    axes('Position',[0.05 0.35 .3 .3])
    box on
    hold
    h(1) = plot(f1_star, f2_star,'b','LineWidth',3);
    xlabel('$f_{1}(x)$');
    ylabel('$f_{2}(x)$');
    title('B -- Cross-correlation');
    axis square
    set(gca,'FontSize',16);
    set(gca,'Xtick',[]);
    set(gca,'Ytick',[]);
    set(gcf, 'Color', 'w');
    
    if save_plots  == 1
        export_fig ./Figs/solution.png -r300
    end
    
elseif D == 2
    
    nn = 50;
    x1 = linspace(lb(1), ub(1), nn)';
    x2 = linspace(lb(2), ub(2), nn)';
    [Xplot, Yplot] = meshgrid(x1,x2);
    x_star = reshape([Xplot Yplot], nn^2, 2);
    n_star = size(x_star,1);
    
    [mean_f2_star, var_f2_star] = predictor_u(x_star);
    [pred_f_star, var_f2_star] = predictor_f(x_star);
    
    u_star = u(x_star);
    f2_star = f(x_star);
    
    fprintf(1,'Relative L2 error u: %e\n', (norm(mean_f2_star-u_star,2)/norm(u_star,2)));
    fprintf(1,'Relative L2 error f: %e\n', (norm(pred_f_star-f2_star,2)/norm(f2_star,2)));
    
    u_star_plot = griddata(x_star(:,1),x_star(:,2),u_star,Xplot,Yplot,'cubic');
    pred_u_star_plot = griddata(x_star(:,1),x_star(:,2),mean_f2_star,Xplot,Yplot,'cubic');
    var_u_star_plot = griddata(x_star(:,1),x_star(:,2),var_f2_star,Xplot,Yplot,'cubic');
    
    f_star_plot = griddata(x_star(:,1),x_star(:,2),f2_star,Xplot,Yplot,'cubic');
    pred_f_star_plot = griddata(x_star(:,1),x_star(:,2),pred_f_star,Xplot,Yplot,'cubic');
    var_f_star_plot = griddata(x_star(:,1),x_star(:,2),var_f2_star,Xplot,Yplot,'cubic');
    
    fig = figure(1);
    set(fig,'units','normalized','outerposition',[0 0 1 1])
    clf
    clear h;
    clear leg;
    subplot(1,2,1)
    hold
    
    s1 = surf(Xplot,Yplot,u_star_plot);
    set(s1,'FaceColor',[100,100,100]/255,'EdgeColor',[100,100,100]/255, 'LineWidth',0.001,'FaceAlpha',0.1);
    
    plot3(ModelInfo.x_u(:,1), ModelInfo.x_u(:,2), ModelInfo.y_u,'s', 'MarkerEdgeColor','k', 'MarkerSize',10,'LineWidth',2);
    view(3)
    
    leg = cell(1,2);
    if time_dep == 1
        leg{1} = '$u(t,x)$';
    else
        leg{1} = '$u(x)$';
    end
    leg{2} = sprintf('%d training data',size(ModelInfo.x_u,1));
    
    hl = legend(leg,'Location','south');
    legend boxoff
    set(hl,'Interpreter','latex')
    
    if time_dep ==1
        xlabel('$t$')
        ylabel('$x$')
        zlabel('$u(t,x)$')
    else
        xlabel('$x_1$')
        ylabel('$x_2$')
        zlabel('$u(x)$')
    end
    title('(A)');
    if time_dep == 1
        set(get(gca,'title'),'Position',[0 0 diff(zlim)*(1+0.1)])
    end
    
    axis square
    set(gca,'FontSize',18);
    set(gcf, 'Color', 'w');
    
    clear h;
    clear leg;
    subplot(1,2,2)
    hold
    
    s1 = surf(Xplot,Yplot,f_star_plot);
    set(s1,'FaceColor',[100,100,100]/255,'EdgeColor',[100,100,100]/255, 'LineWidth',0.001,'FaceAlpha',0.1);
    
    plot3(ModelInfo.x_f(:,1), ModelInfo.x_f(:,2), ModelInfo.y_f,'o', 'MarkerEdgeColor','k','MarkerSize',10,'LineWidth',2);
    view(3)
    
    
    if time_dep == 1
        leg{1} = '$f(t,x)$';
    else
        leg{1} = '$f(x)$';
    end
    leg{2} = sprintf('%d training data', size(ModelInfo.x_f,1));
    
    hl = legend(leg,'Location','south');
    legend boxoff
    set(hl,'Interpreter','latex')
    
    if time_dep == 1
        xlabel('$t$')
        ylabel('$x$')
        zlabel('$f(t,x)$')
    else
        xlabel('$x_1$')
        ylabel('$x_2$')
        zlabel('$f(x)$')
    end
    title('(B)')
    if time_dep == 1
        set(get(gca,'title'),'Position',[0 0 diff(zlim)*(1+0.1)])
    end
    
    axis square
    set(gca,'FontSize',18);
    set(gcf, 'Color', 'w');
    
    if save_plots  == 1
        export_fig ./Figs/data.png -r300
    end
    
    fig = figure(2);
    set(fig,'units','normalized','outerposition',[0 0 1 1])
    clf
    clear h
    clear leg
    subplot(1,2,1)
    hold
    contourf(Xplot,Yplot,abs(u_star_plot - pred_u_star_plot),25);
    colormap('cool')
    plot(ModelInfo.x_f(:,1), ModelInfo.x_f(:,2), 'o', 'MarkerEdgeColor','k','MarkerSize',10,'LineWidth',2);
    plot(ModelInfo.x_u(:,1), ModelInfo.x_u(:,2), 's', 'MarkerEdgeColor','k','MarkerSize',10,'LineWidth',2);
    
    
    
    if time_dep == 1
        title('(C) Error $|\overline{u}(t,x) - u(t,x)|$');
    else
        title('(C) Error $|\overline{u}(x) - u(x)|$');
    end
    
    colorbar
    if time_dep == 1
        xlabel('$t$')
        ylabel('$x$')
    else
        xlabel('$x_1$');
        ylabel('$x_2$');
    end
    set(gca,'XTick',[lb(1) (lb(1)+ub(1))/2 ub(1)]);
    set(gca,'YTick',[lb(2) (lb(2)+ub(2))/2 ub(2)]);
    axis square
    set(gca,'FontSize',18);
    set(gcf, 'Color', 'w');
    
    clear h
    clear leg
    subplot(1,2,2)
    hold
    contourf(Xplot,Yplot,abs(f_star_plot - pred_f_star_plot),25);
    plot(ModelInfo.x_f(:,1), ModelInfo.x_f(:,2), 'o', 'MarkerEdgeColor','k','MarkerSize',10,'LineWidth',2);
    plot(ModelInfo.x_u(:,1), ModelInfo.x_u(:,2), 's', 'MarkerEdgeColor','k','MarkerSize',10,'LineWidth',2);
    
    
    
    if time_dep == 1
        title('(D) Error $|\overline{f}(t,x) - f(t,x)|$');
    else
        title('(D) Error $|\overline{f}(x) - f(x)|$');
    end
    
    colorbar
    if time_dep == 1
        xlabel('$t$')
        ylabel('$x$')
    else
        xlabel('$x_1$');
        ylabel('$x_2$');
    end
    set(gca,'XTick',[lb(1) (lb(1)+ub(1))/2 ub(1)]);
    set(gca,'YTick',[lb(2) (lb(2)+ub(2))/2 ub(2)]);
    axis square
    set(gca,'FontSize',18);
    set(gcf, 'Color', 'w');
    
    if save_plots  == 1
        export_fig ./Figs/error.png -r300
    end
    
    fig = figure(3);
    set(fig,'units','normalized','outerposition',[0 0 1 1])
    clf
    clear h
    clear leg
    subplot(1,2,1)
    hold
    contourf(Xplot,Yplot,sqrt(var_u_star_plot),25);
    colormap('cool')
    plot(ModelInfo.x_f(:,1), ModelInfo.x_f(:,2), 'o', 'MarkerEdgeColor','k','MarkerSize',10,'LineWidth',2);
    plot(ModelInfo.x_u(:,1), ModelInfo.x_u(:,2), 's', 'MarkerEdgeColor','k','MarkerSize',10,'LineWidth',2);
    
    
    
    title({'(E) Standard deviation for $u$'});
    colorbar
    if time_dep == 1
        xlabel('$t$')
        ylabel('$x$')
    else
        xlabel('$x_1$')
        ylabel('$x_2$')
    end
    set(gca,'XTick',[lb(1) (lb(1)+ub(1))/2 ub(1)]);
    set(gca,'YTick',[lb(2) (lb(2)+ub(2))/2 ub(2)]);
    axis square
    set(gca,'FontSize',18);
    set(gcf, 'Color', 'w');
    
    clear h
    clear leg
    subplot(1,2,2)
    hold
    contourf(Xplot,Yplot,sqrt(var_f_star_plot),25);
    plot(ModelInfo.x_f(:,1), ModelInfo.x_f(:,2), 'o', 'MarkerEdgeColor','k','MarkerSize',10,'LineWidth',2);
    plot(ModelInfo.x_u(:,1), ModelInfo.x_u(:,2), 's', 'MarkerEdgeColor','k','MarkerSize',10,'LineWidth',2);
    
    
    
    title({'(F) Standard deviation for $f$'});
    colorbar
    if time_dep == 1
        xlabel('$t$')
        ylabel('$x$')
    else
        xlabel('$x_1$')
        ylabel('$x_2$')
    end
    set(gca,'XTick',[lb(1) (lb(1)+ub(1))/2 ub(1)]);
    set(gca,'YTick',[lb(2) (lb(2)+ub(2))/2 ub(2)]);
    axis square
    set(gca,'FontSize',18);
    set(gcf, 'Color', 'w');
    
    if save_plots  == 1
        export_fig ./Figs/variance.png -r300
    end
end
