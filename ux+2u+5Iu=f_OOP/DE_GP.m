classdef DE_GP
    properties
        x_u, y_u, x_f, y_f, D, N_u, N_f, hyp, jitter
    end
    
    methods
        function obj = DE_GP(x_u, y_u, x_f, y_f)
            [obj.N_u, obj.D] = size(x_u);
            obj.N_f = size(x_f,1);
            obj.x_u = x_u;
            obj.y_u = y_u;
            obj.x_f = x_f;
            obj.y_f = y_f;
            
            hyp = [log(1) log(1*ones(1,obj.D))];
            alpha = 1;
            beta = 1;
            obj.hyp = [hyp alpha beta -5 -5];
            
            obj.jitter = 1e-8;
            
            fprintf('Total number of parameters: %d\n', length(obj.hyp));
        end
        
        function [NLML,D_NLML] = likelihood(obj, hyp)
                        
            y=[obj.y_u;obj.y_f];
            
            sigma_n_f = exp(hyp(end));
            sigma_n_u = exp(hyp(end-1));
            
            N = obj.N_u + obj.N_f;
            
            K_uu = k_uu(obj.x_u, obj.x_u, hyp(1:obj.D+1),0) + eye(obj.N_u).*sigma_n_u;
            K_uf = k_uf(obj.x_u, obj.x_f, hyp(1:end-2),0);
            K_fu = K_uf';
            K_ff = k_ff(obj.x_f, obj.x_f, hyp(1:end-2),0) + eye(obj.N_f).*sigma_n_f;
            
            K = [K_uu K_uf;
                K_fu K_ff];
            
            K = K + eye(N).*obj.jitter;
            
            % Cholesky factorisation
            [L,p]=chol(K,'lower');
            
            if p > 0
                fprintf(1,'Covariance is ill-conditioned\n');
            end
            
            alpha = L'\(L\y);
            NLML = 0.5*y'*alpha + sum(log(diag(L))) + log(2*pi)*N/2;
            
            
            D_NLML = 0*hyp;
            Q =  L'\(L\eye(N)) - alpha*alpha';
            for i=1:obj.D+1
                DK_uu = k_uu(obj.x_u, obj.x_u, hyp(1:obj.D+1),i);
                DK_uf = k_uf(obj.x_u, obj.x_f, hyp(1:end-2),i);
                DK_fu = DK_uf';
                DK_ff = k_ff(obj.x_f, obj.x_f, hyp(1:end-2),i);
                
                DK = [DK_uu DK_uf;
                    DK_fu DK_ff];
                
                D_NLML(i) = sum(sum(Q.*DK))/2;
            end
            
            for i=obj.D+2:length(hyp)-2
                DK_uu = zeros(obj.N_u,obj.N_u);
                DK_uf = k_uf(obj.x_u, obj.x_f, hyp(1:end-2),i);
                DK_fu = DK_uf';
                DK_ff = k_ff(obj.x_f, obj.x_f, hyp(1:end-2),i);
                
                DK = [DK_uu DK_uf;
                    DK_fu DK_ff];
                
                D_NLML(i) = sum(sum(Q.*DK))/2;
            end
            
            D_NLML(end-1) = sigma_n_u*trace(Q(1:obj.N_u,1:obj.N_u))/2;
            D_NLML(end) = sigma_n_f*trace(Q(obj.N_u+1:end,obj.N_u+1:end))/2;
        end
        
        function obj = train(obj)
            options = optimoptions('fminunc','GradObj','on','Display','iter',...
                'Algorithm','trust-region','Diagnostics','on','DerivativeCheck','on',...
                'FinDiffType','central');
            obj.hyp = fminunc(@obj.likelihood,obj.hyp,options);
        end
        
        function [pred_u_star, var_u_star] = predict_u(obj, x_star)
            
            y=[obj.y_u;obj.y_f];
            
            sigma_n_f = exp(obj.hyp(end));
            sigma_n_u = exp(obj.hyp(end-1));
            
            N = obj.N_u + obj.N_f;
            
            K_uu = k_uu(obj.x_u, obj.x_u, obj.hyp(1:obj.D+1),0) + eye(obj.N_u).*sigma_n_u;
            K_uf = k_uf(obj.x_u, obj.x_f, obj.hyp(1:end-2),0);
            K_fu = K_uf';
            K_ff = k_ff(obj.x_f, obj.x_f, obj.hyp(1:end-2),0) + eye(obj.N_f).*sigma_n_f;
            
            K = [K_uu K_uf;
                K_fu K_ff];
            
            K = K + eye(N).*obj.jitter;
            
            % Cholesky factorisation
            [L,p]=chol(K,'lower');
            
            if p > 0
                fprintf(1,'Covariance is ill-conditioned\n');
            end
            
            psi1 = k_uu(x_star, obj.x_u, obj.hyp(1:obj.D+1),0);
            psi2 = k_uf(x_star, obj.x_f, obj.hyp(1:end-2),0);
            psi = [psi1 psi2];

            % calculate prediction
            pred_u_star = psi*(L'\(L\y));

            var_u_star = k_uu(x_star, x_star, obj.hyp(1:obj.D+1),0) ...
                - psi*(L'\(L\psi'));

            var_u_star = abs(diag(var_u_star));
        end
        
        function [pred_f_star, var_f_star] = predict_f(obj, x_star)
            
            y=[obj.y_u;obj.y_f];
            
            sigma_n_f = exp(obj.hyp(end));
            sigma_n_u = exp(obj.hyp(end-1));
            
            N = obj.N_u + obj.N_f;
            
            K_uu = k_uu(obj.x_u, obj.x_u, obj.hyp(1:obj.D+1),0) + eye(obj.N_u).*sigma_n_u;
            K_uf = k_uf(obj.x_u, obj.x_f, obj.hyp(1:end-2),0);
            K_fu = K_uf';
            K_ff = k_ff(obj.x_f, obj.x_f, obj.hyp(1:end-2),0) + eye(obj.N_f).*sigma_n_f;
            
            K = [K_uu K_uf;
                K_fu K_ff];
            
            K = K + eye(N).*obj.jitter;
            
            % Cholesky factorisation
            [L,p]=chol(K,'lower');
            
            if p > 0
                fprintf(1,'Covariance is ill-conditioned\n');
            end
            
            psi1 = k_uf(obj.x_u, x_star, obj.hyp(1:end-2),0)';
            psi2 = k_ff(x_star, obj.x_f, obj.hyp(1:end-2),0);
            psi = [psi1 psi2];

            % calculate prediction
            pred_f_star = psi*(L'\(L\y));

            var_f_star = k_ff(x_star, x_star, obj.hyp(1:end-2),0) ...
                - psi*(L'\(L\psi'));

            var_f_star = abs(diag(var_f_star));
        end
    end
end

%% Kernels
function K = k_uu(X, Xp, hyp, i)
    logsigma = hyp(1);
    logtheta = hyp(2:end);
    sqrt_theta = sqrt(exp(logtheta));

    function C = sq_dist(a, b)
        C = bsxfun(@plus,sum(a.*a,1)',bsxfun(@minus,sum(b.*b,1),2*a'*b));
        C = max(C,0);          % numerical noise can cause C to negative i.e. C > -1e-14
    end

    K = sq_dist(diag(1./sqrt_theta)*X',diag(1./sqrt_theta)*Xp');

    K = exp(logsigma)*exp(-0.5*K);

    if i > 1
        d = i-1;
        kk = @(x,y) 0.5*(x-y).^2/exp(logtheta(d));
        K = K.*bsxfun(kk,X(:,d),Xp(:,d)');
    end
end

function K_uf = k_uf(x, y, hyp,i)

    logsigma = hyp(1);
    logtheta = hyp(2);
    alpha = hyp(3);
    beta = hyp(4);

    if i == 0 || i == 1
        kk = @(x,y) exp(1).^(logsigma+(-1).*logtheta+(-1/2).*exp(1).^((-1).*logtheta) ...
            .*(x+(-1).*y).^2).*(x+(-1).*y+exp(1).^logtheta.*alpha)+exp(1).^( ...
            logsigma+(1/2).*logtheta).*((1/2).*pi).^(1/2).*beta.*(erf(2.^(-1/2).* ...
            exp(1).^((-1/2).*logtheta).*x)+erf(2.^(-1/2).*exp(1).^((-1/2).* ...
            logtheta).*((-1).*x+y)));

        K_uf = bsxfun(kk,x,y');

    elseif i == 2
        kk = @(x,y) (1/4).*exp(1).^(logsigma+(-1/2).*exp(1).^((-1).*logtheta).*x.^2).* ...
            ((-2).*x.*beta+2.*exp(1).^((-2).*logtheta+(-1/2).*exp(1).^((-1).*...
            logtheta).*y.*((-2).*x+y)).*(x+(-1).*y).*((x+(-1).*y).^2+exp(1)...
            .^logtheta.*((-2)+x.*alpha+(-1).*y.*alpha+exp(1).^logtheta.*beta))+exp(1).^(( ...
            1/2).*(logtheta+exp(1).^((-1).*logtheta).*x.^2)).*(2.*pi).^(1/2).* ...
            beta.*(erf(2.^(-1/2).*exp(1).^((-1/2).*logtheta).*x)+(-1).*erf(2.^( ...
            -1/2).*exp(1).^((-1/2).*logtheta).*(x+(-1).*y))));

        K_uf = bsxfun(kk,x,y');

    elseif i == 3
        kk = @(x,y) exp(1).^(logsigma+(-1/2).*exp(1).^((-1).*logtheta).*(x+(-1).*y) ...
            .^2);
        K_uf = bsxfun(kk,x,y');

    elseif i == 4
        kk = @(x,y) exp(1).^(logsigma+(1/2).*logtheta).*((1/2).*pi).^(1/2).*(erf(2.^( ...
            -1/2).*exp(1).^((-1/2).*logtheta).*x)+erf(2.^(-1/2).*exp(1).^(( ...
            -1/2).*logtheta).*((-1).*x+y)));

        K_uf = bsxfun(kk,x,y');
    end

end

function K_ff = k_ff(x, y, hyp,i)

    logsigma = hyp(1);
    logtheta = hyp(2);
    alpha = hyp(3);
    beta = hyp(4);

    if i == 0 || i == 1

        kk = @(x,y) (1/2).*exp(1).^(logsigma+(-2).*logtheta+(-1/2).*exp(1).^((-1).* ...
            logtheta).*(x.^2+y.^2)).*(2.*exp(1).^(2.*logtheta+(1/2).*exp(1).^( ...
            (-1).*logtheta).*x.^2).*beta.*(1+exp(1).^logtheta.*beta)+(-2).*exp(1).^( ...
            exp(1).^((-1).*logtheta).*x.*y).*((x+(-1).*y).^2+exp(1) ...
            .^logtheta.*((-1)+exp(1).^logtheta.*((-1).*alpha.^2+beta.*(2+exp(1) ...
            .^logtheta.*beta))))+exp(1).^(2.*logtheta+(1/2).*exp(1).^((-1).* ...
            logtheta).*y.^2).*beta.*(2+2.*exp(1).^logtheta.*beta+exp(1).^((1/2).*( ...
            logtheta+exp(1).^((-1).*logtheta).*x.^2)).*((-2).*exp(1).^((1/2).* ...
            logtheta).*beta+(2.*pi).^(1/2).*((alpha+x.*beta).*erf(2.^(-1/2).*exp(1).^(( ...
            -1/2).*logtheta).*x)+((-1).*x+y).*beta.*erf(2.^(-1/2).*exp(1).^(( ...
            -1/2).*logtheta).*(x+(-1).*y))+(alpha+y.*beta).*erf(2.^(-1/2).*exp(1).^(( ...
            -1/2).*logtheta).*y)))));

        K_ff = bsxfun(kk,x,y');

    elseif i==2

        kk = @(x,y) (1/4).*exp(1).^((-3).*logtheta+(-1/2).*exp(1).^((-1).*logtheta).* ...
            x.^2).*((-2).*exp(1).^(logsigma+(-1/2).*exp(1).^((-1).*logtheta).* ...
            y.^2).*((-1).*exp(1).^(2.*logtheta+(1/2).*exp(1).^((-1).*logtheta) ...
            .*x.^2).*beta.*(y.^2+(-1).*exp(1).^logtheta.*y.*alpha+2.*exp(1).^(2.* ...
            logtheta).*beta)+exp(1).^(exp(1).^((-1).*logtheta).*x.*y).*((-5).* ...
            exp(1).^logtheta.*(x+(-1).*y).^2+(x+(-1).*y).^4+exp(1).^(2.* ...
            logtheta).*(2+(-1).*(x+(-1).*y).^2.*(alpha.^2+(-2).*beta))+2.*exp(1).^( ...
            4.*logtheta).*beta.^2))+exp(1).^(logsigma+2.*logtheta).*beta.*(2.*x.^2+( ...
            -2).*exp(1).^logtheta.*x.*alpha+(-4).*exp(1).^(2.*logtheta).*((-1)+...
            exp(1).^((1/2).*exp(1).^((-1).*logtheta).*x.^2)).*beta+exp(1).^((1/2) ...
            .*(3.*logtheta+exp(1).^((-1).*logtheta).*x.^2)).*(2.*pi).^(1/2).*( ...
            (alpha+x.*beta).*erf(2.^(-1/2).*exp(1).^((-1/2).*logtheta).*x)+((-1).*x+ ...
            y).*beta.*erf(2.^(-1/2).*exp(1).^((-1/2).*logtheta).*(x+(-1).*y))+(alpha+ ...
            y.*beta).*erf(2.^(-1/2).*exp(1).^((-1/2).*logtheta).*y))));

        K_ff = bsxfun(kk,x,y');

    elseif i==3
        kk = @(x,y) 2.*exp(1).^(logsigma+(-1/2).*exp(1).^((-1).*logtheta).*(x+(-1).*y) ...
            .^2).*alpha+exp(1).^(logsigma+(1/2).*logtheta).*((1/2).*pi).^(1/2).* ...
            beta.*(erf(2.^(-1/2).*exp(1).^((-1/2).*logtheta).*x)+erf(2.^(-1/2).* ...
            exp(1).^((-1/2).*logtheta).*y));

        K_ff = bsxfun(kk,x,y');

    elseif i ==4
        kk = @(x,y) (1/2).*exp(1).^logsigma.*(2.*(exp(1).^((-1/2).*exp(1).^((-1).* ...
            logtheta).*x.^2)+(-2).*exp(1).^((-1/2).*exp(1).^((-1).*logtheta).* ...
            (x+(-1).*y).^2)+exp(1).^((-1/2).*exp(1).^((-1).*logtheta).*y.^2)+( ...
            -2).*exp(1).^logtheta.*beta+2.*exp(1).^(logtheta+(-1/2).*exp(1).^(( ...
            -1).*logtheta).*x.^2).*beta+(-2).*exp(1).^(logtheta+(-1/2).*exp(1).^( ...
            (-1).*logtheta).*(x+(-1).*y).^2).*beta+2.*exp(1).^(logtheta+(-1/2).* ...
            exp(1).^((-1).*logtheta).*y.^2).*beta)+exp(1).^((1/2).*logtheta).*( ...
            2.*pi).^(1/2).*((alpha+2.*x.*beta).*erf(2.^(-1/2).*exp(1).^((-1/2).* ...
            logtheta).*x)+2.*((-1).*x+y).*beta.*erf(2.^(-1/2).*exp(1).^((-1/2).* ...
            logtheta).*(x+(-1).*y))+(alpha+2.*y.*beta).*erf(2.^(-1/2).*exp(1).^(( ...
            -1/2).*logtheta).*y)));

        K_ff = bsxfun(kk,x,y');
    end

end