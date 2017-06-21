classdef GP
    properties
        X, y, D, N, hyp, jitter
    end
    
    methods
        function obj = GP(X, y)
            [obj.N, obj.D] = size(X);
            obj.X = X;
            obj.y = y;
            obj.hyp = [log(1) log(0.01*ones(1,obj.D)) -5];
            
            obj.jitter = 1e-8;
            
            fprintf('Total number of parameters: %d\n', length(obj.hyp));
        end
        
        function [NLML,D_NLML] = likelihood(obj, hyp)
            
            sigma_eps = exp(hyp(end));
            
            K = kernel(obj.X, obj.X, hyp(1:obj.D+1),0) + eye(obj.N)*sigma_eps;
            K = K + eye(obj.N).*obj.jitter;
            
            % Cholesky factorisation
            [L,p]=chol(K,'lower');
            
            if p > 0
                fprintf(1,'Covariance is ill-conditioned\n');
            end
            
            alpha = L'\(L\obj.y);
            NLML = 0.5*obj.y'*alpha + sum(log(diag(L))) + log(2*pi)*obj.N/2;
            
            D_NLML = 0*hyp;
            Q =  L'\(L\eye(obj.N)) - alpha*alpha';
            for i=1:obj.D+1
                DK = kernel(obj.X, obj.X, hyp(1:obj.D+1),i);
                
                D_NLML(i) = trace(Q*DK)/2;
            end
            
            D_NLML(end) = sigma_eps*trace(Q)/2;
        end
        
        function obj = train(obj)
            options = optimoptions('fminunc','GradObj','on','Display','iter',...
                'Algorithm','trust-region','Diagnostics','on','DerivativeCheck','on',...
                'FinDiffType','central');
            obj.hyp = fminunc(@obj.likelihood,obj.hyp,options);
        end
        
        function [mean_star, var_star] = predict(obj, x_star)
            
            sigma_eps = exp(obj.hyp(end));
            
            K = kernel(obj.X, obj.X, obj.hyp(1:obj.D+1),0) + eye(obj.N)*sigma_eps;
            K = K + eye(obj.N).*obj.jitter;
            
            % Cholesky factorisation
            [L,p]=chol(K,'lower');
            
            if p > 0
                fprintf(1,'Covariance is ill-conditioned\n');
            end
            
            psi = kernel(x_star, obj.X, obj.hyp(1:obj.D+1),0);
            
            % calculate prediction
            mean_star = psi*(L'\(L\obj.y));
            
            var_star = kernel(x_star, x_star, obj.hyp(1:obj.D+1),0) ...
                - psi*(L'\(L\psi'));
            
            var_star = abs(diag(var_star));
        end
    end
end

function K = kernel(X, Xp, hyp, i)
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