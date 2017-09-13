classdef GP
    properties
        X, y, hyp, NLML
    end
    
    methods
        function obj = GP(X, y, hyp)
            obj.X = X;
            obj.y = y;
            
            obj.hyp = hyp;
                        
            fprintf('Total number of parameters: %d\n', length(obj.hyp));
        end
        
        function [NLML,D_NLML] = likelihood(obj, hyp)
            
            X_ = obj.X;
            y_ = obj.y;
            
            N = size(y_,1);
            
            sigma = exp(hyp(end));
            hyp_ = hyp(1:end-1);
            
            K = kernel(X_, X_, hyp_ ,0) + eye(N)*sigma;
            
            % Cholesky factorisation
            L = jit_chol(K);
            
            alpha = L'\(L\obj.y);
            NLML = 0.5*y_'*alpha + sum(log(diag(L))) + log(2*pi)*N/2;
            
            D_NLML = 0*hyp;
            Q =  L'\(L\eye(N)) - alpha*alpha';
            for i=1:length(hyp_)
                DK = kernel(X_, X_, hyp_, i);
                
                D_NLML(i) = sum(sum(Q.*DK))/2;
            end
            
            D_NLML(end) = sigma*trace(Q)/2;
        end
        
        function obj = train(obj, n_iter)                        
            [obj.hyp,~,~] = minimize(obj.hyp, @obj.likelihood, -n_iter);
            obj.NLML = obj.likelihood(obj.hyp);
        end
        
        function [mean_star, var_star] = predict(obj, X_star)
            
            X_ = obj.X;
            y_ = obj.y;
            
            N = size(y_,1);
            
            sigma = exp(obj.hyp(end));
            hyp_ = obj.hyp(1:end-1);
            
            K = kernel(X_, X_, hyp_ ,0) + eye(N)*sigma;
            
            % Cholesky factorisation
            L = jit_chol(K);
            
            psi = kernel(X_star, obj.X, hyp_, 0);
            
            % calculate prediction
            mean_star = psi*(L'\(L\obj.y));
            
            var_star = kernel(X_star, X_star, hyp_, 0) ...
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