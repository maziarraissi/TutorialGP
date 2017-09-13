classdef Multifidelity_GP
    properties
        X_L, y_L, X_H, y_H, D, N_L, N_H, hyp
    end
    
    methods
        function obj = Multifidelity_GP(X_L, y_L, X_H, y_H)
            [obj.N_L, obj.D] = size(X_L);
            obj.N_H = size(X_H,1);
            obj.X_L = X_L;
            obj.y_L = y_L;
            obj.X_H = X_H;
            obj.y_H = y_H;
            
            hyp_L = [log(1) log(0.1*ones(1,obj.D))];
            hyp_H = [log(1) log(0.1*ones(1,obj.D))];
            rho = 1;
            obj.hyp = [hyp_L hyp_H rho -4 -4];
                        
            fprintf('Total number of parameters: %d\n', length(obj.hyp));
        end
        
        function [NLML,D_NLML] = likelihood(obj, hyp)
            
            y=[obj.y_L;obj.y_H];
            
            sigma_eps_H = exp(hyp(end));
            sigma_eps_L = exp(hyp(end-1));
            rho = hyp(end-2);
            
            N = obj.N_L + obj.N_H;
            
            K_LL = kernel(obj.X_L, obj.X_L, hyp(1:obj.D+1),0);
            K_LH = rho*kernel(obj.X_L, obj.X_H, hyp(1:obj.D+1),0);
            K_HL = rho*kernel(obj.X_H, obj.X_L, hyp(1:obj.D+1),0);
            K_HH = (rho^2)*kernel(obj.X_H, obj.X_H, hyp(1:obj.D+1),0) + ...
                kernel(obj.X_H, obj.X_H, hyp(obj.D+2:2*obj.D+2),0);
            
            K_LL = K_LL + eye(obj.N_L).*sigma_eps_L;
            K_HH = K_HH + eye(obj.N_H).*sigma_eps_H;
            
            K = [K_LL K_LH;
                K_HL K_HH];
             
            % Cholesky factorisation
            L = jit_chol(K);
            
            alpha = L'\(L\y);
            NLML = 0.5*y'*alpha + sum(log(diag(L))) + log(2*pi)*N/2;
            
            %% Derivatives
            D_NLML = 0*hyp;
            Q =  L'\(L\eye(N)) - alpha*alpha';
            for i=1:obj.D+1
                DK_LL = kernel(obj.X_L, obj.X_L, hyp(1:obj.D+1),i);
                DK_LH = rho*kernel(obj.X_L, obj.X_H, hyp(1:obj.D+1),i);
                DK_HL = rho*kernel(obj.X_H, obj.X_L, hyp(1:obj.D+1),i);
                DK_HH = rho^2*kernel(obj.X_H, obj.X_H, hyp(1:obj.D+1),i);
                
                DK = [DK_LL DK_LH;
                    DK_HL DK_HH];
                D_NLML(i) = sum(sum(Q.*DK))/2;
            end
            
            for i=obj.D+2:2*obj.D+2
                DK_LL = zeros(obj.N_L,obj.N_L);
                DK_LH = zeros(obj.N_L,obj.N_H);
                DK_HL = zeros(obj.N_H,obj.N_L);
                DK_HH = kernel(obj.X_H, obj.X_H, hyp(obj.D+2:2*obj.D+2),i-(obj.D+1));
                
                DK = [DK_LL DK_LH;
                    DK_HL DK_HH];
                D_NLML(i) = sum(sum(Q.*DK))/2;
            end
            
            DK_LL = zeros(obj.N_L,obj.N_L);
            DK_LH = kernel(obj.X_L, obj.X_H, hyp(1:obj.D+1),0);
            DK_HL = kernel(obj.X_H, obj.X_L, hyp(1:obj.D+1),0);
            DK_HH = (2*rho)*kernel(obj.X_H, obj.X_H, hyp(1:obj.D+1),0);
            DK = [DK_LL DK_LH;
                DK_HL DK_HH];
            
            D_NLML(end-2) = sum(sum(Q.*DK))/2;
            
            D_NLML(end-1) = sigma_eps_L*trace(Q(1:obj.N_L,1:obj.N_L))/2;
            D_NLML(end) = sigma_eps_H*trace(Q(obj.N_L+1:end,obj.N_L+1:end))/2;
        end
        
        function obj = train(obj)
            options = optimoptions('fminunc','GradObj','on','Display','iter',...
                'Algorithm','trust-region','Diagnostics','on','DerivativeCheck','on',...
                'FinDiffType','central');
            obj.hyp = fminunc(@obj.likelihood,obj.hyp,options);
        end
        
        function [mean_star, var_star] = predict_H(obj, x_star)
            
            y=[obj.y_L;obj.y_H];
            
            sigma_eps_H = exp(obj.hyp(end));
            sigma_eps_L = exp(obj.hyp(end-1));
            rho = obj.hyp(end-2);
            
            N = obj.N_L + obj.N_H;
            
            K_LL = kernel(obj.X_L, obj.X_L, obj.hyp(1:obj.D+1),0);
            K_LH = rho*kernel(obj.X_L, obj.X_H, obj.hyp(1:obj.D+1),0);
            K_HL = rho*kernel(obj.X_H, obj.X_L, obj.hyp(1:obj.D+1),0);
            K_HH = (rho^2)*kernel(obj.X_H, obj.X_H, obj.hyp(1:obj.D+1),0) + ...
                kernel(obj.X_H, obj.X_H, obj.hyp(obj.D+2:2*obj.D+2),0);
            
            K_LL = K_LL + eye(obj.N_L).*sigma_eps_L;
            K_HH = K_HH + eye(obj.N_H).*sigma_eps_H;
            
            K = [K_LL K_LH;
                K_HL K_HH];
            
            % Cholesky factorisation
            L = jit_chol(K);
            
            psi1 = rho*kernel(x_star, obj.X_L, obj.hyp(1:obj.D+1),0);
            psi2 = rho^2*kernel(x_star, obj.X_H, obj.hyp(1:obj.D+1),0) + ...
                kernel(x_star, obj.X_H, obj.hyp(obj.D+2:2*obj.D+2),0);
            psi = [psi1 psi2];
            
            % calculate prediction
            mean_star = psi*(L'\(L\y));
            
            var_star = rho^2*kernel(x_star, x_star, obj.hyp(1:obj.D+1),0) ...
                + kernel(x_star, x_star, obj.hyp(obj.D+2:2*obj.D+2),0) ...
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