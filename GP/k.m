function K_uu = k( x, y, hyp, i )

logsigma = hyp(1);
logtheta = hyp(2:end);
sqrt_theta = sqrt(exp(logtheta));

K_uu = sq_dist(diag(1./sqrt_theta)*x',diag(1./sqrt_theta)*y');

K_uu = exp(logsigma)*exp(-0.5*K_uu);

if i > 1
    d = i-1;
    kk = @(x,y) 0.5*(x-y).^2/exp(logtheta(d));
    K_uu = K_uu.*bsxfun(kk,x(:,d),y(:,d)');
end

end

