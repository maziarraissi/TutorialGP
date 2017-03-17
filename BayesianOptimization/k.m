function [ G ] = k( x, y, hyp, i )

logsigma = hyp(1);
logtheta = hyp(2:end);
sqrt_theta = sqrt(exp(logtheta));

G = sq_dist(diag(1./sqrt_theta)*x',diag(1./sqrt_theta)*y');

G = exp(logsigma)*exp(-0.5*G);

if i>1
    d = i-1;
    gg = @(x,y) 0.5*(x-y).^2/exp(logtheta(d));
    G = G.*bsxfun(gg,x(:,d),y(:,d)');
end

end

