function K = k( x, y, hyp, i )

logsigma = hyp(1);
logtheta = hyp(2:end);
sqrt_theta = sqrt(exp(logtheta));

K = sq_dist(diag(1./sqrt_theta)*x',diag(1./sqrt_theta)*y');

K = exp(logsigma)*exp(-0.5*K);

if i>1
    d = i-1;
    gg = @(x,y) 0.5*(x-y).^2/exp(logtheta(d));
    K = K.*bsxfun(gg,x(:,d),y(:,d)');
end

end

