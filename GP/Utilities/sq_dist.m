function C = sq_dist(a, b)

  C = bsxfun(@plus,sum(a.*a,1)',bsxfun(@minus,sum(b.*b,1),2*a'*b));
 
  C = max(C,0);          % numerical noise can cause C to negative i.e. C > -1e-14
end
