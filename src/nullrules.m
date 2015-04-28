function [us] = nullrules(x)
n = numel(x);
% Vandemonde matrix
V = fliplr(vander( x ))'; % x = nodes of quadrature rule
 
% Null-rules
us = zeros(numel(x),numel(x)-1);
for m = 1:n-1
    u = sum(null( V(1:end-m,:) ),2);
% Orthogonalise to previous rules
    for i=1:m-1
        u = u - dot(u,us(:,i))*us(:,i);
    end
% Make equaly strong
    u = u ./norm(u);
    us(:,m) = u;
end
