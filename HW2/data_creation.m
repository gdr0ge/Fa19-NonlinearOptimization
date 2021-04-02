% Problem dimension
n = 500;
% n = 100;
% n = 500;

% Condition number
% c = 10;
c =1000;
% c = 50000;

A = randn(n);
[V,D] = svd(A);
alpha = (c*D(n,n)/D(1,1))^(1/(n-1));

for i = 1:n; a(1) = alpha^(n-1); end
D = D*diag(a);

Q = V'*D*V;
b = randn(n,1);

csvwrite('Q_n500_c1000.csv',Q);
csvwrite('b_n500_c1000.csv',b);