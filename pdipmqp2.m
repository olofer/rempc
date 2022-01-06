function rep = pdipmqp2(H,h,C,d,E,f,kmax,epstop,eta,usesparse,useldl,normtyp)
% function rep = pdipmqp2(H,h,C,d,E,f,kmax,epstop,eta,usesparse,useldl,normtyp)
%
% Solves the general quadratic program
%
%  min_z { 0.5*z'*H*z + h'*z }
%     s.t. C*z = d               (QP)
%          E*z <= f
%
% using a primal-dual interior point method based
% on the standard predictor-corrector technique.
%
% The initial point is automatically set.
% All constraints must be present.
% Uses relative-norm stop tolerances.
% Set usesparse = true if the problem is large & sparse.
% useldl = true works in Matlab, but not in Octave.
% normtyp = 'inf' is default.
%

rep = struct;
rep.creator = mfilename();

if nargin < 7 || (nargin >= 7 && isempty(kmax))
  kmax = 100;
  disp(['warning(',mfilename(),'): defaulting kmax=',num2str(kmax)]);
end

if nargin < 8 || (nargin >= 8 && isempty(epstop))
  epstop = 1e-8;
  disp(['warning(',mfilename(),'): defaulting epstop=',num2str(epstop)]);
end

if numel(epstop) == 1
  epstop = ones(4, 1) * epstop;
end

assert(kmax >= 1 && kmax <= 100); % not supposed to ever run very many iters
assert(min(epstop) > 0 && max(epstop) <= 1e-3);

if nargin < 9 || (nargin >= 9 && isempty(eta))
  % dampening factor eta
  eta = 0.95;
  disp(['warning(',mfilename(),'): defaulting eta=',num2str(eta)]);
end

assert(eta > 0 && eta < 1);

if nargin < 10 || (nargin >= 10 && isempty(usesparse))
  usesparse = false;
end

if nargin < 11 || (nargin >= 11 && isempty(useldl))
  isOctave = (exist('OCTAVE_VERSION', 'builtin') ~= 0);
  useldl = ~isOctave;
end

if nargin < 12 || (nargin >= 12 && isempty(normtyp))
  normtyp = 'inf';
end

assert((ischar(normtyp) && strcmp(normtyp, 'inf')) || ...
       (isnumeric(normtyp) && normtyp == 2));

% Do some size-consistency checking
nx = size(H,1); % # primal variables
assert(size(H,2)==nx);
assert(size(h,1)==nx); assert(size(h,2)==1);
ny = size(C,1); % # equality constraints
assert(size(C,2)==nx);
assert(size(d,1)==ny); assert(size(d,2)==1);
nz = size(E,1); % # inequality constraints
assert(size(E,2)==nx);
assert(size(f,1)==nz); assert(size(f,2)==1);

x = zeros(nx,1);
y = zeros(ny,1);
z = ones(nz,1);
s = ones(nz,1);

e = ones(nz,1);

% Simplified initial residuals
rC = h+E'*z;        % rC = H*x + h + C'*y + E'*z;
rE = -d;            % rE = C*x - d;
rI = s - f;         % rI = E*x + s - f;
rsz = ones(nz,1);   % rsz = s.*z;
mu = sum(rsz)/nz;

thrC = epstop(1) * (1 + norm(h, normtyp));
thrE = epstop(2) * (1 + norm(d, normtyp));
thrI = epstop(3) * (1 + norm(f, normtyp));
thrmu = epstop(4);
k = 0;

use_sparse_lu = ~useldl && usesparse;
use_lu = ~useldl && ~usesparse;

while ( (k < kmax) && ...
  (norm(rC, normtyp) >= thrC || ...
    norm(rE, normtyp) >= thrE || ...
    norm(rI, normtyp) >= thrI || ...
    abs(mu) >= thrmu) )
  % Solve system with a Newton−like method/Factorization
  if usesparse
    lhs = [H,C',E';C,spalloc(ny,ny,0),spalloc(ny,nz,0);E,spalloc(nz,ny,0),spdiags(-s./z,0,nz,nz)];
  else
    lhs = [H,C',E';C,zeros(ny,ny),zeros(ny,nz);E,zeros(nz,ny),diag(-s./z)];
  end
  if use_sparse_lu
    [L, U, P, Q] = lu(lhs);
  elseif use_lu
    [L, U, P] = lu(lhs);
  else
    [L, D, P] = ldl(lhs); % LDL factorization step
  end
  %rhs = [-rC;-rE;-rI+rsz./z];
  rhs = [-rC;-rE;-rI+s];
  if use_sparse_lu
    dxyz_a = Q*(U\(L\(P*rhs)));
  elseif use_lu
    dxyz_a = U\(L\(P*rhs));
  else
    dxyz_a = P*(L'\(D\(L\(P'*rhs))));
  end
  % dx_a = dxyz_a(1:nx);
  % dy_a = dxyz_a((nx+1):(nx+ny));
  dz_a = dxyz_a((nx+ny+1):(nx+ny+nz));
  ds_a = -((rsz+s.*dz_a)./z);
  % Compute alphaa_ff
  alpha_a = 1;
  idx_z = find(dz_a<0);
  if (~isempty(idx_z))
    alpha_a = min(alpha_a,min(-z(idx_z)./dz_a(idx_z)));
  end
  idx_s = find(ds_a<0);
  if (~isempty(idx_s))
    alpha_a = min(alpha_a,min(-s(idx_s)./ds_a(idx_s)));
  end
  % Compute the affine duality gap
  mu_a = ((z+alpha_a*dz_a)'*(s+alpha_a*ds_a))/nz;
  % Compute the centering parameter
  sigma = (mu_a/mu)^3;
  % Solve system again (perturbed rhs)
  rsz = rsz + ds_a.*dz_a-sigma*mu*e;
  rhs = [-rC;-rE;-rI+rsz./z];
  if use_sparse_lu
    dxyz = Q*(U\(L\(P*rhs)));
  elseif use_lu
    dxyz = U\(L\(P*rhs));
  else
    dxyz = P*(L'\(D\(L\(P'*rhs))));
  end
  dx = dxyz(1:nx);
  dy = dxyz((nx+1):(nx+ny));
  dz = dxyz((nx+ny+1):(nx+ny+nz));
  ds = -((rsz+s.*dz)./z);
  % Compute alpha
  alpha = 1;
  idx_z = find(dz<0);
  if (~isempty(idx_z))
    alpha = min(alpha,min(-z(idx_z)./dz(idx_z)));
  end
  idx_s = find(ds<0);
  if (~isempty(idx_s))
    alpha = min(alpha,min(-s(idx_s)./ds(idx_s)));
  end
  ea = eta*alpha;
  % Update x, y, z, s
  x = x + ea*dx;
  y = y + ea*dy;
  z = z + ea*dz;
  s = s + ea*ds;
  k = k + 1;
  % Update rhs
  rC = H*x + h + C'*y + E'*z;
  rE = C*x - d;
  rI = E*x - f + s;
  rsz = s.*z;
  mu = sum(rsz)/nz;
end

% Output
rep.x = x;
rep.fx = 0.5*x'*H*x + h'*x;
rep.iters = k;
rep.epstop = epstop;
rep.maxiters = kmax;
rep.y = y; rep.z = z; rep.s = s;
rep.abs_mu = abs(mu);

end
