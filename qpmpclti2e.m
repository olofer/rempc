function rep = qpmpclti2e(P, solveropts)
% function rep = qpmpclti2e(P, solveropts)
%
% Basic discrete-time MPC quadratic program (QP) solver for LTI systems.
% Time-varying reference trajectory and particular time-varying
% constraints are supported. 
%
% Keeps the quadratic program in large sparse form retaining the
% equality constraints (dynamical evolution equations).
% Internally uses the function spalloc to create the QP.
% 
% Supports special terminal cost P-field: Wn and Qxn
% Can optionally handle "soft constraints" by specification 
% of which rows of the inequality constraints can be relaxed
% and at what cost (see further below).
%
% solveropts (optional, can be omitted or set to []) is a struct
% with fields MaxIter and TolX and others (see code).
%
% Pre-fetch default options struct:
%   solveropts = qpmpclti2e();
%
% The specific MPC problem solved is this:
%
% min sum_(i=0)^(n) (e(i)'*W*e(i)+u(i)'*R*u(i)+x(i)'*Qx*x(i))
% s.t.  x(i+1)=A*x(i)+B*u(i)+w(i), i = 0..n-1
%       y(i)=C*x(i)+D*u(i), e(i)=y(i)-r(i), i=0..n
%       F1*x(i)+F2*u(i)<=f3(i), i=0..n
%       x(0)=x
%
% where u(i) is constrained as either piecewise constant or piecewise
% linear with udim nodes, equidistributed over the horizon n.
%
% Matrices W,Qx,R must form a pos. def. cost function.
% r(i) is a reference trajectory, f3(i) can be used for (simple) 
% time-varying constraint formulations. The inequality constraint must
% not be omitted. (A,B,C,D) is a discrete time LTI system. If C is empty
% then neither of D,W,r are considered in the cost function.
%
% r can be given as an (n+1)-by-ny matrix, or an ny-by-(n+1)-matrix,
% or a column vector of length ny (not time-varying).
% f3 analogous to r, but with size nq instead, nq = #(col) of F1 (and F2).
% Also, w analogous but with dimension nx and n instead.
%
% W, R, and Qx can be given as column vectors (they will be expanded to
% diagonal matrices internally in that case). A scalar will also be
% expanded to a scaling of the identity matrix.
%
% If there are soft constraints then slack terms s(i)>=0 are introduced
% that adds
%
%       sum_(i=0)^(n) (s(i)'*S*s(i))
%
% to the above cost subject to the extended constraints (i=0..n)
%
%       F1*x(i) + F2*u(i) - E*s(i) <= f3(i)
%       -s(i) <= 0
%
% where E has 0/1 entries and row sum 1 and is automatically generated
% based on the P-field: softidx. S is a diagonal matrix generated from
% the P-field: softcost (same dim. as softidx).
%
% An alternative interface is provided by setting P.sc equal to
% a vector of length nq where each positive element indicates 
% the creation of a slack variable (softidx) and its cost (softcost).
% This latter interface is preferred.
%
% The solution is a trajectory (x(i),u(i)[,s(i)]) for i=0..n.
% Only the first control sample u(0) would typically be used
% (returned as rep.u0). 

isOctave = (exist('OCTAVE_VERSION', 'builtin') ~= 0);

if nargout == 1 && nargin == 0
  rep = getDefaultOptions(isOctave);
  return;
end

assert(isstruct(P), 'Expects problem data struct.');

% Need to default the solver options?
if nargin<2 || (nargin==2 && isempty(solveropts))
  solveropts = getDefaultOptions(isOctave);
end

assert(isstruct(solveropts), 'Expects options data struct.');

rep = struct;

n = P.n;    % this is the MPC horizon
assert(n>0);

% Setup local variables and basic dimensional checks
A = P.A; B = P.B; C = P.C; 
nx = size(A,1);
assert(size(A,2)==nx);
nu = size(B,2);
assert(size(B,1)==nx);

OutputIsNonzero=(~isempty(C));
if OutputIsNonzero
    ny = size(C,1);
    assert(ny>=1);
    assert(size(C,2)==nx);
    if isempty(P.D)
        DirectTermIsNonzero=0;
    else
        D = P.D;
        assert(size(D,1)==ny && size(D,2)==nu);
        DirectTermIsNonzero=(norm(D,'fro')>0);
    end
    W=aux_format_square_matrix(P.W,ny);
    % Read in reference trajectory r(i)=0..n
    r=aux_format_signal(P.r,ny,n+1);
end

% Read in cost-function matrices (possibly from column vectors)
R=aux_format_square_matrix(P.R,nu);
Qx=aux_format_square_matrix(P.Qx,nx);

% TODO: handle the simplified special case where F1 and F2 are both empty
% (..meaning that there are no inequality constraints to consider..)

F1 = P.F1;
F2 = P.F2;
if isempty(F1) && ~isempty(F2)
    F1=zeros(size(F2,1),nx);
end
nq = size(F1,1);        % number of inequality contraints per stage
assert(nq>0);
assert(size(F2,1)==nq);
assert(size(F1,2)==nx);
assert(size(F2,2)==nu);

% Read in inequality constraint rhs f3(i), i=0..n
f3=aux_format_signal(P.f3,nq,n+1);
% Read equality constraint (linear dynamics) "offset" term w(i), i=0..n-1
w=aux_format_signal(P.w,nx,n);

% Grab terminal costs (if any)
if isfield(P,'Qxn') && ~isempty(P.Qxn)
    Qxn=aux_format_square_matrix(P.Qxn,nx);
else
    Qxn=Qx;
end
if OutputIsNonzero
    if isfield(P,'Wn') && ~isempty(P.Wn)
        Wn=aux_format_square_matrix(P.Wn,ny);
    else
        Wn=W;
    end
end

% Check whether there are any soft constraints (.sc interface has priority)
hasSoftenedConstraints = false; ns = 0;
if isfield(P,'sc') && ~isempty(P.sc)
    assert(length(P.sc)==nq,'Field sc must contain a vector of length nq.');
    idx = find(P.sc>0);
    hasSoftenedConstraints = ~isempty(idx);
    if hasSoftenedConstraints
        ns = length(idx);
        S = diag(P.sc(idx));
        Es = zeros(nq,ns);
        for jj=1:ns
            Es(idx(jj),jj)=1;
        end
    end
elseif isfield(P,'softidx') && ~isempty(P.softidx)
    assert(isfield(P,'softcost'),'Expected field softcost to exist too.');
    hasSoftenedConstraints = true;
    ns = length(P.softidx);
    assert(all(P.softidx>=1 & P.softidx<=nq),'Soft constraint index is out-of-bounds.');
    assert(length(unique(P.softidx))==ns,'Softened rows cannot be duplicated.');
    assert(ns==length(P.softcost),'Soft constraint data not dimensionally consistent.');
    assert(all(P.softcost>0),'Must specify positive cost for slack variables.')
    S = diag(P.softcost);
    Es = zeros(nq,ns); % setup the indexing matrix that goes into the inequality block..
    for jj=1:ns
        Es(P.softidx(jj),jj)=1;
    end
end

x = P.x(:); % initial state x=x(0)
assert(size(x,1)==nx && size(x,2)==1);

% Compute size of the QP vector variable z
if hasSoftenedConstraints
    nz=nx+nu+ns;
    nqp=(n+1)*nz;
    bigh=zeros(nqp,1);
    IJV=zeros(nz*nz*(n+1),3);
    % Create the cost matrices (H,h,h0): 0.5*z'*H*z + h'*z + h0
    if OutputIsNonzero
        if DirectTermIsNonzero % C!=0 and D!=0
            tmp2=C'*W*D;
            tmp3=D'*W*D;
            tmp5=W*D;
        else % C!=0 and D=0
            tmp2=zeros(nx,nu);
            tmp3=zeros(nu,nu);
            tmp5=zeros(ny,nu);
        end
        tmp1=C'*W*C;
        tmp4=W*C;
        Ht = [  tmp1+Qx,      tmp2,         zeros(nx,ns);
                tmp2',        tmp3+R,       zeros(nu,ns);
                zeros(ns,nx), zeros(ns,nu), S ] * 2;
        % Create h0 and bias term h and sparse bigH
        kk=0; qq=0; ll=0; h0=0;
        for jj=1:n % note: not n+1
            rj=r((qq+1):(qq+ny));
            bigh((kk+1):(kk+nz))=[-2*(rj'*tmp4),-2*(rj'*tmp5),zeros(1,ns)]';
            [iv,jv,vv]=blockvec(kk+1,kk+1,Ht);
            IJV((ll+1):(ll+numel(Ht)),:)=[iv,jv,vv];
            h0=h0+(rj'*W*rj);
            kk=kk+nz; qq=qq+ny; ll=ll+numel(Ht);
        end
        % (possibly) special terminal cost block...        
        if DirectTermIsNonzero % C!=0 and D!=0
            tmp2=C'*Wn*D;
            tmp3=D'*Wn*D;
            tmp5=Wn*D;
        else % C!=0 and D=0
            tmp2=zeros(nx,nu);
            tmp3=zeros(nu,nu);
            tmp5=zeros(ny,nu);
        end
        tmp1=C'*Wn*C;
        tmp4=Wn*C;
        Htn = [ tmp1+Qxn,     tmp2,         zeros(nx,ns);
                tmp2',        tmp3+R,       zeros(nu,ns);
                zeros(ns,nx), zeros(ns,nu), S ] * 2;
        rj=r((qq+1):(qq+ny));
        bigh((kk+1):(kk+nz))=[-2*(rj'*tmp4),-2*(rj'*tmp5),zeros(1,ns)]';
        [iv,jv,vv]=blockvec(kk+1,kk+1,Htn);
        IJV((ll+1):(ll+numel(Ht)),:)=[iv,jv,vv];
        h0=h0+(rj'*Wn*rj);
        bigH=sparse(IJV(:,1),IJV(:,2),IJV(:,3),nqp,nqp);
    else
        % C=0 and D=0; so no W or r used either
        bigh=zeros(nqp,1);
        h0=0;
        Ht = [Qx,zeros(nx,nu),zeros(nx,ns);
              zeros(nu,nx),R,zeros(nu,ns);
              zeros(ns,nx),zeros(ns,nu),S]*2;
        kk=0; ll=0;
        for jj=1:n
            [iv,jv,vv]=blockvec(kk+1,kk+1,Ht);
            IJV((ll+1):(ll+numel(Ht)),:)=[iv,jv,vv];
            kk=kk+nz; ll=ll+numel(Ht);
        end
        Htn = [Qxn,zeros(nx,nu),zeros(nx,ns);
              zeros(nu,nx),R,zeros(nu,ns);
              zeros(ns,nx),zeros(ns,nu),S]*2; % (possibly) special terminal weight Qxn for last block
        [iv,jv,vv]=blockvec(kk+1,kk+1,Htn);
        IJV((ll+1):(ll+numel(Htn)),:)=[iv,jv,vv];
        bigH=sparse(IJV(:,1),IJV(:,2),IJV(:,3),nqp,nqp);
    end
    % Create the inequality constraint matrices (E,f): E*z<=f
    Et = [F1,F2,-Es;zeros(ns,nx),zeros(ns,nu),-eye(ns)];
    bigf=zeros((n+1)*(nq+ns),1);
    assert(numel(Et)==nz*(nq+ns));
    IJV=zeros((n+1)*numel(Et),3);
    kk=0; qq=0; ll=0; oo=0;
    for jj=1:(n+1)
        f3j=f3((qq+1):(qq+nq));
        bigf((kk+1):(kk+nq+ns))=[f3j;zeros(ns,1)];
        [iv,jv,vv]=blockvec(kk+1,ll+1,Et);
        IJV((oo+1):(oo+numel(Et)),:)=[iv,jv,vv];
        kk=kk+(nq+ns);
        qq=qq+nq;
        ll=ll+nz;
        oo=oo+numel(Et);
    end
    bigE=sparse(IJV(:,1),IJV(:,2),IJV(:,3),(nq+ns)*(n+1),nqp);
    % Create the equality constraint matrices (G,g): G*z=g
    bigg=zeros(nx*(n+1),1);
    bigg(1:nx)=x;   % initial state goes here; bigg=[x;w(0);...;w(n-1)]
    kk=nx; qq=0;
    for jj=1:n
        bigg((kk+1):(kk+nx))=w((qq+1):(qq+nx));
        kk=kk+nx; qq=qq+nx;
    end
    IJV=zeros(2*nx*nz*n+nx*nz,3);
    Bt=[eye(nx),zeros(nx,nu),zeros(nx,ns);-A,-B,zeros(nx,ns)];
    kk=0; ll=0; oo=0;
    for jj=1:n
        [iv,jv,vv]=blockvec(kk+1,ll+1,Bt);
        IJV((oo+1):(oo+numel(Bt)),:)=[iv,jv,vv];
        kk=kk+nx;
        ll=ll+nz;
        oo=oo+numel(Bt);
    end
    assert(oo+nx*nz==size(IJV,1)); % last block is a bit different (truncated)
    [iv,jv,vv]=blockvec(kk+1,ll+1,[eye(nx),zeros(nx,nu),zeros(nx,ns)]);
    IJV((oo+1):(oo+nx*nz),:)=[iv,jv,vv];
    bigG=sparse(IJV(:,1),IJV(:,2),IJV(:,3),nx*(n+1),nqp);
else
    % No stage slack variables are needed without soft constraints.
    nz=nx+nu;
    nqp=(n+1)*nz;
    assert(ns==0);
    % Create the cost matrices (H,h,h0): 0.5*z'*H*z + h'*z + h0
    bigh=zeros(nqp,1);
    h0=0;
    IJV=zeros(nz*nz*(n+1),3);
    if OutputIsNonzero
        if DirectTermIsNonzero % C!=0 and D!=0
            tmp2=C'*W*D;
            tmp3=D'*W*D;
            tmp5=W*D;
        else % C!=0 and D=0
            tmp2=zeros(nx,nu);
            tmp3=zeros(nu,nu);
            tmp5=zeros(ny,nu);
        end
        tmp1=C'*W*C;
        tmp4=W*C;
        Ht = [  tmp1+Qx, tmp2;
                tmp2',   tmp3+R]*2;
        % Create h0 and bias term h and sparse bigH
        kk=0; qq=0; ll=0; h0=0;
        for jj=1:n % n+1 is special
            rj=r((qq+1):(qq+ny));
            bigh((kk+1):(kk+nz))=[-2*(rj'*tmp4),-2*(rj'*tmp5)]';
            [iv,jv,vv]=blockvec(kk+1,kk+1,Ht);
            IJV((ll+1):(ll+numel(Ht)),:)=[iv,jv,vv];
            h0=h0+(rj'*W*rj);
            kk=kk+nz; qq=qq+ny; ll=ll+numel(Ht);
        end
        % terminal cost block        
        if DirectTermIsNonzero % C!=0 and D!=0
            tmp2=C'*Wn*D;
            tmp3=D'*Wn*D;
            tmp5=Wn*D;
        else % C!=0 and D=0
            tmp2=zeros(nx,nu);
            tmp3=zeros(nu,nu);
            tmp5=zeros(ny,nu);
        end
        tmp1=C'*Wn*C;
        tmp4=Wn*C;
        Htn = [ tmp1+Qxn, tmp2;
                tmp2',    tmp3+R]*2;
        rj=r((qq+1):(qq+ny));
        bigh((kk+1):(kk+nz))=[-2*(rj'*tmp4),-2*(rj'*tmp5)]';
        [iv,jv,vv]=blockvec(kk+1,kk+1,Htn);
        IJV((ll+1):(ll+numel(Htn)),:)=[iv,jv,vv];
        h0=h0+(rj'*Wn*rj);
        bigH=sparse(IJV(:,1),IJV(:,2),IJV(:,3),nqp,nqp);
    else
        % C=0 and D=0; so no W or r used either
        Ht=[Qx,zeros(nx,nu);zeros(nu,nx),R]*2;
        kk=0; ll=0;
        for jj=1:n % n+1 is (possibly) different here; see below
            [iv,jv,vv]=blockvec(kk+1,kk+1,Ht);
            IJV((ll+1):(ll+numel(Ht)),:)=[iv,jv,vv];
            kk=kk+nz; ll=ll+numel(Ht);
        end
        Htn=[Qxn,zeros(nx,nu);zeros(nu,nx),R]*2;
        [iv,jv,vv]=blockvec(kk+1,kk+1,Htn);
        IJV((ll+1):(ll+numel(Htn)),:)=[iv,jv,vv];
        bigH=sparse(IJV(:,1),IJV(:,2),IJV(:,3),nqp,nqp);
    end
    % Create the inequality constraint matrices (E,f): E*z<=f
    Et = [F1,F2];
    bigf=zeros((n+1)*nq,1);
    assert(numel(Et)==nq*nz);
    IJV=zeros((n+1)*numel(Et),3);
    kk=0; qq=0; ll=0; oo=0; % kk and qq are identical here; delete one of them
    for jj=1:(n+1)
        f3j=f3((qq+1):(qq+nq));
        bigf((kk+1):(kk+nq))=f3j;
        [iv,jv,vv]=blockvec(kk+1,ll+1,Et);
        IJV((oo+1):(oo+numel(Et)),:)=[iv,jv,vv];
        kk=kk+nq;
        qq=qq+nq;
        ll=ll+nz;
        oo=oo+numel(Et);
    end
    bigE=sparse(IJV(:,1),IJV(:,2),IJV(:,3),nq*(n+1),nqp);
    % Create the equality constraint matrices (G,g): G*z=g
    bigg=zeros(nx*(n+1),1);
    bigg(1:nx)=x;   % initial state goes here; bigg=[x;w(0);...;w(n-1)]
    kk=nx; qq=0;
    for jj=1:n
        bigg((kk+1):(kk+nx))=w((qq+1):(qq+nx));
        kk=kk+nx; qq=qq+nx;
    end
    IJV=zeros(2*nx*nz*n+nx*nz,3);
    Bt=[eye(nx),zeros(nx,nu);-A,-B];
    kk=0; ll=0; oo=0;
    for jj=1:n
        [iv,jv,vv]=blockvec(kk+1,ll+1,Bt);
        IJV((oo+1):(oo+numel(Bt)),:)=[iv,jv,vv];
        kk=kk+nx;
        ll=ll+nz;
        oo=oo+numel(Bt);
    end
    assert(oo+nx*nz==size(IJV,1)); % last block is a bit different (truncated)
    [iv,jv,vv]=blockvec(kk+1,ll+1,[eye(nx),zeros(nx,nu)]);
    IJV((oo+1):(oo+nx*nz),:)=[iv,jv,vv];
    bigG=sparse(IJV(:,1),IJV(:,2),IJV(:,3),nx*(n+1),nqp);
end

rep.isOctave = isOctave;
rep.solverprogram = mfilename();
rep.opts = solveropts;

assert(strcmpi(solveropts.solver, 'pdipmqp2') || ...
       strcmpi(solveropts.solver, 'quadprog'), ...
       'Unrecognized QP solver option');

ttt=tic;
if strcmpi(solveropts.solver, 'pdipmqp2')
  optout = pdipmqp2( ...
             bigH, bigh, ...
             bigG, bigg, ...
             bigE, bigf, ...
             solveropts.MaxIter, ...
             solveropts.TolX, ...
             0.95, ...
             true, ...
             ~isOctave);
  z = optout.x;
  fz = optout.fx;
  testObjective = (optout.iters < optout.maxiters);
  iters = optout.iters;
elseif strcmpi(solveropts.solver, 'quadprog')
  % Call quadprog to solve, min (0.5*z'*H*z+h'*z), s.t. E*z<=f, A*x=b
  optopts = optimset(...
      'Algorithm','interior-point-convex',...
      'Display','off',...
      'MaxIter',solveropts.MaxIter,...
      'TolX',solveropts.TolX,...
      'TolFun',solveropts.TolFun,...
      'TolCon',solveropts.TolCon);
  [z,fz,exitqp,outqp] = quadprog(bigH,bigh,bigE,bigf,bigG,bigg,[],[],[],optopts);
  rep.exit_quadprog = exitqp;
  rep.out_quadprog = outqp;
  iters = outqp.iterations;
  testObjective=(exitqp==1);
end
ttt=toc(ttt);

rep.solveriters = iters;
rep.solvertime = ttt;
rep.isConverged = testObjective;

if testObjective % do not create solution output unless it appears converged
    assert(fz+h0>=0);
    % Unpack the solution vector (stacked stage variables 0..n)
    if hasSoftenedConstraints
        % stage variable is [x;u;s]
        xbar=zeros((n+1)*nx,1);
        ubar=zeros((n+1)*nu,1);
        sbar=zeros((n+1)*ns,1);
        qq=0; kk=0; ll=0; oo=0; fzs=0;
        for jj=1:(n+1)
            xbar((qq+1):(qq+nx))=z((kk+1):(kk+nx));
            ubar((ll+1):(ll+nu))=z((kk+nx+1):(kk+nx+nu));
            sbar((oo+1):(oo+ns))=z((kk+nx+nu+1):(kk+nx+nu+ns));
            fzs=fzs+sbar((oo+1):(oo+ns))'*S*sbar((oo+1):(oo+ns)); % sum up the slack cost explicitly (diagnosis)
            qq=qq+nx; kk=kk+nz; ll=ll+nu; oo=oo+ns;
        end
        rep.straj=reshape(sbar,ns,n+1)';
        rep.fzs=fzs;
    else
        % stage variable is [x;u]
        xbar=zeros((n+1)*nx,1);
        ubar=zeros((n+1)*nu,1);
        qq=0; kk=0; ll=0;
        for jj=1:(n+1)
            xbar((qq+1):(qq+nx))=z((kk+1):(kk+nx));
            ubar((ll+1):(ll+nu))=z((kk+nx+1):(kk+nx+nu));
            qq=qq+nx; kk=kk+nz; ll=ll+nu;
        end
        rep.fzs=0;
    end
    % Expand solution to full format and return a report
    umat = reshape(ubar,nu,n+1)';
    xmat = reshape(xbar,nx,n+1)';
    rep.x0 = xmat(1,:)';    % this should be equal to x
    rep.u0 = umat(1,:)';    % this is the MPC control sample
    rep.zqp = z;
    rep.fzqp = fz;
    rep.fzofs = h0;
    rep.utraj = umat;       % solution trajectory
    rep.xtraj = xmat;
end

end

function [iv,jv,vv]=blockvec(i1,j1,V)
% Returns vectorized indices and elements for a block matrix V
% to be positioned with its upper left corner at (i1,j1).
% Useful when preparing data for a call to SPARSE(..)
[mb,nb]=size(V);
vv=V(:);
iv=repmat((i1:(i1+mb-1))',[1,nb]);
iv=iv(:);
jv=repmat(j1:(j1+nb-1),[mb,1]);
jv=jv(:);
end

function y=aux_format_signal(x,nx,m)
% Read in reference trajectory x(i)=0..n, dim(x(i))==nx, i=1..m
if size(x,1)==nx && size(x,2)==1
  y = repmat(x,m,1);
elseif size(x,1)==nx && size(x,2)==m
  y = x(:);
elseif size(x,1)==m && size(x,2)==nx
  y = x';
  y = y(:);
elseif numel(x)==1
  y = ones(m*nx,1)*x;
else
  y = x;
end
assert(size(y,1)==m*nx && size(y,2)==1);
end

function Y=aux_format_square_matrix(X,nx)
if size(X,1)==nx && size(X,2)==1
  Y = diag(X);
elseif size(X,1)==1 && size(X,2)==nx
  Y = diag(X);
elseif size(X,1)==nx && size(X,2)==nx
  Y = X;
elseif numel(X)==1
  Y = X*eye(nx);
else
  Y = X;
end
assert(size(Y,1)==nx && size(Y,2)==nx);
end

function solveropts = getDefaultOptions(isOctave)
solveropts = struct;
solveropts.solver = 'pdipmqp2';
solveropts.MaxIter = 50;
if ~isOctave
  V = ver;
  VName = {V.Name};
  if any(strcmpi(VName, 'Optimization Toolbox'))
    solveropts.solver = 'quadprog';
  end
end
solveropts.TolX = 1e-8;     % defaults for quadprog
solveropts.TolFun = 1e-8;
solveropts.TolCon = 1e-8;
end
