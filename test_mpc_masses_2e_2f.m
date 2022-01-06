% Script to thoroughly test 2e and 2f (m and c) MPC codes
% on the Boyd/Yang "masses" test problem.

% First setup system data [See Yang/Boyd 2010]

if exist('use_state_constraints','var')==0
  use_state_constraints = false;
else
  assert(islogical(use_state_constraints),'must be logical variable.');
end

if use_state_constraints
  fprintf(1,'State constraints WILL be included in formulation.\n');
else
  fprintf(1,'State constraints WILL NOT be included in formulation.\n');
end

k = 1;          % spring constant 
lam = 0;        % damping constant 
a = -2*k;
b = -2*lam;
c = k;
d = lam;

n = 12; % state dimension
m = 3; % input dimension

Acts = [zeros(n/2) eye(n/2);
    [a,c,0,0,0,0,b,d,0,0,0,0;
     c,a,c,0,0,0,d,b,d,0,0,0;
     0,c,a,c,0,0,0,d,b,d,0,0;
     0,0,c,a,c,0,0,0,d,b,d,0;
     0,0,0,c,a,c,0,0,0,d,b,d;
     0,0,0,0,c,a,0,0,0,0,d,b]];

Bcts = [zeros(n/2,m);
    [1, 0, 0;
    -1, 0, 0;
     0, 1, 0;
     0, 0, 1;
     0,-1, 0;
     0, 0,-1]];

% convert to discrete-time system
ts = 0.5;       % sampling time
A = expm(ts*Acts);
B = (Acts\(expm(ts*Acts)-eye(n)))*Bcts;

% objective matrices
Q = eye(n);      
R = eye(m);        

% state and control limits
Xmax = 4; 
Xmin = -4;
Umax = 0.5;
Umin = -0.5;

% process noise trajectory
nsim = 1e3;
tvec=(0:1:(nsim-1))*ts;
tvec=tvec(:);
w = 2*rand(n,nsim)-1;
w(1:n/2,:) = 0;
w = 0.5*w;

W = (1/12)*diag([0;0;0;0;0;0;1;1;1;1;1;1]);

% initial state
x0 = zeros(n,1);

X=zeros(n,nsim); X(:,1)=x0; u=zeros(m,1); J=zeros(nsim,1);
% uncontrolled simulation (driven by noise only)
for jj=2:nsim
  xprev=X(:,jj-1);
  xnext=A*xprev+B*u+w(:,jj-1);
  X(:,jj)=xnext;
  J(jj)=xprev'*Q*xprev+u'*R*u;
end

figure(1); clf;
subplot(311); stairs(1:nsim,X(1:6,:)');
xlabel('time index'); ylabel('positions'); title('uncontrolled (u=0)');
subplot(312); stairs(1:nsim,X(7:12,:)');
xlabel('time index'); ylabel('velocities'); title('uncontrolled (u=0)');
subplot(313); stairs(1:nsim,J);
xlabel('time index'); ylabel('stage cost'); title('uncontrolled (u=0)');
drawnow;

% Then create full trajectory solutions with input and state-bounds.
% Solve using 2e and 2f and compare; zero initial condition;
% a step-like reference trajectory; no special terminal cost option.

% Variations on this problem: terminal cost option(s) and 
% different initial states.

% If 2f and 2e check out against each other then proceed to run 
% a full feedback simulation with the 2f C code; horizon of 30 time-steps.

optx = struct;
optx.xreturn=0; % do not return state trajectory prediction
optx.ureturn=1; % only return first control sample from solution path
optx.eps=1e-8;
optx.eta=0.96;
optx.verbose=0;
optx.cholupd=0; % =1 is a bit slower here
optx.blasopt=0;
optx.sparsity=1;

opte = qpmpclti2e();

N = 30;

P = [];
P.A = A;
P.B = B;
P.C = [];   % this tells qpmpclti2d.m to ignore also W,r,D fields
%P.C = [eye(n/2),zeros(n/2)];
%P.D = zeros(n/2,m);
P.W = zeros(n/2);
P.Qx = Q;
P.R = R;
%P.r = zeros(n/2,1);
P.w = zeros(n,1);
P.n = N;
if use_state_constraints
  P.F1 = [eye(n);-eye(n);zeros(2*m,n)];
  P.F2 = [zeros(2*n,m);eye(m);-eye(m)];
  P.f3 = [ones(n,1)*Xmax;-ones(n,1)*Xmin;ones(m,1)*Umax;-ones(m,1)*Umin];
else
  P.F1 = zeros(2*m,n);
  P.F2 = [eye(m);-eye(m)];
  P.f3 = [ones(m,1)*Umax;-ones(m,1)*Umin];
end

%
% Sample a few random initial states x and verify 2f against the 2e
% code; require (numerical) equivalence, otherwise stop.
%

optx.xreturn=N;
optx.ureturn=N;

repsx=1e-2;
repsu=1e-2;
repsf=1e-2;

maxrerr=-Inf;

ntest = 100;

fprintf(1,'2e/2f numerical equivalence tests (x%i) on problem P...\n',ntest);

for jj=1:ntest
  P.x=(Xmin+Xmax)*0.5 + randn(n,1)*(Xmax-Xmin)/16;
  if use_state_constraints
    %assert(false,'Component clamping not implemented.');
    P.x=min(Xmax,max(Xmin,P.x));
    %disp(P.x');
  end
  rep1=qpmpclti2f(P,optx);
  rep0=qpmpclti2e(P,opte);
  assert(rep1.isconverged==1);
  errx=rep0.xtraj(1:N,:)-rep1.xtraj;
  erru=rep0.utraj(1:N,:)-rep1.utraj;
  errf=(rep0.fzofs+rep0.fzqp)-(rep1.fxofs+rep1.fxopt);
  rerrx=max(max(abs(errx)))/max(max(abs(rep0.xtraj(1:N,:))));
  rerru=max(max(abs(erru)))/max(max(abs(rep0.utraj(1:N,:))));
  rerrf=abs(errf)/(1+rep0.fzofs+rep0.fzqp);
  if rerrx>=repsx || rerru>=repsu || rerrf>=repsf    
    fprintf(1,'FAILED TEST (%i)\n',jj);
    fprintf(1,'max(|errx|)=%e\n',max(max(abs(errx))));
    fprintf(1,'max(|erru|)=%e\n',max(max(abs(erru))));
    fprintf(1,'|errf|=%e\n',abs(errf));
    fprintf(1,'rerrx=%e, rerru=%e, rerrf=%e\n',rerrx,rerru,rerrf);
    return;
  end
  maxrerr=max([maxrerr,rerrf,rerru,rerrx]);
end

fprintf('TESTS PASSED: max(rel.err(f,x,u))=%e\n',maxrerr);

optx.xreturn=0;
optx.ureturn=1;

%
%
%

disp(['#',num2str(n),' states, #',num2str(m),...
    ' controls, #',num2str(length(P.f3)),' constraints/stage.']);
disp(['simulating #',num2str(nsim),...
    ' time-steps using MPC with horizon=#',num2str(P.n),'...']);

mpctimes=zeros(nsim,2); ipmiters=zeros(nsim,1);
X=zeros(n,nsim); X(:,1)=x0; u=zeros(m,1); J=zeros(nsim,1); U=zeros(m,nsim);
for jj=2:nsim
  xprev=X(:,jj-1);    % "measure" state
  ttt=tic;
  P.x=xprev;
  repj=qpmpclti2f(P,optx);
  assert(repj.isconverged==1,'Failed to converge.');
  u=repj.utraj(1,:)';
  mpctimes(jj,1)=toc(ttt);
  mpctimes(jj,2)=repj.solveclock;
  ipmiters(jj)=repj.iterations;
  xnext=A*xprev+B*u+w(:,jj-1);
  U(:,jj-1)=u;
  X(:,jj)=xnext;
  J(jj)=xprev'*Q*xprev+u'*R*u;
end
medmpctime=median(mpctimes(:,1))*1e3;
medipmtime=median(mpctimes(:,2))*1e3;
disp(['...median MEX execution time was ',num2str(medmpctime),' ms.']);
disp(['...median IPM execution time was ',num2str(medipmtime),' ms.']);

figure(2); clf; ttlstr=['controlled (',repj.solverprogram,')'];
subplot(311); stairs(1:nsim,X(1:6,:)');
xlabel('time index'); ylabel('positions'); title(ttlstr);
if use_state_constraints
    hold on;
    line([1,nsim],Xmax*[1,1],'Color','k','LineWidth',2);
    line([1,nsim],Xmin*[1,1],'Color','k','LineWidth',2);
end
subplot(312); stairs(1:nsim,X(7:12,:)');
xlabel('time index'); ylabel('velocities'); title(ttlstr);
if use_state_constraints
    hold on;
    line([1,nsim],Xmax*[1,1],'Color','k','LineWidth',2);
    line([1,nsim],Xmin*[1,1],'Color','k','LineWidth',2);
end
subplot(313); stairs(1:nsim,U');
xlabel('time index'); ylabel('controls'); title(ttlstr);
hold on;
line([1,nsim],Umax*[1,1],'Color','k','LineWidth',2);
line([1,nsim],Umin*[1,1],'Color','k','LineWidth',2);

figure(3); clf;
subplot(311); stairs(1:nsim,J);
xlabel('time index'); ylabel('stage-cost'); title(ttlstr);
subplot(312); stairs(1:nsim,mpctimes*1e3);
xlabel('time index'); ylabel('MEX time and IPM time [ms]'); title(ttlstr);
hold on;
line([1,nsim],medmpctime*[1,1],'Color','g','LineWidth',2);
subplot(313); stairs(1:nsim,ipmiters);
xlabel('time index'); ylabel('IPM iterations'); title(ttlstr);
