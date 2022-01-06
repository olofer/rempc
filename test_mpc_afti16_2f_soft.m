fprintf(1,'START [%s]\n',mfilename());
% Script to run the AFTI-16 benchmark problem using two different
% MPC solvers: one (2e) based on matlab's quadprog routine, and the 
% other one (2f) a fully standalone MEX function written in standard C.
%
% qpmpclti2e.m [reference quadprog called with sparse matrices]
% qpmpclti2f.c [C standalone, library free]
%
% The reference signal will be generated on-the-fly in a relay-fashion.
% The dynamical constraints will implicitly
% impose a rate-limit on the relay switches.
% The constraint involving state variables is softened (feasibility).
%

post_simulation_micro_solver_check = true;
post_simulation_macro_solver_check = true;
check_complexity_along_horizon = true;

if exist('optx', 'var') == 0
  optx=struct;
  optx.eps=1e-8;
  optx.eta=0.96;
  optx.verbose=0;
  optx.cholupd=0; % =1 is a bit slower here
  optx.blasopt=0;
  optx.sparsity=1;
else
  assert(isstruct(optx), 'optx workspace variable must be a struct.');
end

% Define system and discretise it using the time-step Ts
Ac=[-.151,-60.5651,0,-32.174;
    -.0001,-1.3411,.9929,0;
    .00018,43.2541,-.86939,0;
    0,0,1,0];

Bc=[-2.516,-13.136;
    -.1689,-.2514;
    -17.251,-1.5766;
    0,0];

Cc=[0,1,0,0;
    0,0,0,1];

Ts=0.05;
Tstop=7.0;  % there will be about 3.5 s per "cycle", so 2 cycles

sysd=c2d(ss(Ac,Bc,Cc,0),Ts,'zoh');

fprintf(1,'max(abs(eig(A)))=%.4f\n',max(abs(eig(sysd.A))));

% Now define the LTI/MPC problem data
Pmpc=struct;
Pmpc.n=15;   % this means a horizon k=0..n of n+1 stages
Pmpc.A=sysd.A;
Pmpc.B=sysd.B;
Pmpc.C=sysd.C;
Pmpc.D=sysd.D;  % this will be zero so could be set to [] also
Pmpc.w=0; % dynamics bias term
y2ref=10;
Pmpc.r=[0;y2ref];
Pmpc.x=zeros(4,1);
Pmpc.W=1;
Pmpc.Qx=1e-6; %diag([1e-4;0;1e-3;0]);
Pmpc.R=1e-3;
% the state constraints are essential for this problem
% both inputs are box-bounded abs(.)<=25
umax=25; umin=-25;
% and the output y2 is boxed such that -1/2<=y2<=1/2
y1max=1/2; y1min=-1/2;
Pmpc.F1=[zeros(2,4);zeros(2,4);Pmpc.C(1,:);-Pmpc.C(1,:)];
Pmpc.F2=[eye(2);-eye(2);zeros(2)];
Pmpc.f3=[umax;umax;-umin;-umin;y1max;-y1min];
% Specify constraint softening for rows 5 and 6
% (corresponding to the y1 limits); this is indicated
% by positive elements in the .sc vector (the slack costs).
Pmpc.sc=-ones(length(Pmpc.f3),1);
Pmpc.sc(5)=5e2;
Pmpc.sc(6)=5e2;
%Pmpc.sc(5:6)=1;

% Input noise (uncertainty/error) std about 1% of max allowed range.
% Generate as w=B*d, d=randn(2,1)*sgmd;
sgmd=0.01*(umax-umin)/2;

% During the simulation; do not even bother
% to return the full MPC solution; just output the first control sample.
% And show nothing of the state trajectory.

optx.xreturn=0; % do not return state trajectory prediction
optx.ureturn=1; % only return first control sample from solution path
optx.sreturn=0;

% Start the control loop (!) run until Tstop
kmax=ceil(Tstop/Ts);

tsim=Ts*(0:(kmax-1))';
xsim=zeros(kmax,4);
usim=zeros(kmax,2);
ysim=zeros(kmax,2);
rsim=zeros(kmax,2);
wsim=zeros(kmax,4);
tclk=zeros(kmax,4);
itrs=zeros(kmax,1);
k=1;

fprintf(1,'Closed-loop simulating #%i time-steps with Ts=%f...\n',kmax,Ts);

while true
    x0=xsim(k,:)';
    ysim(k,:)=(sysd.C*x0)';
    Pmpc.x=x0;
    if Pmpc.r(2)==y2ref && abs(ysim(k,2)-y2ref)/y2ref<1e-2
        Pmpc.r(2)=0;    % go to zero
    elseif Pmpc.r(2)==0 && abs(ysim(k,2))<2e-2
        Pmpc.r(2)=y2ref;    % go back up
    end
    rsim(k,:)=Pmpc.r';
    tic;
    rep1=qpmpclti2f(Pmpc,optx);
    tclk(k,1)=toc;
    tclk(k,2)=rep1.totalclock;
    tclk(k,3)=rep1.solveclock;
    tclk(k,4)=rep1.cholyclock;
    assert(rep1.isconverged==1,'Failed to converge.');
    itrs(k)=rep1.iterations;
    u0=rep1.utraj(1,:)';
    d=sgmd*randn(2,1);
    w=Pmpc.B*d;
    x1=sysd.A*x0+sysd.B*u0+w;
    usim(k,:)=u0';
    wsim(k,:)=w';
    if k==kmax
        break;
    end
    k=k+1;
    xsim(k,:)=x1';
end

fprintf(1,'...done.\n');
avgclk=mean(tclk(:,3));
fprintf(1,'avg. solveclock=%f ms -> %f x realtime (@Ts=%f sec)\n',1e3*avgclk,Ts/avgclk,Ts);
avgitr=mean(itrs);
fprintf(1,'avg. PDIPM iterations=%f (@epstol=%e)\n',avgitr,rep1.epsopt);

% Present the results
figure(1); clf;
subplot(411); hold on;
line([tsim(1),tsim(end)],y1min*[1,1],'Color','k','LineStyle','--');
line([tsim(1),tsim(end)],y1max*[1,1],'Color','k','LineStyle','--');
stairs(tsim,ysim);
ylabel('y_1, y_2');
title(sprintf('(Feedback) Outputs (MPC n=%i)',Pmpc.n));

subplot(412); hold on;
line([tsim(1),tsim(end)],umin*[1,1],'Color','k','LineStyle','--');
line([tsim(1),tsim(end)],umax*[1,1],'Color','k','LineStyle','--');
stairs(tsim,usim);
ylabel('u_1, u_2');
title('(Feedback) Inputs');

subplot(413);
stairs(tsim,rsim);
ylabel('r_1, r_2');
title('(Feedback) Output reference (dynamically generated)');

subplot(414);
stairs(tsim,[tclk*1e3,itrs/10]);
legend('mex-call [ms]','internal [ms]','solver [ms]','cholesky [ms]','iterations [#/10]',...
    'location','best');
xlabel('time [sec]');
title(sprintf('Timing and convergence (epstol=%e)',rep1.epsopt));

if exist('opte', 'var') == 0
  opte = qpmpclti2e();
else
  assert(isstruct(opte), 'opte workspace variable must be a struct.');
end

if post_simulation_micro_solver_check
    % Step through each time-point and re-solve for the (x,r)
    % data recorded at each step; verify that the solution is numerically
    % matched for both 2e and 2f codes.
    
    fprintf(1,'Verification of #%i mini-trajectories (2f vs. 2e) ...\n',kmax);
    
    % compare full trajectories
    optx.xreturn=Pmpc.n+1;
    optx.ureturn=Pmpc.n+1;
    optx.sreturn=Pmpc.n+1;
    
    epxufosc=NaN(kmax,6); % relative errors btw. 2e and 2f for the full-solution at each time-step
    slvclk=NaN(kmax,2);
    
    for ii=1:kmax
        Pmpc.r=rsim(ii,:)';
        Pmpc.x=xsim(ii,:)';
        rep0=qpmpclti2e(Pmpc,opte);
        rep1=qpmpclti2f(Pmpc,optx);
        % check and compare MPC reports
        assert(rep0.isConverged,'MICRO: Failed to converge (2e).');
        assert(rep1.isconverged==1,'MICRO: Failed to converge (2f).');
        errx=rep1.xtraj-rep0.xtraj;
        erru=rep1.utraj-rep0.utraj;
        errf=rep1.fxopt-rep0.fzqp;
        erro=rep1.fxofs-rep0.fzofs;
        errs=rep1.straj-rep0.straj;
        errc=rep1.fxoft-rep0.fzs;
        % compute and store relative inf-norm errors
        errx=max(abs(errx(:)))/(1+max(abs(rep0.xtraj(:))));
        erru=max(abs(erru(:)))/(1+max(abs(rep0.utraj(:))));
        errf=abs(errf)/(1+abs(rep0.fzqp));
        erro=abs(erro)/(1+abs(rep0.fzofs));
        errs=max(abs(errs(:)))/(1+max(abs(rep0.straj(:))));
        errc=abs(errc)/(1+abs(rep0.fzs));
        epxufosc(ii,:)=[errx,erru,errf,erro,errs,errc];
        slvclk(ii,:)=[rep0.solvertime,rep1.totalclock];
    end
    fprintf(1,'...done (<2e>=%.3f ms, <2f>=%.3f ms).\n',...
        1e3*median(slvclk(:,1)),1e3*median(slvclk(:,2)));
    figure(2); clf;
    plot(1:kmax,log10(epxufosc),'s-');
    hl=legend('err(x)','err(u)','err(fopt)','err(fofs)','err(s)','err(fs)');
    set(hl,'FontSize',12);
    xlabel('timestep');
    ylabel('log-10 relative inf-norm error');
    title('mini-trajectory comparison 2e vs. 2f');
end


if post_simulation_macro_solver_check
    
    fprintf(1,'Verification of macro-trajectory (2f vs. 2e, n=%i) ...\n',kmax);
    % Next create a large MPC problem over the entire time range simulated
    Pmpc.n=kmax-1;
    
    % compare full trajectories
    optx.xreturn=Pmpc.n+1;
    optx.ureturn=Pmpc.n+1;
    optx.sreturn=Pmpc.n+1;
    
    % NOTE: transposed or not does not (should not) make any difference here
    Pmpc.r=rsim;
    Pmpc.w=wsim(1:Pmpc.n,:); % one row less than rsim
    
    Pmpc.x=xsim(1,:).';
    rep0=qpmpclti2e(Pmpc,opte);
    rep1=qpmpclti2f(Pmpc,optx);
    % check MPC reports
    assert(rep0.isConverged,'MACRO: Failed to converge (2e).');
    assert(rep1.isconverged==1,'MACRO: Failed to converge (2f).');
    % compare solutions to each other
    errx=rep1.xtraj-rep0.xtraj;
    erru=rep1.utraj-rep0.utraj;
    errf=rep1.fxopt-rep0.fzqp;
    erro=rep1.fxofs-rep0.fzofs;
    errs=rep1.straj-rep0.straj;
    errc=rep1.fxoft-rep0.fzs;
    % compute and store relative inf-norm errors
    errx=max(abs(errx(:)))/(1+max(abs(rep0.xtraj(:))));
    erru=max(abs(erru(:)))/(1+max(abs(rep0.utraj(:))));
    errf=abs(errf)/(1+abs(rep0.fzqp));
    erro=abs(erro)/(1+abs(rep0.fzofs));
    errs=max(abs(errs(:)))/(1+max(abs(rep0.straj(:))));
    errc=abs(errc)/(1+abs(rep0.fzs));
    % print out 
    fprintf(1,'...done (<2e>=%.3f ms, <2f>=%.3f ms).\n',...
        1e3*rep0.solvertime,1e3*rep1.solveclock);
    fprintf(1,'macro(errx)=%e\n',errx);
    fprintf(1,'macro(erru)=%e\n',erru);
    fprintf(1,'macro(errf)=%e\n',errf);
    fprintf(1,'macro(erro)=%e\n',erro);
    fprintf(1,'macro(errs)=%e\n',errs);
    fprintf(1,'macro(errc)=%e\n',errc);
    
    % display both 2e and 2f solution trajectories; for (y,u,s)
    figure(3); clf;
    
    subplot(3,2,1); stairs(tsim,(Pmpc.C*rep0.xtraj')'); title('2e:y (ref)');
    subplot(3,2,3); stairs(tsim,rep0.utraj); title('2e:u (ref)');
    subplot(3,2,5); stairs(tsim,rep0.straj); title('2e:s (ref)');
    
    subplot(3,2,2); stairs(tsim,(Pmpc.C*rep1.xtraj')'); title('2f:y (black=2f-2e)');
    subplot(3,2,4); stairs(tsim,rep1.utraj); title('2f:u (black=2f-2e)');
    subplot(3,2,6); stairs(tsim,rep1.straj); title('2f:s (black=2f-2e)');
    
    % add in residual 2f-2e into the second set of panels (n black)
    subplot(3,2,2); hold on; stairs(tsim,(Pmpc.C*rep1.xtraj')'-(Pmpc.C*rep0.xtraj')','k');
    subplot(3,2,4); hold on; stairs(tsim,rep1.utraj-rep0.utraj,'k');
    subplot(3,2,6); hold on; stairs(tsim,rep1.straj-rep0.straj,'k');
    
end

if check_complexity_along_horizon
    % Solve MPC problem for increasing horizons nmin..nmax
    % using a replication of rsim to extend the target output;
    % compare 2e and 2f solution times (median over nrep repeated solves)
    
    nmin=5;
    nmax=floor((2*kmax-1)*0.75);
    nstride=5;
    nrep=30;
    rref=[rsim;rsim];
    
    nvec=nmin:nstride:nmax;
    ntot=length(nvec);
    T2e=NaN(ntot,nrep);
    T2f=NaN(ntot,nrep);
    
    fprintf(1,...
        'Sampling (%i reps) solve-speed as a function of n=%i..%i...\n',...
        nrep,nmin,nmax);
    
    for kk=1:ntot
        n=nvec(kk);
        % full trajectory return requested;
        Pmpc.n=n;
        optx.ureturn=Pmpc.n+1;
        optx.xreturn=Pmpc.n+1;
        optx.sreturn=Pmpc.n+1;
        % set the target from truncation of rref
        Pmpc.r=rref(1:(Pmpc.n+1),:);
        Pmpc.w=0; % do not bother with this one for now..
        for rr=1:nrep
            % draw sample from states xsim as initial state;
            Pmpc.x=xsim(randi(kmax),:);
            %fprintf(1,'(%i,%i)\n',n,rr);
            rep0=qpmpclti2e(Pmpc,opte);
            rep1=qpmpclti2f(Pmpc,optx);
            % check MPC reports
            assert(rep0.isConverged,'SPEED: Failed to converge (2e).');
            assert(rep1.isconverged==1,'SPEED: Failed to converge (2f).');
            % store timing
            T2e(kk,rr)=rep0.solvertime;
            T2f(kk,rr)=rep1.solveclock;
            % NOTE: might be a good idea to also verify numerical
            % equivlance along the way at each step..
        end
        fprintf(1,'n=%i ',n);
    end
    fprintf(1,'\n');
    
    % Compute the mean ratio of the median solution clocks
    % ...
    
    figure(4); clf;
    plot(nvec,[1e3*median(T2e,2),1e3*median(T2f,2)],'o--');
    hl=legend('quadprog(2e) [ms]','standalone(2f) [ms]');
    set(hl,'FontSize',15);
    xlabel('MPC horizon n','FontSize',15);
    ylabel(sprintf('Median solve-time (over %i samples) [ms]',nrep),'FontSize',15);
    title('MPC solve time as a function of horizon n');
    grid on;
end

fprintf(1,'END [%s]\n',mfilename());
