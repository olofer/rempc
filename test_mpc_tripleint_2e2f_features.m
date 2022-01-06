%
% Script for combinatorial testing of "features" comparing Matlab/Octave 
% code 2e against the standalone C code 2f (MEX interface).
%
% TODO: implement softened constraint option in 2f code and include
%       its testing in this program.
% TODO: implement a second test suite based on randomized systems.
%

RELERRXOK = 1e-4;
RELERRUOK = 1e-3;
RELERRFOK = 1e-5;

% Simple triple-integrator system definition.
Ac=[0,1,0;
    0,0,1;
    0,0,0];
Bc=[0;0;1];
Cc=[1,0,0];
Ts=0.1;
csys=ss(Ac,Bc,Cc,0);
dsys=c2d(csys,Ts,'zoh');
N=200;
tvec=(0:N)';

% Setup the constrained QP
Pmpc=[];
Pmpc.A=dsys.a;
Pmpc.B=dsys.b;
Pmpc.C=dsys.c;
Pmpc.D=dsys.d;
Pmpc.n=N;
Pmpc.R=1;
Pmpc.Qx=1e-6;
Pmpc.W=1;

U=1.5;
X3=3.0;

% Options for 2f code 
optx=struct;
optx.xreturn=N+1;
optx.ureturn=N+1;
optx.eps=1e-8;
optx.eta=0.96;

% Options for 2e code
opte = qpmpclti2e();

NTST=64; % 6 bits one cycle
relinferrs=NaN(NTST,3);
objoffsets=NaN(NTST,3);

% 6 different features are toggled on/off by looping
% over ii=0..63 and checking the bit patterns of ii.

for ii=0:1:(NTST-1)
    
    % Setup the "featureless" test problem
    Pmpc.x=zeros(3,1);
    Pmpc.r=0;
    Pmpc.w=0;
    Pmpc.F1=[];
    Pmpc.F2=[1;-1];
    Pmpc.f3=[U;U];
    Pmpc.Qxn=[];
    Pmpc.Wn=[];
    
    ff=bitget(ii,6:-1:1);
    
    %disp(ff);
    for jj=1:length(ff)
        fprintf(1,'%i',ff(jj));
    end
    fprintf(1,':');
    
    % Interpret ff(j), j=1..6 as follows (binary values).
    
    % ff(1) = 1 : x0 random (otherwise 0)
    % ff(2) = 1 : r nonzero, otherwise zero.
    % ff(3) = 1 : w nonzero, otherwise zero.
    % ff(4) = 1 : include state constraint on x3 otherwise only for input.
    % ff(5) = 1 : include terminal cost Wn otherwise empty
    % ff(6) = 1 : include terminal cost Qn otherwise empty
    
    % ...
    if ff(1)==1
        %Pmpc.x=[0;0.1*randn(1,1);0.1*randn(1,1)];
        Pmpc.x=(2*rand(3,1)-1);
    end
    
    if ff(2)==1
        Pmpc.r=2+randn(1,1);
    end
    
    if ff(3)==1
        Pmpc.w=[0;0;randn(1,1)*0.1]; % acts as input disturbance on u, w=B*d basically
    end
    
    if ff(4)==1
        % extend constraints set here with bound abs(x3)<=X3
        % fprintf(1,' (!) ');
        Pmpc.F1=[zeros(2,3);[0,0,1;0,0,-1]];
        Pmpc.F2=[Pmpc.F2;zeros(2,1)];
        Pmpc.f3=[Pmpc.f3;X3;X3];
    end
    
    if ff(5)==1
        Pmpc.Wn=Pmpc.W*20;
        %fprintf(1,' (Wn ignored) ');
    end
    
    if ff(6)==1
        Pmpc.Qxn=N*Pmpc.Qx*1e2;
    end
    
    r2e=qpmpclti2e(Pmpc,opte);
    r2f=qpmpclti2f(Pmpc,optx);
    
    % Check that both codes converged
    % and compare the solutions..
    
    if r2e.isConverged && r2f.isconverged
        
        % Compare objective values
        
        % Compare objective values and "offsets" separately
        
        fobje=r2e.fzqp;
        fobjf=r2f.fxopt;
        errf=fobjf-fobje;
        relinferrf=abs(errf)/max([1,abs(fobje)]);
        
        errfofs=r2f.fxofs-r2e.fzofs;
        relinferrfofs=abs(errfofs)/max([1,abs(r2e.fzofs)]);
        
        % Compare solutions
        
        maxx=max(max(abs(r2e.xtraj)));
        maxu=max(max(abs(r2e.utraj)));
        
%        rmsx=sqrt(sum(sum(r2e.xtraj.^2))/numel(r2e.xtraj));
%        rmsu=sqrt(sum(sum(r2e.xtraj.^2))/numel(r2e.xtraj));
        
        errx=r2f.xtraj-r2e.xtraj;
        erru=r2f.utraj-r2e.utraj;
        
%        rmsex=sqrt(sum(sum(errx.^2))/numel(errx));
%        rmseu=sqrt(sum(sum(errx.^2))/numel(errx));
        
        maxerrx=max(max(abs(errx)));
        maxerru=max(max(abs(erru)));
        
%        relerrx=rmsex/max([1,rmsx]);
%        relerru=rmseu/max([1,rmsu]);
        
        relinferrx=maxerrx/max([1,maxx]);
        relinferru=maxerru/max([1,maxu]);
        
        fprintf(1,' relx=%.4e,relu=%.4e,relf=%.4e: ',relinferrx,relinferru,relinferrf);
        
        if relinferrx<RELERRXOK && relinferru<RELERRUOK && relinferrf<RELERRFOK
            fprintf(1,' OK.\n');
        else
            fprintf(1,' *** BAD ***.\n');
        end
        
        relinferrs(ii+1,:)=[relinferrf,relinferrx,relinferru];
        objoffsets(ii+1,:)=[r2e.fzofs,r2f.fxofs,relinferrfofs];
        
        % Also log solvertime (2e) and totalclock (2f)
        % ...
        
    elseif ~r2e.isConverged && ~r2f.isconverged && ff(4)==1
        
        % This can be perfectly true; might be infeasible.
        
        fprintf(' BOTH 2e and 2f failed to converge; and F1 was nonzero.\n');
        
    else
        % Print warning message and halt.
        fprintf(1,'\nERROR: disagree on convergence: 2e[%i], 2f[%i].\n',...
            r2e.isConverged,r2f.isconverged);
        
        return;
    end
    
end

% Create a plot of the relative errors on a logscale

figure(1); clf;
plot(0:(NTST-1),log10(relinferrs),'o-.');
legend('errf','errx','erru');
title('Relative inf-norm errors');
xlabel('Test Index');
ylabel('Base-10 logarithm of (relative) error');
hold on;
line([0,NTST-1],[1,1]*log10(RELERRFOK),'Color','k','LineStyle','-');
line([0,NTST-1],[1,1]*log10(RELERRXOK),'Color','b','LineStyle','-');
line([0,NTST-1],[1,1]*log10(RELERRUOK),'Color','r','LineStyle','-');
A=axis(); axis([0,NTST-1,A(3),A(4)]);
