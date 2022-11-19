/*
 * qpmpclti2f.c
 *
 * Compile suggestions:
 *
 * mex -largeArrayDims CFLAGS="\$CFLAGS -std=c99" -lrt qpmpclti2f.c
 * mex -largeArrayDims CFLAGS="\$CFLAGS -std=c99 -O2 -Wall" -lrt qpmpclti2f.c
 *
 * (add -v after mex to be verbose; need -c99 for "static inline")
 * (only need -lrt since using the nanosecond timer fastclock.h)
 * 
 * MPC optimization solver for LTI systems using plain ANSI C code
 * for the implementation of a structure exploiting block-based
 * O(n) algorithm; where n is the MPC horizon. The code solves a
 * quadratic program which is never explicitly formed.
 *
 * USAGE: rep = qpmpclti2f(P,opts)
 *
 * P : is a struct containing all the information and data to setup the
 *     quadratic program for MPC. 
 * opts : a struct with QP-solver specific options.
 * rep : a struct with the solution data.
 *
 * Handles also the special case where P does not define any linear
 * inequality constraint. Also handles softened linear inequalitites
 * by automatic introduction of stage slack variables, when applicable.
 *
 * K E J Olofsson, Aug./Sept./Oct./Nov./Dec. 2016
 *
 * TODO: (a) [x] special case with no inequalitites
 *       (b) [x] extension to slack costs + sreturn option
 *       (c) [x] special terminal cost options
 *       (d) [x] parsing of sparse blocks (especially J & D)
 *       (e) [x] verbosity option (dynamical, ie. not preprocessor dir.)
 *       (f) [ ] one-step iterative refinement option (for the cholesky solver)
 *       (g) [x] remove x-vector output 
 *       (h) [x] return convergence epsilons (at final iteration)
 *       (i) [x] "cached factor" support and infrastructure
 *       (j) [x] nq<<nx e.g., serious need of caching/updating
 *       (k) [ ] return meta-data on allocated buffer sizes (# doubles)
 *       (l) [ ] improved cache usage for block ops
 *       (m) [ ] override to return data after maxiters even if not converged
 */

#include <math.h>
#include <memory.h>
#include <time.h>
#include <string.h>

#include "mex.h"

#define __DEVELOPMENT_TEXT_OUTPUT__ /* #undef (#define); silent (verbose) */
#define __CLUMSY_ASSERTIONS__
#define __COMPILE_WITH_INTERNAL_TICTOC__    /* include nanosecond timer tic/toc? */

/* To support these; need the -c99 compiler option switched on */
#include "vectorops.h"
#include "matrixopsc.h"

#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
#include "fastclock.h"
#endif

/* NOTE: memory allocation should be done using the
 * mxMalloc(.) and mxFree(.) routines supplied by the MATLAB API.
 */

#define __MULTISOLVER_MALLOC mxMalloc
#define __MULTISOLVER_FREE   mxFree
#define __MULTISOLVER_PRINTF mexPrintf

#include "multisolver.h"  // <-- "core" solver code

#define PARSE_VERBOSITY 1

#define PRMSTRUCT prhs[0]
#define OPTSTRUCT prhs[1]
#define REPSTRUCT plhs[0]

/* The ordering of variables in the list below will be used to populate
   a vector of indices into the actual struct variable P for the
   corresponding data. All of the listed parameters are required to
   be of class "double". The field names are case-sensitive.
 */
#define PRMSTRUCT_NUMLOOKUP 17
const char *prmstruct_lookupnames[PRMSTRUCT_NUMLOOKUP]={
    "A",
    "B",
    "C",
    "D",
    "Qx",
    "W",
    "R",
    "n",
    "F1",
    "F2",
    "f3",
    "w",
    "r",
    "x",
    "Qxn",
    "Wn",
    "sc"
};

/* This listing must be the zero-based order of the above fieldnames */
#define IDX_A 0
#define IDX_B 1
#define IDX_C 2
#define IDX_D 3
#define IDX_Q 4
#define IDX_W 5
#define IDX_R 6
#define IDX_n 7
#define IDX_F1 8
#define IDX_F2 9
#define IDX_f3 10
#define IDX_w 11
#define IDX_r 12
#define IDX_x 13
#define IDX_Qn 14 /* special terminal cost option */
#define IDX_Wn 15 /* ditto */
#define IDX_sc 16 /* soft/slack cost vector (optional) */

/*
 * Declare which fields are considered for the options
 * struct input argument.
 *
 * NOTE: case sensitive and all of them must be scalar doubles.
 *
 */

#define OPTSTRUCT_NUMLOOKUP 11
const char *optstruct_lookupnames[OPTSTRUCT_NUMLOOKUP]={
    "eps",
    "eta",
    "maxiters",
    "ureturn",
    "xreturn",
    "verbose",
    "initopt",
    "sparsity", /* try to exploit sparsity of J and C, D blocks */
    "cholupd",
    "blasopt",
    "sreturn"
};

#define IDX_OPT_EPS 0
#define IDX_OPT_ETA 1
#define IDX_OPT_MAXITERS 2
#define IDX_OPT_URETURN 3
#define IDX_OPT_XRETURN 4
#define IDX_OPT_VERBOSE 5
#define IDX_OPT_SPARSITY 7
#define IDX_OPT_CHOLUPD 8
#define IDX_OPT_BLASOPT 9
#define IDX_OPT_SRETURN 10

/* Declare the field names of the return struct object */

#define REPSTRUCT_NUMFIELDS 21
const char *repstruct_fieldnames[REPSTRUCT_NUMFIELDS]={
    "solverprogram",
    "mextimestamp",
    "numstages",
    "totalclock",
    "solveclock",
    "clockresol",
    "fxopt",
    "fxofs",
    "qpclass",
    "isconverged",
    "iterations",
    "utraj",
    "xtraj",
    "epsopt",
    "etaopt",
    "cholyclock",
    "straj",
    "inftuple",
    "cholfail",
    "nxnunqns",
    "fxoft"
};
#define REP_SOLVERPROGRAM 0
#define REP_MEXTIMESTAMP 1
#define REP_NUMSTAGES 2
#define REP_TOTALCLOCK 3
#define REP_SOLVECLOCK 4
#define REP_CLOCKRESOL 5
#define REP_FXOPT 6
#define REP_FXOFS 7
#define REP_QPCLASS 8
#define REP_ISCONVERGED 9
#define REP_ITERATIONS 10
#define REP_UTRAJ 11
#define REP_XTRAJ 12
#define REP_EPSOPT 13
#define REP_ETAOPT 14
#define REP_CHOLYCLOCK 15
#define REP_STRAJ 16
#define REP_INFTUPLE 17
#define REP_CHOLFAIL 18
#define REP_NXNUNQNS 19
#define REP_FXOFT 20

#define NUM_PRBLM_CLASSES 3
const char *prblm_class_names[NUM_PRBLM_CLASSES]={
    "qp-eq",
    "qp-eq-ineq",
    "qp-eq-ineq-slack"
};

/* These types are used to decide the type of calculation needed 
 * when using the cost matrices W,R,Q,[Wn,Qn]
 */
#define TYP_UNDEF -1
#define TYP_SCALAR 0
#define TYP_VECTOR 1
#define TYP_MATRIX 2
#define TYP_MATRIXT 3   /* transposed matrix type */

/*
 * Declare auxiliary functions (defined below mexFunction)
 *
 */

int aux_search_for_numeric_struct_fields(const mxArray *,const char **,int,int *,int);
int aux_read_square_matrix(mxArray *,int,int *,double *,double **);
int aux_read_signal_matrix(mxArray *,int,int,int *,double *,double **);
void aux_print_array(double *,int,int);
void aux_print_array_sparsity(double *,int,int);

void aux_compute_sym_cost_matrix(
        int nd,int nx,int nu,int ny,double *pQstg,
        int typQ,double sQ,double *pQ,
        int typR,double sR,double *pR,
        int typW,double sW,double *pW,
        double *pC,double *pD);

/*
 * mexFunction : the MATLAB main / entry point / interface
 */

void mexFunction(int nlhs,mxArray *plhs[],
        int nrhs,const mxArray *prhs[]) {
    
    #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    static fclk_timespec _tic1,_tic2;
    static fclk_timespec _toc1,_toc2;
    #endif
    
    #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_tic1);
    #endif
    
    /* Local aux. variables used during input/output parsing */
    mxArray *pmx;
    int qq,ll,kk,mm;
    double dd;
    double *pdd;
    
    int numReturnStepsU,numReturnStepsX,numReturnStepsS;
    
    static int idxp[PRMSTRUCT_NUMLOOKUP];
    static int idxo[OPTSTRUCT_NUMLOOKUP];
        
    /* Local MPC specific variables */
    static qpdatStruct qpDat;
    static qpoptStruct qpOpt;
    static qpretStruct qpRet;
    
    /* Meta-data storage for sparse J and D (if needed) */
    static sparseMatrix spJay;
    static sparseMatrix spDee;
    
    int n,nx,nu,ny,nq,ns,nd,ni; /* stagecount and problem (sub-)block dimension(s) */
    double *pA,*pB,*pC,*pD;     /* System matrices */
    
    double *pW,*pR,*pQ,*pS;     /* Cost term matrices */
    int typW,typR,typQ; /*,typS;*/
    double sW,sR,sQ;/*,sS;*/         /* Only used if typX = TYP_SCALAR */
    
    double *pF1,*pF2;

    double *pw,*pr,*px,*pf3,*psc=NULL;	/* vectors & vector signals */
    int typw,typr,typf3;
    double sw,sr,sf3;
    
    double *pWn,*pQn;       /* used only with special terminal costs */
    int typWn,typQn;
    double sWn,sQn,q0tmp;
    
    /* These are constructed aux. data matrices */
    double *pJ=NULL,*pCstg=NULL,*pDstg=NULL,*pQstg=NULL;
    double *pCstg0=NULL,*pDstg1=NULL,*pQNstg=NULL;
    double *pCC1,*pCC2;
    
    /* These are constructed aux. data vectors */
    double *pvecd=NULL,*pvecq=NULL,*pvecf=NULL,
            *pvecq0=NULL,*pvecr=NULL,*prtmp=NULL;
    
    int hasOutput=0;            /* is C!=0 ? */
    int hasDirectTerm=0;        /* is D!=0 ? */
    int hasF1=0,hasF2=0;
    int hasInequalities=0;      /* is F1!=0 or F2!=0 ? */
    int hasSlackCost=0;         /* slack variable extension required? ns>0? */
    int hasTerminalQ=0;         /* Qn exists? */
    int hasTerminalW=0;         /* Wn exists? */
    int prblmClass=-1;
    
    stageStruct *pStages=NULL;
    stageStruct *thisStage=NULL;
    
    int auxbufsz=0,bufofs=0;
    double *pauxbuf=NULL;

		spJay.buf=NULL; /* mark the sparse structs as unused */
		spDee.buf=NULL;
    
    ns = 0;
    
    /* Create an empty output struct */
    REPSTRUCT = mxCreateStructMatrix(1,1,REPSTRUCT_NUMFIELDS,repstruct_fieldnames);
    
    if (nrhs!=2)
        mexErrMsgTxt("exactly 2 input arguments required.");
    
    if (!(nlhs==1 || nlhs==0))
        mexErrMsgTxt("0 or 1 output arguments required.");
    
    /* Start to parse the MPC data input struct P */
    if (aux_search_for_numeric_struct_fields(
            PRMSTRUCT,prmstruct_lookupnames,
            PRMSTRUCT_NUMLOOKUP,idxp,PARSE_VERBOSITY)!=0) {
        /* error message and exit */
        mexErrMsgTxt("Error during parsing of problem/parameter struct input argument.");
    }
        
    /* Check what numerical data was found and that it is sized consistently */
    if (idxp[IDX_n]<0)
        mexErrMsgTxt("Field n not found or is empty.");
    pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_n]);
    if (mxGetNumberOfElements(pmx)!=1) /*mxIsScalar(.) ? */
        mexErrMsgTxt("Field n must be a scalar.");
    n=(int)round(mxGetScalar(pmx));
    if (n+1<__MULTISOLVER_MINIMUM_STAGES)   /* confirm that n+1>=__MULTISOLVER_MINIMUM_STAGES */
        mexErrMsgTxt("Horizon is too short (n+1>=3 required).");
    
    /* Read A-matrix; get dimension nx and pointer pA to column-major matrix data */
    if (idxp[IDX_A]<0)
        mexErrMsgTxt("Field A not found or is empty.");
    pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_A]);
    nx=(int)mxGetM(pmx);
    if (nx!=mxGetN(pmx))
        mexErrMsgTxt("A must be a square matrix.");
    pA=mxGetPr(pmx);
    
    /* Read B-matrix; get dimension nu and pointer pB */
    if (idxp[IDX_B]<0)
        mexErrMsgTxt("Field B not found or is empty.");
    pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_B]);
    nu=(int)mxGetN(pmx);
    if (nx!=mxGetM(pmx))
        mexErrMsgTxt("B must have same number of rows as A.");
    pB=mxGetPr(pmx);
    
    /* Read C-matrix; get dimension ny and pointer pC */
    if (idxp[IDX_C]>=0) {
        pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_C]);
        ny=(int)mxGetM(pmx);
        if (nx!=mxGetN(pmx))
            mexErrMsgTxt("C must have same number of columns as A.");
        pC=mxGetPr(pmx);
        hasOutput=1;
    } else { /* C is allowed to be empty/omitted */
        ny=0;
        hasOutput=0;
        pC=NULL;
    }
    
    /* Check D-matrix if it exists; but only when C exists also */
    if (idxp[IDX_D]>=0 && hasOutput!=0) {
        pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_D]);
        if (!(mxGetM(pmx)==ny && mxGetN(pmx)==nu))
            mexErrMsgTxt("D has inconsistent dimensions.");
        pD=mxGetPr(pmx);
        hasDirectTerm=0; /* Check whether D is actually all zeros. */
        for (qq=0;qq<ny*nu;qq++) {
            if (pD[qq]!=0.0) { hasDirectTerm=1; break; }
        }
    } else {
        hasDirectTerm=0;
        pD=NULL;
    }
    
    if (idxp[IDX_x]<0)
        mexErrMsgTxt("Input field x not found or is empty.");
    pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_x]);
    /* Make sure x is nx-by-1 or 1-by-nx; both are OK (same memory arrangement) */
    if (!(((mxGetM(pmx)==nx && mxGetN(pmx)==1)) || (mxGetM(pmx)==1 && mxGetN(pmx)==nx)))
        mexErrMsgTxt("Input field x must be a vector with nx elements.");
    px=mxGetPr(pmx);
    
    /* Check F1,F2 if any; check dims; get nq */
    if (idxp[IDX_F1]>=0) {
        pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_F1]);
        if (mxGetN(pmx)!=nx)
            mexErrMsgTxt("F1 has inconsistent dimensions.");
        nq=mxGetM(pmx);
        pF1=mxGetPr(pmx);
        hasF1=0; /* set to 1 only if nonzero */
        for (qq=0;qq<nq*nx;qq++) {
            if (pF1[qq]!=0.0) { hasF1=1; break; }
        }
    } else {
        hasF1=0;
        nq=0;
        pF1=NULL;
    }
    
    if (idxp[IDX_F2]>=0) {
        pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_F2]);
        if (mxGetN(pmx)!=nu)
            mexErrMsgTxt("F2 has inconsistent dimensions.");
        if (hasF1) {
            if (mxGetM(pmx)!=nq)
                mexErrMsgTxt("F2 row-count does not match that of F1.");
        } else {
            nq=mxGetM(pmx);
        }
        pF2=mxGetPr(pmx);
        hasF2=0;    /* set to 1 only if nonzero */
        for (qq=0;qq<nq*nu;qq++) {
            if (pF2[qq]!=0.0) { hasF2=1; break; }
        }
    } else {
        hasF2=0;
        pF2=NULL;
    }
    
    /* If either of F1 or F2 is nonzero; there are inequalities present */
    if (hasF1!=0 || hasF2!=0) {
        hasInequalities=1;
    } else {
        hasInequalities=0;
    }
    
    /* If hasInequalitites, then check whether constraint softening is requested;
       update ns accordingly
     */
    if (hasInequalities>0 && idxp[IDX_sc]>=0) {
    	pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_sc]);
    	/* Make sure sc is nq-by-1 or 1-by-nq; both are OK */
	    if (!(((mxGetM(pmx)==nq && mxGetN(pmx)==1)) || (mxGetM(pmx)==1 && mxGetN(pmx)==nq)))
        mexErrMsgTxt("Soft/slack cost vector length must have nq elements.");
      /* next extract the implied slack vector dimension (maximum nq, minimum 0) */
      psc=mxGetPr(pmx);
      for (qq=0;qq<nq;qq++)
      	if (psc[qq]>0.0) ns++;
      /* ... it is enough to just find the value of ns here;
         then populate S and J later when req. storage is allocated
       */
      if (ns>0) hasSlackCost=1;
    }
    
    /* nd = dimension of stage-variable */
    nd=nx+nu+ns;	/* also add ns, if ns>0 */
    ni=nq+ns;
    
    /* Since the dimensions are known at this point; allocate aux. space */
    auxbufsz=(n+1)*nx+(n+1)*nd+(n+1)*ni+ni*nd+nx*nd+nx*nd
            +2*nx*nd+2*nx*nd+nd*nd+nd*nd+(n+1)+(n+1)*ny+ny
            +nd*(nd+1)+nd*(nd+1)+ns;
    pauxbuf=mxMalloc(auxbufsz*sizeof(double));
    if (pauxbuf==NULL)
        mexErrMsgTxt("Failed to allocate aux. memory block.");
    bufofs=0;
    pvecd=&pauxbuf[bufofs];  bufofs+=(n+1)*nx;
    pvecq=&pauxbuf[bufofs];  bufofs+=(n+1)*nd;
    pvecf=&pauxbuf[bufofs];  bufofs+=(n+1)*ni;
    pJ=&pauxbuf[bufofs];     bufofs+=ni*nd;
    pCstg=&pauxbuf[bufofs];  bufofs+=nx*nd;
    pDstg=&pauxbuf[bufofs];  bufofs+=nx*nd;
    pCstg0=&pauxbuf[bufofs]; bufofs+=2*nx*nd;
    pDstg1=&pauxbuf[bufofs]; bufofs+=2*nx*nd;
    pQstg=&pauxbuf[bufofs];  bufofs+=nd*nd;
    pQNstg=&pauxbuf[bufofs]; bufofs+=nd*nd; /* this is unused unless Qxn or Wn are specified */
    pvecq0=&pauxbuf[bufofs]; bufofs+=(n+1);
    pvecr=&pauxbuf[bufofs];  bufofs+=(n+1)*ny;
    prtmp=&pauxbuf[bufofs];  bufofs+=ny;
    pCC1=&pauxbuf[bufofs];   bufofs+=nd*(nd+1); /* storage for cached common Cholesky factors, when applicable */
    pCC2=&pauxbuf[bufofs];   bufofs+=nd*(nd+1);
    pS=&pauxbuf[bufofs];     bufofs+=ns; /* NOTE: cannot be used if ns=0 */   
    
    if (bufofs!=auxbufsz) {
        mexPrintf("WARNING: auxbufsz=%i, bufofs=%i\n",auxbufsz,bufofs);
    }
    
    if (hasInequalities>0) {
        if (idxp[IDX_f3]<0)
            mexErrMsgTxt("Field f3 not found or is empty.");
        pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_f3]);
        /* Check/read f3 nq-by-(n+1) data */
        if (aux_read_signal_matrix(pmx,nq,n+1,&typf3,&sf3,&pf3)!=1)
            mexErrMsgTxt("Field f3 has inconsistent size.");
        /* NOTE: J may have significant sparsity.
         * Typically there will only be a single element per row which is
         * nonzero in J=[F1,F2] so this is important to handle efficiently.
         */
        if (hasSlackCost>0) {
            matopc_zeros(pJ,ni,nd);
            if (hasF1>0) matopc_sub_assign(pJ,ni,nd,0,0,pF1,nq,nx,+1);
            if (hasF2>0) matopc_sub_assign(pJ,ni,nd,0,nx,pF2,nq,nu,+1);
            /* ... assemble the rest ... indices for psc[.]>0.0: (nq+qq,nx+nu+qq)=-1*/
            for (ll=0,qq=0;qq<nq;qq++) {
            	if (psc[qq]>0.0) {
            		pJ[ni*(nx+nu+ll)+(qq)]=-1.0;
            		pJ[ni*(nx+nu+ll)+(nq+ll)]=-1.0;
            		pS[ll]=psc[qq];
            		ll++;
            	}
            }
        } else {
       		/* Create J by copying submatrices into proper patches .. */
        	matopc_zeros(pJ,nq,nd);
        	if (hasF1>0) matopc_sub_assign(pJ,nq,nd,0,0,pF1,nq,nx,+1);
        	if (hasF2>0) matopc_sub_assign(pJ,nq,nd,0,nx,pF2,nq,nu,+1);
        }
    }
    
    /* Always check w (part of evolution equation); data should be
     * a matrix of size nx-by-n ideally; but could also be scalar, vector
     * or a transposed matrix.
     */
    if (idxp[IDX_w]<0)
        mexErrMsgTxt("Field w not found or is empty.");
    pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_w]);
    if (aux_read_signal_matrix(pmx,nx,n,&typw,&sw,&pw)!=1)
        mexErrMsgTxt("Field w has inconsistent size.");
    
    /* Check W if hasOutput>0 */
    if (hasOutput>0) {
        if (idxp[IDX_W]<0)
            mexErrMsgTxt("Field W not found or is empty.");
        pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_W]);
        if (aux_read_square_matrix(pmx,ny,&typW,&sW,&pW)!=1)
            mexErrMsgTxt("Field W has inconsistent size.");
        
        if (idxp[IDX_r]<0)
            mexErrMsgTxt("Field r not found or is empty.");
        pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_r]);
        /* Field r should ideally be a ny-by-(n+1) matrix.
         * It can also be a ny-vector interpreted as repeated n+1 times.
         * Or it can be a scalar; assumed to fill up the full matrix.
         * A transposed matrix (n+1)-by-ny is also allowed; if r does not
         * fit any of the above sizes. 
         */
        if (aux_read_signal_matrix(pmx,ny,n+1,&typr,&sr,&pr)!=1)
            mexErrMsgTxt("Field r has inconsistent size.");
        
        /* Check whether special terminal cost matrix Wn is present among the arguments */
        if (idxp[IDX_Wn]>=0) {
            pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_Wn]);
            if (aux_read_square_matrix(pmx,ny,&typWn,&sWn,&pWn)!=1)
                mexErrMsgTxt("Field Wn has inconsistent size.");
            hasTerminalW=1;
        }
    }
    
    /* Check Q always; must not be undefined */
    if (idxp[IDX_Q]<0)
        mexErrMsgTxt("Field Q not found or is empty.");
    pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_Q]);
    if (aux_read_square_matrix(pmx,nx,&typQ,&sQ,&pQ)!=1)
        mexErrMsgTxt("Field Q has inconsistent size.");
    
    /* Check R always; must not be undefined */
    if (idxp[IDX_R]<0)
        mexErrMsgTxt("Field R not found or is empty.");
    pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_R]);
    if (aux_read_square_matrix(pmx,nu,&typR,&sR,&pR)!=1)
        mexErrMsgTxt("Field R has inconsistent size.");
    
    /* Special terminal cost matrix Qxn exists? */
    if (idxp[IDX_Qn]>=0) {
        pmx=mxGetFieldByNumber(PRMSTRUCT,0,idxp[IDX_Qn]);
        if (aux_read_square_matrix(pmx,nx,&typQn,&sQn,&pQn)!=1)
            mexErrMsgTxt("Field Qxn has inconsistent size.");
        hasTerminalQ=1;
    }
    
    #ifdef __DEVELOPMENT_TEXT_OUTPUT__
    /*mexPrintf("mpc horizon: 0..%i.\n",n);
    mexPrintf("system: (states,inputs,outputs):[hasout,dterm] = (%i,%i,%i):[%i,%i].\n",
            nx,nu,ny,hasOutput,hasDirectTerm);
    mexPrintf("inequalities: (hasF1,hasF2,nq) = (%i,%i,%i).\n",
            hasF1,hasF2,nq);*/
    /*mexPrintf("initial condition vector x:\n");
    for (qq=0;qq<nx;qq++) {
        mexPrintf("%f\n",px[qq]);
    }*/
    /* NOTE: order of elements same as in matlab with A(:) */
    /*for (qq=0;qq<nx*nx;qq++) {
        mexPrintf("%f\n",pA[qq]);
    }*/
    /*mexPrintf("R[0]=%e;typ=%i, Q[0]=%e;typ=%i\n",*pR,typR,*pQ,typQ);
    if (hasOutput>0)
        mexPrintf("W[0]=%e;typ=%i\n",*pW,typW);*/
    #endif
       
    /* Setup big d vector */
    matopc_copy(pvecd,px,nx,1);
    if (typw==TYP_SCALAR) {
        for (qq=nx;qq<nx*(n+1);qq++) pvecd[qq]=sw;
    } else if (typw==TYP_VECTOR) {
        for (qq=0,ll=0;qq<n;qq++,ll+=nx) {
            matopc_copy(&pvecd[nx+ll],pw,nx,1);
        }
    } else if (typw==TYP_MATRIX) {
        /*for (qq=0,ll=0;qq<n;qq++,ll+=nx) {
            matopc_copy(&pvecd[nx+ll],&pw[ll],nx,1);
        }*/
        matopc_copy(&pvecd[nx],pw,nx,n);
    } else {
        /*matopc_zeros(pvecd,nx*(n+1),1);
        mexErrMsgTxt("TYP_MATRIXT not yet supported @ d.");*/
        matopc_copy_transpose(&pvecd[nx],pw,n,nx);
    }
    /* aux_print_array(pvecd,nx*(n+1),1); */
    
    /* Setup big q vector (also known as vector h).
     * Start by expanding/rearranging to obtain a stacked vector of r.
     */
    matopc_zeros(pvecq,nd*(n+1),1);
    matopc_zeros(pvecq0,(n+1),1);
    if (hasOutput>0) {
        /* Setup a full-size reference stacked vector */ 
        if (typr==TYP_SCALAR) {
            /* Treat scalar sr as the vector ones(ny,1)*sr at each stage. */
            for (qq=0;qq<ny*(n+1);qq++) pvecr[qq]=sr;
        } else if (typr==TYP_VECTOR) {
            /* Use vector ny-by-1 pr for each stage. */
            for (qq=0,ll=0;qq<n+1;qq++,ll+=ny) {
                matopc_copy(&pvecr[ll],pr,ny,1);
            }
        } else if (typr==TYP_MATRIX) {
            /* Each column in matrix ny-by-(n+1) pr is a unique reference vector */
            matopc_copy(pvecr,pr,ny,n+1);
        } else {
            /* Each row in matrix (n+1)-by-ny pr is a (transposed) unique reference vector */
            matopc_copy_transpose(pvecr,pr,n+1,ny);
        }
        
        /* Save last reference vector if it needs to be used for modification of last stage below */
        if (hasTerminalW>0) {
            matopc_copy(prtmp,&pvecr[n*ny],ny,1); /* copy last ny-vector r(n) of pvecr to prtmp */
        }
                
        /* Apply in-place transformation r(i) <- -2*W*r(i) on the buffer pvecr */
        /* Evaluate q0(i)=r(i)'*W*r(i) for each stage i=0..n; before the inplace op. (!) */
        if (typW==TYP_SCALAR) {
            kk=0;
            for (qq=0;qq<n+1;qq++) {
                dd=0.0;
                for (ll=0;ll<ny;ll++)
                    dd+=sW*pvecr[kk+ll]*pvecr[kk+ll];
                pvecq0[qq]=dd;
                kk+=ny;
            }
            dd=-2.0*sW; for (qq=0;qq<ny*(n+1);qq++) pvecr[qq]*=dd;
        } else if (typW==TYP_VECTOR) {
            kk=0;
            for (qq=0;qq<n+1;qq++) {
                dd=0.0;
                for (ll=0;ll<ny;ll++)
                    dd+=pW[ll]*pvecr[kk+ll]*pvecr[kk+ll];
                pvecq0[qq]=dd;
                kk+=ny;
            }
            kk=0;
            for (qq=0;qq<n+1;qq++) {
                for (ll=0;ll<ny;ll++) pvecr[kk++]*=(-2.0*pW[ll]);
            }
        } else { /* TYP_MATRIX */
            if (ny>MATOPC_TEMPARRAYSIZE)
                mexErrMsgTxt("ERROR: ny>MATOPC_TEMPARRAYSIZE");
            kk=0;
            for (qq=0;qq<n+1;qq++) {
                pvecq0[qq]=matopc_xtax_sym(pW,ny,&pvecr[kk]);
                /* Multiply by general ny-by-ny matrix W */
                matopc_inplace_scaled_ax(&pvecr[kk],pW,ny,ny,-2.0);
                /* TODO: actually never access lower triangle of W */
                /* ATTN: uses static temp. array of fixed size internally */
                kk+=ny;
            }
        }
        
        /* Need modification of linear cost term and offset for last stage if Wn exists */
        if (hasTerminalW) {
            /* evaluate q0n <- r(n)'*Wn*r(n) and r(n) <- -2*Wn*r(n) for usage later */
            if (typWn==TYP_SCALAR) {
                q0tmp=0.0;
                for (ll=0;ll<ny;ll++)
                    q0tmp+=prtmp[ll]*prtmp[ll];
                q0tmp*=sWn;
                dd=-2.0*sWn;
                for (ll=0;ll<ny;ll++)
                    prtmp[ll]*=dd;
            } else if (typWn==TYP_VECTOR) {
                q0tmp=0.0;
                for (ll=0;ll<ny;ll++)
                    q0tmp+=pWn[ll]*prtmp[ll]*prtmp[ll];
                for (ll=0;ll<ny;ll++)
                    prtmp[ll]*=(-2.0*pWn[ll]);
            } else { /* TYP_MATRIX */
                if (ny>MATOPC_TEMPARRAYSIZE)
                    mexErrMsgTxt("ERROR: ny>MATOPC_TEMPARRAYSIZE");
                q0tmp=matopc_xtax_sym(pWn,ny,prtmp);
                matopc_inplace_scaled_ax(prtmp,pWn,ny,ny,-2.0);
                /* TODO: symmetrized version that never accessed lower part */
            }
            /* overwrite last vector in pvecr and last element in pvecq0 */
            pvecq0[n]=q0tmp;
            matopc_copy(&pvecr[n*ny],prtmp,ny,1);
        }
        
        /* Apply C' and D' transformations to obtain q(i) from W*r(i) vectors */
        if (hasDirectTerm>0) { /* q(i) = -2*[C';D']*W*r(i) */
            kk=0; ll=0;
            for (qq=0;qq<n+1;qq++) {
                matopc_atx(&pvecq[kk],pC,ny,nx,&pvecr[ll]);
                matopc_atx(&pvecq[kk+nx],pD,ny,nu,&pvecr[ll]);
                kk+=nd; ll+=ny;
            }
        } else { /* q(i) = -2*[C';zeros(nu,ny)]*W*r(i) */
            kk=0; ll=0;
            for (qq=0;qq<n+1;qq++) {
                matopc_atx(&pvecq[kk],pC,ny,nx,&pvecr[ll]);
                kk+=nd; ll+=ny;
            }
        }
        
        /*mexPrintf("[pvecq0-new]\n"); 
        aux_print_array(pvecq0,n+1,1);*/
        
    }
    
    /* NOTE: pvecf must be padded correctly when ns>0 */
    /* Setup big f vector; length of pvecf is ni*(n+1), with ni=nq+ns, where ns might be zero */
    if (hasInequalities>0) {
    		matopc_zeros(pvecf,ni*(n+1),1);
        if (typf3==TYP_SCALAR) {
        		if (ns>0) {
	        		for (qq=0,ll=0;qq<n+1;qq++,ll+=ni) {
	        			vecop_assign(&pvecf[ll],nq,sf3);
  	          }
        		} else {
            	/*for (qq=0;qq<nq*(n+1);qq++) pvecf[qq]=sf3;*/
            	vecop_assign(&pvecf[0],nq*(n+1),sf3);
            }
        } else if (typf3==TYP_VECTOR) {
            for (qq=0,ll=0;qq<n+1;qq++,ll+=ni) {
                matopc_copy(&pvecf[ll],pf3,nq,1);
            }
        } else if (typf3==TYP_MATRIX) {
            for (qq=0,ll=0,kk=0;qq<n+1;qq++,ll+=ni,kk+=nq) {
                matopc_copy(&pvecf[ll],&pf3[kk],nq,1);
            }
        } else {
            /*mexErrMsgTxt("TYP_MATRIXT not yet supported @ f.");*/
            /* NOTE: should read in as if pf3 is a column-major (n+1)-by-nq matrix */
            for (qq=0,ll=0;qq<n+1;qq++,ll+=ni) {
              for (kk=0;kk<nq;kk++)
                pvecf[ll+kk]=pf3[qq+kk*(n+1)];
            }
        }
    }
    /* aux_print_array(pvecf,ni*(n+1),1); */
    
/*    if (ns>0) {
			aux_print_array(pJ,ni,nd);
			aux_print_array(pS,ns,1);
			mexPrintf("pvecf[0]:\n");
			aux_print_array(pvecf,2*ni,1);
			mexErrMsgTxt("Slack cost not yet supported.");
    } */
    
    /* Cstg=[-A,-B] */
    matopc_sub_assign(pCstg,nx,nd,0,0,pA,nx,nx,-1);
    matopc_sub_assign(pCstg,nx,nd,0,nx,pB,nx,nu,-1);
    if (ns>0) matopc_sub_assign_zeros(pCstg,nx,nd,0,nx+nu,nx,ns); /* [-A,-B,0] needed here */
    
    /*mexPrintf("[C]=[-A,-B]\n");
    aux_print_array(pCstg,nx,nd);*/
    
    /* Dstg=[eye(nx),zeros(nx,nu)] */
    matopc_zeros(pDstg,nx,nd);
    matopc_sub_assign_diag_scalar(pDstg,nx,nd,0,0,1.0,nx,-1);
    
    /*mexPrintf("[D]=[I,0]\n");
    aux_print_array(pDstg,nx,nd);*/
    
    /* C0=[I,0;-A,-B] */
    matopc_zeros(pCstg0,2*nx,nd);
    matopc_sub_assign_diag_scalar(pCstg0,2*nx,nd,0,0,1.0,nx,-1);
    matopc_sub_assign(pCstg0,2*nx,nd,nx,0,pCstg,nx,nd,+1); /* Do not change sign again! */
    
    /*mexPrintf("[C0]\n");
    aux_print_array(pCstg0,2*nx,nd);*/
    
    /* D1=[0,0;I,0] */
    matopc_zeros(pDstg1,2*nx,nd);
    matopc_sub_assign_diag_scalar(pDstg1,2*nx,nd,nx,0,1.0,nx,-1);
    
    /*mexPrintf("[D1]\n");
    aux_print_array(pDstg1,2*nx,nd);*/
    
    /* If there are no slack variables;
     * the cost matrix is square symmetric with side = nx+nu.
     * If there are slack avariables the cost matrix has side = nx+nu+ns.
     * This dimension is stored in stage size: nd.
     */
    
    aux_compute_sym_cost_matrix(
        nd,nx,nu,ny,pQstg,typQ,sQ,pQ,typR,sR,pR,typW,sW,pW,pC,pD);
    
    /* Setup special stage cost matrix for last stage as needed */
    if (hasTerminalQ>0 && hasTerminalW==0) {
        /* create a Qn cost matrix which only need a new upper left block */
        aux_compute_sym_cost_matrix(
            nd,nx,nu,ny,pQNstg,typQn,sQn,pQn,typR,sR,pR,typW,sW,pW,pC,pD);
    } else if (hasTerminalQ==0 && hasTerminalW>0) {
        /* use previous/common Q but need to recreate the rest of Qn */
        aux_compute_sym_cost_matrix(
            nd,nx,nu,ny,pQNstg,typQ,sQ,pQ,typR,sR,pR,typWn,sWn,pWn,pC,pD);
    } else if (hasTerminalQ>0 && hasTerminalW>0) {
        /* both Qn and Wn are special; new stage cost from */
        aux_compute_sym_cost_matrix(
            nd,nx,nu,ny,pQNstg,typQn,sQn,pQn,typR,sR,pR,typWn,sWn,pWn,pC,pD);
    }
    
    /* May need to insert the slack cost on the bottom-right part of the diagonal */
    if (ns>0) {
      /* insert diag(ps)*2.0 */
    	matopc_sub_assign_scaled_diag_vector(pQstg,nd,nd,nx+nu,nx+nu,pS,ns,-1,2.0);
    	if (hasTerminalQ>0 || hasTerminalW>0) {
    		/* pQNstg update required too */
    		matopc_sub_assign_scaled_diag_vector(pQNstg,nd,nd,nx+nu,nx+nu,pS,ns,-1,2.0);
    	}
    }
    
    /* TEMPORARY DEVELOPMENT CHECK FOR NS>0 */
    /*if (ns>0) {
    	mexPrintf("[Cstg]=[-A,-B,0]\n");
    	aux_print_array(pCstg,nx,nd);
    	mexPrintf("[Dstg]=[I,0,0]\n");
    	aux_print_array(pDstg,nx,nd);
    	mexPrintf("[Qstg]\n");
    	aux_print_array(pQstg,nd,nd);
    }*/
    
    /* NOTE: lower triangle of Q is actually not assigned */
    /*#ifdef __DEVELOPMENT_TEXT_OUTPUT__
    mexPrintf("[Qstg]\n");
    aux_print_array(pQstg,nd,nd);
    #endif*/
    
    /* Set default algorithm options.
     * Overwritten later if an options struct provides a copy.
     */
    setup_qpopt_defaults(&qpOpt);
    
    /* Set the default return variable options: full control trajectory
     * but nothing for the state trajectory.
     */
    numReturnStepsU=n+1;
    numReturnStepsX=0;
    numReturnStepsS=0;
        
    /* Parse the options struct input if any */
    if (mxIsEmpty(OPTSTRUCT)) {
        /* nothing to do here */
        
        /*#ifdef __DEVELOPMENT_TEXT_OUTPUT__
        mexPrintf("algo. options is empty.\n");
        #endif*/
        
    } else {
        
        /* Parse input options structure */
        
        if (aux_search_for_numeric_struct_fields(
                OPTSTRUCT,optstruct_lookupnames,
                OPTSTRUCT_NUMLOOKUP,idxo,PARSE_VERBOSITY)!=0) {
            /* error message and exit */
            mexErrMsgTxt("Error during parsing of options struct input argument.");
        }
        
        /*for (ll=0;ll<OPTSTRUCT_NUMLOOKUP;ll++) {
            if (idxo[ll]!=-1) {
                mexPrintf("opts[%s] provided.\n",optstruct_lookupnames[ll]);
            }
        }*/
        
        if (idxo[IDX_OPT_EPS]>=0) {
            pmx=mxGetFieldByNumber(OPTSTRUCT,0,idxo[IDX_OPT_EPS]);
            if (mxGetNumberOfElements(pmx)!=1)
                mexErrMsgTxt("Field eps must be a scalar.");
            qpOpt.ep=mxGetScalar(pmx);
            if (qpOpt.ep<0.0) qpOpt.ep=0.0;
        }
        
        if (idxo[IDX_OPT_ETA]>=0) {
            pmx=mxGetFieldByNumber(OPTSTRUCT,0,idxo[IDX_OPT_ETA]);
            if (mxGetNumberOfElements(pmx)!=1)
                mexErrMsgTxt("Field eta must be a scalar.");
            qpOpt.eta=mxGetScalar(pmx);
            if (qpOpt.eta<0.50) qpOpt.eta=0.50;
            if (qpOpt.eta>1.0) qpOpt.eta=1.0;
        }
        
        if (idxo[IDX_OPT_MAXITERS]>=0) {
            pmx=mxGetFieldByNumber(OPTSTRUCT,0,idxo[IDX_OPT_MAXITERS]);
            if (mxGetNumberOfElements(pmx)!=1)
                mexErrMsgTxt("Field maxiters must be a scalar.");
            qpOpt.maxiters=(int)round(mxGetScalar(pmx));
            if (qpOpt.maxiters<1) qpOpt.maxiters=1;
            if (qpOpt.maxiters>CLIP_MAXITERS) qpOpt.maxiters=CLIP_MAXITERS;
        }
        
        if (idxo[IDX_OPT_URETURN]>=0) {
            pmx=mxGetFieldByNumber(OPTSTRUCT,0,idxo[IDX_OPT_URETURN]);
            if (mxGetNumberOfElements(pmx)!=1)
                mexErrMsgTxt("Field ureturn must be a scalar.");
            numReturnStepsU=(int)round(mxGetScalar(pmx));
        }
        
        if (idxo[IDX_OPT_XRETURN]>=0) {
            pmx=mxGetFieldByNumber(OPTSTRUCT,0,idxo[IDX_OPT_XRETURN]);
            if (mxGetNumberOfElements(pmx)!=1)
                mexErrMsgTxt("Field xreturn must be a scalar.");
            numReturnStepsX=(int)round(mxGetScalar(pmx));
        }
        
        if (idxo[IDX_OPT_SRETURN]>=0) {
            pmx=mxGetFieldByNumber(OPTSTRUCT,0,idxo[IDX_OPT_SRETURN]);
            if (mxGetNumberOfElements(pmx)!=1)
                mexErrMsgTxt("Field sreturn must be a scalar.");
            numReturnStepsS=(int)round(mxGetScalar(pmx));
        }
        
        if (idxo[IDX_OPT_VERBOSE]>=0) {
            pmx=mxGetFieldByNumber(OPTSTRUCT,0,idxo[IDX_OPT_VERBOSE]);
            if (mxGetNumberOfElements(pmx)!=1)
                mexErrMsgTxt("Field verbose must be a scalar.");
            qpOpt.verbosity=(int)round(mxGetScalar(pmx));
        }
        
        if (idxo[IDX_OPT_SPARSITY]>=0) {
            pmx=mxGetFieldByNumber(OPTSTRUCT,0,idxo[IDX_OPT_SPARSITY]);
            if (mxGetNumberOfElements(pmx)!=1)
                mexErrMsgTxt("Field sparsity must be a scalar.");
            qpOpt.expl_sparse=(int)round(mxGetScalar(pmx));
        }
        
        if (idxo[IDX_OPT_CHOLUPD]>=0) {
            pmx=mxGetFieldByNumber(OPTSTRUCT,0,idxo[IDX_OPT_CHOLUPD]);
            if (mxGetNumberOfElements(pmx)!=1)
                mexErrMsgTxt("Field cholupd must be a scalar.");
            qpOpt.chol_update=(int)round(mxGetScalar(pmx));
        }
        
        if (idxo[IDX_OPT_BLASOPT]>=0) {
            pmx=mxGetFieldByNumber(OPTSTRUCT,0,idxo[IDX_OPT_BLASOPT]);
            if (mxGetNumberOfElements(pmx)!=1)
                mexErrMsgTxt("Field blasopt must be a scalar.");
            qpOpt.blas_suite=(int)round(mxGetScalar(pmx));
            /* silent adjustment if needed */
            if (qpOpt.blas_suite<0) qpOpt.blas_suite=0;
            if (qpOpt.blas_suite>1) qpOpt.blas_suite=1; /* make sure the selection is in set {0,1} */
        }
    }
    
    if (qpOpt.verbosity>0) {
        mexPrintf("mpc horizon: 0..%i.\n",n);
        mexPrintf("system: (states,inputs,outputs):[hasout,dterm] = (%i,%i,%i):[%i,%i].\n",
            nx,nu,ny,hasOutput,hasDirectTerm);
        mexPrintf("inequalities: (hasF1,hasF2,nq,ns) = (%i,%i,%i,%i).\n",
            hasF1,hasF2,nq,ns);
    }
    
    /* Now create sparse block representations; if requested
     * Note that this may fail e.g. if matrices are deemed too dense; let the failure
     * be silent; the rest of the code runs through anyway.
     */
    if (qpOpt.expl_sparse>0) {
    	if (hasInequalities>0) {
    		sparseMatrixCreate(&spJay,pJ,ni,nd); /* was nq,nd */
    	}
    	sparseMatrixCreate(&spDee,pDstg,nx,nd);
    }
    
    /* Allocate stage structure and initialize the pointers */
    pStages=(stageStruct *)mxMalloc((n+1)*sizeof(stageStruct));
    if (pStages==NULL)
        mexErrMsgTxt("Stage struct array memory allocation failure.");
    
    /* Initialize the stage struct array */
    ll=0; kk=0; mm=0;
    for (qq=0;qq<=n;qq++) {
        thisStage=&pStages[qq];
        thisStage->idx=qq;
        thisStage->nd=nd;   /* in general: nd=nx+nu+ns */
        thisStage->niq=ni; /* nq+ns = ni (was ...=nq) */
        if (qq==0) {
            /* Stage 0 includes initial condition equality constraint */
            thisStage->neq=2*nx;
        } else if (qq==n) {
            thisStage->neq=0;
        } else {
            thisStage->neq=nx;
        }
        /* Setup data pointers */
        thisStage->ptrQ=pQstg;
        thisStage->ptrJ=pJ; /* ... */
        thisStage->ptrq=&pvecq[qq*nd];
        thisStage->ptrq0=&pvecq0[qq];
        thisStage->ptrf=&pvecf[qq*ni]; /* was qq*nq*/
        thisStage->ptrd=&pvecd[ll];
        ll+=thisStage->neq;
        kk+=thisStage->niq;
        mm+=(thisStage->nd)*(thisStage->nd)+(thisStage->nd);
        thisStage->ptrC=NULL;  /* C does not exist for stage n */
        if (qq==0) {
            thisStage->ptrC=pCstg0;
        } else if (qq<n) {
            thisStage->ptrC=pCstg;
        }
        thisStage->ptrD=NULL; /* D does not exist for stage 0 */
        if (qq==1) {
            thisStage->ptrD=pDstg1;
            thisStage->ptrDsp=NULL; /* TODO: special sparse matrix object for first stage (may not be worth it though) */
        } else if (qq>1) {
            thisStage->ptrD=pDstg;
            thisStage->ptrDsp=(spDee.buf==NULL ? NULL : &spDee);
        }
        thisStage->ptrL=NULL; /* must be initialized separately, when needed */
        /*mexPrintf("stage=%i: (neq=%i)\n",pStages[qq].idx,pStages[qq].neq);*/
        
        /* J is shared for every stage in the present formulation */
        thisStage->ptrJsp = (spJay.buf==NULL ? NULL : &spJay);
    }
    /*mexPrintf("ofs=%i (should be %i)\n",ll,n*nx+nx);
     * mexPrintf("sizeof(stageStruct)=%i bytes.\n",sizeof(stageStruct));*/
    #ifdef __CLUMSY_ASSERTIONS__
    if (kk!=ni*(n+1) || ll!=nx*(n+1))
        mexErrMsgTxt("Eq. or ineq. sizing error(s).");
    /*if (hasTerminalQ>0 || hasTerminalW>0)
        mexErrMsgTxt("Did not adjust ptrQ for last stage; but terminal cost options do exist.");*/
    #endif
    
    /* Modify matrix pointer for the last stage if there is a special terminal cost matrix */
    if (hasTerminalQ>0 || hasTerminalW>0)
        pStages[n].ptrQ=pQNstg;
    
    /* Setup problem meta data structure (collection of pointers mostly) */
    qpDat.ndec=(n+1)*nd;
    qpDat.neq=(n+1)*nx;
    qpDat.niq=(n+1)*ni; /* (was nq*(n+1)) */
    qpDat.ph=pvecq;
    qpDat.pf=pvecf;
    qpDat.pd=pvecd;
    qpDat.nstg=n+1;         /* qq=0..n; there are n+1 stages */
    qpDat.pstg=pStages;
    
    /* Initialize working memory for algorithm [NOTE: depends on algo. choice below] */
    if (!msqp_pdipm_init(&qpDat,qpOpt.verbosity))
        mexErrMsgTxt("Failed to initialize working vectors memory buffer.");
    
    /* Setup memory required for block Cholesky matrix factorizations.
     * Both programs (with or without inequalities) use the same block
     * Cholesky factorization program and can be initialized the same way.
     * But the simpler program does use less working memory (not adjusted for).
     */
    if (!InitializePrblmStruct(&qpDat,0x001|0x002|0x004,qpOpt.verbosity))
        mexErrMsgTxt("Failed to initialize Cholesky working memory buffer(s).");
    
    #ifdef __CLUMSY_ASSERTIONS__
    if (qpDat.neq!=qpDat.netot || qpDat.ndec!=qpDat.ndtot)
        mexPrintf("Eq. or ineq. sizing error(s).");
    if (mm!=qpDat.blkphsz)
        mexPrintf("Phi blk buffer sizing mismatch.");
    #endif
    
    /*#ifdef __DEVELOPMENT_TEXT_OUTPUT__
        mexPrintf("blklysz=%i, blkphsz=%i, blkwrsz=%i\n",
                qpDat.blklysz,qpDat.blkphsz,qpDat.blkwrsz);
    #endif*/
    
    if (hasInequalities>0) {
        /* Full IPM required; equality and inequality constrained QP */
        
        if (qpOpt.chol_update>0) {
            /* TODO: pre-factor the only two unique cost matrices (before modification) */
            /* pCC1, pCC2 to be prepared with factorizations of pQstg and pQNstg (if it exists) */
            /* then the stage meta-data should be updated accordingly... ptrL field */
            qq=CreateCholeskyCache(&qpDat,pCC1,pCC2,nd);
            if (qq!=0) mexErrMsgTxt("Fatal block Cholesky factorization failure.\n");
        }
        
        if (ns>0) {
            prblmClass=2; /* slack variable extension on this one (same interface) */
        } else {
            prblmClass=1;
        }
        
        /* Slack extension or not; it is the same code invokation here */
        #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
		        fclk_timestamp(&_tic2);
		    #endif
		        qq=msqp_pdipm_solve(&qpDat,&qpOpt,&qpRet); /* Call main PDIPM algorithm */
		    #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
		        fclk_timestamp(&_toc2);
		    #endif
        
        if (qq!=0 && qpOpt.verbosity>0)
            mexPrintf("WARNING: main PDIPM solver did not converge.\n");
        
    } else {
        /* No need to use the full IPM iteration; equality-constrained QP only */
        
        prblmClass=0;
        
        /* Always pre-factor the stage cost for the simplified solver.
         * The simplified solver will force itself to use the Cholesky cache always.
         * This is possible since the factors never need to be "updated".
         */
        if (qpOpt.chol_update>0) {
          qq=CreateCholeskyCache(&qpDat,pCC1,pCC2,nd);
          if (qq!=0) mexErrMsgTxt("Fatal block Cholesky factorization failure.\n");
        }
        
        #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
        fclk_timestamp(&_tic2);
        #endif
        qq=msqp_solve_niq(&qpDat,&qpOpt,&qpRet);    /* call KKT solver code */
        #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
        fclk_timestamp(&_toc2);
        #endif
        
        if (qq!=0 && qpOpt.verbosity>0)
            mexPrintf("WARNING: main PDIPM solver did not converge.\n");
    }
    
    /* Free Cholesky working memory */
    FreePrblmStruct(&qpDat,qpOpt.verbosity);
    
    mxSetFieldByNumber(REPSTRUCT,0,REP_ITERATIONS,mxCreateDoubleScalar(qpRet.iters));
    /* these guys can be useless if qq!=0 */
    mxSetFieldByNumber(REPSTRUCT,0,REP_FXOPT,mxCreateDoubleScalar(qpRet.fxopt));
    mxSetFieldByNumber(REPSTRUCT,0,REP_FXOFS,mxCreateDoubleScalar(qpRet.fxofs));
    
    /* qq==0 implies converged solution;
       qq==1 implies not converged after max iters;
       qq<0 implies Cholesky error
       
			 Flag Cholesky failure with the "cholfail" return struct field below.
			 
       TODO: check if this is also true for the NIQ solver?
     */
    
    if (qq<0) mxSetFieldByNumber(REPSTRUCT,0,REP_CHOLFAIL,mxCreateDoubleScalar(1.0));
      else mxSetFieldByNumber(REPSTRUCT,0,REP_CHOLFAIL,mxCreateDoubleScalar(0.0));
      
    /* Always return a residual 4-tuple */
    pmx=mxCreateUninitNumericMatrix(1,4,mxDOUBLE_CLASS,mxREAL);
    mxSetFieldByNumber(REPSTRUCT,0,REP_INFTUPLE,pmx);
    memcpy((double *)mxGetPr(pmx),qpRet.inftuple,4*sizeof(double));
        /*pdd=(double *)mxGetPr(pmx);
    pdd[0]=qpRet.inftuple[0];
    pdd[1]=qpRet.inftuple[1];
    pdd[2]=qpRet.inftuple[2];
    pdd[3]=qpRet.inftuple[3];*/
     
    if (qq==0) {
        /* PRESUMABLY Converged to solution (but may have been forced to return _something_ "early") */
        mxSetFieldByNumber(REPSTRUCT,0,REP_ISCONVERGED,mxCreateDoubleScalar(1.0));
        
        /* Create output vector x here; before destroying the working memory */
        /* pmx=mxCreateDoubleMatrix(qpRep.nx,1,mxREAL); */
        
        /*pmx=mxCreateUninitNumericMatrix(qpRet.nx,1,mxDOUBLE_CLASS,mxREAL);
        mxSetFieldByNumber(REPSTRUCT,0,REP_XVECTOR,pmx);
        vecop_copy((double *)mxGetPr(pmx),qpRet.nx,qpRet.x);*/
        
        if (numReturnStepsU>0) {
            if (numReturnStepsU>n+1) numReturnStepsU=n+1; /* clip if needed */
            pmx=mxCreateUninitNumericMatrix(numReturnStepsU,nu,mxDOUBLE_CLASS,mxREAL);
            mxSetFieldByNumber(REPSTRUCT,0,REP_UTRAJ,pmx);
            kk=nx; pdd=(double *)mxGetPr(pmx);
            for (ll=0;ll<numReturnStepsU;ll++) {
                /* control vector for stage ll should go into row ll */
                for (mm=0;mm<nu;mm++)
                    pdd[mm*numReturnStepsU+ll]=qpRet.x[kk+mm];
                kk+=nd;
            }
        }
        
        if (numReturnStepsX>0) {
            if (numReturnStepsX>n+1) numReturnStepsX=n+1; /* clip if needed */
            pmx=mxCreateUninitNumericMatrix(numReturnStepsX,nx,mxDOUBLE_CLASS,mxREAL);
            mxSetFieldByNumber(REPSTRUCT,0,REP_XTRAJ,pmx);
            kk=0; pdd=(double *)mxGetPr(pmx);
            for (ll=0;ll<numReturnStepsX;ll++) {
                /* state vector for stage ll should go into row ll */
                for (mm=0;mm<nx;mm++)
                    pdd[mm*numReturnStepsX+ll]=qpRet.x[kk+mm];
                kk+=nd;
            }
        }
        
        /* slack variable trajectory return; if it even exists */
        if (ns>0 && numReturnStepsS>0) {
        	if (numReturnStepsS>n+1) numReturnStepsS=n+1; /* clip if needed */
            pmx=mxCreateUninitNumericMatrix(numReturnStepsS,ns,mxDOUBLE_CLASS,mxREAL);
            mxSetFieldByNumber(REPSTRUCT,0,REP_STRAJ,pmx);
            kk=nx+nu; pdd=(double *)mxGetPr(pmx);
            for (ll=0;ll<numReturnStepsS;ll++) {
                /* state vector for stage ll should go into row ll */
                for (mm=0;mm<ns;mm++)
                    pdd[mm*numReturnStepsS+ll]=qpRet.x[kk+mm];
                kk+=nd;
            }
        }
        
    } else {
        /* Did not converge to solution; still set the "iterations" field though */
        mxSetFieldByNumber(REPSTRUCT,0,REP_ISCONVERGED,mxCreateDoubleScalar(0.0));
    }
    
    /* Evaluate the part of the cost that comes from the slack terms only;
       Return it in the report field "fxoft"; it is very easy to compute 
       due to the implicit diagonal structure of the S matrix.
     */
    if (ns>0) {
      dd=0.0; kk=nx+nu;
      for (qq=0;qq<n+1;qq++) {
      	pdd=(double *)&qpRet.x[kk];
      	for (ll=0;ll<ns;ll++) dd+=pdd[ll]*pS[ll]*pdd[ll];
      	kk+=nd;
      }
      mxSetFieldByNumber(REPSTRUCT,0,REP_FXOFT,mxCreateDoubleScalar(dd));
    }
    
    /* Free working vector buffer (NOTE: invalidates pointer in qpRet) */
    msqp_pdipm_free(&qpDat,qpOpt.verbosity);
    
    /* Free up stage memory */
    if (pStages!=NULL) mxFree(pStages);
    /* Free up aux. memory block(s) */
    if (pauxbuf!=NULL) mxFree(pauxbuf);
    
    if (qpOpt.expl_sparse>0) {
    	sparseMatrixDestroy(&spJay);
			sparseMatrixDestroy(&spDee);
    }
    
    /* pointless return of [nx nu nq ns] (REMOVE?) */
    pmx=mxCreateUninitNumericMatrix(1,4,mxDOUBLE_CLASS,mxREAL);
    mxSetFieldByNumber(REPSTRUCT,0,REP_NXNUNQNS,pmx);
    pdd=(double *)mxGetPr(pmx);
    pdd[0]=(double)nx;
    pdd[1]=(double)nu;
    pdd[2]=(double)nq;
    pdd[3]=(double)ns;
    
    /* Fill up the output report struct */
    mxSetFieldByNumber(REPSTRUCT,0,REP_SOLVERPROGRAM,mxCreateString(__FILE__));
    mxSetFieldByNumber(REPSTRUCT,0,REP_MEXTIMESTAMP,mxCreateString(__TIMESTAMP__));
    mxSetFieldByNumber(REPSTRUCT,0,REP_NUMSTAGES,mxCreateDoubleScalar((double)(n+1))); /* qpDat.nstg */
    mxSetFieldByNumber(REPSTRUCT,0,REP_QPCLASS,mxCreateString(prblm_class_names[prblmClass]));
    mxSetFieldByNumber(REPSTRUCT,0,REP_EPSOPT,mxCreateDoubleScalar(qpOpt.ep));
    mxSetFieldByNumber(REPSTRUCT,0,REP_ETAOPT,mxCreateDoubleScalar(qpOpt.eta));
    
    #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_toc1);
    dd=fclk_delta_timestamps(&_tic1,&_toc1);
    mxSetFieldByNumber(REPSTRUCT,0,REP_TOTALCLOCK,mxCreateDoubleScalar(dd));
    dd=fclk_delta_timestamps(&_tic2,&_toc2);
    mxSetFieldByNumber(REPSTRUCT,0,REP_SOLVECLOCK,mxCreateDoubleScalar(dd));
    fclk_get_resolution(&_tic1); dd=fclk_time(&_tic1);
    mxSetFieldByNumber(REPSTRUCT,0,REP_CLOCKRESOL,mxCreateDoubleScalar(dd));
    dd=qpRet.cholytime;
    if (dd>=0.0) { /* set to -1 if not used by solver; diagnostics/profilinf of code speed */
        mxSetFieldByNumber(REPSTRUCT,0,REP_CHOLYCLOCK,mxCreateDoubleScalar(dd));
    }
    #endif
    
    /*#ifdef __DEVELOPMENT_TEXT_OUTPUT__
    mexPrintf("auxbufsz=%i\n",auxbufsz);
    #endif*/

    return;
}

/*
 * Aux. sub-program to create symmetric cost-matrices.
 */

void aux_compute_sym_cost_matrix(
        int nd,int nx,int nu,int ny,double *pQstg,
        int typQ,double sQ,double *pQ,
        int typR,double sR,double *pR,
        int typW,double sW,double *pW,
        double *pC,double *pD) {
    
    /* pQstg points to the nd-by-nd storage for this matrix */
    matopc_zeros(pQstg,nd,nd);
    
    if (typQ==TYP_SCALAR) {
        matopc_sub_assign_diag_scalar(pQstg,nd,nd,0,0,2.0*sQ,nx,-1);
    } else if (typQ==TYP_VECTOR) {
        matopc_sub_assign_scaled_diag_vector(pQstg,nd,nd,0,0,pQ,nx,-1,2.0);
    } else { /*TODO: this could be made faster with a symmetrized assign */
        matopc_sub_assign_scaled(pQstg,nd,nd,0,0,pQ,nx,nx,-1,2.0);
    }
    
    if (typR==TYP_SCALAR) {
        matopc_sub_assign_diag_scalar(pQstg,nd,nd,nx,nx,2.0*sR,nu,-1);
    } else if (typR==TYP_VECTOR) {
        matopc_sub_assign_scaled_diag_vector(pQstg,nd,nd,nx,nx,pR,nu,-1,2.0);
    } else { /*TODO: this could be made faster with a symmetrized assign */
        matopc_sub_assign_scaled(pQstg,nd,nd,nx,nx,pR,nu,nu,-1,2.0);
    }
    
    if (pC!=NULL) {  /* C ny-by-nx exists */
        /* Add (2x) C'*W*C to upper-left nx-by-nx block (only add in upper triangle) */
        if (typW==TYP_SCALAR) { /* add sW*C'*C to block */
            matopc_sub_assign_sym_scaled_ctc(pQstg,nd,0,pC,ny,nx,1,2.0*sW);
        } else if (typW==TYP_VECTOR) { /* add C'*W*C, W=diag(w), w of length ny */
            matopc_sub_assign_sym_scaled_ctwc(pQstg,nd,0,pC,ny,nx,pW,1,2.0);
        } else { /* add C'*W*C, with W a general sym. matrix */
            matopc_sub_assign_sym_scaled_ctwc_gen(pQstg,nd,0,pC,ny,nx,pW,1,2.0);
        }
        if (pD!=NULL) { /* D ny-by-nu exists */
            /* General Q block should be 2*[Qx+C'*W*C,C'*W*D;D'*W*C,R+D'*W*D] */
            if (typW==TYP_SCALAR) {
                matopc_sub_assign_sym_scaled_ctc(pQstg,nd,nx,pD,ny,nu,1,2.0*sW);
                /* add off-diagonal block C'*W*D */
                matopc_sub_assign_scaled_ctd(pQstg,nd,0,nx,pC,ny,nx,pD,nu,1,2*sW);
            } else if (typW==TYP_VECTOR) {
                matopc_sub_assign_sym_scaled_ctwc(pQstg,nd,nx,pD,ny,nu,pW,1,2.0);
                matopc_sub_assign_scaled_ctwd(pQstg,nd,0,nx,pC,ny,nx,pD,nu,pW,1,2.0);
            } else {
                matopc_sub_assign_sym_scaled_ctwc_gen(pQstg,nd,nx,pD,ny,nu,pW,1,2.0);
                /* Assign (i.e. not add) upper-triangle off-diagonal block C'*W*D */
                matopc_sub_assign_scaled_ctwd_gen(pQstg,nd,0,nx,pC,ny,nx,pD,nu,pW,-1,2.0);
            }
            
            /* C,D exist; reference r already processed above */
            
        } else { /* D does not exist */
            /* Q block should be 2*[Qx+C'*W*C,0;0,R]; it already is! */
            
            /* C exists; reference r already processed above */
        }
    } else {    /* no W,r or C or D */
        /* Q block should be 2*[Qx,0;0,R]; and it already is! */
    }
}


/*
 * Auxiliary I/O functions
 *
 */

int aux_search_for_numeric_struct_fields(
        const mxArray *STRU,const char **lookupnames,
        int numlookupnames,int *idx,int verbose) {
    mxArray *pmx;
    int numstrufields,qq,ll,errc;
    char *tmpstr;
    for (ll=0;ll<numlookupnames;ll++)
        idx[ll]=-1; /* mark each name as not found */
    if (!mxIsStruct(STRU)) {
        if (verbose>0)
            mexPrintf("[%s]: struct argument expected (2).\n",__func__);
        return -1;
    }
    if (mxGetNumberOfElements(STRU)!=1) {
        if (verbose>0)
            mexPrintf("[%s]: Struct argument should have 1 element only.\n",__func__);
        return -2;
    }
    errc=0;
    numstrufields=mxGetNumberOfFields(STRU);
    for (qq=0;qq<numstrufields;qq++) {
        tmpstr=(char *)mxGetFieldNameByNumber(STRU,qq);
        /* All input fields must be real-valued, non-sparse, non-complex */
        for (ll=0;ll<numlookupnames; ll++) {
            if (idx[ll]==-1) {
                if (strcmp(tmpstr,lookupnames[ll])==0) {
                    idx[ll]=qq;
                    pmx=mxGetFieldByNumber(STRU,0,qq);
                    if (mxGetNumberOfDimensions(pmx)!=(mwSize)2) {
                        idx[ll]=-3;
                        if (verbose>0)
                            mexPrintf("[%s]: data must not be multidimensional.\n",lookupnames[ll]);
                    } else {
                        if (mxGetN(pmx)>0 && mxGetM(pmx)>0) {
                            if (!mxIsDouble(pmx) || mxIsComplex(pmx) || mxIsSparse(pmx)) {
                                idx[ll]=-4;
                                if (verbose>0)
                                    mexPrintf("[%s]: expects double-type/real-valued/non-sparse data.\n",lookupnames[ll]);
                            }
                        } else {
                            /* field name found but it is empty; this does not provoke the error counter */
                            idx[ll]=-2;
                        }
                    }
                    /* Field was found and it is non-empty;
                     * but it did not have the correct properties;
                     * therefore bump up the error counter.
                     */
                    if (idx[ll]!=qq && idx[ll]!=-2) errc++;
                }
            }
        }
    }
    /* "total" success means errc=0 */
    return errc;
}

int aux_read_square_matrix(
        mxArray *pmx,int n,
        int *typ,double *sclr,double **ptr) {
    int retval=1;
    if (mxGetM(pmx)==1 && mxGetN(pmx)==1) {
        *typ=TYP_SCALAR; *sclr=mxGetScalar(pmx); *ptr=sclr;
    } else if ((mxGetM(pmx)==n && mxGetN(pmx)==1) ||
            (mxGetM(pmx)==1 && mxGetN(pmx)==n)) {
        /* interpret the n numbers as the diagonal in a matrix */
        *typ=TYP_VECTOR; *sclr=-1; *ptr=mxGetPr(pmx);
    } else if (mxGetM(pmx)==n && mxGetN(pmx)==n) {
        /* full matrix n-by-n */
        *typ=TYP_MATRIX; *sclr=-1; *ptr=mxGetPr(pmx);
    } else {
        *typ=TYP_UNDEF; *sclr=0; *ptr=NULL;
        retval=0;
    }
    return retval;
}

int aux_read_signal_matrix(
        mxArray *pmx,int n,int nt,
        int *typ,double *sclr,double **ptr) {
    int retval=1;
    if (mxGetM(pmx)==1 && mxGetN(pmx)==1) {
        /* Check first if scalar */
        *typ=TYP_SCALAR; *sclr=mxGetScalar(pmx); *ptr=sclr;
    } else if ((mxGetM(pmx)==n && mxGetN(pmx)==1) ||
            (mxGetM(pmx)==1 && mxGetN(pmx)==n)) {
        /* n-vector, row or column; assumed to be constant over nt timesteps */
        *typ=TYP_VECTOR; *sclr=-1; *ptr=mxGetPr(pmx);
    } else if (mxGetM(pmx)==n && mxGetN(pmx)==nt) {
        /* full matrix n-by-nt */
        *typ=TYP_MATRIX; *sclr=-1; *ptr=mxGetPr(pmx);
    } else if (mxGetM(pmx)==nt && mxGetN(pmx)==n) {
        /* full matrix nt-by-n (transposed, non-transposed has precedence) */
        *typ=TYP_MATRIXT; *sclr=-1; *ptr=mxGetPr(pmx);
    } else {
        *typ=TYP_UNDEF; *sclr=0; *ptr=NULL;
        retval=0;
    }
    return retval;
}

void aux_print_array(double *A,int m,int n) {
    int rr,cc;
    for (rr=0;rr<m;rr++) {
        for (cc=0;cc<n;cc++) {
            mexPrintf("%f ",A[cc*m+rr]);
        }
        mexPrintf("\n");
    }
}

void aux_print_array_sparsity(double *A,int m,int n) {
    int rr,cc;
    for (rr=0;rr<m;rr++) {
        for (cc=0;cc<n;cc++) {
            if (A[cc*m+rr]!=0.0)
                mexPrintf("* ");
            else
                mexPrintf("- ");
        }
        mexPrintf("\n");
    }
}

