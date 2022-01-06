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

//#include "matrix.h"
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

#define PARSE_VERBOSITY 1
#define MINIMUM_REQUIRED_STAGES 3

/* NOTE: memory allocation should be done using the
 * mxMalloc(.) and mxFree(.) routines supplied by the MATLAB API.
 */

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

/* Redundant sparse matrix CCS/CRS representation */
typedef struct sparseMatrix {
		int m;
		int n;
		char *buf;
    int nnz;
    double *ccsval;
    int *rowind;
    int *colptr;
    double *crsval;
    int *colind;
    int *rowptr;
} sparseMatrix;

void sparseMatrixDestroy(sparseMatrix *M) {
	if (M->buf!=NULL) { mxFree(M->buf); M->buf=NULL; }
}

/* return 1 if OK, 0 if allocation error, -1 if refused to create ("too dense" matrix) */
int sparseMatrixCreate(sparseMatrix *M,double *A,int m,int n) {
	M->buf=NULL;
	int nnz=matopc_nnz(A,m,n);
	if ((double)nnz/(double)(m*n)>0.50) return -1;
	int numbytes=
		2*(nnz)*sizeof(double)+2*(nnz)*sizeof(int)+
		(m+1)*sizeof(int)+(n+1)*sizeof(int);
	char *buf=mxMalloc(numbytes);
	if (buf==NULL) return 0;
	M->nnz=nnz;
	M->m=m;
	M->n=n;
	M->buf=buf;
	int byteofs=0;
	M->ccsval=(double *)&buf[byteofs]; byteofs+=nnz*sizeof(double);
	M->rowind=(int *)&buf[byteofs]; byteofs+=nnz*sizeof(int);
	M->colptr=(int *)&buf[byteofs]; byteofs+=(n+1)*sizeof(int);
	M->crsval=(double *)&buf[byteofs]; byteofs+=nnz*sizeof(double);
	M->colind=(int *)&buf[byteofs]; byteofs+=nnz*sizeof(int);
	M->rowptr=(int *)&buf[byteofs]; byteofs+=(m+1)*sizeof(int);
	if (byteofs!=numbytes) {
		mexPrintf("ERROR: byteofs=%i (it should be=%i)\n",byteofs,numbytes);
		sparseMatrixDestroy(M);
		return 0;
	}
	/* Buffers ready; create CCS (compressed column storage) */
	matopc_create_ccs(A,m,n,
		M->ccsval,M->rowind,M->colptr);
	/* Based on the CCS: create CRS (compressed row storage) */
	matopc_ccs2crs(m,n,M->ccsval,M->rowind,M->colptr,
		M->crsval,M->colind,M->rowptr);
	return 1;
}

/* Internally creates Jstg, Cstg, Dstg and auto-detects basic sparsity and
 * exploits the special structure of Dstg and Jstg.
 *
 * Internally allocates all the needed memory based on the problem dimensions.
 * All linear algebra subprograms should be contained in the "matrixopsc.h" header file
 * as static inline code that supposedly is expanded in place as macros. 
 *
 * Aux. memory required is allocated with only few calls to mxMalloc(..).
 * pbuf points to storage with this arrangement:
 *
 * --------------------------------------------------
 * NAME             SIZE (# DOUBLES)        EXPL.
 * --------------------------------------------------
 * d                nx*(n+1)
 * q                nd*(n+1)
 * f                ni*(n+1)
 * J                ni*nd                   [F1, F2]
 * Cstg             nx*nd                   [-A, -B]
 * Dstg             nx*nd                   [I, 0]
 * Cstg0            2*nx*nd                 [I,0;Cstg]
 * Dstg1            2*nx*nd                 [0,0;Dstg]
 * Q                nd*nd                   sym.
 * Qn               nd*nd                   sym. terminal
 * q0               n+1
 * r                ny*(n+1)                aux.ref.
 * rtmp             ny
 * CC1              nd*(nd+2)               Chol. cache #1
 * CC2              nd*(nd+2)               -- #2
 * --------------------------------------------------
 *
 * NOTE 1: ni=nq+ns, where nq is the # rows of F1 and F2
 * and ns is the number of slack variables per stage.
 * NOTE 2: nd=nx+nu+ns.
 * NOTE 3: with no "softened" constraints, ns=0.
 *
 * If ns>0 the J matrix gets a different structure;
 * but most of the problem data is simply padded with zeros.
 *
 */

/* Meta-data for each time-stage idx=0..n.
 *
 * The number crunching routines operate on an array of 
 * these structures; the pointers typically point to the same
 * matrices for each stage if LTI system (except maybe terminal/initial
 * stages with special cost/structure).
 *
 */
typedef struct stageStruct {
    int idx;
    int nd;
    int neq;
    int niq;
    double *ptrQ;
    double *ptrq;
    double *ptrq0;
    double *ptrJ;
    double *ptrf;
    double *ptrC;
    double *ptrd;
    double *ptrD;
    /* pointer to cached factorization of Q; not always used */
    double *ptrL;
    /* sparse representations; not always used */
    sparseMatrix *ptrJsp;
    sparseMatrix *ptrDsp;
} stageStruct;

typedef struct qpdatStruct {
    int nstg;
    stageStruct *pstg;
    double *ph;
    double *pd;
    double *pf;
    int ndec;
    int neq;
    int niq;
    /* All fields below are initialized from (nstg,pstg) */
	int netot;
	int ndtot;
	int nemax;
	int ndmax;
    /* These fields are for Cholesky factor blocks */
	int blklysz;
	double *blkly;
	int blkphsz;
	double *blkph;
	int blkwrsz;
	double *blkwr;
    /* These fields are for PDIPM working vectors */
    int vecwrsz;
    double *vecwr;
} qpdatStruct;

#define CLIP_MAXITERS 250
#define DEFAULT_OPT_MAXITERS 50
#define DEFAULT_OPT_EP 1.0e-8
#define DEFAULT_OPT_ETA 0.96
#define DEFAULT_OPT_VERBOSITY 0

typedef struct qpoptStruct {
    int maxiters;
    double ep;
    double eta;
    int chol_update;    /* if nq<nx, can be faster */
    int expl_sparse;    /* exploit sparsity of J,D if pos. */
    int verbosity;
    int blas_suite;     /* which suite of basic lin.alg. subprogs. ? */
} qpoptStruct;

/* Return meta-data */
typedef struct qpretStruct {
    int nx;
    double *x;
    double fxofs;
    double fxopt;
    int iters;
    int converged;
    double cholytime;
    double inftuple[4];
} qpretStruct;

/*
 * Main algorithm(s) interface function declarations.
 */

int InitializePrblmStruct(qpdatStruct *dat,int whichmem,int verbosity);
void FreePrblmStruct(qpdatStruct *dat,int verbosity);
int CreateCholeskyCache(qpdatStruct *dat,double *pCC1,double *pCC2,int ndassert);

int msqp_pdipm_init(qpdatStruct *qpd,int verbosity);
void msqp_pdipm_free(qpdatStruct *qpd,int verbosity);

int msqp_pdipm_solve(qpdatStruct *qpd,qpoptStruct *qpo,qpretStruct *qpr);
int msqp_solve_niq(qpdatStruct *qpd,qpoptStruct *qpo,qpretStruct *qpr);

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
    if (n+1<MINIMUM_REQUIRED_STAGES)   /* confirm that n+1>=MINIMUM_REQUIRED_STAGES */
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
    qpOpt.verbosity=DEFAULT_OPT_VERBOSITY;
    qpOpt.maxiters=DEFAULT_OPT_MAXITERS;
    qpOpt.ep=DEFAULT_OPT_EP;
    qpOpt.eta=DEFAULT_OPT_ETA;
    qpOpt.expl_sparse=0;
    qpOpt.chol_update=0;
    qpOpt.blas_suite=0;
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
        qq=CreateCholeskyCache(&qpDat,pCC1,pCC2,nd);
        if (qq!=0) mexErrMsgTxt("Fatal block Cholesky factorization failure.\n");
        
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
 * Aux. computational routines that operates on the stage struct array.
 * These sub-programs operate with the block structure defined through
 * the array of pointers and meta-data stored in the QP struct.
 */

/* Given decision vector x, evaluate the QP cost objective 
 * store bias in y0, linear term in y1, quadratic in y2... */
void qpcost(qpdatStruct *qpd,double *px,double *py) {
    double y0=0.0,y1=0.0,y2=0.0;
    stageStruct *pstg=qpd->pstg;
    int ii,ll=0,nd;
    for (ii=0;ii<(qpd->nstg);ii++) {
        nd=pstg[ii].nd;
        y0+=*(pstg[ii].ptrq0);
        /* y1+=vecop_dot(nd,&(qpd->ph[ll]),&px[ll]); */
        y1+=vecop_dot(nd,pstg[ii].ptrq,&px[ll]);
        y2+=matopc_xtax_sym(pstg[ii].ptrQ,nd,&px[ll]);
        ll+=nd;
    }
    py[0]=y0; py[1]=y1; py[2]=0.5*y2;
    #ifdef __CLUMSY_ASSERTIONS__
    if (ll!=qpd->ndec) mexPrintf("ERROR:[%s]\n",__func__);
    #endif
}

/* y <- J*x, J block structured [ineq. constraints] */
void Jmult(qpdatStruct *qpd,double *px,double *py) {
    stageStruct *pstg=qpd->pstg;
    sparseMatrix *psp; /* Exploit J's CRS data for stage if it exists */
    int ii,ll=0,rr=0,nd,niq;
    for (ii=0;ii<(qpd->nstg);ii++) {
        nd=pstg[ii].nd; niq=pstg[ii].niq;
        psp=pstg[ii].ptrJsp;
        /*matopc_ypax(&py[rr],0.0,1.0,pstg[ii].ptrJ,niq,nd,&px[ll]);*/
        if (psp!=NULL) {
        	matopc_crs_ypax(&py[rr],0.0,1.0,niq,nd,psp->crsval,psp->colind,psp->rowptr,&px[ll]);
        } else {
        	matopc_ax(&py[rr],pstg[ii].ptrJ,niq,nd,&px[ll]);
        }
        ll+=nd; rr+=niq;
    }
    #ifdef __CLUMSY_ASSERTIONS__
    if (ll!=qpd->ndec) mexPrintf("ERROR(1):[%s]\n",__func__);
    if (rr!=qpd->niq) mexPrintf("ERROR(2):[%s]\n",__func__);
    #endif
}

/* y <- J'*x, J block structured [ineq. constraints] */
void Jtmult(qpdatStruct *qpd,double *px,double *py) {
    stageStruct *pstg=qpd->pstg;
    sparseMatrix *psp; /* Exploit J's CCS data for stage if it exists */
    int ii,ll=0,rr=0,nd,niq;
    for (ii=0;ii<(qpd->nstg);ii++) {
        nd=pstg[ii].nd; niq=pstg[ii].niq;
        psp=pstg[ii].ptrJsp;
        /*matopc_ypatx(&py[ll],0.0,1.0,pstg[ii].ptrJ,niq,nd,&px[rr]);*/
        if (psp!=NULL) {
        	matopc_ccs_ypatx(&py[ll],0.0,1.0,niq,nd,psp->ccsval,psp->rowind,psp->colptr,&px[rr]);
        } else {
        	matopc_atx(&py[ll],pstg[ii].ptrJ,niq,nd,&px[rr]);
        }
        ll+=nd; rr+=niq;
    }
    #ifdef __CLUMSY_ASSERTIONS__
    if (ll!=qpd->ndec) mexPrintf("ERROR(1):[%s]\n",__func__);
    if (rr!=qpd->niq) mexPrintf("ERROR(2):[%s]\n",__func__);
    #endif
}

/* y <- C*x, C block structured [eq. constraints] */
void Cmult(qpdatStruct *qpd,double *px,double *py) {
    stageStruct *pstg=qpd->pstg;
    sparseMatrix *psp; /* Exploit D's CRS data for stage if it exists */
    int ii,ll=pstg[0].nd,rr=0,nd,nd0,neq;
    for (ii=1;ii<(qpd->nstg);ii++) {
        nd0=pstg[ii-1].nd; nd=pstg[ii].nd; neq=pstg[ii-1].neq;
        matopc_ax(&py[rr],pstg[ii-1].ptrC,neq,nd0,&px[ll-nd0]);
        psp=pstg[ii].ptrDsp;
        if (psp!=NULL) {
        	matopc_crs_ypax(&py[rr],1.0,1.0,neq,nd,psp->crsval,psp->colind,psp->rowptr,&px[ll]);
        } else {
	        matopc_ypax(&py[rr],1.0,1.0,pstg[ii].ptrD,neq,nd,&px[ll]);
        }
        rr+=neq; ll+=nd;
    }
    #ifdef __CLUMSY_ASSERTIONS__
    if (ll!=qpd->ndec) mexPrintf("ERROR(1):[%s]\n",__func__);
    if (rr!=qpd->neq) mexPrintf("ERROR(2):[%s]\n",__func__);
    #endif
}

/* y <- C'*x, C block structured [eq. constraints] */
void Ctmult(qpdatStruct *qpd,double *px,double *py) {
    stageStruct *pstg=qpd->pstg;
    sparseMatrix *psp; /* Exploit D's CCS data for stage if it exists */
    int ndi=pstg[0].nd,neqi=pstg[0].neq,neqi0=-1,ii,jj,kk; /* neqi0 dummy init to fool -Wall */
    matopc_atx(&py[0],pstg[0].ptrC,neqi,ndi,&px[0]); /* 1st mult */
    jj=ndi; kk=neqi;
    for (ii=1;ii<(qpd->nstg-1);ii++) {
        neqi0=pstg[ii-1].neq; ndi=pstg[ii].nd; neqi=pstg[ii].neq;
        /* TODO: here neqi0==neqi always so eliminate redundant local variable yeah ?! */
        matopc_atx(&py[jj],pstg[ii].ptrC,neqi,ndi,&px[kk]);
        psp=pstg[ii].ptrDsp;
        if (psp!=NULL) {
        	matopc_ccs_ypatx(&py[jj],1.0,1.0,neqi0,ndi,psp->ccsval,psp->rowind,psp->colptr,&px[kk-neqi0]);
        } else {
        	matopc_ypatx(&py[jj],1.0,1.0,pstg[ii].ptrD,neqi0,ndi,&px[kk-neqi0]);
        }
        jj+=ndi; kk+=neqi;
    }
    #ifdef __CLUMSY_ASSERTIONS__
    if (ii!=qpd->nstg-1) mexPrintf("ERROR(0):[%s]\n",__func__);
    #endif
    ndi=pstg[qpd->nstg-1].nd;
    matopc_atx(&py[jj],pstg[qpd->nstg-1].ptrD,neqi,ndi,&px[kk-neqi0]); /* last mult */
    jj+=ndi;
    #ifdef __CLUMSY_ASSERTIONS__
    if (jj!=qpd->ndec) mexPrintf("ERROR(1):[%s]\n",__func__);
    if (kk!=qpd->neq) mexPrintf("ERROR(2):[%s]\n",__func__);
    #endif
}

/* y <- H*x, H block structured [Hessian matrix] */
void Hmult(qpdatStruct *qpd,double *px,double *py) {
    /* !!! TODO: candidate for sparsity exploitation also, autodetect diagonal Hessian at least... */
    stageStruct *pstg=qpd->pstg;
    int ii,ll=0,nd;
    for (ii=0;ii<(qpd->nstg);ii++) {
        nd=pstg[ii].nd;
        matopc_sym_ax_upper(&py[ll],pstg[ii].ptrQ,nd,&px[ll]);
        ll+=nd;
    }
    #ifdef __CLUMSY_ASSERTIONS__
    if (ll!=qpd->ndec) mexPrintf("ERROR:[%s]\n",__func__);
    #endif
}

/* ************************************************* */
/* Given a prblmStruct pointer which is initialized with nstg/pstg; fill in the other fields
 * If bit 1 is set for whichmem, then allocate memory for blocks of Phi.
 * If bit 2 is set for whichmem, then allocate memory for blocks of LY.
 * If bit 3 is set for whichmem, then allocate memory for temporary local block variables.
 */
int InitializePrblmStruct(qpdatStruct *dat,int whichmem,int verbosity) {
	int ii,N,errc;
	int ne,nemax,nd,ndmax;
	int ofsl,ofsp;
	stageStruct *pstg;
	if (dat==NULL) return 0;
	if (dat->pstg==NULL) return 0;
	if (dat->nstg<MINIMUM_REQUIRED_STAGES) return 0;
	errc=0;
	dat->blkly=NULL;
	dat->blkph=NULL;
	dat->blkwr=NULL;
    /* TODO: break and exit if ne=0 or nd=0 somewhere... */
	pstg=dat->pstg;
	N=dat->nstg-1;
	nd=0; ne=0; nemax=-1; ndmax=-1;
	ofsp=0;
	for (ii=0;ii<(N+1);ii++) {
		nd+=pstg[ii].nd;
		ofsp+=pstg[ii].nd*(pstg[ii].nd+1); /* space for nd-by-nd square plus a nd-by-1 column */
		if (pstg[ii].nd>ndmax) ndmax=pstg[ii].nd;
		if (ii!=N) {
			ne+=pstg[ii].neq;
			if (pstg[ii].neq>nemax) nemax=pstg[ii].neq;
		}
	}
	dat->ndtot=nd;
	dat->netot=ne;
	dat->ndmax=ndmax;
	dat->nemax=nemax;
	dat->blkwrsz=ndmax*(ndmax+1)+2*ndmax*nemax+2*nemax*(nemax+1)+ndmax;
	dat->blkphsz=ofsp;
	ofsl=0;
	for (ii=0;ii<N;ii++) {
		/* square block LY[i,i]; plus space for one extra column */
		ofsl+=pstg[ii].neq*(pstg[ii].neq+1);
		if (ii<N-1) {
			/* for rectangular block LY[i+1,i] */
			ofsl+=pstg[ii+1].neq*pstg[ii].neq;
		}
	}
	dat->blklysz=ofsl;
    if (verbosity>2) {
        mexPrintf("[%s]: ndmax=%i, nemax=%i, ndtot=%i, netot=%i\n",
                __func__,dat->ndmax,dat->nemax,dat->ndtot,dat->netot);
        mexPrintf("[%s]: blkphsz=%i, blklysz=%i,blkwrsz=%i\n",
                __func__,dat->blkphsz,dat->blklysz,dat->blkwrsz);
    }
	/* Allocate memory if requested based on bit pattern of argument whichmem */
	if (whichmem & 0x0001) {
		dat->blkph=(double *)mxMalloc(dat->blkphsz*sizeof(double));
		if (dat->blkph==NULL) errc++;
		/*mexPrintf("[%s]: malloc(blkph).\n",__func__);*/
	}
	if (whichmem & 0x0002) {
		dat->blkly=(double *)mxMalloc(dat->blklysz*sizeof(double));
		if (dat->blkly==NULL) errc++;
		/*mexPrintf("[%s]: malloc(blkly).\n",__func__);*/
	}
	if (whichmem & 0x0004) {
		dat->blkwr=(double *)mxMalloc(dat->blkwrsz*sizeof(double));
		if (dat->blkwr==NULL) errc++;
		/*mexPrintf("[%s]: malloc(blkwr).\n",__func__);*/
	}
	return (errc==0 ? 1 : 0);
}

/* Free memory; but do not touch nstg/pstg */
void FreePrblmStruct(qpdatStruct *dat,int verbosity) {
	int fc=0;
	if (dat->blkph!=NULL) {
		mxFree(dat->blkph);
		dat->blkph=NULL;
		fc++;
	}
	if (dat->blkly!=NULL) {
		mxFree(dat->blkly);
		dat->blkly=NULL;
		fc++;
	}
	if (dat->blkwr!=NULL) {
		mxFree(dat->blkwr);
		dat->blkwr=NULL;
		fc++;
	}
    if (verbosity>2) {
        mexPrintf("[%s]: num.freed=%i.\n",__func__,fc);
    }
}
/* ************************************************* */

/* Factorize the "unmodified" stage cost matrices;
 * This can be used to save many flops if nq<<nd, using
 * sequences of rank-1 updates instead of full-block re-decomposition.
 * See further the next two functions below.
 *
 * Note that there will be a maximum of 2 blocks (each size nd);
 * the cost Q and the terminal cost Qn (if it exists only).
 *
 */
int CreateCholeskyCache(
        qpdatStruct *dat,
        double *pCC1,double *pCC2,
        int ndassert) {
    int chret,N,ii,ofs;
    stageStruct *pstg=dat->pstg;
    N=dat->nstg-1;
    if (pstg[0].nd!=ndassert) return 1; /* make sure this is a valid use-case */
    matopc_copy(pCC1,pstg[0].ptrQ,ndassert,ndassert);
    ofs=ndassert*ndassert;
    chret=matopc_cholesky_decompose(pCC1,&pCC1[ofs],ndassert);
    if (chret!=0) return 1;
    for (ii=0;ii<=N;ii++) pstg[ii].ptrL=pCC1;
    if (pstg[N].ptrQ!=pstg[0].ptrQ) {
        /* initialize also pCC2; and modify last-stage ptrL */
        if (pstg[N].nd!=ndassert) return 1; /* make sure this is a valid use-case */
        matopc_copy(pCC2,pstg[N].ptrQ,ndassert,ndassert);
        chret=matopc_cholesky_decompose(pCC2,&pCC2[ofs],ndassert);
        if (chret!=0) return 1;
        pstg[N].ptrL=pCC2;
    }
    return 0;
}

/* Given a cholesky factorization L*L'=Q (size nd), represented by
 * data L,d; update (L,d) to represent the Cholesky factorization of
 * instead Q+J'*diag(l)*J, where J is nq-by-nd and l is an nq vector.
 *
 * works through the sequence of rows of J, and uses temporary
 * storage for sqrt(l(j))*row(J) for updating vectors.
 *
 * wrkx is a user provided work array (must have room for nd doubles).
 * wrkx=NULL will default to the static temp array defined in matopsc.h.
 */
static inline void chol_nq_rank1_updates(
        double *L,double *d,int nd,
        double *J,double *l,int nq,
        double *wrkx) {
    double dd;
    double *x=matopc_temparray;
    if (wrkx!=NULL) x=wrkx;
    int jj,cc,ofs;
    for (jj=0;jj<nq;jj++) {
        dd=sqrt(l[jj]); /* TODO: implement updater without the square root */
        for (cc=0,ofs=jj;cc<nd;cc++,ofs+=nq) x[cc]=dd*J[ofs];
        matopc_cholesky_update(L,d,nd,x);
    }
}

/* Cholesky factorization of the implied block-structured matrix Y.
 * See Domahidi's CDC 2012 paper.
 *
 * TODO: "autodetect" diagonal blocks to save lots of comp. if possible.
 *
 * Return 0 if factorization succeeded.
 * Return 1 if Cholesky failure during block factoring Phi.
 * Return 2 if Cholesky failure during block factoring Y.
 *
 */

/* 
 * Block-based Cholesky factorization: Y=C*inv(Phi)*C'=L*L'.
 * Returns 0 if successful and nonzero if block
 * Cholesky factorization error (or other error) occurs.
 * Cholesky errors specifically yield negative error code.
 * Stage index and L or Y factor error can be unpacked if so desired.
 */
int BlkCholFactorizeY(
        qpdatStruct *dat,
        int storePhi,
        int useJay,
        double *v,
        int useCache) {
	stageStruct *pstg=dat->pstg;
	sparseMatrix *psp;
	int nstg=dat->nstg;
	double *tmpbuf=NULL;
	double *blkph,*blkly;
	double *V,*W,*L,*Y0,*Y1,*tmpx;
	int tmpbufsz,ofsl,ofsp,ofsv;
	int ii,chret,dmy,storeph,nd1;
	int N=nstg-1;
	/* Buffer for Cholesky blocks of LY must be provided
	   (otherwise the code has no output!) */
	if (dat->blkly==NULL) {
		return 1;
	}
  /* If J-updates of Phi blocks are activated; then non-null v is required */
  if (useJay>0) if (v==NULL) return 2;
  /* Must provide external memory space for local block matrices */
	if (dat->blkwr==NULL) {
		return 3;
	} else {
		tmpbuf=dat->blkwr;
		tmpbufsz=dat->blkwrsz;
	}
	dmy=0;
	L=&tmpbuf[dmy];	dmy+=dat->ndmax*(dat->ndmax+1);
	V=&tmpbuf[dmy]; dmy+=dat->nemax*dat->ndmax;
	W=&tmpbuf[dmy]; dmy+=dat->nemax*dat->ndmax;
	Y0=&tmpbuf[dmy]; dmy+=dat->nemax*(dat->nemax+1);
	Y1=&tmpbuf[dmy]; dmy+=dat->nemax*(dat->nemax+1);
    tmpx=&tmpbuf[dmy]; dmy+=dat->ndmax; /* use this ndmax-sized vector during Cholesky updating if needed */
	if (tmpbufsz!=dmy) {
		mexPrintf("ERROR[%s]: memory offset mismatch (%i!=%i).\n",
			__func__,dmy,tmpbufsz);
		return 4;
	}
	blkph=dat->blkph; /* This is allowed to be NULL */
	blkly=dat->blkly;
	if (blkph!=NULL && storePhi>0) {
		storeph=1;
	} else {
		storeph=0;
	}

	/* TODO: automatic sparsity exploitation/skip-ahead for the W <- L\D' operation in the main loop */
	/* TODO: integrate special code for efficient handling of diagonal Q blocks */
	
	ofsl=0; ofsp=0; ofsv=0; nd1=pstg[0].nd;
  if (useCache>0) { /* Load cached factorization */
      matopc_copy(L,pstg[0].ptrL,nd1,nd1+1);
      if (useJay>0) { /* rank-1 updates if needed */
          chol_nq_rank1_updates(L,&L[nd1*nd1],nd1,pstg[0].ptrJ,&v[ofsv],pstg[0].niq,tmpx);
          ofsv+=pstg[0].niq;
      }
  } else {
      /*matopc_copy(L,pstg[0].ptrQ,nd1,nd1);*/
      matopc_copy_up2up(L,pstg[0].ptrQ,nd1,1);
      if (useJay>0) { /* L <- L+J'*diag(v)*J */
        psp=pstg[0].ptrJsp;
        if (psp!=NULL) {
	        matopc_ccs_cpatda(
       				L,1.0,pstg[0].niq,nd1,
       				psp->ccsval,psp->rowind,psp->colptr,
       				&v[ofsv]);
        } else {
           matopc_cpatda(L,nd1,pstg[0].ptrJ,pstg[0].niq,&v[ofsv],1.0,1.0,MATOPC_UPPER);
        }
          ofsv+=pstg[0].niq;
      }
      chret=matopc_cholesky_decompose(L,&L[nd1*nd1],nd1);
      if (chret!=0) return -(0+1);
  }
  if (storeph) matopc_copy(&blkph[ofsp],L,nd1,nd1+1); /* storage */
	ofsp+=nd1*(nd1+1);
	matopc_copy_transpose(V,pstg[0].ptrC,pstg[0].neq,nd1); /* V <- C[0]' */
	matopc_cholesky_trisubst_left_matrix(L,&L[nd1*nd1],nd1,V,V,pstg[0].neq); /* V <- L\V */
	for (ii=0;ii<N-1;ii++) {
        nd1=pstg[ii+1].nd;
        if (useCache>0) {
            matopc_copy(L,pstg[ii+1].ptrL,nd1,nd1+1);
            if (useJay>0) { /* rank-1 updates if needed */
                chol_nq_rank1_updates(L,&L[nd1*nd1],nd1,pstg[ii+1].ptrJ,&v[ofsv],pstg[ii+1].niq,tmpx);
                ofsv+=pstg[ii+1].niq;
            }
        } else {
            /*matopc_copy(L,pstg[ii+1].ptrQ,nd1,nd1);*/
            matopc_copy_up2up(L,pstg[ii+1].ptrQ,nd1,1);
            if (useJay>0) { /* L <- L+J'*diag(v)*J */
            		/* Check if sparse J is setup; use its CCS storage in that case; otherwise dense update */
            		psp=pstg[ii+1].ptrJsp;
            		if (psp!=NULL) {
            			matopc_ccs_cpatda(
            				L,1.0,pstg[ii+1].niq,nd1,
            				psp->ccsval,psp->rowind,psp->colptr,
            				&v[ofsv]); /* only updates upper part */
            		} else {
                	matopc_cpatda(L,nd1,pstg[ii+1].ptrJ,pstg[ii+1].niq,&v[ofsv],1.0,1.0,MATOPC_UPPER);
                }
                ofsv+=pstg[ii+1].niq;
            }
    		chret=matopc_cholesky_decompose(L,&L[nd1*nd1],nd1);
    		if (chret!=0) return -(ii+1);
        }
		if (storeph) matopc_copy(&blkph[ofsp],L,nd1,nd1+1); /* storage */
		ofsp+=nd1*(nd1+1);
		psp=pstg[ii+1].ptrDsp;
		if (psp!=NULL) { /* Exploit CRS data for D */
		/*	matopc_cholesky_trisubst_left_tr_matrix(L,&L[nd1*nd1],nd1,pstg[ii+1].ptrD,W,pstg[ii].neq,tmpx); */
			matopc_crs_cholesky_trisubst_left_tr_matrix(
        L,&L[nd1*nd1],nd1,
        W,pstg[ii].neq,
        psp->crsval,psp->colind,psp->rowptr,
        tmpx);
		} else {
		  /* W <- L[ii+1]\D[ii+1]' */
		  matopc_copy_transpose(W,pstg[ii+1].ptrD,pstg[ii].neq,pstg[ii+1].nd); /* W <- D[ii+1]' */
		  matopc_cholesky_trisubst_left_matrix(L,&L[nd1*nd1],nd1,W,W,pstg[ii].neq); /* W <- L[ii+1]\W */
		  /*matopc_cholesky_trisubst_left_tr_matrix(L,&L[nd1*nd1],nd1,pstg[ii+1].ptrD,W,pstg[ii].neq,tmpx);*/
		}
		/* Y[ii,ii] <- V'*V+W'*W */
		matopc_mtm(Y0,V,pstg[ii].nd,pstg[ii].neq,MATOPC_UPPER); /* assign V'*V */
		matopc_spmtm(Y0,W,nd1,pstg[ii].neq,MATOPC_UPPER); /* add W'*W to block */
		/* Y[ii,ii] available: factorize to get L[ii,ii] */
		if (ii>0) { /* "downdate" Y0 <- Y0-U'*U, then factorize Y0 in place (not for first iteration though) */
			matopc_smmtm(Y0,Y1,pstg[ii-1].neq,pstg[ii].neq,MATOPC_UPPER);
		}
		/* Factorize Y0 in-place */
		chret=matopc_cholesky_decompose(Y0,&Y0[pstg[ii].neq*pstg[ii].neq],pstg[ii].neq);
		if (chret!=0) return -(N+ii+1);
		matopc_copy(&blkly[ofsl],Y0,pstg[ii].neq,pstg[ii].neq+1); /* STORAGE LINE */
		ofsl+=pstg[ii].neq*(pstg[ii].neq+1);
		/* Overwrite V with L[ii+1]\C[ii+1]' */
		matopc_copy_transpose(V,pstg[ii+1].ptrC,pstg[ii+1].neq,nd1); /* V <- C[ii+1]' */
		matopc_cholesky_trisubst_left_matrix(L,&L[nd1*nd1],nd1,V,V,pstg[ii+1].neq); /* V <- L\V */
/*		matopc_cholesky_trisubst_left_tr_matrix(L,&L[nd1*nd1],nd1,pstg[ii+1].ptrC,V,pstg[ii+1].neq,tmpx);*/
		/* so that Y[ii,ii+1] is W'*V, then go to next iteration... */
		matopc_atb(Y1,W,nd1,pstg[ii].neq,V,pstg[ii+1].neq); /* Y[ii,ii+1] <- W'*V */
		/* Y[ii,ii+1] is available: backsolve in-place with last L[ii,ii] to get U[ii] rectangle for next iteration. */
		matopc_cholesky_trisubst_left_matrix(Y0,&Y0[pstg[ii].neq*pstg[ii].neq],pstg[ii].neq,Y1,Y1,pstg[ii+1].neq); // U <- L\Y
		matopc_copy_transpose(&blkly[ofsl],Y1,pstg[ii].neq,pstg[ii+1].neq);	/* STORAGE LINE: copy U' as the sub-diagonal Cholesky block */
		ofsl+=pstg[ii+1].neq*pstg[ii].neq;
	}
  #ifdef __CLUMSY_ASSERTIONS__
  if (ii!=N-1) mexPrintf("[%s]: iteration counter error!\n",__func__);
  #endif
	/* Last block Y[N-1,N-1]: */
  nd1=pstg[ii+1].nd;
  if (useCache>0) {
      matopc_copy(L,pstg[ii+1].ptrL,nd1,nd1+1);
      if (useJay>0) { /* rank-1 updates if needed */
          chol_nq_rank1_updates(L,&L[nd1*nd1],nd1,pstg[ii+1].ptrJ,&v[ofsv],pstg[ii+1].niq,tmpx);
          ofsv+=pstg[ii+1].niq;
      }
  } else {
      /*matopc_copy(L,pstg[ii+1].ptrQ,nd1,nd1);*/
      matopc_copy_up2up(L,pstg[ii+1].ptrQ,nd1,1);
      if (useJay>0) { /* L <- L+J'*diag(v)*J */
      	psp=pstg[ii+1].ptrJsp;
      	if (psp!=NULL) {
         	matopc_ccs_cpatda(
    			L,1.0,pstg[ii+1].niq,nd1,
     				psp->ccsval,psp->rowind,psp->colptr,
     				&v[ofsv]);
      	} else {
          matopc_cpatda(L,nd1,pstg[ii+1].ptrJ,pstg[ii+1].niq,&v[ofsv],1.0,1.0,MATOPC_UPPER);
        }
        ofsv+=pstg[ii+1].niq;
      }
      chret=matopc_cholesky_decompose(L,&L[nd1*nd1],nd1);
      if (chret!=0) return -(ii+1);
  }
	if (storeph) matopc_copy(&blkph[ofsp],L,nd1,nd1+1); /* storage */
	ofsp+=nd1*(nd1+1);
	matopc_copy_transpose(W,pstg[ii+1].ptrD,pstg[ii].neq,pstg[ii+1].nd); /* W <- D[ii+1]' */
	matopc_cholesky_trisubst_left_matrix(L,&L[nd1*nd1],nd1,W,W,pstg[ii].neq); /* W <- L[ii+1]\W */
	matopc_mtm(Y0,V,pstg[ii].nd,pstg[ii].neq,MATOPC_UPPER); /* assign V'*V */
	matopc_spmtm(Y0,W,nd1,pstg[ii].neq,MATOPC_UPPER); /* add W'*W to block Y[N-1,N-1] */
	/* Factorize downdated Y[N-1,N-1] for last block factor L[N-1,N-1] */
	matopc_smmtm(Y0,Y1,pstg[ii-1].neq,pstg[ii].neq,MATOPC_UPPER);
	chret=matopc_cholesky_decompose(Y0,&Y0[pstg[ii].neq*pstg[ii].neq],pstg[ii].neq);
	if (chret!=0) return -(N+ii+1);
	/* Store last block L[N-1,N-1], the extra column is for the diagonal of the Cholesky factor */
	matopc_copy(&blkly[ofsl],Y0,pstg[ii].neq,pstg[ii].neq+1); /* STORAGE LINE */
	ofsl+=pstg[ii].neq*(pstg[ii].neq+1);
  #ifdef __CLUMSY_ASSERTIONS__
	if (ofsl!=dat->blklysz)
		mexPrintf("[%s]: memory offset counting error LY!\n",__func__);
	if (ofsp!=dat->blkphsz)
		mexPrintf("[%s]: memory offset counting error PH!\n",__func__);
    if (useJay>0)
        if (ofsv!=dat->niq)
            mexPrintf("[%s]: memory offset counting error (%i=ofsv!=niq=%i)\n",
                    __func__,ofsv,dat->niq);
  #endif
    /*#ifdef __DEVELOPMENT_TEXT_OUTPUT__
	printf("[%s] @ ii=%i, ofsl=%i, blklysz=%i, ofsp=%i, blkphsz=%i\n",
		__func__,ii,ofsl,dat->blklysz,ofsp,dat->blkphsz);
    #endif*/
	return 0;
}

/* The below function performs the same job as the above function.
 * But it uses different block-operation sub programs.
 *
 * TODO: why is the updating of L slow? sqrt()'s removable?
 *
 */
int BlkCholFactorizeY2(
        qpdatStruct *dat,
        int storePhi,
        int useJay,
        double *v,
        int useCache) {
	stageStruct *pstg=dat->pstg;
	sparseMatrix *psp;
	int nstg=dat->nstg;
	double *tmpbuf=NULL;
	double *blkph,*blkly;
	double *V,*W,*L,*Y0,*Y1,*tmpx;
	int tmpbufsz,ofsl,ofsp,ofsv;
	int ii,chret,dmy,storeph,nd1;
	int N=nstg-1;
	/* Buffer for Cholesky blocks of LY must be provided
	   (otherwise the code has no output!) */
	if (dat->blkly==NULL) {
		return 1;
	}
    /* If J-updates of Phi blocks are activated; then non-null v is required */
    if (useJay>0)
        if (v==NULL) return 2;
    /* Must provide external memory space for local block matrices */
	if (dat->blkwr==NULL) {
		return 3;
	} else {
		tmpbuf=dat->blkwr;
		tmpbufsz=dat->blkwrsz;
	}
	dmy=0;
	L=&tmpbuf[dmy];	dmy+=dat->ndmax*(dat->ndmax+1);
	V=&tmpbuf[dmy]; dmy+=dat->nemax*dat->ndmax;
	W=&tmpbuf[dmy]; dmy+=dat->nemax*dat->ndmax;
	Y0=&tmpbuf[dmy]; dmy+=dat->nemax*(dat->nemax+1);
	Y1=&tmpbuf[dmy]; dmy+=dat->nemax*(dat->nemax+1);
    tmpx=&tmpbuf[dmy]; dmy+=dat->ndmax; /* use this ndmax-sized vector during Cholesky updating if needed */
	if (tmpbufsz!=dmy) {
		mexPrintf("ERROR[%s]: memory offset mismatch (%i!=%i).\n",
			__func__,dmy,tmpbufsz);
		return 4;
	}
	blkph=dat->blkph; /* This is allowed to be NULL */
	blkly=dat->blkly;
	if (blkph!=NULL && storePhi>0) {
		storeph=1;
	} else {
		storeph=0;
	}

	ofsl=0; ofsp=0; ofsv=0; nd1=pstg[0].nd;
    if (useCache>0) { /* Load cached factorization */
        matopc_copy(L,pstg[0].ptrL,nd1,nd1+1);
        if (useJay>0) { /* rank-1 updates if needed */
            chol_nq_rank1_updates(L,&L[nd1*nd1],nd1,pstg[0].ptrJ,&v[ofsv],pstg[0].niq,tmpx);
            ofsv+=pstg[0].niq;
        }
    } else {
        matopc_copy_up2lo(L,pstg[0].ptrQ,nd1,1);
        if (useJay>0) { /* L <- L+J'*diag(v)*J */
            /*matopc_cpatda(L,nd1,pstg[0].ptrJ,pstg[0].niq,&v[ofsv],1.0,1.0,MATOPC_UPPER);*/
            matopc_blk_atwa_4x4(
                    L,nd1,pstg[0].ptrJ,pstg[0].niq,nd1,
                    pstg[0].niq,&v[ofsv],+1,MATOPC_LOWER);
            ofsv+=pstg[0].niq;
        }
        /*chret=matopc_cholesky_decompose(L,&L[nd1*nd1],nd1);*/
        chret=matopc_cholesky_dclo4(L,nd1);
        matopc_getdiag(&L[nd1*nd1],L,nd1);
        if (chret!=0) return -(0+1);
    }
    if (storeph) matopc_copy(&blkph[ofsp],L,nd1,nd1+1); /* storage */
	ofsp+=nd1*(nd1+1);
	/*matopc_copy_transpose(V,pstg[0].ptrC,pstg[0].neq,nd1);*/ /* V <- C[0]' */
	/*matopc_cholesky_trisubst_left_matrix(L,&L[nd1*nd1],nd1,V,V,pstg[0].neq);*/ /* V <- L\V */
	matopc_cholesky_trisubst_left_tr_matrix(L,&L[nd1*nd1],nd1,pstg[0].ptrC,V,pstg[0].neq,tmpx);
	for (ii=0;ii<N-1;ii++) {
        nd1=pstg[ii+1].nd;
        if (useCache>0) {
            matopc_copy(L,pstg[ii+1].ptrL,nd1,nd1+1);
            if (useJay>0) { /* rank-1 updates if needed */
              chol_nq_rank1_updates(L,&L[nd1*nd1],nd1,pstg[ii+1].ptrJ,&v[ofsv],pstg[ii+1].niq,tmpx);
              ofsv+=pstg[ii+1].niq;
            }
        } else {
            matopc_copy_up2lo(L,pstg[ii+1].ptrQ,nd1,1);
            if (useJay>0) { /* L <- L+J'*diag(v)*J (lower triangle only) */
              psp=pstg[ii+1].ptrJsp;
		        	if (psp!=NULL) {
	  		      	matopc_ccs_cpatda_lo(
    		   				L,pstg[ii+1].niq,nd1,
    		   				psp->ccsval,psp->rowind,psp->colptr,
    		   				&v[ofsv]);
    		    	} else {
/*    		        matopc_cpatda(L,nd1,pstg[ii+1].ptrJ,pstg[ii+1].niq,&v[ofsv],1.0,1.0,MATOPC_UPPER); */
    		        matopc_blk_atwa_4x4(
                    L,nd1,pstg[ii+1].ptrJ,pstg[ii+1].niq,nd1,
                    pstg[ii+1].niq,&v[ofsv],+1,MATOPC_LOWER);
    		      }
              ofsv+=pstg[ii+1].niq;
            }
    		/*chret=matopc_cholesky_decompose(L,&L[nd1*nd1],nd1);*/
            chret=matopc_cholesky_dclo4(L,nd1);
            matopc_getdiag(&L[nd1*nd1],L,nd1);
    				if (chret!=0) return -(ii+1);
        }
		if (storeph) matopc_copy(&blkph[ofsp],L,nd1,nd1+1); /* storage */
		ofsp+=nd1*(nd1+1);
		psp=pstg[ii+1].ptrDsp;
		if (psp!=NULL) { /* Exploit CRS data for D */
			matopc_crs_cholesky_trisubst_left_tr_matrix(
        L,&L[nd1*nd1],nd1,
        W,pstg[ii].neq,
        psp->crsval,psp->colind,psp->rowptr,
        tmpx);
		} else {
		  /* W <- L[ii+1]\D[ii+1]' */
		  /*matopc_copy_transpose(W,pstg[ii+1].ptrD,pstg[ii].neq,pstg[ii+1].nd);*/ /* W <- D[ii+1]' */
		  /*matopc_cholesky_trisubst_left_matrix(L,&L[nd1*nd1],nd1,W,W,pstg[ii].neq);*/ /* W <- L[ii+1]\W */
		  matopc_cholesky_trisubst_left_tr_matrix(L,&L[nd1*nd1],nd1,pstg[ii+1].ptrD,W,pstg[ii].neq,tmpx);
		}
		/* Y[ii,ii] <- V'*V+W'*W */
		/*matopc_mtm(Y0,V,pstg[ii].nd,pstg[ii].neq,MATOPC_LOWER);*/
        /* assign V'*V */
        matopc_blk_ata_4x4(
                Y0,pstg[ii].neq,
                V,pstg[ii].nd,pstg[ii].neq,
                pstg[ii].nd,0,MATOPC_LOWER);
		/*matopc_spmtm(Y0,W,nd1,pstg[ii].neq,MATOPC_LOWER);*/
        /* add W'*W to block */
        matopc_blk_ata_4x4(
                Y0,pstg[ii].neq,
                W,nd1,pstg[ii].neq,
                nd1,+1,MATOPC_LOWER);
		/* Y[ii,ii] available: factorize to get L[ii,ii] */
		if (ii>0) { /* "downdate" Y0 <- Y0-U'*U, then factorize Y0 in place (not for first iteration though) */
			matopc_smmtm(Y0,Y1,pstg[ii-1].neq,pstg[ii].neq,MATOPC_LOWER);
            /* why is the below func. slower? */
            /*matopc_blk_ata_4x4(
                Y0,pstg[ii].neq,
                Y1,pstg[ii-1].neq,pstg[ii].neq,
                pstg[ii-1].neq,-1,MATOPC_LOWER);*/
		}
		/* Factorize Y0 in-place */
		/*chret=matopc_cholesky_decompose(Y0,&Y0[pstg[ii].neq*pstg[ii].neq],pstg[ii].neq);*/
    chret=matopc_cholesky_dclo4(Y0,pstg[ii].neq);
    matopc_getdiag(&Y0[pstg[ii].neq*pstg[ii].neq],Y0,pstg[ii].neq);
		if (chret!=0) return -(N+ii+1);
		matopc_copy(&blkly[ofsl],Y0,pstg[ii].neq,pstg[ii].neq+1); /* STORAGE LINE */
		ofsl+=pstg[ii].neq*(pstg[ii].neq+1);
		/* Overwrite V with L[ii+1]\C[ii+1]' */
		/*matopc_copy_transpose(V,pstg[ii+1].ptrC,pstg[ii+1].neq,nd1);*/ /* V <- C[ii+1]' */
		/*matopc_cholesky_trisubst_left_matrix(L,&L[nd1*nd1],nd1,V,V,pstg[ii+1].neq);*/ /* V <- L\V */
		matopc_cholesky_trisubst_left_tr_matrix(L,&L[nd1*nd1],nd1,pstg[ii+1].ptrC,V,pstg[ii+1].neq,tmpx);
		/* so that Y[ii,ii+1] is W'*V, then go to next iteration... */
		/*matopc_atb(Y1,W,pstg[ii+1].nd,pstg[ii].neq,V,pstg[ii+1].neq);*/
        /* Y[ii,ii+1] <- W'*V */
        matopc_blk_atb_4x4(
                Y1,pstg[ii].neq,
                W,nd1,pstg[ii].neq,
                V,nd1,pstg[ii+1].neq,
                nd1,0);
		/* Y[ii,ii+1] is available: backsolve in-place with last L[ii,ii] to get U[ii] rectangle for next iteration. */
		matopc_cholesky_trisubst_left_matrix(Y0,&Y0[pstg[ii].neq*pstg[ii].neq],pstg[ii].neq,Y1,Y1,pstg[ii+1].neq); // U <- L\Y
		matopc_copy_transpose(&blkly[ofsl],Y1,pstg[ii].neq,pstg[ii+1].neq);	/* STORAGE LINE: copy U' as the sub-diagonal Cholesky block */
		ofsl+=pstg[ii+1].neq*pstg[ii].neq;
	}
	/* Last block Y[N-1,N-1]: */
    nd1=pstg[ii+1].nd;
    if (useCache>0) {
        matopc_copy(L,pstg[ii+1].ptrL,nd1,nd1+1);
        if (useJay>0) { /* rank-1 updates if needed */
            chol_nq_rank1_updates(L,&L[nd1*nd1],nd1,pstg[ii+1].ptrJ,&v[ofsv],pstg[ii+1].niq,tmpx);
            ofsv+=pstg[ii+1].niq;
        }
    } else {
        matopc_copy_up2lo(L,pstg[ii+1].ptrQ,nd1,1);
        if (useJay>0) { /* L <- L+J'*diag(v)*J */
            /*matopc_cpatda(L,nd1,pstg[ii+1].ptrJ,pstg[ii+1].niq,&v[ofsv],1.0,1.0,MATOPC_UPPER);*/
            matopc_blk_atwa_4x4(
                    L,nd1,pstg[ii+1].ptrJ,pstg[ii+1].niq,nd1,
                    pstg[ii+1].niq,&v[ofsv],+1,MATOPC_LOWER);
            ofsv+=pstg[ii+1].niq;
        }
        /*chret=matopc_cholesky_decompose(L,&L[nd1*nd1],nd1);*/
        chret=matopc_cholesky_dclo4(L,nd1);
        matopc_getdiag(&L[nd1*nd1],L,nd1);
        if (chret!=0) return -(ii+1);
    }
	if (storeph) matopc_copy(&blkph[ofsp],L,nd1,nd1+1); /* storage */
	ofsp+=nd1*(nd1+1);
	matopc_copy_transpose(W,pstg[ii+1].ptrD,pstg[ii].neq,nd1); /* W <- D[ii+1]' */
	matopc_cholesky_trisubst_left_matrix(L,&L[nd1*nd1],nd1,W,W,pstg[ii].neq); /* W <- L[ii+1]\W */
	matopc_mtm(Y0,V,pstg[ii].nd,pstg[ii].neq,MATOPC_LOWER); /* assign V'*V */
	matopc_spmtm(Y0,W,nd1,pstg[ii].neq,MATOPC_LOWER); /* add W'*W to block Y[N-1,N-1] */
	/* Factorize downdated Y[N-1,N-1] for last block factor L[N-1,N-1] */
	matopc_smmtm(Y0,Y1,pstg[ii-1].neq,pstg[ii].neq,MATOPC_LOWER);
	/*chret=matopc_cholesky_decompose(Y0,&Y0[pstg[ii].neq*pstg[ii].neq],pstg[ii].neq);*/
    chret=matopc_cholesky_dclo4(Y0,pstg[ii].neq);
    matopc_getdiag(&Y0[pstg[ii].neq*pstg[ii].neq],Y0,pstg[ii].neq);
	if (chret!=0) return -(N+ii+1);
	/* Store last block L[N-1,N-1], the extra column is for the diagonal of the Cholesky factor */
	matopc_copy(&blkly[ofsl],Y0,pstg[ii].neq,pstg[ii].neq+1); /* STORAGE LINE */
	ofsl+=pstg[ii].neq*(pstg[ii].neq+1);
	return 0;
}

/* Multiply vector x by the inverse of the implied block-diagonal
 * matrix Phi: x<-inv(Phi)*y 
 * Use the block Cholesky factors of Phi to accomplish this.
 * It is allowed to use in-place op. x=y.
   TODO: special treatment for blocks where Phi is diagonal.. etc..
 */
void PhiInverseMult(qpdatStruct *dat,double *y,double *x) {
	stageStruct *pstg=dat->pstg;
	int nstg=dat->nstg;
	int ii,ofsp,ofsy,nd;
	double *L00;
	double *blkph=dat->blkph;
	if (blkph==NULL || x==NULL || y==NULL) {
		mexPrintf("[%s]: early exit due to NULL pointer(s).\n",__func__);
		return;
	}
	ofsp=0; ofsy=0;
	for (ii=0;ii<nstg;ii++) {
		nd=pstg[ii].nd;
		L00=&blkph[ofsp];
		/* block forward backward subst., in-place OK if x=y */
		matopc_cholesky_solve(L00,&L00[nd*nd],nd,&y[ofsy],&x[ofsy]);
		ofsp+=nd*(nd+1);
		ofsy+=nd;
	}
    #ifdef __CLUMSY_ASSERTIONS__
	if (ofsp!=dat->blkphsz || ofsy!=dat->ndtot) {
		mexPrintf("[%s]: ofsy=%i, ndtot=%i\n",__func__,ofsy,dat->ndtot);
		mexPrintf("[%s]: ofsp=%i, blkphsz=%i\n",__func__,ofsp,dat->blkphsz);
	}
    #endif
}

/* Solve Y*x=b using the block Cholesky factors of Y stored
 * in the buffer accessed through problem data structure dat.
 * Can be done in-place (b=x).
 */
void BlkCholSolve(qpdatStruct *dat,double *b,double *x) {
	stageStruct *pstg=dat->pstg;
	int nstg=dat->nstg;
	int N=nstg-1;
	int ii,ofsl,ofsx;
	double *L10,*L00;
	double *blkly=dat->blkly;
	if (blkly==NULL || x==NULL || b==NULL) return;
	/* solution vector is x, unless x==b already, copy b into x and work in-place below */
	if (x!=b) matopc_copy(x,b,dat->netot,1);
	/* First solve LY*z=b for intermediate z (forward block substitution) */
	ofsl=0; ofsx=0;
	L00=&blkly[ofsl];
	ofsl+=pstg[0].neq*(pstg[0].neq+1);
	/* forward in-place block substitution x[0] <- L00\x[0] */
	matopc_cholesky_trisubst_left(L00,&L00[pstg[0].neq*pstg[0].neq],pstg[0].neq,&x[ofsx],&x[ofsx]);
	ofsx+=pstg[0].neq;
	for (ii=1;ii<N;ii++) {
		L10=&blkly[ofsl];
		ofsl+=pstg[ii].neq*pstg[ii-1].neq;
		/* Now subtract L10*x[ii-1] from x[ii] then backsolve in-place for x[ii] */
		matopc_addax(&x[ofsx],L10,pstg[ii].neq,pstg[ii-1].neq,&x[ofsx-pstg[ii-1].neq],-1);
		L00=&blkly[ofsl];
		ofsl+=pstg[ii].neq*(pstg[ii].neq+1);
		matopc_cholesky_trisubst_left(L00,&L00[pstg[ii].neq*pstg[ii].neq],pstg[ii].neq,&x[ofsx],&x[ofsx]);
		ofsx+=pstg[ii].neq;
	}
    #ifdef __CLUMSY_ASSERTIONS__
	if (ofsx!=dat->netot || ofsl!=dat->blklysz || ii!=N) {
		mexPrintf("[%s]: ofsx=%i, netot=%i\n",__func__,ofsx,dat->netot);
		mexPrintf("[%s]: ofsl=%i, blklysz=%i\n",__func__,ofsl,dat->blklysz);
		return;
	}
    #endif
	/* Then solve LY'*x=z to get x (backward block substitution) */
	ofsl-=pstg[N-1].neq*(pstg[N-1].neq+1);
	L00=&blkly[ofsl];
	ofsx-=pstg[N-1].neq;
	/* in-place L'\(.) operation */
	matopc_cholesky_trisubst_tr_left(L00,&L00[pstg[N-1].neq*pstg[N-1].neq],pstg[N-1].neq,&x[ofsx],&x[ofsx]);
	for (ii=N-2;ii>=0;ii--) {
		ofsl-=pstg[ii+1].neq*pstg[ii].neq;
		L10=&blkly[ofsl];
		/* subtract L10'*x[ii+1] from x[ii] */
		matopc_addatx(&x[ofsx-pstg[ii].neq],L10,pstg[ii+1].neq,pstg[ii].neq,&x[ofsx],-1);
		ofsx-=pstg[ii].neq;
		ofsl-=pstg[ii].neq*(pstg[ii].neq+1);
		L00=&blkly[ofsl];
		matopc_cholesky_trisubst_tr_left(L00,&L00[pstg[ii].neq*pstg[ii].neq],pstg[ii].neq,&x[ofsx],&x[ofsx]);
	}
    #ifdef __CLUMSY_ASSERTIONS__
	if (ofsx!=0 || ofsl!=0 || ii!=-1) {
		mexPrintf("[%s]: ofsx=%i, netot=%i\n",__func__,ofsx,dat->netot);
		mexPrintf("[%s]: ofsl=%i, blklysz=%i\n",__func__,ofsl,dat->blklysz);
	}
    #endif
}

/*
 * Main Primal-Dual Interior-Point Method interface.
 * The multi-stage quadratic program meta data is accessed
 * exclusively through the struct array pointer.
 *
 * The working memory needed must be allocated outside of the
 * pdipm_solve(..) call. 
 *
 */

/* Based on the neq, niq and ndec values; determine a buffer size
 * which is large enough to house all the working vectors to be used
 * by the msqp_pdipm_solve(..) code below. Allocate this buffer and
 * store the pointer/size in the asociated problem struct fields.
 * Return 0 if fail; otherwise 1 when OK.
 *
 * Buffer "vecwr" organized as follows.
 * Here read nx=ndec, ny=neq, nz=niq
 *
 * --------------------------------------------------
 * NAME             SIZE (# DOUBLES)        EXPL.
 * --------------------------------------------------
 * x                nx                      Stages
 * y                ny                      Lagrange
 * z                nz
 * s                nz
 * rC               nx
 * rE               ny
 * rI               nz
 * rsz              nz
 * tmp1             nz
 * rd               nx
 * tmp2             nx
 * tmp3             nz
 * bet              ny
 * dy               ny
 * dx               nx
 * dza              nz
 * dsa              nz
 * drsz             nz
 * dz               nz
 * ds               nz
 * --------------------------------------------------
 *
 * SUM TOTAL # DOUBLES = 5*nx + 4*ny + 11*nz = vecwrsz.
 *
 */
int msqp_pdipm_init(qpdatStruct *qpd,int verbosity) {
    qpd->vecwrsz=5*(qpd->ndec)+4*(qpd->neq)+11*(qpd->niq);
    qpd->vecwr=mxMalloc((qpd->vecwrsz)*sizeof(double));
    if (verbosity>2) {
        if (qpd->vecwr!=NULL) {
            mexPrintf("[%s]: vecwrsz=%i\n",__func__,qpd->vecwrsz);
        }
    }
    return (qpd->vecwr!=NULL ? 1 : 0);
}

/* Free memory that was allocated by the previous subprogram */
void msqp_pdipm_free(qpdatStruct *qpd,int verbosity) {
    if (qpd->vecwr!=NULL) {
        mxFree(qpd->vecwr);
        qpd->vecwr=NULL;
    }
}

/* Search direction clipping template for the main PDIPM loop */
static inline double aux_alpha_clip(double a,int n,double *v,double *dv) {
    double tmp,aret;
    int ii;
    aret=a;
    for (ii=0;ii<n;ii++) {
        if (dv[ii]<0.0) {
            tmp=-v[ii]/dv[ii];
            if (tmp<aret) aret=tmp;
        }
    }
    return aret;
}

/* Special inner product used in the main PDIPM loop */
static inline double aux_alpha_dot(
        double a,int n,double *u,double *du,double *v,double *dv) {
    int ii;
    double s=0.0;
    for (ii=0;ii<n;ii++) s+=(u[ii]+a*du[ii])*(v[ii]+a*dv[ii]);
    return s;
}

/* 
 * This is the main PDIPM algorithm program.
 *
 * Integer (minimal) return code j as follows:
 *
 * j=0: converged to a solution.
 * j<0: Cholesky factorization error at iteration -j
 * j=1: not converged after max. allowed iterations.
 * j>1: other error(s)
 *
 * Returns optimal decision vector and objective value
 * information via the qpretStruct pointer (if non-NULL).
 *
 */
int msqp_pdipm_solve(qpdatStruct *qpd,qpoptStruct *qpo,qpretStruct *qpr) {
    
    /*
     * TODO: based on the value of qpo->blas_suite {1,2}
     *       make choice of the associated suite:
     *       { BlkCholFactorizeY, BlkCholSolve, PhiInverseMult }
     *  or   { BlkCholFactorizeY2, BlkCholSolve2, PhiInverseMult2 }
     *  
     *  select using function pointer assignment.
     *  Suite 2 is probably significantly faster for intermediate stage
     *  dimensions; but for small stage dimension this may be a dead race.
     *
     */
    
    static double fobj[3];
    
    #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    static fclk_timespec __tica,__ticb;
    double cholytimesum=0.0;
    #endif
    
    int kk,ofs,oktostop,chret=0;
    double thrC,thrE,thrI,thrmu;
    double infC,infE,infI; /* infinity norms of the respective residuals */
    double mu,mua,ea,sigma;
    double alpha,alphaa;
    int nx,ny,nz;
    
    int (*CHOLY)(qpdatStruct *,int,int,double *,int) = BlkCholFactorizeY;
    void (*CHOLSOLVE)(qpdatStruct *,double *,double *) = BlkCholSolve;
    void (*PHIINV)(qpdatStruct *,double *,double *) = PhiInverseMult;
    
    double *vecwr;
    double *x,*y,*z,*s;
    double *rC,*rE,*rI,*rsz;
    double *tmp1,*rd,*tmp2,*tmp3,*bet;
    double *dx,*dy,*dza,*dsa;
    double *drsz,*dz,*ds;
    
    double *h=qpd->ph;
    double *d=qpd->pd;
    double *f=qpd->pf;
    
    if (qpo->blas_suite==0) {
        /* no need to do anything; this is the default */
    } else if (qpo->blas_suite==1) {
        /* need to change the function pointers */
        CHOLY = BlkCholFactorizeY2;
    } else {
        return -1; /* ERROR: unrecognized blas_suite option; so refuse to continue */
    }
    
    nx=qpd->ndec;
    ny=qpd->neq;
    nz=qpd->niq;
    
    /*#ifdef __DEVELOPMENT_TEXT_OUTPUT__
    mexPrintf("[%s]\n",__func__);
    mexPrintf("qpo->maxiters=%i\n",qpo->maxiters);
    mexPrintf("qpo->eta=%f\n",qpo->eta);
    mexPrintf("qpo->ep=%e\n",qpo->ep);
    mexPrintf("nx=%i, ny=%i, nz=%i\n",nx,ny,nz);
    #endif*/
    
    if (qpd->vecwr==NULL) return -1;
    vecwr=qpd->vecwr; ofs=0;
    x=&vecwr[ofs]; ofs+=nx;
    y=&vecwr[ofs]; ofs+=ny;
    z=&vecwr[ofs]; ofs+=nz;
    s=&vecwr[ofs]; ofs+=nz;
    rC=&vecwr[ofs]; ofs+=nx;
    rE=&vecwr[ofs]; ofs+=ny;
    rI=&vecwr[ofs]; ofs+=nz;
    rsz=&vecwr[ofs]; ofs+=nz;
    tmp1=&vecwr[ofs]; ofs+=nz;
    rd=&vecwr[ofs]; ofs+=nx;
    tmp2=&vecwr[ofs]; ofs+=nx;
    tmp3=&vecwr[ofs]; ofs+=nz;
    bet=&vecwr[ofs]; ofs+=ny;
    dy=&vecwr[ofs]; ofs+=ny;
    dx=&vecwr[ofs]; ofs+=nx;
    dza=&vecwr[ofs]; ofs+=nz;
    dsa=&vecwr[ofs]; ofs+=nz;
    drsz=&vecwr[ofs]; ofs+=nz;
    dz=&vecwr[ofs]; ofs+=nz;
    ds=&vecwr[ofs]; ofs+=nz;
    
    #ifdef __CLUMSY_ASSERTIONS__
    if (ofs!=qpd->vecwrsz) {
		  mexPrintf("[%s]: MISMATCH: ofs=%i, vecwrsz=%i\n",__func__,ofs,qpd->vecwrsz);
    }
    #endif
    
    /* Setup relative stop conditions based on QP data vector norms */
    thrmu=qpo->ep;
    thrC=(qpo->ep)*(1.0+vecop_norm(h,nx,0)); /* 0 = inf-norm */
    thrE=(qpo->ep)*(1.0+vecop_norm(d,ny,0));
    thrI=(qpo->ep)*(1.0+vecop_norm(f,nz,0));
    
    /* NOTE: WARNING: the initial condition here assumes
     * that x=0 is feasible; it is very possible that there are 
     * some situations where this is inappropriate!
     */
    
    /* Initialize vectors x,y,z,s */
    vecop_zeros(x,nx);
    vecop_zeros(y,ny);
    vecop_ones(z,nz);
    vecop_ones(s,nz);
    
    /* Initialize residuals rC,rE,rI,rsz; assuming x=0, y=0, z=1, s=1
     *
     * General init. is:
     * rC <- H*x + h + C'*y + J'*z
     * rE <- C*x - d
     * rI <- J*x + s - f
     *
     */
    Jtmult(qpd,z,rC);               /* rC <- J'*z */
    /*vecop_macc(rC,nx,1.0,h,1.0);*/    /* rC <- rC+h = h+J'*z */
    vecop_addx(rC,nx,h);
    vecop_neg_copy(rE,ny,d);        /* rE <- -d */
    vecop_sub(rI,nz,s,f);           /* rI <- s-f */
    vecop_mul(rsz,nz,s,z);          /* rsz <- s.*z */
    mu=vecop_sum(nz,rsz)/nz;        /* mu <- sum(rsz)/nz */
    
    infC=vecop_norm(rC,nx,0);
    infE=vecop_norm(rE,ny,0);
    infI=vecop_norm(rI,nz,0);
    
    kk=0;
    oktostop=(infC<thrC) && (infE<thrE) && (infI<thrI) && (mu<thrmu);
    
    while ( (kk<qpo->maxiters) && !oktostop ) {
        vecop_div(tmp1,nz,z,s); /* tmp1 <- z./s */
        /* Begin: rd =  */
        Hmult(qpd,x,rd); /* rd <- H*x */
        /*vecop_macc(rd,nx,1.0,h,1.0);*/ /* rd <- rd+h */
        vecop_addx(rd,nx,h);
        Ctmult(qpd,y,tmp2); /* tmp2 <- C'*y */
        /*vecop_macc(rd,nx,1.0,tmp2,1.0);*/ /* rd <- rd+tmp2 */
        vecop_addx(rd,nx,tmp2);
        vecop_mul(tmp3,nz,tmp1,rI); /* tmp3 <- tmp1.*rI */
        Jtmult(qpd,tmp3,tmp2); /* tmp2 <- J'*tmp3 */
        /*vecop_macc(rd,nx,1.0,tmp2,1.0);*/ /* rd <- rd+tmp2 */
        vecop_addx(rd,nx,tmp2);
        /* End: rd = H*x+h+C'*y+J'*(diag(tmp1)*rI)  */
        /* Factorize Y=C'*inv(Phi)*C, Phi=H+J'*diag(tmp1)*J */
        #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
        fclk_timestamp(&__tica);
        #endif
        chret=CHOLY(qpd,1,1,tmp1,qpo->chol_update); /* This is the costly one. */
        #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
        fclk_timestamp(&__ticb);
        cholytimesum+=fclk_delta_timestamps(&__tica,&__ticb);
        #endif
        if (chret!=0) {
            /*#ifdef __DEVELOPMENT_TEXT_OUTPUT__
            mexPrintf("[%s]: ERROR: chret=%i\n",__func__,chret);
            #endif*/
            break;
        }
        PHIINV(qpd,rd,tmp2);    /* tmp2 <- inv(Phi)*rd */
            /* NOTE: rd is modified below; Mehrotra predictor-corrector */
        /* bet <- rE-C*tmp2 (in 2 steps) */
        Cmult(qpd,tmp2,bet); /* bet <- C*tmp2 */
        vecop_sub(bet,ny,rE,bet); /* bet <- rE-bet, elementwise */
        /* Solve Y*dy=bet for dy, using obtained L, L*L'=Y */
        CHOLSOLVE(qpd,bet,dy);
        /* dx <- -tmp2-inv(Phi)*(C'*dy) */
        Ctmult(qpd,dy,dx); /* dx <- C'*y */
        PHIINV(qpd,dx,dx);  /* dx <- inv(Phi)*dx in-place */
        vecop_macc(dx,nx,-1.0,tmp2,-1.0); /* dx <- -dx-tmp2 */
        /* dza <- tmp1.*(J*dx+rI)-z */
        Jmult(qpd,dx,tmp3); /* tmp3 <- J*dx */
        vecop_addx(tmp3,nz,rI); /* tmp3 <- tmp3+rI */
        vecop_mulx(tmp3,nz,tmp1); /* tmp3 <- tmp3.*tmp1 */
        vecop_sub(dza,nz,tmp3,z); /* dza <- tmp3-z */
        /* dsa <- -s-(s./z).*dza, but this is also -s-dza./tmp1 */
        vecop_div(dsa,nz,dza,tmp1); /* dsa <- dza./tmp1 */
        vecop_macc(dsa,nz,-1.0,s,-1.0); /* dsa <- -dsa-s */
        /* Trial step */
        alphaa=1.0;
        alphaa=aux_alpha_clip(alphaa,nz,z,dza);
        alphaa=aux_alpha_clip(alphaa,nz,s,dsa);
        mua=aux_alpha_dot(alphaa,nz,z,dza,s,dsa)/nz;
        sigma=mua/mu; sigma*=(sigma*sigma); /* sigma <- (mua/mu)^3 */
        /* drsz <- dsa.*dza-sigma*mu*ones(nz,1) */
        vecop_mul(drsz,nz,dsa,dza);
        vecop_adda(drsz,nz,-sigma*mu);
        /* rd <- rd-J'*(drsz./s) */
        vecop_div(tmp3,nz,drsz,s);
        Jtmult(qpd,tmp3,tmp2);
        vecop_subx(rd,nx,tmp2);
        /* tmp2 <- inv(Phi)*rd */
        PHIINV(qpd,rd,tmp2);
        /* bet <- rE-C*tmp2 (in 2 steps) */
        Cmult(qpd,tmp2,bet); /* bet <- C*tmp2 */
        vecop_sub(bet,ny,rE,bet); /* bet <- rE-bet, elementwise */
        /* Solve again for dy; now with modified RHS */
        CHOLSOLVE(qpd,bet,dy);
        /* dx <- -tmp2-inv(Phi)*(C'*dy) */
        Ctmult(qpd,dy,dx); /* dx <- C'*y */
        PHIINV(qpd,dx,dx);  /* dx <- inv(Phi)*dx in-place */
        vecop_macc(dx,nx,-1.0,tmp2,-1.0); /* dx <- -dx-tmp2 */
        /* dz <- tmp1.*(J*dx+rI)-z-drsz./s, but drsz./s is now in tmp3 */
        Jmult(qpd,dx,dz); /* dz <- J*dx */
        vecop_addx(dz,nz,rI);
        vecop_mulx(dz,nz,tmp1);
        vecop_subx(dz,nz,z);
        vecop_subx(dz,nz,tmp3);
        /* ds <- -s-drsz./z-(s./z).*dz, but use -dz./tmp1 for last term */
        vecop_div(tmp3,nz,dz,tmp1); /* tmp3 <- dz./tmp1 */
        vecop_div(ds,nz,drsz,z);    /* ds <- drsz./z */
        vecop_addx(ds,nz,s);        /* ds <- ds+s */
        vecop_macc(ds,nz,-1.0,tmp3,-1.0); /* ds <- -ds-tmp3 */
        /* Corrected step */
        alpha=1.0;
        alpha=aux_alpha_clip(alpha,nz,z,dz);
        alpha=aux_alpha_clip(alpha,nz,s,ds);
        /* Take a step (x,y,z,s) += ea*(dx,dy,dz,ds) */
        ea=alpha*(qpo->eta);
        vecop_addax(x,nx,ea,dx);
        vecop_addax(y,ny,ea,dy);
        vecop_addax(z,nz,ea,dz);
        vecop_addax(s,nz,ea,ds);
/*vecop_macc(x,nx,1.0,dx,ea);
  vecop_macc(y,ny,1.0,dy,ea);
  vecop_macc(z,nz,1.0,dz,ea);
  vecop_macc(s,nz,1.0,ds,ea);*/
        /* re-evaluate residual vectors */
        /* rC <- H*x+h+C'*y+J'*z */
        Hmult(qpd,x,rC);
        vecop_addx(rC,nx,h);
        Ctmult(qpd,y,tmp2);
        vecop_addx(rC,nx,tmp2);
        Jtmult(qpd,z,tmp2);
        vecop_addx(rC,nx,tmp2);
        /* rE <- C*x-d */
        Cmult(qpd,x,rE);
        vecop_subx(rE,ny,d);
        /* rI <- J*x-f+s */
        Jmult(qpd,x,rI);
        vecop_subx(rI,nz,f);
        vecop_addx(rI,nz,s);
        /* rsz <- s.*z, and mu=sum(rsz)/nz */
        vecop_mul(rsz,nz,s,z);
        mu=vecop_sum(nz,rsz)/nz;
        /* did we converge; compute "oktostop" flag */
        infC=vecop_norm(rC,nx,0);
    		infE=vecop_norm(rE,ny,0);
    		infI=vecop_norm(rI,nz,0);
        oktostop=(infC<thrC) && (infE<thrE) && (infI<thrI) && (mu<thrmu);
        kk++;   /* next iteration */
    }
    
    /*#ifdef __DEVELOPMENT_TEXT_OUTPUT__
    mexPrintf("[%s]: stop after %i itrs. (oktostop=%i, chret=%i).\n",
            __func__,kk,oktostop,chret);
    #endif*/
    
    /* Evaluate the final objective function value @ current x */
    /*if (chret==0) qpcost(qpd,x,fobj);*/
    
    qpcost(qpd,x,fobj);
   
    /* Fill in return struct even if not converged... */
    if (qpr!=NULL) {
        qpr->nx=nx;
        qpr->x=x;
        qpr->fxofs=fobj[0];
        qpr->fxopt=fobj[1]+fobj[2];
        qpr->iters=kk;
        qpr->converged=oktostop;
    #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
        qpr->cholytime=cholytimesum;
    #else
        qpr->cholytime=-1.0;
    #endif
    		/* Return the 4-tuple (infC,infE,infI,mu) of final residuals */
    		qpr->inftuple[0]=infC;
    		qpr->inftuple[1]=infE;
    		qpr->inftuple[2]=infI;
    		qpr->inftuple[3]=mu;
    }
   
    if (chret!=0) return -(kk+1);
    if (!oktostop) return 1;
    
    /*#ifdef __DEVELOPMENT_TEXT_OUTPUT__
    mexPrintf("[%s]: fx*=%e, fofs=%e\n",
            __func__,fobj[1]+fobj[2],fobj[0]);
    #endif*/
    
    return 0;
}

/* Simplified version that ignores the inequalitites
 * (they can be left unspecified)
 *
 * Returns 0 if "converged" (residuals below relative epsilon).
 * Returns 1 if not "converged" at given epsilon
 * Returns negative if Cholesky factorization failed.
 *
 * This is typically an order of magnitude faster than the full
 * PDIPM code (ie. with inequalities) for a same size problem.
 *
 * TODO: iterative refinement?
 *
 * TODO: note that typically the same diagonal block is
 *       decomposed many times (as many as the timesteps).
 *       This is quite stupid so add some technique to auto-detect
 *       this redundancy (checking pointers are the same; and copy/paste).
 */
int msqp_solve_niq(qpdatStruct *qpd,qpoptStruct *qpo,qpretStruct *qpr) {
    
    static int NIQ_USE_CACHE=1;
    static double fobj[3];
    int kk,ofs,oktostop,chret;
    double thr1,thr2;
    double inf1,inf2;
    double *vecwr;
    double *x,*y,*tmp,*r1,*r2,*bet;
    
    double *h=qpd->ph;
    double *d=qpd->pd;
    int nx=qpd->ndec;
    int ny=qpd->neq;
    
    #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    static fclk_timespec __tica,__ticb;
    double cholytimesum=0.0;
    #endif
    
    /* Setup the working vector pointers tmp is size nx, bet is ny */
    if (qpd->vecwr==NULL) return -1;
    vecwr=qpd->vecwr; ofs=0;
    x=&vecwr[ofs]; ofs+=nx;
    y=&vecwr[ofs]; ofs+=ny;
    tmp=&vecwr[ofs]; ofs+=nx;
    r1=&vecwr[ofs]; ofs+=nx;
    r2=&vecwr[ofs]; ofs+=ny;
    bet=&vecwr[ofs]; ofs+=ny;
    
    #ifdef __CLUMSY_ASSERTIONS__
    if (ofs>qpd->vecwrsz) {
        /* This should never be possible since more memory is allocated than needed
         * (allocation as if running the full PDIPM code; subset of memory used here).
         */
		    mexPrintf("[%s]: MISMATCH: ofs=%i, vecwrsz=%i\n",
                __func__,ofs,qpd->vecwrsz);
    }
    #endif
    
    thr1=(qpo->ep)*(1.0+vecop_norm(h,nx,0)); /* 0 = inf-norm */
    thr2=(qpo->ep)*(1.0+vecop_norm(d,ny,0));
    
    #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&__tica);
    #endif
    chret=BlkCholFactorizeY(qpd,1,0,NULL,NIQ_USE_CACHE); /* factorize Y=C*inv(H)*C'=L*L' */
    #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&__ticb);
    cholytimesum+=fclk_delta_timestamps(&__tica,&__ticb);
    #endif
    if (chret!=0) return -1;
    
    /* bet <- -d-C*inv(H)*h */
    PhiInverseMult(qpd,h,tmp);  /* tmp <- inv(H)*h */
    Cmult(qpd,tmp,bet);         /* bet <- C*tmp */
    vecop_macc(bet,ny,-1.0,d,-1.0); /* bet <- -bet-d */
    /* solve (L*L')*y=bet for y (y are the lagrange multipliers) */
    BlkCholSolve(qpd,bet,y);
    /* obtain solution x <- -inv(H)*(h+C'*y) */
    Ctmult(qpd,y,tmp); /* tmp <- C'*y */
    vecop_macc(tmp,nx,-1.0,h,-1.0); /* here: tmp <- -tmp-h */
    PhiInverseMult(qpd,tmp,x);  /* x <- inv(H)*tmp */
    /* compute r1 <- H*x+h+C'*y (but h+C'*y is already equal to -tmp) */
    Hmult(qpd,x,r1);
    vecop_subx(r1,nx,tmp);
    /* compute r2 <- C*x-d */
    Cmult(qpd,x,r2);
    vecop_subx(r2,ny,d);
    
    /* check whether inf norm r1 < thr1 and inf norm r2 < thr2 */
    inf1=vecop_norm(r1,nx,0);
    inf2=vecop_norm(r2,ny,0);
    oktostop=(inf1<thr1) && (inf2<thr2);
    
    /* TODO: proceed with 1-step iterative refinement if !oktostop, or better, always? */
    /* ... */
    
    kk=0;
    
    /* evaluate cost function at the solution x (supposedly) */
    qpcost(qpd,x,fobj);
    
    if (qpr!=NULL) {
        qpr->nx=nx;
        qpr->x=x;
        qpr->fxofs=fobj[0];
        qpr->fxopt=fobj[1]+fobj[2];
        qpr->iters=kk;
        qpr->converged=oktostop;
    #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
        qpr->cholytime=cholytimesum;
    #else
        qpr->cholytime=-1.0;
    #endif
        /* Return the inf-norm residual 2-tuple as the 4-tuple (inf1,inf2,0,0) */
        qpr->inftuple[0]=inf1;
        qpr->inftuple[1]=inf2;
        qpr->inftuple[2]=0.0;
        qpr->inftuple[3]=0.0;
    }
    
    if (!oktostop) return 1;
    
    return 0;
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

