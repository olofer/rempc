#ifndef __MULTISOLVER_H__
#define __MULTISOLVER_H__

#ifndef __MULTISOLVER_MALLOC
#define __MULTISOLVER_MALLOC malloc
#endif

#ifndef __MULTISOLVER_FREE
#define __MULTISOLVER_FREE free
#endif

#ifndef __MULTISOLVER_PRINTF
#define __MULTISOLVER_PRINTF printf
#endif

#ifndef __MULTISOLVER_MINIMUM_STAGES
#define __MULTISOLVER_MINIMUM_STAGES 3
#endif

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
	if (M->buf!=NULL) { __MULTISOLVER_FREE(M->buf); M->buf=NULL; }
}

/* return 1 if OK, 0 if allocation error, -1 if refused to create ("too dense" matrix) */
int sparseMatrixCreate(sparseMatrix *M,double *A,int m,int n) {
	M->buf=NULL;
	int nnz=matopc_nnz(A,m,n);
	if ((double)nnz/(double)(m*n)>0.50) return -1;
	int numbytes=
		2*(nnz)*sizeof(double)+2*(nnz)*sizeof(int)+
		(m+1)*sizeof(int)+(n+1)*sizeof(int);
	char *buf=__MULTISOLVER_MALLOC(numbytes);
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
		__MULTISOLVER_PRINTF("ERROR: byteofs=%i (it should be=%i)\n",byteofs,numbytes);
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
 * Aux. memory required is allocated with only few calls to __MULTISOLVER_MALLOC(..).
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
    if (ll!=qpd->ndec) __MULTISOLVER_PRINTF("ERROR:[%s]\n",__func__);
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
    if (ll!=qpd->ndec) __MULTISOLVER_PRINTF("ERROR(1):[%s]\n",__func__);
    if (rr!=qpd->niq) __MULTISOLVER_PRINTF("ERROR(2):[%s]\n",__func__);
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
    if (ll!=qpd->ndec) __MULTISOLVER_PRINTF("ERROR(1):[%s]\n",__func__);
    if (rr!=qpd->niq) __MULTISOLVER_PRINTF("ERROR(2):[%s]\n",__func__);
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
    if (ll!=qpd->ndec) __MULTISOLVER_PRINTF("ERROR(1):[%s]\n",__func__);
    if (rr!=qpd->neq) __MULTISOLVER_PRINTF("ERROR(2):[%s]\n",__func__);
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
    if (ii!=qpd->nstg-1) __MULTISOLVER_PRINTF("ERROR(0):[%s]\n",__func__);
    #endif
    ndi=pstg[qpd->nstg-1].nd;
    matopc_atx(&py[jj],pstg[qpd->nstg-1].ptrD,neqi,ndi,&px[kk-neqi0]); /* last mult */
    jj+=ndi;
    #ifdef __CLUMSY_ASSERTIONS__
    if (jj!=qpd->ndec) __MULTISOLVER_PRINTF("ERROR(1):[%s]\n",__func__);
    if (kk!=qpd->neq) __MULTISOLVER_PRINTF("ERROR(2):[%s]\n",__func__);
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
    if (ll!=qpd->ndec) __MULTISOLVER_PRINTF("ERROR:[%s]\n",__func__);
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
	if (dat->nstg<__MULTISOLVER_MINIMUM_STAGES) return 0;
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
        __MULTISOLVER_PRINTF("[%s]: ndmax=%i, nemax=%i, ndtot=%i, netot=%i\n",
                __func__,dat->ndmax,dat->nemax,dat->ndtot,dat->netot);
        __MULTISOLVER_PRINTF("[%s]: blkphsz=%i, blklysz=%i,blkwrsz=%i\n",
                __func__,dat->blkphsz,dat->blklysz,dat->blkwrsz);
    }
	/* Allocate memory if requested based on bit pattern of argument whichmem */
	if (whichmem & 0x0001) {
		dat->blkph=(double *)__MULTISOLVER_MALLOC(dat->blkphsz*sizeof(double));
		if (dat->blkph==NULL) errc++;
		/*__MULTISOLVER_PRINTF("[%s]: malloc(blkph).\n",__func__);*/
	}
	if (whichmem & 0x0002) {
		dat->blkly=(double *)__MULTISOLVER_MALLOC(dat->blklysz*sizeof(double));
		if (dat->blkly==NULL) errc++;
		/*__MULTISOLVER_PRINTF("[%s]: malloc(blkly).\n",__func__);*/
	}
	if (whichmem & 0x0004) {
		dat->blkwr=(double *)__MULTISOLVER_MALLOC(dat->blkwrsz*sizeof(double));
		if (dat->blkwr==NULL) errc++;
		/*__MULTISOLVER_PRINTF("[%s]: malloc(blkwr).\n",__func__);*/
	}
	return (errc==0 ? 1 : 0);
}

/* Free memory; but do not touch nstg/pstg */
void FreePrblmStruct(qpdatStruct *dat,int verbosity) {
	int fc=0;
	if (dat->blkph!=NULL) {
		__MULTISOLVER_FREE(dat->blkph);
		dat->blkph=NULL;
		fc++;
	}
	if (dat->blkly!=NULL) {
		__MULTISOLVER_FREE(dat->blkly);
		dat->blkly=NULL;
		fc++;
	}
	if (dat->blkwr!=NULL) {
		__MULTISOLVER_FREE(dat->blkwr);
		dat->blkwr=NULL;
		fc++;
	}
    if (verbosity>2) {
        __MULTISOLVER_PRINTF("[%s]: num.freed=%i.\n",__func__,fc);
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
		__MULTISOLVER_PRINTF("ERROR[%s]: memory offset mismatch (%i!=%i).\n",
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
  if (ii!=N-1) __MULTISOLVER_PRINTF("[%s]: iteration counter error!\n",__func__);
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
		__MULTISOLVER_PRINTF("[%s]: memory offset counting error LY!\n",__func__);
	if (ofsp!=dat->blkphsz)
		__MULTISOLVER_PRINTF("[%s]: memory offset counting error PH!\n",__func__);
    if (useJay>0)
        if (ofsv!=dat->niq)
            __MULTISOLVER_PRINTF("[%s]: memory offset counting error (%i=ofsv!=niq=%i)\n",
                    __func__,ofsv,dat->niq);
  #endif
    /*#ifdef __DEVELOPMENT_TEXT_OUTPUT__
	__MULTISOLVER_PRINTF("[%s] @ ii=%i, ofsl=%i, blklysz=%i, ofsp=%i, blkphsz=%i\n",
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
		__MULTISOLVER_PRINTF("ERROR[%s]: memory offset mismatch (%i!=%i).\n",
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
		__MULTISOLVER_PRINTF("[%s]: early exit due to NULL pointer(s).\n",__func__);
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
		__MULTISOLVER_PRINTF("[%s]: ofsy=%i, ndtot=%i\n",__func__,ofsy,dat->ndtot);
		__MULTISOLVER_PRINTF("[%s]: ofsp=%i, blkphsz=%i\n",__func__,ofsp,dat->blkphsz);
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
		__MULTISOLVER_PRINTF("[%s]: ofsx=%i, netot=%i\n",__func__,ofsx,dat->netot);
		__MULTISOLVER_PRINTF("[%s]: ofsl=%i, blklysz=%i\n",__func__,ofsl,dat->blklysz);
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
		__MULTISOLVER_PRINTF("[%s]: ofsx=%i, netot=%i\n",__func__,ofsx,dat->netot);
		__MULTISOLVER_PRINTF("[%s]: ofsl=%i, blklysz=%i\n",__func__,ofsl,dat->blklysz);
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
    qpd->vecwr=__MULTISOLVER_MALLOC((qpd->vecwrsz)*sizeof(double));
    if (verbosity>2) {
        if (qpd->vecwr!=NULL) {
            __MULTISOLVER_PRINTF("[%s]: vecwrsz=%i\n",__func__,qpd->vecwrsz);
        }
    }
    return (qpd->vecwr!=NULL ? 1 : 0);
}

/* Free memory that was allocated by the previous subprogram */
void msqp_pdipm_free(qpdatStruct *qpd,int verbosity) {
    if (qpd->vecwr!=NULL) {
        __MULTISOLVER_FREE(qpd->vecwr);
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
    __MULTISOLVER_PRINTF("[%s]\n",__func__);
    __MULTISOLVER_PRINTF("qpo->maxiters=%i\n",qpo->maxiters);
    __MULTISOLVER_PRINTF("qpo->eta=%f\n",qpo->eta);
    __MULTISOLVER_PRINTF("qpo->ep=%e\n",qpo->ep);
    __MULTISOLVER_PRINTF("nx=%i, ny=%i, nz=%i\n",nx,ny,nz);
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
		  __MULTISOLVER_PRINTF("[%s]: MISMATCH: ofs=%i, vecwrsz=%i\n",__func__,ofs,qpd->vecwrsz);
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
            __MULTISOLVER_PRINTF("[%s]: ERROR: chret=%i\n",__func__,chret);
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
    __MULTISOLVER_PRINTF("[%s]: stop after %i itrs. (oktostop=%i, chret=%i).\n",
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
    __MULTISOLVER_PRINTF("[%s]: fx*=%e, fofs=%e\n",
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
		    __MULTISOLVER_PRINTF("[%s]: MISMATCH: ofs=%i, vecwrsz=%i\n",
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

#endif  // __MULTISOLVER_H__
