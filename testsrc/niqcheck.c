/*

Tests for the block-structured multi-stage solver elements (some of the elements).
Sets up a multi-stage structure with random matrix data and with random-size stage vectors.
Also test of "no inequality" (NIQ) solver specifically defined in ../multisolver.h
Complete test analysis requires running the script: niqcheck.py

*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <memory.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include "mt19937ar.h"
#include "vectorops.h"
#include "matrixopsc.h"
#include "textio.h"
#include "multisolver.h"

/* Uniform integer on 1..q */
int genrand_index1q(int q) {
	return (int)ceil(q*genrand_real3());
}

/* Main entry-point */
int main(int argc,const char **argv)
{
	unsigned long _mt_init[4]={0x123, 0x234, 0x345, 0x456};
	unsigned long _mt_length=4;
	time_t epochtime;
	epochtime = time(NULL);
	if (epochtime==((time_t)-1)) {
		printf("WARNING: time() failed. MT PRNG will be seeded with a default.\n");
		init_by_array(_mt_init,_mt_length);
	} else {
		init_genrand((unsigned long)epochtime);
	}

  const char bigPhiTextfilename[] = "bigPhi.txt";
  const char bigCeeTextfilename[] = "bigCee.txt";
  const char bigEllTextfilename[] = "bigEll.txt";
  //const char bigWhyTextFilename[] = "bigWhy.txt"; // this is created by "nondense" block based method...
  //const char bigEllWhyTextFilename[] = "bigEllWhy.txt"; // this is created by "nondense" block based method...
  const int minimumStages = __MULTISOLVER_MINIMUM_STAGES;
  const double PHI_EPSILON = 1.0e-1;

	int numStages;
	int dmin,dmax,tmp;
	int N,ii,ofs1,ofs2,ofs3,rr,cc;
	int nd,ne,ndmax;
	int do_text_output_dump;

	double dd1,dd2;

	int *stageDims=NULL;
	int *cnstrDims=NULL;
	stageStruct *pstg=NULL;

	double *pqbuf=NULL;
	double *pqtmp=NULL;
	double *pcdbuf=NULL;

	double *bigPhi=NULL;
	double *bigCee=NULL;
	double *bigWhy=NULL;

	bool return_ok = true;

	if (argc<4 || argc>5) {
		printf("usage: %s numstages dmin dmax [textfile]\n",argv[0]);
		printf("sourcefile: %s, timestamp: %s\n",__FILE__,__TIMESTAMP__);
		return 1;
	}

	numStages=(int)atof(argv[1]); // numStages=N+1
	dmin=(int)atof(argv[2]);
	dmax=(int)atof(argv[3]);
	if (argc>=5) {
		do_text_output_dump = (int)atof(argv[4]);
	} else {
		do_text_output_dump = 0;
	}

	if (do_text_output_dump>0) {
		printf("Will dump big text files.\n");
	}

	if (numStages<minimumStages) {
		printf("At least %i stages required; bumping up (from %i).\n",minimumStages,numStages);
		numStages=minimumStages;
	}

	if (dmax<dmin) {
		printf("dmin<=dmax required; swapping.\n");
		tmp=dmin; dmin=dmax; dmax=tmp;
	}

	N=numStages-1;
	printf("N=%i\n",N);
	printf("dmin=%i, dmax=%i\n",dmin,dmax);

	// Create an integer vector with N+1 stage-variable sizes in [dmin,dmax]...
	// And also create a vector of equality constraint sizes (N) with size constraints...
	// Then calculate how much memory is needed to fill up the blocks with random data...
	// And also reqs. for the full matrices [for testing !!!]

	stageDims=(int *)malloc(numStages*sizeof(int));
	cnstrDims=(int *)malloc((numStages-1)*sizeof(int));
	pstg=(stageStruct *)malloc(numStages*sizeof(stageStruct));
  memset(pstg, 0, numStages*sizeof(stageStruct));

	/* ii=0..N */
	nd=0; ndmax=-1; ofs1=0;
	for (ii=0;ii<numStages;ii++) {
		stageDims[ii]=dmin-1+genrand_index1q(dmax-dmin+1);	/* uniform on [dmin,dmax] */
		nd+=stageDims[ii];
		if (stageDims[ii]>ndmax) {
			ndmax=stageDims[ii];
		}
		ofs1+=stageDims[ii]*(stageDims[ii]+1); /* accumulate the memory required for diagonal blocks of Phi + extra diag.. */
	//	printf("[dim#%i]: %i\n",ii,stageDims[ii]);
	}

	pqbuf=(double *)malloc(ofs1*sizeof(double));
	pqtmp=(double *)malloc(ndmax*ndmax*sizeof(double));

	/* ii=0..N-1 */
	ne=0; ofs1=0;
	for (ii=0;ii<(numStages-1);ii++) {
		cnstrDims[ii]=(int)ceil((double)(stageDims[ii]+stageDims[ii+1])/3.0);
		ne+=cnstrDims[ii];
		ofs1+=cnstrDims[ii]*(stageDims[ii]+stageDims[ii+1]);	/* Aggreg. mem. needed for [C,D] full matrices */
	//	printf("[eq#%i]: %i\n",ii,cnstrDims[ii]);
	}

	printf("nd=%i (ndmax=%i), ne=%i\n",nd,ndmax,ne);

	pcdbuf=(double *)malloc(ofs1*sizeof(double));

	/* Allocate the big full dense matrices for reference calculations */
	bigPhi=(double *)malloc(nd*(nd+1)*sizeof(double));
	bigCee=(double *)malloc(ne*nd*sizeof(double));
	bigWhy=(double *)malloc(ne*(ne+1)*sizeof(double));

	if (bigPhi==NULL || bigCee==NULL || bigWhy==NULL) {
		printf("Allocation failure!\n");
		return 1;
	}

	matopc_zeros(bigPhi,nd,nd);
	matopc_zeros(bigCee,ne,nd);

	/* Initialize the stage struct array */
	ofs1=0; ofs2=0; ofs3=0; rr=0; cc=0;
	for (ii=0;ii<numStages;ii++) {
		pstg[ii].idx=ii;
		pstg[ii].nd=stageDims[ii];
		if (ii<numStages-1) {
			pstg[ii].neq=cnstrDims[ii];
		} else {
			pstg[ii].neq=0;
		}
		pstg[ii].ptrQ=&pqbuf[ofs1];
		/* Create a random pos. def. sym. matrix block */
		matopc_randn(pqtmp,pstg[ii].nd,pstg[ii].nd);
		matopc_mtm(pstg[ii].ptrQ,pqtmp,pstg[ii].nd,pstg[ii].nd,MATOPC_UPPER);
		matopc_mpeye(pstg[ii].ptrQ,pstg[ii].nd,PHI_EPSILON); // add scaled identity matrix into Q to control the numerical condition..
		matopc_symmetrize(pstg[ii].ptrQ,pstg[ii].nd,MATOPC_UPPER);
		/* Copy it into the big block diagonal matrix Phi */
		matopc_sub_assign(bigPhi,nd,nd,ofs2,ofs2,pstg[ii].ptrQ,pstg[ii].nd,pstg[ii].nd,+1);
		/* Move forward to next block */
		ofs1+=(pstg[ii].nd)*(pstg[ii].nd+1);
		ofs2+=pstg[ii].nd;
		/* Next create random [C,D] matrix */
		if (ii==0) {
			pstg[ii].ptrC=&pcdbuf[ofs3];
			pstg[ii].ptrD=NULL;
		} else if (ii==numStages-1) {
			pstg[ii].ptrC=NULL;
			pstg[ii].ptrD=&pcdbuf[ofs3-cnstrDims[ii-1]*stageDims[ii]];
		} else {
			pstg[ii].ptrC=&pcdbuf[ofs3];
			pstg[ii].ptrD=&pcdbuf[ofs3-cnstrDims[ii-1]*stageDims[ii]];
		}
		if (ii<numStages-1) {
			matopc_randn(&pcdbuf[ofs3],cnstrDims[ii],stageDims[ii]+stageDims[ii+1]);
			/* Copy [C,D] into bigCee */
			matopc_sub_assign(bigCee,ne,nd,rr,cc,&pcdbuf[ofs3],cnstrDims[ii],stageDims[ii]+stageDims[ii+1],+1);
			rr+=cnstrDims[ii];
			cc+=stageDims[ii];
			ofs3+=cnstrDims[ii]*(stageDims[ii]+stageDims[ii+1]);
		}
	}

	/* Run a verification code that checks the equality of bigCee blocks to
	   those that are supposed to exactly the same based on the stage struct array pointer fields.
	 */
	rr=0; cc=0; dd2=0.0;
	for (ii=0;ii<numStages-1;ii++) {
		/* Look into bigCee @ row rr and column cc */

		// First extract pqtmp <- block of bigCee : compare to C[ii]
		matopc_sub_extract(pqtmp,pstg[ii].neq,pstg[ii].nd,rr,cc,bigCee,ne,nd);
		dd1=matopc_frob_norm_diff(pqtmp,pstg[ii].ptrC,pstg[ii].neq,pstg[ii].nd);
		//dd2=matopc_frob_norm(pqtmp,pstg[ii].neq,pstg[ii].nd);
		if (dd1!=0.0) {
			printf("WARNING[stage=%i]: frob.norm.diff[C]=%e\n",ii,dd1);
		}
		dd2+=dd1;
		// Then extract another block of bigCee : compare to D[ii+1]
		matopc_sub_extract(pqtmp,pstg[ii].neq,pstg[ii+1].nd,rr,cc+pstg[ii].nd,bigCee,ne,nd);
		dd1=matopc_frob_norm_diff(pqtmp,pstg[ii+1].ptrD,pstg[ii].neq,pstg[ii+1].nd);
		if (dd1!=0.0) {
			printf("WARNING[stage=%i]: frob.norm.diff[D]=%e\n",ii+1,dd1);
		}
		dd2+=dd1;
		rr+=pstg[ii].neq;
		cc+=pstg[ii].nd;
		//ofs3+=pstg[ii].neq*(pstg[ii].nd+pstg[ii+1].nd);
	}
	printf("Total frob.err.sum=%e\n",dd2); /* MUST BE ZERO! */

	return_ok = (return_ok && (dd2 == 0.0));

	/* C:	ne-by-nd
	 * Phi:	nd-by-nd	Phi = blkdiag(Q(i)) = L0*L0', inv(Phi)=L0'\(L0\I)
	 * Y:	ne-by-ne	Y = C*inv(Phi)*C' = Z'*Z, Z = L0\C'
	 * L:	ne-by-ne	Y = L*L', L lower (block) triangular Cholesky factor
	 *
	 * Form the above matrices (as full dense matrices).
	 * This is the referance result for the block-based factorization code.
	 * Also dump the full dense matrices to text files for external verification.
	 *
	 */

	if (do_text_output_dump>0) {
		tmp=textio_write_double_array_matrix(bigPhiTextfilename,bigPhi,nd,nd,"%.16e");
		return_ok = (return_ok && (tmp == 1));
		if (tmp!=1) {
			printf("Failed to write text file: %s\n",bigPhiTextfilename);
		} else {
			printf("Wrote text file: %s\n",bigPhiTextfilename);
		}
		tmp=textio_write_double_array_matrix(bigCeeTextfilename,bigCee,ne,nd,"%.16e");
		return_ok = (return_ok && (tmp == 1));
		if (tmp!=1) {
			printf("Failed to write text file: %s\n",bigCeeTextfilename);
		} else {
			printf("Wrote text file: %s\n",bigCeeTextfilename);
		}
	}

	/* Transpose bigCee in place, factorize bigPhi in place,
	   backsolve for Z in place, then create bigWhy and factorize bigWhy in place... neat!
	 */

	printf("Creating and factorizing full dense Y (%i-by-%i) for reference...",ne,ne);
	matopc_inplace_transpose(bigCee,ne,nd);
	tmp=matopc_cholesky_decompose(bigPhi,&bigPhi[nd*nd],nd);
	return_ok = (return_ok && (tmp == 0));
	if (tmp!=0) {
		printf("Cholesky decomposition of bigPhi failed.\n");
	}
	matopc_cholesky_trisubst_left_matrix(bigPhi,&bigPhi[nd*nd],nd,bigCee,bigCee,ne); // Z=L0\C'
	matopc_mtm(bigWhy,bigCee,nd,ne,MATOPC_UPPER); // upper triangle of Y=Z'*Z
	tmp=matopc_cholesky_decompose(bigWhy,&bigWhy[ne*ne],ne);
	return_ok = (return_ok && (tmp == 0));
	if (tmp!=0) {
		printf("Cholesky decomposition of bigWhy failed.\n");
	}
	printf("done.\n");

	if (return_ok) {
		/*
		Testdrive the multisolver.h solver "API" (for the case with no inequalities -> block structured linear equation once)
		*/
		const int local_qpd_verbosity = 3;
		qpdatStruct qpd;
		memset(&qpd, 0, sizeof(qpdatStruct));
		qpd.pstg = pstg;
		qpd.nstg = numStages;
		qpd.ndec = nd;
		qpd.neq = ne;
		qpd.ph = NULL; // should be random nd vector
    qpd.pf = NULL; // can be NULL
    qpd.pd = NULL; // should be random ne vector

		tmp = InitializePrblmStruct(&qpd, 0x07, local_qpd_verbosity);
		return_ok = (return_ok && (tmp == 1));
		tmp = msqp_pdipm_init(&qpd, local_qpd_verbosity);
		return_ok = (return_ok && (tmp == 1));

		qpoptStruct qpo;
		memset(&qpo, 0, sizeof(qpoptStruct));

		qpretStruct qpr;
		memset(&qpr, 0, sizeof(qpretStruct));

		qpo.verbosity=DEFAULT_OPT_VERBOSITY;
    qpo.maxiters=DEFAULT_OPT_MAXITERS;
    qpo.ep=DEFAULT_OPT_EP;
    qpo.eta=DEFAULT_OPT_ETA;
    qpo.expl_sparse=0;
    qpo.chol_update=0;
    qpo.blas_suite=0;

		// TODO: the main task is now to run msqp_solve_niq(..) ; using a few random RHSs, and saving the results to disk; for numpy dense verification
		//tmp = msqp_solve_niq(&qpd, &qpo, &qpr);
		//return_ok = (return_ok && (tmp == 0));

    msqp_pdipm_free(&qpd, local_qpd_verbosity);
		FreePrblmStruct(&qpd, local_qpd_verbosity);
	}

	if (do_text_output_dump>0) {
		matopc_zero_triangle(bigWhy,ne,MATOPC_UPPER);
		matopc_setdiag(bigWhy,&bigWhy[ne*ne],ne);
		tmp=textio_write_double_array_matrix(bigEllTextfilename,bigWhy,ne,ne,"%.16e");
		return_ok = (return_ok && (tmp == 1));
		if (tmp!=1) {
			printf("Failed to write text file: %s\n",bigEllTextfilename);
		} else {
			printf("Wrote text file: %s\n",bigEllTextfilename);
		}
	}

	free(bigCee);
	free(bigPhi);
	free(bigWhy);

	free(pcdbuf);
	free(pqbuf);
	free(pqtmp);
	free(pstg);
	free(stageDims);
	free(cnstrDims);
	
	return (return_ok ? 0 : 2);
}
