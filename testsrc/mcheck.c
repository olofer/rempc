/*
 * Code for checking basic matrix routines.
 */

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <assert.h>

/* High-quality random numbers */
#include "mt19937ar.h"
/* Include basic generic vector operation utility functions: vectorops.h */
#include "vectorops.h"
/* Include the basic column-major matrix algebra routines */
#include "matrixopsc.h"
/* Need a few text input/output functions */
#include "textio.h"

int test_transpose(matopc_real *A,int m,int n);
int test_at_mult_a(matopc_real *A,int m,int n);
int test_a_mult_at(matopc_real *A,int m,int n);
int test_cholesky(matopc_real *A,int m,int n,matopc_real eptol);
int test_chol_solve(matopc_real *A,int m,int n,matopc_real eptol);
int test_gemm_and_gemv(int m,int n,int k,matopc_real eptol);
int test_chol_subst_inv(int m,int n,matopc_real eptol);
int test_chol_subst_leftright(int m,int n,matopc_real eptol);
int test_chol_rank1_update(int n,int q,matopc_real eptol);

/* Utility function for in-place residual check */
matopc_real frob_diff_chol(int n,matopc_real *L1,matopc_real *d1,matopc_real *L2,matopc_real *d2);

int main(int argc,char **argv) {

	const char xfilename[]="X.dat";
	const char xtfilename[]="Xt.dat";
	const char tablespec[]="%.16e";	/* full capacity for doubles */
	const char wfilename[]="W.dat";
	const char xwfilename[]="XW.dat";
	const double epstol=1.0e-14;
	int retval,cholret;

	unsigned long _mt_init[4]={0x123, 0x234, 0x345, 0x456};
	unsigned long _mt_length=4;

	if (argc<2 || argc>4) {
		printf("*** Test program for column-major plain C matrix library routines ***\n");
		printf("*** Source file (%s) timestamp: %s ***\n",__FILE__,__TIMESTAMP__);
		printf("usage: %s m [n] [k]\n",argv[0]);
		return 1;
	}
	
	int m;	/* # rows for test matrices, where applicable */
	int n;	/* # columns for test matrices, where applicable */
	int k;	/* # columns for GEMM test second matrix */
	int q;	/* # rank-1 updates */
	
	m = atoi(argv[1]);
	
	if (argc>=3) {
		n = atoi(argv[2]);
	} else {
		n = m;
		k = m;
	}
	
	if (argc>=4) {
		k = atoi(argv[3]);
	} else {
		k = n;
	}

	q=k;
	
	if (n<=0 || m<=0 || k<=0 || q<=0) {
		printf("m,n,k>0 required.\n");
		return 1;
	}
	
	printf("m=%i, n=%i, k=%i, q=%i\n",m,n,k,q);
	
	time_t epochtime;
	epochtime = time(NULL);
	if (epochtime==((time_t)-1)) {
		printf("WARNING: time() failed. MT PRNG will be seeded with a default.\n");
		init_by_array(_mt_init,_mt_length);
	} else {
		init_genrand((unsigned long)epochtime);
	}
	
	/* Allocate a big enough chunk to hold all the matrices needed for later */
	double *heap,*p0,*p1,*p2,*p3,*pd,*px,*p4,*p5;
	heap=(double *)malloc(sizeof(double)*(m*n+m*m+n*n+m*n+(m+n)*2+n*k+m*k));
	
	if (heap==NULL) {
		printf("Failed to allocate memory.\n");
		return 1;
	}
	
	p0=&heap[0];
	p1=&heap[0+m*n];
	p2=&heap[0+m*n+m*m];
	p3=&heap[0+m*n+m*m+n*n];
	pd=&heap[0+m*n+m*m+n*n+m*n];	/* diagonal of Cholesky factor */
	px=&heap[0+m*n+m*m+n*n+m*n+(m+n)]; /* solution vector storage (and rhs) */
	p4=&heap[0+m*n+m*m+n*n+m*n+(m+n)*2];
	p5=&heap[0+m*n+m*m+n*n+m*n+(m+n)*2+n*k];
	
	matopc_randn(p0,m,n);	/* normal standard variates */
	/* matopc_rand(p0,m,n,-1.0,+1.0); */		/* p0 = rand(m,n), elements uniform on (-1,+1) */
	matopc_copy_transpose(p3,p0,m,n);	/* p3 = p0' */
	
	retval=test_transpose(p0,m,n); if (retval!=1) printf("test failed.\n");
	retval=test_at_mult_a(p0,m,n); if (retval!=1) printf("test failed.\n");
	retval=test_a_mult_at(p0,m,n); if (retval!=1) printf("test failed.\n");
	retval=test_cholesky(p0,m,n,epstol); if (retval!=1) printf("test failed.\n");
	retval=test_chol_solve(p0,m,n,epstol); if (retval!=1) printf("test failed.\n");
	retval=test_gemm_and_gemv(m,n,k,epstol); if (retval!=1) printf("test failed.\n");
	retval=test_chol_subst_inv(m,n,epstol); if (retval!=1) printf("test failed.\n");
	retval=test_chol_subst_leftright(m,n,epstol); if (retval!=1) printf("test failed.\n");
	retval=test_chol_rank1_update(m,q,epstol); if (retval!=1) printf("test failed.\n");
	
	retval=textio_write_double_array_matrix(xfilename,p0,m,n,tablespec);
	if (retval!=1) {
		printf("Failed to write %s\n",xfilename);
	}
	retval=textio_write_double_array_matrix(xtfilename,p3,n,m,tablespec);
	if (retval!=1) {
		printf("Failed to write %s\n",xtfilename);
	}
	
	matopc_mmt(p1,p0,m,n,MATOPC_UPPER); /* p1 = p0*p0' (m-by-m) only compute the upper triangle */
	matopc_mpeye(p1,m,1.0);	/* p1 = p1 + eye(m)*1.0 */	
	matopc_symmetrize(p1,m,MATOPC_UPPER);
	textio_write_double_array_matrix("YpI.dat",p1,m,m,tablespec);
	cholret=matopc_cholesky_decompose(p1,pd,m);
	if (cholret!=0) {
		printf("[%s]: cholesky failed at line %i.\n",__func__,__LINE__);
		free(heap);
		return cholret;
	}
	/* Solve equation YpI*y=ones(m,1) for y and write y.dat */
	matopc_ones(px,m,1);
	matopc_cholesky_solve(p1,pd,m,px,px);
	textio_write_double_array_matrix("y.dat",px,m,1,tablespec);
	/* Now assign diagonal of p1 and zero the upper part -> then write cholesky factor to L1.dat */
	matopc_setdiag(p1,pd,m);
	matopc_zero_triangle(p1,m,MATOPC_UPPER);
	textio_write_double_array_matrix("L1.dat",p1,m,m,tablespec);
	
	matopc_mtm(p2,p0,m,n,MATOPC_UPPER); /* p2 = p0'*p0 (n-by-n) only compute the upper triangle */
	matopc_mpeye(p2,n,1.0); /* p2 = p2 + eye(n)*1.0 */
	matopc_symmetrize(p2,n,MATOPC_UPPER);
	textio_write_double_array_matrix("ZpI.dat",p2,n,n,tablespec);
	cholret=matopc_cholesky_decompose(p2,pd,n);
	if (cholret!=0) {
		printf("[%s]: cholesky failed at line %i.\n",__func__,__LINE__);
		free(heap);
		return cholret;
	}
	/* Solve equation ZpI*z=ones(n,1) for z and write z.dat */
	matopc_ones(px,n,1);
	matopc_cholesky_solve(p2,pd,n,px,px);
	textio_write_double_array_matrix("z.dat",px,n,1,tablespec);
	/* Now assign diagonal of p1 and zero the upper part -> then write cholesky factor to L2.dat */
	matopc_setdiag(p2,pd,n);
	matopc_zero_triangle(p2,n,MATOPC_UPPER);
	textio_write_double_array_matrix("L2.dat",p2,n,n,tablespec);
	
	/* Now allow an external check of the GEMM routine */
	matopc_rand(p4,n,k,-1.0,+1.0);	/* p4=rand(n,k) */
	retval=textio_write_double_array_matrix(wfilename,p4,n,k,tablespec);
	if (retval!=1) {
		printf("Failed to write %s\n",wfilename);
	}
	matopc_gemm(p5,m,k,p0,n,p4,0.0,1.0);
	retval=textio_write_double_array_matrix(xwfilename,p5,m,k,tablespec);
	if (retval!=1) {
		printf("Failed to write %s\n",xwfilename);
	}
	
	free(heap);
	
	return 0;
}

/*
 * Self-contained test routines follow below.
 * They are isolated in the sense that they manage their own local memory.
 */

int test_transpose(matopc_real *A,int m,int n) {
	matopc_real *buf=malloc(sizeof(matopc_real)*m*n*2);
	matopc_real *B,*C;
	matopc_real diff;
	if (!buf) return 0;
	B=&buf[0]; C=&buf[m*n];
	matopc_copy(B,A,m,n);
	matopc_copy(C,A,m,n);
	diff = matopc_frob_norm_diff(B,A,m,n);
	diff += matopc_frob_norm_diff(C,A,m,n);
	diff += matopc_frob_norm_diff(C,B,m,n);	/* check copy routine */
	matopc_copy_transpose(B,A,m,n);
	matopc_copy_transpose(C,B,n,m);
	diff += matopc_frob_norm_diff(C,A,m,n);	/* two transposes should equal a copy */
	/* Now do an in-place transpose of C and compare it to B; accumulate the error: diff+=... */
	matopc_inplace_transpose(C,m,n);
	diff += matopc_frob_norm_diff(C,B,n,m);
	free(buf);
	printf("[%s]: diff.=%.16e\n",__func__,diff);
	return (diff==0.0 ? 1 : 0);
}

int test_at_mult_a(matopc_real *A,int m,int n) {
	matopc_real *buf=malloc(sizeof(matopc_real)*n*n*2);
	matopc_real *B,*C;
	matopc_real diff;
	if (!buf) return 0;
	B=&buf[0]; C=&buf[n*n];
	/* Produce A'*A (n-by-n) in two ways: form upper,
	   copy to lower or vice versa; then compare the two results */
	matopc_mtm(B,A,m,n,MATOPC_UPPER);
	matopc_symmetrize(B,n,MATOPC_UPPER);
	matopc_mtm(C,A,m,n,MATOPC_LOWER);
	matopc_symmetrize(C,n,MATOPC_LOWER);
	diff = matopc_frob_norm_diff(B,C,n,n);
	free(buf);
	printf("[%s]: diff.=%.16e\n",__func__,diff);
	return (diff==0.0 ? 1 : 0);
}

int test_a_mult_at(matopc_real *A,int m,int n) {
	matopc_real *buf=malloc(sizeof(matopc_real)*m*m*2);
	matopc_real *B,*C;
	matopc_real diff;
	if (!buf) return 0;
	B=&buf[0]; C=&buf[m*m];
	/* Produce A*A' (m-by-m) in two ways: form upper,
	   copy to lower or vice versa; then compare the two results */
	matopc_mmt(B,A,m,n,MATOPC_UPPER);
	matopc_symmetrize(B,m,MATOPC_UPPER);
	matopc_mmt(C,A,m,n,MATOPC_LOWER);
	matopc_symmetrize(C,m,MATOPC_LOWER);
	diff = matopc_frob_norm_diff(B,C,m,m);
	free(buf);
	printf("[%s]: diff.=%.16e\n",__func__,diff);
	return (diff==0.0 ? 1 : 0);
}

/* Form Y=A*A'+eye, form the lower Cholesky factor L: Y=L*L', then
   re-multiply the factor and check the elementwise difference: Y-L*L' */
int test_cholesky(matopc_real *A,int m,int n,matopc_real eptol) {
	matopc_real *buf=malloc(sizeof(matopc_real)*(m*m*3+m));
	matopc_real *Y1,*Y2,*Y3,*P;
	matopc_real diff,reldiff;
	int cholret;
	if (!buf) return 0;
	Y1=&buf[0]; Y2=&buf[m*m]; Y3=&buf[m*m*2]; P=&buf[m*m*3];
	matopc_mmt(Y1,A,m,n,MATOPC_UPPER);	/* Y1 = A*A' (upper half only) */
	matopc_mpeye(Y1,m,1.0);	/* Y1=A*A'+eye(m) (upper half only) */
	matopc_copy(Y2,Y1,m,m);	/* Copy Y2=Y1 (only upper half is useful) */
	matopc_symmetrize(Y2,m,MATOPC_UPPER);	/* After copy, Y2 is expanded to a full matrix */
	cholret=matopc_cholesky_decompose(Y1,P,m);	/* Cholesky factor will be in lower part of Y1 and its diagonal will be in P */
	if (cholret!=0) {
		printf("[%s]: cholesky failed at line %i.\n",__func__,__LINE__);
		free(buf);
		return 0;
	}
	/* Expand the factor L to a full matrix with zeros in the upper triangle */
	matopc_setdiag(Y1,P,m);
	matopc_zero_triangle(Y1,m,MATOPC_UPPER);	/* Y1=L */
	matopc_mmt(Y3,Y1,m,m,MATOPC_UPPER);	/* Y3 = L*L' (upper half only) */
	matopc_symmetrize(Y3,m,MATOPC_UPPER); /* Y3 is now a full symmetric matrix, and should be a reconstruction of Y2 */
	diff=matopc_frob_norm_diff(Y2,Y3,m,m);
	reldiff=diff/(1+matopc_frob_norm(Y2,m,m)); /* Form the relative difference in the Frobenius norm sense... */
	free(buf);
	printf("[%s]: rel.diff.=%.16e\n",__func__,reldiff);
	return (reldiff>=eptol ? 0 : 1);
}

/* Cholesky factorize (A'*A+eye(n))=L*L' and solve (L*L')*X=rand(n,m) for X n-by-m.
   Solve with and without buffer overwrite. And check the solution accuracy with the residual
   calculation Y=rhs; Y=Y-(L*L')*X (gemm update).
 */
int test_chol_solve(matopc_real *A,int m,int n,matopc_real eptol) {
	matopc_real *buf=malloc(sizeof(matopc_real)*(n*n*2+n+m*n*3));
	matopc_real *M,*M2,*P,*Y,*X,*X2;
	matopc_real normY,diff,reldiff;
	int cholret;
	if (!buf) return 0;
	M=&buf[0]; M2=&buf[n*n]; P=&buf[2*n*n];
	Y=&buf[2*n*n+n]; X=&buf[2*n*n+n+m*n]; X2=&buf[2*n*n+n+m*n*2];
	matopc_mtm(M,A,m,n,MATOPC_UPPER);	/* M = A'*A (upper half only) */
	matopc_mpeye(M,n,1.0);	/* M = A'*A + 1.0*eye(n) (upper half only) */
	matopc_copy(M2,M,n,n);	/* Copy M2=M, lower half is garbage */
	matopc_symmetrize(M2,n,MATOPC_UPPER);	/* Extend M2 to full symmetry by a triangle copy */
	cholret=matopc_cholesky_decompose(M,P,n);	/* M=L*L' */
	if (cholret!=0) {
		printf("[%s]: cholesky failed at line %i.\n",__func__,__LINE__);
		free(buf);
		return 0;
	}
	matopc_rand(Y,n,m,-1.0,+1.0);	/* Y = rand(n,m), elements uniform on (-1,+1) */
	matopc_copy(X2,Y,n,m);
	normY=matopc_frob_norm(Y,n,m);
	matopc_cholesky_solve_matrix(M,P,n,Y,X,m);	/* solve matrix equation M*X=Y column by column */
	matopc_cholesky_solve_matrix(M,P,n,X2,X2,m);	/* solve matrix equation M*X=Y column by column, with overwrite */
	diff=matopc_frob_norm_diff(X,X2,n,m);
	printf("[%s]: solve diff.=%.16e\n",__func__,diff);
	if (diff!=0.0) {	/* X and X2 must be exactly the same otherwise fail this test immediately. */
		free(buf);
		return 0;
	}
	/* Now compare M2*X and Y (Frobenius norm of difference) to evaluate how good the solution was. */
	/* matopc_zeros(X2,n,m); */
	matopc_gemm(Y,n,m,M2,n,X2,1.0,-1.0);	/* X2=M2*X with M2=n-by-n, X=n-by-m so X2 is n-by-m as is Y */
	/*diff=matopc_frob_norm_diff(X2,Y,n,m);*/
	diff=matopc_frob_norm(Y,n,m);
	reldiff=diff/(1+normY);
	free(buf);
	printf("[%s]: rel.diff.=%.16e\n",__func__,reldiff);
	return (reldiff>=eptol ? 0 : 1);
}

/* Test of GEMM and GEMV routines and transposed variations of these.
   Let X be m-by-k and Y k-by-n so that X*Y=Z is m-by-n. Two versions of Z are to be maintained for comparisons. 
 */
int test_gemm_and_gemv(int m,int n,int k,matopc_real eptol) {
	/* TASK: generate random matrices X and Y then form Z=X*Y and Z=(Y'*X')' using the GEMM routine and inplace transposition.. */
	matopc_real *buf=malloc(sizeof(matopc_real)*(m*k+k*n+2*m*n));
	matopc_real *X,*Y,*Z1,*Z2;
	matopc_real diff,reldiff;
	int j;
	if (!buf) return 0;
	X=&buf[0]; Y=&buf[m*k]; Z1=&buf[m*k+k*n]; Z2=&buf[m*k+k*n+m*n];
	matopc_rand(X,m,k,-1.0,+1.0);
	matopc_rand(Y,k,n,-1.0,+1.0);
	matopc_gemm(Z1,m,n,X,k,Y,0.0,1.0);
	matopc_inplace_transpose(X,m,k);
	matopc_inplace_transpose(Y,k,n);
	matopc_gemm(Z2,n,m,Y,k,X,0.0,1.0);
	matopc_inplace_transpose(Z2,n,m);
	diff=matopc_frob_norm_diff(Z1,Z2,m,n);	/* must be exactly zero here */
	/* NEXT TASK: alternative Z2 calculation by applying a GEMV routine repeatedly... */
	matopc_inplace_transpose(X,k,m);
	matopc_inplace_transpose(Y,n,k);
	/* OK so Z is m-by-n, i.e. n columns: n tests for GEMV zj=X*yj, where y is the j-th column */
	matopc_zeros(Z2,m,n);
	for (j=0;j<n;j++) {
		/* overwrite Z2 here: z2(j) <- z2(j)+X*y(j) for column j, with Z2 pre-zeroed */
		matopc_ypax(&Z2[j*m],1.0,1.0,X,m,k,&Y[j*k]);
	}
	diff+=matopc_frob_norm_diff(Z1,Z2,m,n);	/* accumulate the error */
	/* FINAL TASK: the same thing but with transposed multiplication "ypatx" */
	matopc_zeros(Z2,m,n);
	matopc_inplace_transpose(X,m,k);
	for (j=0;j<n;j++) {
		/* overwrite Z2 here: z2(j) <- z2(j)+(X')'*y(j) for column j, with Z2 pre-zeroed */
		matopc_ypatx(&Z2[j*m],1.0,1.0,X,k,m,&Y[j*k]);
	}
	diff+=matopc_frob_norm_diff(Z1,Z2,m,n);	/* accumulate the error */
	reldiff=diff/(1.0+matopc_frob_norm(Z1,m,n));
	free(buf);
	printf("[%s]: rel.diff.=%.16e\n",__func__,reldiff);
	return (reldiff>=eptol ? 0 : 1);
}

/* Invert a "covariance matrix" using Cholesky forward and backward substitutions.
   Do this using the standalone subst procedures and the integrated solver and compare the results.
   Finally check that the inverse is actually an inverse by both left and right multiplications.
 */
int test_chol_subst_inv(int m,int n,matopc_real eptol) {
	matopc_real *buf;
	matopc_real *X,*M,*Mcpy,*P,*Y,*Z;
	matopc_real reldiff;
	int cholret,tmp;
	if (m<n) { /* require m>=n so swap if n>m */
		tmp=m; m=n; n=tmp;
	}
	buf=malloc(sizeof(matopc_real)*(m*n+n*n+n*n+n+n*n+n*n));
	if (!buf) return 0;
	X=&buf[0]; M=&buf[m*n]; Mcpy=&buf[m*n+n*n]; P=&buf[m*n+n*n+n*n];
	Y=&buf[m*n+n*n+n*n+n]; Z=&buf[m*n+n*n+n*n+n+n*n];
	printf("[%s]: m=%i,n=%i,eptol=%.16e\n",__func__,m,n,eptol);
	/* (...) form X (m-by-n), then "full" matrix M=X'*X (n-by-n) */
	/* matopc_rand(X,m,n,-1.0,+1.0); */
	matopc_randn(X,m,n);
	matopc_mtm(M,X,m,n,MATOPC_UPPER);
	matopc_copy(Mcpy,M,n,n);
	matopc_symmetrize(Mcpy,n,MATOPC_UPPER);
	/* then Cholesky-decompose M ... */
	cholret=matopc_cholesky_decompose(M,P,n);	/* M=L*L' */
	if (cholret!=0) {
		printf("[%s]: cholesky failed at line %i.\n",__func__,__LINE__);
		free(buf);
		return 0;
	}
	/* ... and solve the matrix equation (L*L')*Y=eye(n), so that inv(M)=Y using forward-backward substitution inplace */
	matopc_eye(Y,n);
	matopc_cholesky_solve_matrix(M,P,n,Y,Y,n);
	/* check that this is actually the inverse formed by multiplication using GEMM */
	matopc_gemm(Z,n,n,Mcpy,n,Y,0.0,1.0); /* Here Z should be the identity matrix ... */
	matopc_mpeye(Z,n,-1.0);	/* subtract 1.0 from the diagonal of Z */
	printf("[%s]: inv.res.norm=%.16e\n",__func__,matopc_frob_norm(Z,n,n));
	/* and solve a second time using the "manual" approach of first L\(.) and then L'\(.) explicitly */
	matopc_eye(Z,n);
	matopc_cholesky_trisubst_left_matrix(M,P,n,Z,Z,n);	/* Z <- L\eye(n) */
	matopc_cholesky_trisubst_tr_left_matrix(M,P,n,Z,Z,n);	/* Z <- L'\Z */
	/* ... and then evaluate the relative difference norm(Z-Y)/(1+norm(Y)) */
	reldiff=matopc_frob_norm_diff(Z,Y,n,n)/(1.0+matopc_frob_norm(Y,n,n));
	printf("[%s]: rel.diff.=%.16e\n",__func__,reldiff);
	/* Then multiply this Z with Mcpy from the other side with GEMM */
	matopc_gemm(Y,n,n,Z,n,Mcpy,0.0,1.0); /* Here Y should become the identity matrix ... */
	matopc_mpeye(Y,n,-1.0);	/* subtract 1.0 from the diagonal of Z */
	printf("[%s]: inv.res.norm=%.16e\n",__func__,matopc_frob_norm(Y,n,n));
	free(buf);
	return (reldiff>=eptol ? 0 : 1);
}

/* Check the matrix Cholesky triangular substitutions L\X, X/L, L'\X, X/L' */
int test_chol_subst_leftright(int m,int n,matopc_real eptol) {
	matopc_real *buf;
	matopc_real *X,*M,*P,*Y,*W,*Z;
	matopc_real reldiff1,reldiff2;
	int cholret,tmp;
	if (m<n) { /* require m>=n so swap if n>m */
		tmp=m; m=n; n=tmp;
	}
	buf=malloc(sizeof(matopc_real)*(m*n+n*n+n+3*n*m));
	if (!buf) return 0;
	X=&buf[0]; M=&buf[m*n]; P=&buf[m*n+n*n];
	Y=&buf[m*n+n*n+n]; W=&buf[m*n+n*n+n+n*m]; Z=&buf[m*n+n*n+n+n*m+n*m];
	printf("[%s]: m=%i,n=%i,eptol=%.16e\n",__func__,m,n,eptol);
	/* (...) form X (m-by-n), then "full" matrix M=X'*X (n-by-n) */
	matopc_randn(X,m,n);
	matopc_mtm(M,X,m,n,MATOPC_UPPER);
	cholret=matopc_cholesky_decompose(M,P,n);	/* M=L*L' */
	if (cholret!=0) {
		printf("[%s]: cholesky failed at line %i.\n",__func__,__LINE__);
		free(buf);
		return 0;
	}
	/* So M,P holds Cholesky factor L and the original matrix M at this point.
	   Proceed by creating an n-by-m rectangular matrix W and solve L*Y=W for Y, Y=L\W
	   Check solution by also doing the transposed problem: Y'*L'=W', Y=(W'/L')'
	 */
	matopc_randn(W,n,m);	/* random matrix W n-by-m, m>n */
	matopc_cholesky_trisubst_left_matrix(M,P,n,W,Y,m);	/* Y <- L\W */
	matopc_inplace_transpose(W,n,m);	/* W <- W' */
	matopc_cholesky_trisubst_tr_right_matrix(M,P,n,W,Z,m);	/* Z <- W'/L' */
	matopc_inplace_transpose(Z,m,n);	/* Z <- Z' */	
	reldiff1=matopc_frob_norm_diff(Z,Y,n,m)/(1.0+matopc_frob_norm(Y,n,m));
	printf("[%s]: rel.diff.=%.16e\n",__func__,reldiff1);
	
	/* Repeat the above using the upper triangular factor L': Y=L'\W and compare to Z=(W'/L)' */
	matopc_inplace_transpose(W,m,n);	/* restore random matrix W n-by-m, m>n */
	matopc_cholesky_trisubst_tr_left_matrix(M,P,n,W,Y,m);	/* Y <- L'\W */
	matopc_inplace_transpose(W,n,m);	/* W <- W' */
	matopc_cholesky_trisubst_right_matrix(M,P,n,W,Z,m);	/* Z <- W'/L */
	matopc_inplace_transpose(Z,m,n);	/* Z <- Z' */	
	reldiff2=matopc_frob_norm_diff(Z,Y,n,m)/(1.0+matopc_frob_norm(Y,n,m));
	printf("[%s]: rel.diff.=%.16e\n",__func__,reldiff2);
	
	free(buf);
	return ((reldiff1>=eptol || reldiff2>=eptol) ? 0 : 1);
}

/* Create a pos. def. n-by-n square matrix A; Cholesky factorize it L*L'=A;
   Then update it with q times rank-1 amendments; A+x(q)*x(q)', x(q) n-by-1 vectors.
   After each update; evaluate the full Cholesky refactorization and compare to the updated
   factor L; monitor the relative errors; and return the final relative error after q rank-1
   corrections.
 */
int test_chol_rank1_update(int n,int q,matopc_real eptol) {
	matopc_real *buf;
	matopc_real reldiff=1.0,diff,sumdiff=0.0,normA;
	matopc_real *pQ,*pA,*pd,*pX,*pd0;
	int cholret,i;

	/* Need: Q n-by-n, A=Q'*Q n-by-n, d n-by-1, X n-by-q, p0 n-by-1 */
	buf=(matopc_real *)malloc(sizeof(matopc_real)*(n*n+n*n+n+n*q+n));
	if (!buf) return 0;
	pQ=&buf[0];
	pA=&buf[n*n];
	pd=&buf[n*n+n*n];
	pX=&buf[n*n+n*n+n];
	pd0=&buf[n*n+n*n+n+n*q];

	matopc_randn(pQ,n,n);	/* Create n-by-n random matrix Q */
	matopc_randn(pX,n,q);	/* Create n-by-q random matrix X */
	matopc_mtm(pA,pQ,n,n,MATOPC_UPPER); /* A=Q'*Q (only compute upper triangle) */
	/* Next destroy Q by copying the original A into its location */
	matopc_symmetrize(pA,n,MATOPC_UPPER); /* First fill up lower triangle of A */
	matopc_copy(pQ,pA,n,n);
	normA=matopc_frob_norm(pA,n,n);
	/* Now  factorize A in-place */
	cholret=matopc_cholesky_decompose(pA,pd,n);	/* A=L*L', L in lower triangle, diagonal in d */
	if (cholret!=0) {
		printf("[%s]: cholesky failed at line %i.\n",__func__,__LINE__);
		free(buf);
		return 0;
	}

	/* Do q times rank-1 updates and check against a full redactorization at each step */
	for (i=0;i<q;i++) {
		/* Update Cholesky factor by the i-th column outer produce of X */
		matopc_copy(pd0,&pX[n*i],n,1);
		matopc_cholesky_update(pA,pd,n,pd0); /* pd0 is temporary storage for the i-th column of X (it is overwritten) */
		/* Next update the matrix A explicitly (located at Q) A += x[q]*x[q]' */
		matopc_cpata(pQ,n,&pX[i*n],1,1.0,1.0,MATOPC_UPPER); /* use A'*A update for row matrix 1-by-n mock-up */
		cholret=matopc_cholesky_decompose(pQ,pd0,n); /* lower triangle of Q set to complete refactor of updated matrix */
		if (cholret!=0) {
			printf("[%s]: cholesky failed at line %i.\n",__func__,__LINE__);
			break;
		}
		/* Now check whether the factor (A,d) is equal to that at (Q,d0) */
		/* ... print the Frobenius norm of the difference... */
		diff=frob_diff_chol(n,pA,pd,pQ,pd0);
		if (diff > 1e-12) {
			printf("WARNING: rank-1 update %i: factor frob. diff.=%e.\n",i,diff);
		}
		sumdiff+=diff;
	}

	free(buf);
	reldiff=((sumdiff)/((matopc_real)q))/(1.0+sqrt(normA)); /* TODO: not exactly the correct normalization... */
	printf("[%s]: rel.diff.=%.16e\n",__func__,reldiff);
	return (reldiff>=eptol ? 0 : 1);
}

/* Sum up the elementwise squared differences for the lower triangles and the diagonals 
 * and return the square root of the sum
 */
matopc_real frob_diff_chol(
	int n,
	matopc_real *L1,matopc_real *d1,
	matopc_real *L2,matopc_real *d2) {
	int r,c,idx;
	matopc_real sum=0.0,e;
	for (r=0;r<n;r++)
		sum += (d1[r]-d2[r])*(d1[r]-d2[r]);
	for (c=0;c<n;c++) {
		for (r=c+1;r<n;r++) {
			idx=c*n+r;
			e=L1[idx]-L2[idx];
			sum+=2*e*e;
		}
	}
	return sqrt(sum);
}
