/*
 * matrixopsc.h
 *
 * Assortment of utility functions for operations with column-major
 * matrices and matrix-vectors.
 *
 * Note: these subroutines are intended for small-scale block
 * operations. It can be fast for small matrices.
 * It can be very slow for large(r) matrices.
 *
 * Basic small-scale sparsity exploitation routines
 * are included also; CCS and CRS storage.
 *
 * Protocol: all utility functions have the prefix "matopc_"
 * the first argument is (mostly) the destination (if applicable).
 *
 * WARNING: no input checking in any code fragment.
 *
 * K Erik J Olofsson,
 * 
 * Created: January 2016
 * Updated: August-October 2016
 * Updated: December 2016 (very minor)
 */
 
#ifndef __MATRIXOPSC_H__
#define __MATRIXOPSC_H__

/* TODO: QR, LU, SVD factorization routines */

/* Define the below symbol and the matopc_real type
   before including this file, if it should be a float instead. */
#ifndef __MATRIXOPSC_REAL_TYPE__
typedef double matopc_real;
#endif

#define MATOPC_LOWER 1
#define MATOPC_UPPER 2
#define MATOPC_FULL 3
#define MATOPC_LEFT 10
#define MATOPC_RIGHT 11
#define MATOPC_TWOPI (2.0 * 3.14159265358979323846264338327950288)
#define MATOPC_TEMPARRAYSIZE 2000

/*#define MATOPC_MAX_PANEL 16
#define MATOPC_MAX_CHOL_PANEL 8*/

/* General purpose temporary static array */
static matopc_real matopc_temparray[MATOPC_TEMPARRAYSIZE];

/* Set matrix M=zeros(mM,nM) */
static inline void matopc_zeros(matopc_real *M,int m,int n) {
	memset(M,0,m*n*sizeof(matopc_real));
}

/* Count number of nonzeros in dense matrix M */
static inline int matopc_nnz(matopc_real *M,int m,int n) {
	int i,nnz=0;
    for (i=0;i<m*n;i++) if (M[i]!=0.0) nnz++;
    return nnz;
}

/* Set upper or lower triangle of M to zeros (excluding diagonal), only for square matrices */
static inline void matopc_zero_triangle(matopc_real *M,int mM,int which) {
	int r,c;
	if (which==MATOPC_UPPER) {
		for (r=0;r<mM;r++)
			for (c=r+1;c<mM;c++) M[mM*c+r]=0.0;
	} else if (which==MATOPC_LOWER) {
		for (r=0;r<mM;r++)
			for (c=0;c<r;c++) M[mM*c+r]=0.0;
	}
}

/* Copy general matrix A <- B using a single call to memcpy() */
static inline void matopc_copy(matopc_real *A,matopc_real *B,int mB,int nB) {
	memcpy(A,B,mB*nB*sizeof(matopc_real));
}

/* Copy matrix transpose A <- B' (NB: not an in-place transpose, A!=B) */
static inline void matopc_copy_transpose(matopc_real *A,matopc_real *B,int mB,int nB) {
	int r,c; for (r=0;r<mB;r++) for (c=0;c<nB;c++) A[nB*r+c]=B[mB*c+r];
}

/* Copy upper triangle of B into upper triangle of A; incl>0 to include diagonal */
static inline void matopc_copy_up2up(matopc_real *A,matopc_real *B,int m,int incl) {
	int i,j,ofs;
	if (incl>0) {
		for (i=0;i<m;i++) for (j=i,ofs=m*j+i;j<m;j++,ofs+=m) A[ofs]=B[ofs];
	} else {
		for (i=0;i<m;i++) for (j=i+1,ofs=m*j+i;j<m;j++,ofs+=m) A[ofs]=B[ofs];
	}
}

/* Copy upper triangle of B into lower triangle of A (transp.); incl>0 to include diagonal */
static inline void matopc_copy_up2lo(matopc_real *A,matopc_real *B,int m,int incl) {
	int i,j,ofsb,ofsa;
	if (incl>0) {
		for (i=0;i<m;i++) for (j=i,ofsa=m*i+j,ofsb=m*j+i;j<m;j++,ofsb+=m) A[ofsa++]=B[ofsb];
	} else {
		for (i=0;i<m;i++) for (j=i+1,ofsa=m*i+j,ofsb=m*j+i;j<m;j++,ofsb+=m) A[ofsa++]=B[ofsb];
	}
}

/* Multiply "in-place"; x <- a*A*x (using the static buffer as temp.) */
static inline void matopc_inplace_scaled_ax(
        matopc_real *x,matopc_real *A,int mA,int nA,matopc_real a) {
    matopc_real *y=&matopc_temparray[0];
    matopc_real s;
    int r,c,ofs;
    for (r=0;r<mA;r++) {
        s=0.0;
        for (c=0,ofs=r;c<nA;c++,ofs+=mA) s+=A[ofs]*x[c];
        y[r]=a*s;
    }
    matopc_copy(x,y,mA,1);
}

/* y <- A*x, A is mA-by-nA, so y is mA and x is nA, y!=x */
static inline void matopc_ax(
        matopc_real *y,matopc_real *A,int mA,int nA,matopc_real *x) {
    matopc_real s; int r,c,ofs;
    for (r=0;r<mA;r++) {
        s=0.0; for (c=0,ofs=r;c<nA;c++,ofs+=mA) s+=A[ofs]*x[c];
        y[r]=s;
    }
}

/* Does these ops: y <- y+A*x (if sign>=0), or y <- y-A*x (if sign<0) */
static inline void matopc_addax(
		matopc_real *y,matopc_real *A,int mA,int nA,
		matopc_real *x,int sign) {
	matopc_real s; int r,c,ofs;
	if (sign>=0) {
		for (r=0;r<mA;r++) {
        	s=0.0; for (c=0,ofs=r;c<nA;c++,ofs+=mA) s+=A[ofs]*x[c];
        	y[r]+=s;
    	}
	} else {
		for (r=0;r<mA;r++) {
        	s=0.0; for (c=0,ofs=r;c<nA;c++,ofs+=mA) s+=A[ofs]*x[c];
        	y[r]-=s;
    	}
	}
}

/* y <- A'*x, A is mA-by-nA, so y is nA and x is mA, y!=x */
static inline void matopc_atx(
        matopc_real *y,matopc_real *A,int mA,int nA,matopc_real *x) {
    matopc_real s; int r,c,ofs;
    for (c=0;c<nA;c++) {
        s=0.0; ofs=c*mA; for (r=0;r<mA;r++) s+=A[ofs++]*x[r];
        y[c]=s;
    }
}

/* Does these ops: y <- y+A'*x (if sign>=0), or y <- y-A'*x (if sign<0) */
static inline void matopc_addatx(
		matopc_real *y,matopc_real *A,int mA,int nA,
		matopc_real *x,int sign) {
	matopc_real s; int r,c,ofs;
	if (sign>=0) {
		for (c=0;c<nA;c++) {
        	s=0.0; ofs=c*mA; for (r=0;r<mA;r++) s+=A[ofs++]*x[r];
        	y[c]+=s;
    	}
	} else {
		for (c=0;c<nA;c++) {
        	s=0.0; ofs=c*mA; for (r=0;r<mA;r++) s+=A[ofs++]*x[r];
        	y[c]-=s;
    	}
	}
}

/* Evaluation of x'*A*x where A is symmetric A=A'.
 * Lower triangle of A is never accessed
 */
static inline matopc_real matopc_xtax_sym(
        matopc_real *A,int n,matopc_real *x) {
    int i,j,ofs;
    matopc_real s=0.0;
    for (j=0;j<n;j++) {
        for (i=0,ofs=n*j;i<j;i++,ofs++) s+=A[ofs]*x[i]*x[j];
    }
    s*=2.0;
    for (i=0,ofs=0;i<n;i++,ofs+=(n+1)) s+=A[ofs]*x[i]*x[i];
    return s;
}

/* Evaluation of y'*A*x where A is symmetric A=A'.
 * Lower triangle of A is never accessed
 */
static inline matopc_real matopc_ytax_sym(
        matopc_real *A,int n,matopc_real *y,matopc_real *x) {
    int i,j,ofs; matopc_real s=0.0;
    for (j=0;j<n;j++)
        for (i=0,ofs=n*j;i<j;i++,ofs++)
        	s+=A[ofs]*(y[i]*x[j]+y[j]*x[i]);
    for (i=0,ofs=0;i<n;i++,ofs+=(n+1))
    	s+=A[ofs]*y[i]*x[i];
    return s;
}

/* Evaluation of y <- A*x where A is symmetric.
 * Only the upper triangle of A is accessed.
 */
static inline void matopc_sym_ax_upper(
        matopc_real *y,matopc_real *A,int n,matopc_real *x) {
    int i,l,ofs; matopc_real s;
    for (i=0;i<n;i++) {
        s=0.0; ofs=n*i;
        for (l=0;l<i;l++) s+=A[ofs++]*x[l];
        for (l=i;l<n;l++,ofs+=n) s+=A[ofs]*x[l];
        y[i]=s;
    }
}

/* Assign either the lower or upper triangle of
 * a smaller panel B of size b to the block diagonal
 * of the larger matrix A of size m starting at (k,k).
 * NOTE: excludes the actual diagonal.
 */
static inline void matopc_sub_assign_blkdiag_triangle_nodiag(
        matopc_real *A,int m,int k,
        matopc_real *B,int b,int which) {
    int i,j,ofsa,ofsb;
    if (which==MATOPC_LOWER) {
        for (i=0;i<b;i++) {
            for (j=0;j<i;j++) {
                ofsa=(k+j)*m+(k+i);
                ofsb=j*b+i;
                A[ofsa]=B[ofsb];
            }
        }
    } else if (which==MATOPC_UPPER) {
        for (i=0;i<b;i++) {
            for (j=i+1;j<b;j++) {
                ofsa=(k+j)*m+(k+i);
                ofsb=j*b+i;
                A[ofsa]=B[ofsb];
            }
        }
    }
}

/* Insert zeros in a submatrix of A; nA is actually not needed */
static inline void matopc_sub_assign_zeros(
        matopc_real *A,int mA,int nA,int i,int j,
        int mB,int nB) {
  int ofs,c,r;
  for (c=j;c<j+nB;c++) {
    ofs=c*mA+i;
    for (r=0;r<mB;r++) A[ofs++]=0.0;
  }
}

/* Assign B to a sub-matrix of A: A[i..,j..] <- B, B is mB-by-nB.
 * Copies -B into the block of A if sign<0, otherwise +B.
 */
static inline void matopc_sub_assign(
        matopc_real *A,int mA,int nA,int i,int j,
        matopc_real *B,int mB,int nB,int sign) {
    /* if (i<0 || j<0 || (i+mB>mA) || (j+nB>nA)) return; */
    int ofs,c,r;
    int ofsb=0;
    if (sign>=0) {
        for (c=j;c<j+nB;c++) {
            ofs=c*mA+i;
            for (r=0;r<mB;r++) A[ofs++]=B[ofsb++];
        }
    } else {
        for (c=j;c<j+nB;c++) {
            ofs=c*mA+i;
            for (r=0;r<mB;r++) A[ofs++]=-B[ofsb++];
        }
    }
}

/* Assign B to a sub-matrix of A: A[i..,j..] <- B', B is mB-by-nB.
 */
static inline void matopc_sub_assign_transposed(
        matopc_real *A,int mA,int nA,int i,int j,
        matopc_real *B,int mB,int nB) {
    int p,q,ofsa,ofsb;
    for (p=0;p<nB;p++) {
        for (q=0;q<mB;q++) {
            ofsa=(j+q)*mA+(i+p);
            ofsb=p*mB+q;
            A[ofsa]=B[ofsb];
        }
    }
}

/* Same as above but multiplies matrix by a scalar a; assigns or adds */
static inline void matopc_sub_assign_scaled(
        matopc_real *A,int mA,int nA,int i,int j,
        matopc_real *B,int mB,int nB,int add,matopc_real a) {
    int ofs,c,r;
    int ofsb=0;
    if (add>=0) {
        for (c=j;c<j+nB;c++) {
            ofs=c*mA+i;
            for (r=0;r<mB;r++) A[ofs++]+=a*B[ofsb++];
        }
    } else {
        for (c=j;c<j+nB;c++) {
            ofs=c*mA+i;
            for (r=0;r<mB;r++) A[ofs++]=a*B[ofsb++];
        }
    }
}

/* Assign (or add) diagonal matrix D to a block in matrix A; where D=diag(d), d scalar */
static inline void matopc_sub_assign_diag_scalar(
        matopc_real *A,int mA,int nA,int i,int j,
        matopc_real d,int nd,int add) {
    int ofs=j*mA+i,k;
    if (add>=0) { /* add to existing element */
        for (k=0;k<nd;k++) {
            A[ofs]+=d;
            ofs+=(mA+1);
        }
    } else { /* overwrite existing element */
        for (k=0;k<nd;k++) {
            A[ofs]=d;
            ofs+=(mA+1);
        }
    }
}

/* Assign (or add) diagonal matrix D to a block in matrix A; where D=diag(d), d vector */
static inline void matopc_sub_assign_diag_vector(
        matopc_real *A,int mA,int nA,int i,int j,
        matopc_real *d,int nd,int add) {
    int ofs=j*mA+i,k;
    if (add>=0) { /* add to existing element */
        for (k=0;k<nd;k++) {
            A[ofs]+=d[k];
            ofs+=(mA+1);
        }
    } else { /* overwrite existing element */
        for (k=0;k<nd;k++) {
            A[ofs]=d[k];
            ofs+=(mA+1);
        }
    }
}

/* Assign (or add) scaled diagonal matrix D to a block in matrix A;
 * where D=a*diag(d), d vector, a scalar */
static inline void matopc_sub_assign_scaled_diag_vector(
        matopc_real *A,int mA,int nA,int i,int j,
        matopc_real *d,int nd,int add,matopc_real a) {
    int ofs=j*mA+i,k;
    if (add>=0) { /* add to existing element */
        for (k=0;k<nd;k++) {
            A[ofs]+=a*d[k];
            ofs+=(mA+1);
        }
    } else { /* overwrite existing element */
        for (k=0;k<nd;k++) {
            A[ofs]=a*d[k];
            ofs+=(mA+1);
        }
    }
}

/* Assign (or add) a scaled symmetric a*C'*C block to a larger matrix A.
 * C is m-by-n, so C'*C is n-by-n; its upper left corner will be at (k,k)
 * of matrix A; which is p-by-p; only the upper triangle will be assigned.
 */
static inline void matopc_sub_assign_sym_scaled_ctc(
        matopc_real *A,int p,int k,
        matopc_real *C,int m,int n,
        int add,matopc_real a) {
    matopc_real s;
    matopc_real *c1,*c2;
    int i,j,l,ofs;
    if (add>=0) { /* add */
        for (j=0;j<n;j++) {
            c1=&C[j*m]; ofs=p*(k+j)+k;
            for (i=0;i<=j;i++) {
                c2=&C[i*m];
                s=0.0; for (l=0;l<m;l++) s+=c1[l]*c2[l];
                A[ofs++]+=a*s;
            }
        }
    } else { /* overwrite */
        for (j=0;j<n;j++) {
            c1=&C[j*m]; ofs=p*(k+j)+k;
            for (i=0;i<=j;i++) {
                c2=&C[i*m];
                s=0.0; for (l=0;l<m;l++) s+=c1[l]*c2[l];
                A[ofs++]=a*s;
            }
        }
    }
}

/* Assign (or add) a scaled sym. a*C'*diag(w)*C block to a larger matrix A.
 * C is m-by-n, so C'*C is n-by-n; its upper left corner will be at (k,k)
 * of matrix A; which is p-by-p; only the upper triangle will be assigned.
 * w is a vector of length m.
 */
static inline void matopc_sub_assign_sym_scaled_ctwc(
        matopc_real *A,int p,int k,
        matopc_real *C,int m,int n,
        matopc_real *w,int add,matopc_real a) {
    matopc_real s;
    matopc_real *c1,*c2;
    int i,j,l,ofs;
    if (add>=0) { /* add */
        for (j=0;j<n;j++) {
            c1=&C[j*m]; ofs=p*(k+j)+k;
            for (i=0;i<=j;i++) {
                c2=&C[i*m];
                s=0.0; for (l=0;l<m;l++) s+=c1[l]*c2[l]*w[l];
                A[ofs++]+=a*s;
            }
        }
    } else { /* overwrite */
        for (j=0;j<n;j++) {
            c1=&C[j*m]; ofs=p*(k+j)+k;
            for (i=0;i<=j;i++) {
                c2=&C[i*m];
                s=0.0; for (l=0;l<m;l++) s+=c1[l]*c2[l]*w[l];
                A[ofs++]=a*s;
            }
        }
    }
}

/* Assign (or add) a scaled sym. a*C'*W*C block to a larger matrix A.
 * C is m-by-n, so C'*W*C is n-by-n; its upper left corner is put at (k,k)
 * of matrix A; which has p rows; only its upper triangle is assigned.
 * W is a symmetric m-by-m matrix; only its upper triangle is accessed.
 */
static inline void matopc_sub_assign_sym_scaled_ctwc_gen(
        matopc_real *A,int p,int k,
        matopc_real *C,int m,int n,
        matopc_real *W,int add,matopc_real a) {
    matopc_real s;
    matopc_real *ci,*cj;
    int i,j,ofs;
    if (add>=0) {
        for (i=0;i<n;i++) {
            ofs=p*(i+k)+(i+k); ci=&C[i*m];
            for (j=i;j<n;j++) {
                /* Comp. element (i,j) of C'*W*C, j>=i */
                cj=&C[j*m]; s=matopc_ytax_sym(W,m,ci,cj);
                A[ofs]+=a*s; ofs+=p; /* here ofs=p*(j+k)+(i+k) */
            }
        }
    } else { /* assign/overwrite */
        for (i=0;i<n;i++) {
            ofs=p*(i+k)+(i+k); ci=&C[i*m];
            for (j=i;j<n;j++) {
                cj=&C[j*m]; s=matopc_ytax_sym(W,m,ci,cj);
                A[ofs]=a*s; ofs+=p;
            }
        }
    }
}

/* Assign (or add) a scaled block a*C'*D to a larger matrix A.
 * C is m-by-n, D is m-by-u, so C'*D is n-by-u. The scaled block
 * is put with its upper right corner at (p,q) in A (which has mA rows).
 */
static inline void matopc_sub_assign_scaled_ctd(
        matopc_real *A,int mA,int p,int q,
        matopc_real *C,int m,int n,
        matopc_real *D,int u,
        int add,matopc_real a) {
    matopc_real s;
    matopc_real *c,*d;
    int i,j,l,ofs;
    if (add>=0) { /* add */
        for (i=0;i<n;i++) {
            c=&C[i*m]; ofs=q*mA+(p+i);
            for (j=0;j<u;j++) {
                d=&D[j*m];
                s=0.0; for (l=0;l<m;l++) s+=c[l]*d[l];
                A[ofs]+=a*s; ofs+=mA;
            }
        }
    } else { /* overwrite */
        for (i=0;i<n;i++) {
            c=&C[i*m]; ofs=q*mA+(p+i);
            for (j=0;j<u;j++) {
                d=&D[j*m];
                s=0.0; for (l=0;l<m;l++) s+=c[l]*d[l];
                A[ofs]=a*s; ofs+=mA;
            }
        }
    }
}

/* Assign (or add) a scaled block a*C'*diag(w)*D to a larger matrix A.
 * C is m-by-n, D is m-by-u, so C'*D is n-by-u. The scaled block
 * is put with its upper right corner at (p,q) in A (which has mA rows).
 * w is a vector of length m.
 */
static inline void matopc_sub_assign_scaled_ctwd(
        matopc_real *A,int mA,int p,int q,
        matopc_real *C,int m,int n,
        matopc_real *D,int u,
        matopc_real *w,int add,matopc_real a) {
    matopc_real s;
    matopc_real *c,*d;
    int i,j,l,ofs;
    if (add>=0) { /* add */
        for (i=0;i<n;i++) {
            c=&C[i*m]; ofs=q*mA+(p+i);
            for (j=0;j<u;j++) {
                d=&D[j*m];
                s=0.0; for (l=0;l<m;l++) s+=c[l]*d[l]*w[l];
                A[ofs]+=a*s; ofs+=mA;
            }
        }
    } else { /* overwrite */
        for (i=0;i<n;i++) {
            c=&C[i*m]; ofs=q*mA+(p+i);
            for (j=0;j<u;j++) {
                d=&D[j*m];
                s=0.0; for (l=0;l<m;l++) s+=c[l]*d[l]*w[l];
                A[ofs]=a*s; ofs+=mA;
            }
        }
    }
}

/* Assign (or add) a scaled block a*C'*W*D to a larger matrix A.
 * C is m-by-n, D is m-by-u, so C'*D is n-by-u. The scaled block
 * is put with its upper right corner at (p,q) in A (which has mA rows).
 * W is a symmetric m-by-m matrix.
 */
static inline void matopc_sub_assign_scaled_ctwd_gen(
        matopc_real *A,int mA,int p,int q,
        matopc_real *C,int m,int n,
        matopc_real *D,int u,
        matopc_real *W,int add,matopc_real a) {
    matopc_real s;
    matopc_real *c,*d;
    int i,j,ofs;
    if (add>=0) { /* add */
        for (i=0;i<n;i++) {
            c=&C[i*m]; ofs=q*mA+(p+i);
            for (j=0;j<u;j++) {
                d=&D[j*m]; s=matopc_ytax_sym(W,m,c,d);
                A[ofs]+=a*s; ofs+=mA;
            }
        }
    } else { /* overwrite */
        for (i=0;i<n;i++) {
            c=&C[i*m]; ofs=q*mA+(p+i);
            for (j=0;j<u;j++) {
                d=&D[j*m]; s=matopc_ytax_sym(W,m,c,d);
                A[ofs]=a*s; ofs+=mA;
            }
        }
    }
}

/* Extract A as a mA-by-nA sub-matrix of B: upper-left corner at (i,j) */
static inline void matopc_sub_extract(
        matopc_real *A,int mA,int nA,int i,int j,
        matopc_real *B,int mB,int nB) {
    /* if (i<0 || j<0 || (i+mA>mB) || (j+nA>nB)) return; */
    int ofs,c,r;
    int ofsa=0;
    for (c=j;c<j+nA;c++) {
        ofs=c*mB+i;
        for (r=0;r<mA;r++) A[ofsa++]=B[ofs++];
    }
}

/* Transpose matrix A, m-by-n, in-place, to A', n-by-m */
static inline void matopc_inplace_transpose(matopc_real *A,int m,int n) {
	int start,next,i;
	matopc_real tmp;
	for (start=0;start<=n*m-1;start++) {
		next=start; i=0;
		do { i++; next=(next%n)*m+next/n; } while (next>start);
		if (next<start || i==1) continue;
		tmp=A[next=start];
		do {
			i=(next%n)*m+next/n;
			A[next]=(i==start) ? tmp : A[i];
			next=i;
		} while (next>start);
	}
}

/* Set matrix M=ones(mM,nM) */
static inline void matopc_ones(matopc_real *M,int mM,int nM) {
	int i; for (i=0;i<mM*nM;i++) M[i]=1.0;
}

/* Set matrix M=eye(mM) */
static inline void matopc_eye(matopc_real *M,int mM) {
	matopc_zeros(M,mM,mM);
	int i,j; for (i=0,j=0;j<mM;j++,i+=(mM+1)) M[i]=1.0;
}

/* Set matrix M=diag(x), x a vector of length mx */
static inline void matopc_diag(matopc_real *M,matopc_real *x,int mx) {
	matopc_zeros(M,mx,mx);
	int i,j; for (i=0,j=0;j<mx;j++,i+=(mx+1)) M[i]=x[j];
}

/* Assign diagonal of matrix M, diag(M)=x, x a vector of length mx */
static inline void matopc_setdiag(matopc_real *M,matopc_real *x,int mx) {
	int i,j; for (i=0,j=0;j<mx;j++,i+=(mx+1)) M[i]=x[j];
}

/* Assign diagonal of matrix M to vector: x <- diag(M), length mx */
static inline void matopc_getdiag(matopc_real *x,matopc_real *M,int mx) {
    int i,j; for (i=0,j=0;j<mx;j++,i+=(mx+1)) x[j]=M[i];
}

/* Compute matrix trace (sum of diagonal elements) */
static inline matopc_real matopc_trace(matopc_real *M,int mM) {
    int i,j; matopc_real s=0.0;
    for (i=0,j=0;j<mM;j++,i+=(mM+1)) s+=M[i];
    return s;
}

#ifdef __MT19937AR_H__
/* Fill matrix M with elementwise random numbers uniform on (a,b) */
static inline void matopc_rand(matopc_real *M,int m,int n,matopc_real a,matopc_real b) {
	int i,imax; imax=m*n; for (i=0;i<imax;i++) M[i]=(matopc_real)(a+(b-a)*genrand_real3());
}
/* Normal standard variates using the Box-Muller transform; generates pairs,
   should correspond to "randn(m,n)" in MATLAB
 */
static inline void matopc_randn(matopc_real *M,int m,int n) {
	int i,imax;
	double u1,u2,R,TH;
	i=0; imax=m*n;
	while(i<imax) {
		u1=genrand_real3(); u2=genrand_real3();
		R=sqrt(-2.0*log(u1)); TH=MATOPC_TWOPI*u2;
		M[i++]=(matopc_real)(R*cos(TH));
		if (i<imax) {
			M[i++]=(matopc_real)(R*sin(TH));
		}
	}
}
#endif

/* Frobenius norm of matrix A */
static inline matopc_real matopc_frob_norm(matopc_real *A,int mA,int nA) {
	int i,imax;
	matopc_real s;
	imax=nA*mA; s=0.0;
	for (i=0;i<imax;i++) s+=A[i]*A[i];
	return sqrt(s);
}

/* Frobenius norm of elementwise matrix difference A-B */
static inline matopc_real matopc_frob_norm_diff(matopc_real *A,matopc_real *B,int m,int n) {
	int i,imax;
	matopc_real s,d;
	imax=m*n; s=0.0;
	for (i=0;i<imax;i++) {
		d=A[i]-B[i]; s+=d*d;
	}
	return sqrt(s);
}

/* Frobenius norm of upper or lower triangle of a square matrix (inc. or excl. diagonal) */
static inline matopc_real matopc_frob_norm_triangle(
        matopc_real *A,int m,
        int which,int incl) {
    int i,j; matopc_real s=0.0,e;
    if (which==MATOPC_UPPER) {
        for (i=0;i<m;i++) {
            for (j=i+1;j<m;j++) {
                e=A[j*m+i]; s+=e*e;
            }
        }
    } else if (which==MATOPC_LOWER) {
        for (i=0;i<m;i++) {
            for (j=0;j<i;j++) {
                e=A[j*m+i]; s+=e*e;
            }
        }
    }
    if (incl>0) {
        for (i=0;i<m;i++) {
            e=A[i*m+i]; s+=e*e;
        }
    }
    return sqrt(s);
}

/* Maximum absolute value of elementwise matrix difference A-B */
static inline matopc_real matopc_max_abs_diff(matopc_real *A,matopc_real *B,int m,int n) {
	int i,imax;
	matopc_real s,d;
	imax=m*n; s=0.0;
	for (i=0;i<imax;i++) {
		d=fabs(A[i]-B[i]);
		if (d>s) s=d;
	}
	return s;
}

/* Maximum absolute value of elementwise matrix difference A-B; only looking at lower or
 * upper triangle, including or excluding diagonal (square matrices only).
 */
static inline matopc_real matopc_max_abs_diff_triangle(
        matopc_real *A,matopc_real *B,int m,
        int which,int incl) {
	int i,j,idx;
	matopc_real s=0.0,tmp;
	for (i=0;i<m;i++) {
        if (which==MATOPC_UPPER) {
            for (j=i+1;j<m;j++) {
                idx=m*j+i; tmp=fabs(A[idx]-B[idx]); if (tmp>s) s=tmp;
            }
        } else if (which==MATOPC_LOWER) {
            for (j=0;j<i;j++) {
                idx=m*j+i; tmp=fabs(A[idx]-B[idx]); if (tmp>s) s=tmp;
            }
        }
	}
    if (incl>0) {
        for (i=0;i<m;i++) {
            idx=i*m+i; tmp=fabs(A[idx]-B[idx]); if (tmp>s) s=tmp;
        }
    }
	return s;
}

/* Calculate M <- M + a*diag(d) */
static inline void matopc_mpdiag(matopc_real *M,int mM,matopc_real *d,matopc_real a) {
	int i,j;
	if (a==0.0) {
		/* M <- M do nothing*/
	} else if (a==1.0) { /* M <- M+diag(d) */
		for (i=0,j=0;j<mM;j++,i+=(mM+1)) M[i]+=d[j];
	} else {
		for (i=0,j=0;j<mM;j++,i+=(mM+1)) M[i]+=a*d[j];
	}
}

/* Calculate M <- M + a*eye(mM) */
static inline void matopc_mpeye(matopc_real *M,int mM,matopc_real a) {
	int i,imax;
	imax=mM*mM;
	if (a==0.0) {
		/* M <- M do nothing*/
	} else if (a==1.0) { /* M <- M+eye(mM) */
		for (i=0;i<imax;i+=(mM+1)) M[i]+=1.0;
	} else {
		for (i=0;i<imax;i+=(mM+1)) M[i]+=a;
	}
}

/* Multiply M (m-by-n) from the left by a diagonal matrix represented by a vector of length m,
   if which==MATOPC_LEFT or multiply from right by a diagonal matrix
   represented by a vector of length n, if which==MATOPC_RIGHT.
 */
static inline void matopc_diag_mult(matopc_real *M,int m,int n,int which,matopc_real *d) {
	int r,c,i;
	matopc_real s;
	if (which==MATOPC_LEFT) { /* Multiply M <- diag(d)*M where d has length m (scaling of rows) */
		for (r=0;r<m;r++) {
			i=r; s=d[r];
			for (c=0;c<n;c++) {
				M[i]*=s; i+=m;
			}
		}
	} else if (which==MATOPC_RIGHT) {	/* Multiply M <- M*diag(d) where d has length n (scaling of columns) */
		for (c=0;c<n;c++) {
			i=c*m; s=d[c];
			for (r=0;r<m;r++) M[i++]*=s;
		}
	}
}

/* ... */

/* Copy the lower (upper) triangle into the upper (lower) triangle, M is mM-by-mM,
   overwriting whatever is there: which is the source triangle and can be either of
   MATOPC_LOWER or MATOPC_UPPER.
 */
static inline void matopc_symmetrize(matopc_real *M,int mM,int which) {
	int r,c;
	if (which==MATOPC_LOWER) { /* copy lower into upper */
		for (r=0;r<mM;r++) for (c=r+1;c<mM;c++) M[mM*c+r]=M[mM*r+c];
	} else if (which==MATOPC_UPPER) { /* copy upper into lower */
		for (r=0;r<mM;r++) for (c=r+1;c<mM;c++) M[mM*r+c]=M[mM*c+r];
	}
}

/* x <- means of columns of M (M is m-by-n) */
static inline void matopc_stats_colmean(matopc_real *x,matopc_real *M,int m,int n) {
    int i,j; matopc_real s; matopc_real *mj;
    for (j=0;j<n;j++) {
        s=0.0; mj=&M[m*j]; for (i=0;i<m;i++) s+=mj[i];
        x[j]=s/m;
    }
}

/* x <- standard deviations of columns of M (M is m-by-n) */
static inline void matopc_stats_colsdev(matopc_real *x,matopc_real *M,int m,int n) {
    int i,j; matopc_real s; matopc_real *mj;
    matopc_stats_colmean(x,M,m,n);
    for (j=0;j<n;j++) {
        s=0.0; mj=&M[m*j]; for (i=0;i<m;i++) s+=(mj[i]-x[j])*(mj[i]-x[j]);
        x[j]=sqrt(s/m);
    }
}

/* Compute the update y <- a*y+b*A*x where A is mA-by-nA and a, b are scalars.
   It is assumed that x is a column of length nA.
   y will be a column of length mA.
   If a,b=0,1 will be treated with special code for speed.
 */
static inline void matopc_ypax(
	matopc_real *y,matopc_real a,matopc_real b,
	matopc_real *A,int mA,int nA,matopc_real *x) {
	matopc_real s;
	int r,c,k;
	if (a==0.0 && b==0.0) {
		matopc_zeros(y,mA,1);
	} else if (a==0.0 && b==1.0) {	/* common case y <- A*x */
		for (r=0;r<mA;r++) {
			s=0.0; k=r;
			for (c=0;c<nA;c++,k+=mA) s+=A[k]*x[c];
			y[r]=s;
		}
	} else if (a==1.0 && b==0.0) {
		/* copy y <- y (do nothing here) */
	} else if (a==1.0 && b==1.0) {	/* accumulation y <- y+A*x */
		for (r=0;r<mA;r++) {
			s=0.0; k=r;
			for (c=0;c<nA;c++,k+=mA) s+=A[k]*x[c];
			y[r]+=s;
		}
	} else {	/* general case y <- a*y+b*A*x */
		for (r=0;r<mA;r++) {
			s=0.0; k=r;
			for (c=0;c<nA;c++,k+=mA) s+=A[k]*x[c];
			y[r]=a*y[r]+b*s;
		}
	}
}

/* Similar to the above GEMV routine but implicit transposition of A:
   y <- a*y+b*A'*x for a,b scalars, A is mA-by-nA, so A' is nA-by-mA
   and thus x will be a column of length mA and y a column of length nA.
 */
static inline void matopc_ypatx(
	matopc_real *y,matopc_real a,matopc_real b,
	matopc_real *A,int mA,int nA,matopc_real *x) {
	matopc_real s;
	int r,c,k;
	if (a==0.0 && b==0.0) {
		matopc_zeros(y,nA,1);
    } else if (a==0.0 && b==1.0) { /* common case y <- A'*x */
        for (r=0;r<nA;r++) {
			s=0.0; k=r*mA;
			for (c=0;c<mA;c++) s+=A[k++]*x[c];
			y[r]=s;
		}
    } else if (a==1.0 && b==0.0) {
        /* y <- y (do nothing) */
    } else if (a==1.0 && b==1.0) { /* basic accumulation y <- y+A'*x */
        for (r=0;r<nA;r++) {
			s=0.0; k=r*mA;
			for (c=0;c<mA;c++) s+=A[k++]*x[c];
			y[r]+=s;
		}
    } else {	/* generic case */
		for (r=0;r<nA;r++) {
			s=0.0; k=r*mA;
			for (c=0;c<mA;c++) s+=A[k++]*x[c];
			y[r]=a*y[r]+b*s;
		}
	}
}

/* Compute symmetric S=M'*M where M is mM-by-nM. S will be nM-by-nM
   which can be MATOPC_LOWER, or MATOPC_UPPER.
 */
static inline void matopc_mtm(
	matopc_real *S,matopc_real *M,int mM,int nM,int which) {
	int r,c,i,j,k;
	matopc_real s;
	if (which==MATOPC_UPPER) {
		for (r=0;r<nM;r++) {
			for (c=r;c<nM;c++) {
				s=0.0; i=mM*r; j=mM*c;
				for (k=0;k<mM;k++) {
					s+=M[i++]*M[j++];
				}
				S[nM*c+r]=s;
			}
		}
	} else if (which==MATOPC_LOWER) {
		for (r=0;r<nM;r++) {
			for (c=0;c<=r;c++) {
				s=0.0; i=mM*r; j=mM*c;
				for (k=0;k<mM;k++) {
					s+=M[i++]*M[j++];
				}
				S[nM*c+r]=s;
			}
		}
	}
}

/* Compute symmetric update S <- S+M'*M where M is mM-by-nM. S will be nM-by-nM
   which can be MATOPC_LOWER, or MATOPC_UPPER (and is assumed to be symmetric)
 */
static inline void matopc_spmtm(
    matopc_real *S,matopc_real *M,int mM,int nM,int which) {
    int r,c,i,j,k;
    matopc_real s;
    if (which==MATOPC_UPPER) {
        for (r=0;r<nM;r++) {
            for (c=r;c<nM;c++) {
                s=0.0; i=mM*r; j=mM*c;
                for (k=0;k<mM;k++) {
                    s+=M[i++]*M[j++];
                }
                S[nM*c+r]+=s;
            }
        }
    } else if (which==MATOPC_LOWER) {
        for (r=0;r<nM;r++) {
            for (c=0;c<=r;c++) {
                s=0.0; i=mM*r; j=mM*c;
                for (k=0;k<mM;k++) {
                    s+=M[i++]*M[j++];
                }
                S[nM*c+r]+=s;
            }
        }
    }
}

/* Compute symmetric "downdate" S <- S-M'*M where M is mM-by-nM. S will be nM-by-nM
   which can be MATOPC_LOWER, or MATOPC_UPPER (and is assumed to be symmetric)
 */
static inline void matopc_smmtm(
    matopc_real *S,matopc_real *M,int mM,int nM,int which) {
    int r,c,i,j,k;
    matopc_real s;
    if (which==MATOPC_UPPER) {
        for (r=0;r<nM;r++) {
            for (c=r;c<nM;c++) {
                s=0.0; i=mM*r; j=mM*c;
                for (k=0;k<mM;k++) {
                    s+=M[i++]*M[j++];
                }
                S[nM*c+r]-=s;
            }
        }
    } else if (which==MATOPC_LOWER) {
        for (r=0;r<nM;r++) {
            for (c=0;c<=r;c++) {
                s=0.0; i=mM*r; j=mM*c;
                for (k=0;k<mM;k++) {
                    s+=M[i++]*M[j++];
                }
                S[nM*c+r]-=s;
            }
        }
    }
}

/* Compute symmetric K=M*M' where M is mM-by-nM. K will be mM-by-mM
   which can be MATOPC_LOWER, or MATOPC_UPPER.
 */
static inline void matopc_mmt(
	matopc_real *K,matopc_real *M,int mM,int nM,int which) {
	int r,c,i,j,k;
	matopc_real s;
	if (which==MATOPC_UPPER) {
		for (r=0;r<mM;r++) {
			for (c=r;c<mM;c++) {
				s=0.0; i=r; j=c;
				for (k=0;k<nM;k++) {
					s+=M[i]*M[j]; i+=mM; j+=mM;
				}
				K[mM*c+r]=s;
			}
		}
	} else if (which==MATOPC_LOWER) {
		for (r=0;r<mM;r++) {
			for (c=0;c<=r;c++) {
				s=0.0; i=r; j=c;
				for (k=0;k<nM;k++) {
					s+=M[i]*M[j]; i+=mM; j+=mM;
				}
				K[mM*c+r]=s;
			}
		}
	}
}

/* S <- S-M*M', S is mM-by-mM, M is mM-by-nM */
static inline void matopc_smmmt(
	matopc_real *S,matopc_real *M,int mM,int nM,int which) {
	int r,c,i,j,k;
	matopc_real s;
	if (which==MATOPC_UPPER) {
		for (r=0;r<mM;r++) {
			for (c=r;c<mM;c++) {
				s=0.0; i=r; j=c;
				for (k=0;k<nM;k++) {
					s+=M[i]*M[j]; i+=mM; j+=mM;
				}
				S[mM*c+r]-=s;
			}
		}
	} else if (which==MATOPC_LOWER) {
		for (r=0;r<mM;r++) {
			for (c=0;c<=r;c++) {
				s=0.0; i=r; j=c;
				for (k=0;k<nM;k++) {
					s+=M[i]*M[j]; i+=mM; j+=mM;
				}
				S[mM*c+r]-=s;
			}
		}
	}
}

/* General matrix multiply routine: C <- a*C+b*A*B.
   Based on a naive non-blocked triple-loop. C is m-by-n, A is m-by-k, B is k-by-n.
   Special code is applied if a,b=0,1
 */
static inline void matopc_gemm(
        matopc_real *C,int m,int n,
        matopc_real *A,int k,matopc_real *B,
        matopc_real a,matopc_real b) {
	int r,c,i,i1,j1,j2;
	matopc_real s;
	if (a==0.0 && b==0.0) {
		matopc_zeros(C,m,n);
	} else if (a==0.0 && b==1.0) {
		for (r=0;r<m;r++) {
			i1=r;
			for (c=0;c<n;c++) {
				s=0.0; j1=r; j2=c*k;
				for (i=0;i<k;i++) {
					s+=A[j1]*B[j2]; j1+=m; j2++;
				}
				C[i1]=s; i1+=m;
			}
		}
	} else {	/* Generic scalars a and b */
		for (r=0;r<m;r++) {
			i1=r;
			for (c=0;c<n;c++) {
				s=0.0; j1=r; j2=c*k;
				for (i=0;i<k;i++) {
					s+=A[j1]*B[j2];
					j1+=m; j2++;
				}
				C[i1]=a*C[i1]+b*s;
				i1+=m;
			}
		}
	}
}

/* Basic symmetric rank-1 matrix update A <- A+x*x', A is n-by-n, x is n-by-1 (column).
   Parameter which is either MATOPC_UPPER or MATOPC_LOWER.
 */
static inline void matopc_apxxt(matopc_real *A,int n,matopc_real *x,int which) {
	/* ... TODO ... */
	/*if (upda>0) {
        for (k=0;k<n;k++) {
        	i=n*k;
            for (l=0;l<=k;l++) {
                A[i+l]+=x[l]*x[k];
            }
        }
    }*/   
}

/* Generic rectangular X <- A'*B, A is m-by-n and B is m-by-k, so X is n-by-k */
static inline void matopc_atb(matopc_real *X,matopc_real *A,int m,int n,matopc_real *B,int k) {
	int p,q,j,idx;
	matopc_real s;
	matopc_real *a,*b;
	for (p=0;p<n;p++) {
		a=&A[p*m]; idx=p;
		for (q=0;q<k;q++) {
			b=&B[q*m]; s=0.0;
			for (j=0;j<m;j++) s+=a[j]*b[j];
			X[idx]=s; idx+=n; /* idx=n*q+p */
		}
	}
}

/* Generic rectangular update X <- X+A'*B, A is m-by-n and B is m-by-k, so X is n-by-k */
static inline void matopc_xpatb(matopc_real *X,matopc_real *A,int m,int n,matopc_real *B,int k) {
	int p,q,j,idx;
	matopc_real s;
	matopc_real *a,*b;
	for (p=0;p<n;p++) {
		a=&A[p*m]; idx=p;
		for (q=0;q<k;q++) {
			b=&B[q*m]; s=0.0;
			for (j=0;j<m;j++) s+=a[j]*b[j];
			X[idx]+=s; idx+=n; /* idx=n*q+p */
		}
	}
}

/* General multiply/update for symmetric matrix C : C <- a*C+b*A'*A, a,b are scalars.
   Here it must be decided whether the upper or lower triangle of C is to be updated.
   C will be mC-by-mC, A will be mA-by-mC; which can be MATOPC_LOWER, MATOPC_UPPER,
   or MATOPC_FULL (when C is not symmetric; but the update is symmetric).
 */
static inline void matopc_cpata(
	matopc_real *C,int mC,matopc_real *A,int mA,matopc_real a,matopc_real b,int which) {
    int r,c,i,j,k,idx;
	matopc_real s, bs;
    /*if (a==0.0 && b==1.0 && which!=MATOPC_FULL) {
        matopc_mtm(C,A,mA,mC,which);
    }*/
	if (which==MATOPC_UPPER) {
		for (r=0;r<mC;r++) {
			for (c=r;c<mC;c++) {
				s=0.0; i=mA*r; j=mA*c;
				for (k=0;k<mA;k++) {
					s+=A[i++]*A[j++];
				}
                idx=mC*c+r;
                C[idx]=a*C[idx]+b*s;
			}
		}
	} else if (which==MATOPC_LOWER) {
		for (r=0;r<mC;r++) {
			for (c=0;c<=r;c++) {
				s=0.0; i=mA*r; j=mA*c;
				for (k=0;k<mA;k++) {
					s+=A[i++]*A[j++];
				}
				idx=mC*c+r;
                C[idx]=a*C[idx]+b*s;
			}
		}
	} else if (which==MATOPC_FULL) {
        for (r=0;r<mC;r++) {
			for (c=0;c<=r;c++) {
				s=0.0; i=mA*r; j=mA*c;
				for (k=0;k<mA;k++) {
					s+=A[i++]*A[j++];
				}
                bs=b*s;
				idx=mC*c+r;
                C[idx]=a*C[idx]+bs;
                if (c!=r) {
                    idx=mC*r+c;
                    C[idx]=a*C[idx]+bs;
                }
			}
		}
    }
}

/* Symmetric "rank-k" (k=mA) update with diagonal weight d (length=mA): C <- a*C+b*A'*diag(d)*A) */
static inline void matopc_cpatda(
        matopc_real *C,int mC,matopc_real *A,int mA,matopc_real *d,
        matopc_real a,matopc_real b,int which) {
    int r,c,i,j,k,idx;
    matopc_real s;
    if (a==0.0 && b==0.0) { /* C <- zeros(mC,mC) */
        matopc_zeros(C,mC,mC);
    } else if (a==0.0 && b==1.0) {  /* C <- A'*diag(d)*A */
        if (which==MATOPC_UPPER) {
            for (r=0;r<mC;r++) {
                for (c=r;c<mC;c++) {
                    s=0.0; i=mA*r; j=mA*c;
                    for (k=0;k<mA;k++) {
    					s+=A[i++]*A[j++]*d[k];
        			}
                    C[mC*c+r]=s;
                }
            }
        } else if (which==MATOPC_LOWER) {
            for (r=0;r<mC;r++) {
    			for (c=0;c<=r;c++) {
        			s=0.0; i=mA*r; j=mA*c;
            		for (k=0;k<mA;k++) {
                		s+=A[i++]*A[j++]*d[k];
                    }
                    C[mC*c+r]=s;
                }
            }
        }
    } else if (a==1.0 && b==1.0) {  /* C <- C + A'*diag(d)*A */
        if (which==MATOPC_UPPER) {
            for (r=0;r<mC;r++) {
                for (c=r;c<mC;c++) {
                    s=0.0; i=mA*r; j=mA*c;
                    for (k=0;k<mA;k++) {
    					s+=A[i++]*A[j++]*d[k];
        			}
                    C[mC*c+r]+=s;
                }
            }
        } else if (which==MATOPC_LOWER) {
            for (r=0;r<mC;r++) {
    			for (c=0;c<=r;c++) {
        			s=0.0; i=mA*r; j=mA*c;
            		for (k=0;k<mA;k++) {
                		s+=A[i++]*A[j++]*d[k];
                    }
                    C[mC*c+r]+=s;
                }
            }
        }
    } else {    /* generic case, a,b general; nonzero, nonunity */
        if (which==MATOPC_UPPER) {
            for (r=0;r<mC;r++) {
                for (c=r;c<mC;c++) {
                    s=0.0; i=mA*r; j=mA*c;
                    for (k=0;k<mA;k++) {
    					s+=A[i++]*A[j++]*d[k];
        			}
                    idx=mC*c+r;
                    C[idx]=a*C[idx]+b*s;
                }
            }
        } else if (which==MATOPC_LOWER) {
            for (r=0;r<mC;r++) {
    			for (c=0;c<=r;c++) {
        			s=0.0; i=mA*r; j=mA*c;
            		for (k=0;k<mA;k++) {
                		s+=A[i++]*A[j++]*d[k];
                    }
                    idx=mC*c+r;
                    C[idx]=a*C[idx]+b*s;
                }
            }
        }
    }
}

/* B'*B updating; simple 4-by-4 unrolling: A <- A+B'*B where B is 4-by-n (A is n-by-n)
 * Uses column strides: cstrA for A and cstrB for B.
 * Only updates the lower triangle of A; including the diagonal.
 */
static inline void matopc_apbtb_lo4(matopc_real *A,int cstrA,matopc_real *B,int cstrB,int n) {
	matopc_real *b=&matopc_temparray[0];	/* need exactly 16 doubles */
	matopc_real *r=&matopc_temparray[16]; /* need (up to) 10 doubles */
	int j,k=0,ntg,ofsA,ofsB;
	while (k<n) {
		ofsA=cstrA*k+k;
		ofsB=cstrB*k;
		ntg=n-k;
		if (ntg>=4) { /* 4x4 matrix B1; diagonal block (plus many 1x4 rows possibly) */
			b[0]=B[ofsB]; b[1]=B[ofsB+1]; b[2]=B[ofsB+2]; b[3]=B[ofsB+3]; /* column 1 of B1 */
			ofsB+=cstrB;
			b[4]=B[ofsB]; b[5]=B[ofsB+1]; b[6]=B[ofsB+2]; b[7]=B[ofsB+3]; /* column 2 of B1 */
			ofsB+=cstrB;
			b[8]=B[ofsB]; b[9]=B[ofsB+1]; b[10]=B[ofsB+2]; b[11]=B[ofsB+3]; /* column 3 of B1 */
			ofsB+=cstrB;
			b[12]=B[ofsB]; b[13]=B[ofsB+1]; b[14]=B[ofsB+2]; b[15]=B[ofsB+3]; /* column 4 of B1 */
			r[0]=b[0]*b[0]+b[1]*b[1]+b[2]*b[2]+b[3]*b[3];
			r[1]=b[4]*b[0]+b[5]*b[1]+b[6]*b[2]+b[7]*b[3];
			r[2]=b[8]*b[0]+b[9]*b[1]+b[10]*b[2]+b[11]*b[3];
			r[3]=b[12]*b[0]+b[13]*b[1]+b[14]*b[2]+b[15]*b[3];
			r[4]=b[4]*b[4]+b[5]*b[5]+b[6]*b[6]+b[7]*b[7];
			r[5]=b[8]*b[4]+b[9]*b[5]+b[10]*b[6]+b[11]*b[7];
			r[6]=b[12]*b[4]+b[13]*b[5]+b[14]*b[6]+b[15]*b[7];
			r[7]=b[8]*b[8]+b[9]*b[9]+b[10]*b[10]+b[11]*b[11];
			r[8]=b[12]*b[8]+b[13]*b[9]+b[14]*b[10]+b[15]*b[11];
			r[9]=b[12]*b[12]+b[13]*b[13]+b[14]*b[14]+b[15]*b[15];
			A[ofsA]+=r[0];
			A[ofsA+1]+=r[1];
			A[ofsA+2]+=r[2];
			A[ofsA+3]+=r[3];
			ofsA+=cstrA;
			A[ofsA+1]+=r[4];
			A[ofsA+2]+=r[5];
			A[ofsA+3]+=r[6];
			ofsA+=cstrA;
			A[ofsA+2]+=r[7];
			A[ofsA+3]+=r[8];
			ofsA+=cstrA;
			A[ofsA+3]+=r[9];
			if (ntg>4) { /* not yet done? */
				ofsA=cstrA*k+k+4;
				ofsB+=cstrB;
				for (j=k+4;j<n;j++) {
					/* read next column of B, multiply by B1' from the left, then store vector as row in A, repeat */
					r[0]=B[ofsB]; r[1]=B[ofsB+1]; r[2]=B[ofsB+2]; r[3]=B[ofsB+3];
					r[4]=b[0]*r[0]+b[1]*r[1]+b[2]*r[2]+b[3]*r[3];
					r[5]=b[4]*r[0]+b[5]*r[1]+b[6]*r[2]+b[7]*r[3];
					r[6]=b[8]*r[0]+b[9]*r[1]+b[10]*r[2]+b[11]*r[3];
					r[7]=b[12]*r[0]+b[13]*r[1]+b[14]*r[2]+b[15]*r[3];
					A[ofsA]+=r[4];
					A[ofsA+cstrA]+=r[5];
					A[ofsA+2*cstrA]+=r[6];
					A[ofsA+3*cstrA]+=r[7];
					ofsB+=cstrB;
					ofsA++;
				}
			}
			k+=4;
		} else if (ntg==3) { /* 4x3 matrix B1 */
			b[0]=B[ofsB]; b[1]=B[ofsB+1]; b[2]=B[ofsB+2]; b[3]=B[ofsB+3]; /* column 1 of B1 */
			ofsB+=cstrB;
			b[4]=B[ofsB]; b[5]=B[ofsB+1]; b[6]=B[ofsB+2]; b[7]=B[ofsB+3]; /* column 2 of B1 */
			ofsB+=cstrB;
			b[8]=B[ofsB]; b[9]=B[ofsB+1]; b[10]=B[ofsB+2]; b[11]=B[ofsB+3]; /* column 3 of B1 */
			r[0]=b[0]*b[0]+b[1]*b[1]+b[2]*b[2]+b[3]*b[3];
			r[1]=b[4]*b[0]+b[5]*b[1]+b[6]*b[2]+b[7]*b[3];
			r[2]=b[8]*b[0]+b[9]*b[1]+b[10]*b[2]+b[11]*b[3];
			r[3]=b[4]*b[4]+b[5]*b[5]+b[6]*b[6]+b[7]*b[7];
			r[4]=b[8]*b[4]+b[9]*b[5]+b[10]*b[6]+b[11]*b[7];
			r[5]=b[8]*b[8]+b[9]*b[9]+b[10]*b[10]+b[11]*b[11];
			A[ofsA]+=r[0];
			A[ofsA+1]+=r[1];
			A[ofsA+2]+=r[2];
			ofsA+=cstrA;
			A[ofsA+1]+=r[3];
			A[ofsA+2]+=r[4];
			ofsA+=cstrA;
			A[ofsA+2]+=r[5];
			k+=3;
		} else if (ntg==2) { /* 4x2 matrix B1, superimpose B1'*B1 on lower diagonal of A */
			b[0]=B[ofsB]; b[1]=B[ofsB+1]; b[2]=B[ofsB+2]; b[3]=B[ofsB+3]; /* column 1 of B1 */
			ofsB+=cstrB;
			b[4]=B[ofsB]; b[5]=B[ofsB+1]; b[6]=B[ofsB+2]; b[7]=B[ofsB+3]; /* column 2 of B1 */
			r[0]=b[0]*b[0]+b[1]*b[1]+b[2]*b[2]+b[3]*b[3];
			r[1]=b[4]*b[0]+b[5]*b[1]+b[6]*b[2]+b[7]*b[3];
			r[2]=b[4]*b[4]+b[5]*b[5]+b[6]*b[6]+b[7]*b[7];
			A[ofsA]+=r[0];
			A[ofsA+1]+=r[1];
			ofsA+=cstrA;
			A[ofsA+1]+=r[2];
			k+=2;
		} else { /* need only 1 4-dim dot-product */
			b[0]=B[ofsB]; b[1]=B[ofsB+1]; b[2]=B[ofsB+2]; b[3]=B[ofsB+3];
			r[0]=b[0]*b[0]+b[1]*b[1]+b[2]*b[2]+b[3]*b[3];
			A[ofsA]+=r[0];
			k++;
		}
	}
}

/* Cholesky decomp. inplace working only on lower triangle including the 
 * diagonal. Does not touch the upper triangle (above diagonal).
 */
static inline int matopc_cholesky_dclo(matopc_real *A,int n) {
	int i,j,k,ofs,ofsa;
	matopc_real d,e;
	for (k=0;k<n;k++) {
		ofs=n*k; d=A[ofs+k];
		if (d<=0.0) return 1;
		d=sqrt(d); A[ofs+k]=d;
		for (i=k+1;i<n;i++) A[ofs+i]=A[ofs+i]/d;
		for (j=k+1;j<n;j++) {
			e=A[ofs+j]; ofsa=n*j+j;
			for (i=j;i<n;i++) {
				d=A[ofsa]; d-=A[ofs+i]*e; A[ofsa++]=d;
			}
		}
	}
	return 0;
}

/* Simple unrolled version of the rank-4 downdate operation A <- A-B*B', where B is n-by-4.
 * It is used as a subroutine in Cholesky decomposition below; but has other uses too.
 * cstrA and cstrB are the column strides (memory offsets) to the next column
 * (not necessarily equal to n, but could be; and never less than n).
 * Only downdates the lower triangle (including the diagonal) of A.
 */
static inline void matopc_ambbt_lo4(matopc_real *A,int cstrA,matopc_real *B,int cstrB,int n) {
/*	static matopc_real b[16];
	static matopc_real r[10];*/
	matopc_real *b=&matopc_temparray[0];	/* need exactly 16 doubles */
	matopc_real *r=&matopc_temparray[16]; /* need (up to) 10 (12) doubles */
	int j,k=0,ntg,ofsA,ofsB,ofs;
	while (k<n) {
		ofsA=cstrA*k+k;
		ofsB=k;
		ntg=n-k;
		if (ntg>=4) {	/* 4x4 = 4-by-4 times its transpose */
			b[0]=B[ofsB]; b[1]=B[ofsB+cstrB]; b[2]=B[ofsB+2*cstrB]; b[3]=B[ofsB+3*cstrB];
			b[4]=B[ofsB+1]; b[5]=B[ofsB+cstrB+1]; b[6]=B[ofsB+2*cstrB+1]; b[7]=B[ofsB+3*cstrB+1];
			b[8]=B[ofsB+2]; b[9]=B[ofsB+cstrB+2]; b[10]=B[ofsB+2*cstrB+2]; b[11]=B[ofsB+3*cstrB+2];
			b[12]=B[ofsB+3]; b[13]=B[ofsB+cstrB+3]; b[14]=B[ofsB+2*cstrB+3]; b[15]=B[ofsB+3*cstrB+3];
			/* column 1: [r0,r1,r2,r3], 2: [-,4,5,6], 3: [-,-,7,8], 4: [-,-,-,9] */
			r[0]=b[0]*b[0]+b[1]*b[1]+b[2]*b[2]+b[3]*b[3];
			r[1]=b[4]*b[0]+b[5]*b[1]+b[6]*b[2]+b[7]*b[3];
			r[2]=b[8]*b[0]+b[9]*b[1]+b[10]*b[2]+b[11]*b[3];
			r[3]=b[12]*b[0]+b[13]*b[1]+b[14]*b[2]+b[15]*b[3];
			r[4]=b[4]*b[4]+b[5]*b[5]+b[6]*b[6]+b[7]*b[7];
			r[5]=b[8]*b[4]+b[9]*b[5]+b[10]*b[6]+b[11]*b[7];
			r[6]=b[12]*b[4]+b[13]*b[5]+b[14]*b[6]+b[15]*b[7];
			r[7]=b[8]*b[8]+b[9]*b[9]+b[10]*b[10]+b[11]*b[11];
			r[8]=b[12]*b[8]+b[13]*b[9]+b[14]*b[10]+b[15]*b[11];
			r[9]=b[12]*b[12]+b[13]*b[13]+b[14]*b[14]+b[15]*b[15];
			A[ofsA]-=r[0];
			A[ofsA+1]-=r[1];
			A[ofsA+2]-=r[2];
			A[ofsA+3]-=r[3];
			A[ofsA+cstrA+1]-=r[4];
			A[ofsA+cstrA+2]-=r[5];
			A[ofsA+cstrA+3]-=r[6];
			A[ofsA+2*cstrA+2]-=r[7];
			A[ofsA+2*cstrA+3]-=r[8];
			A[ofsA+3*cstrA+3]-=r[9];
			if (ntg>4) {
				ofsA+=4;
				for (j=k+4;j<n;j++) {
					ofs=j;
					r[0]=B[ofs]; ofs+=cstrB;
					r[1]=B[ofs]; ofs+=cstrB;
					r[2]=B[ofs]; ofs+=cstrB;
					r[3]=B[ofs];
					r[4]=b[0]*r[0]+b[1]*r[1]+b[2]*r[2]+b[3]*r[3];
					r[5]=b[4]*r[0]+b[5]*r[1]+b[6]*r[2]+b[7]*r[3];
					r[6]=b[8]*r[0]+b[9]*r[1]+b[10]*r[2]+b[11]*r[3];
					r[7]=b[12]*r[0]+b[13]*r[1]+b[14]*r[2]+b[15]*r[3];
					ofs=ofsA;
					A[ofs]-=r[4]; ofs+=cstrA;
					A[ofs]-=r[5]; ofs+=cstrA;
					A[ofs]-=r[6]; ofs+=cstrA;
					A[ofs]-=r[7];
					ofsA++;
				} 
			}
			k+=4;
		} else if (ntg==3) { /* 3x3 = result of 3-by-4 times its transpose */
			b[0]=B[ofsB]; b[1]=B[ofsB+cstrB]; b[2]=B[ofsB+2*cstrB]; b[3]=B[ofsB+3*cstrB];
			b[4]=B[ofsB+1]; b[5]=B[ofsB+cstrB+1]; b[6]=B[ofsB+2*cstrB+1]; b[7]=B[ofsB+3*cstrB+1];
			b[8]=B[ofsB+2]; b[9]=B[ofsB+cstrB+2]; b[10]=B[ofsB+2*cstrB+2]; b[11]=B[ofsB+3*cstrB+2];
			r[0]=b[0]*b[0]+b[1]*b[1]+b[2]*b[2]+b[3]*b[3];
			r[1]=b[4]*b[0]+b[5]*b[1]+b[6]*b[2]+b[7]*b[3];
			r[2]=b[8]*b[0]+b[9]*b[1]+b[10]*b[2]+b[11]*b[3];
			r[3]=b[4]*b[4]+b[5]*b[5]+b[6]*b[6]+b[7]*b[7];
			r[4]=b[8]*b[4]+b[9]*b[5]+b[10]*b[6]+b[11]*b[7];
			r[5]=b[8]*b[8]+b[9]*b[9]+b[10]*b[10]+b[11]*b[11];			
			A[ofsA]-=r[0];
			A[ofsA+1]-=r[1];
			A[ofsA+2]-=r[2];
			A[ofsA+cstrA+1]-=r[3];
			A[ofsA+cstrA+2]-=r[4];
			A[ofsA+2*cstrA+2]-=r[5];
			k+=3;
		} else if (ntg==2) { /* 2x2 = result of 2-by-4 times its transpose */
			b[0]=B[ofsB]; b[1]=B[ofsB+cstrB]; b[2]=B[ofsB+2*cstrB]; b[3]=B[ofsB+3*cstrB];
			b[4]=B[ofsB+1]; b[5]=B[ofsB+cstrB+1]; b[6]=B[ofsB+2*cstrB+1]; b[7]=B[ofsB+3*cstrB+1];
			r[0]=b[0]*b[0]+b[1]*b[1]+b[2]*b[2]+b[3]*b[3];
			r[1]=b[4]*b[0]+b[5]*b[1]+b[6]*b[2]+b[7]*b[3];
			r[2]=b[4]*b[4]+b[5]*b[5]+b[6]*b[6]+b[7]*b[7];
			A[ofsA]-=r[0];
			A[ofsA+1]-=r[1];
			A[ofsA+cstrA+1]-=r[2];
			k+=2;
		} else { /* 1x1 scalar = result of 1-by-4 times its tranpose */
			b[0]=B[ofsB]; b[1]=B[ofsB+cstrB]; b[2]=B[ofsB+2*cstrB]; b[3]=B[ofsB+3*cstrB];
			r[0]=b[0]*b[0]+b[1]*b[1]+b[2]*b[2]+b[3]*b[3];
			A[ofsA]-=r[0];
			k++;
		}
	}
}

#ifdef __AVX__
/* NOTE: needs AVX instruction set & FMA feature (fused multiply-add) */
static inline void matopc_ambbt_lo4_avx(double *A,int cstrA,double *B,int cstrB,int n) {
    static double b[16];
    static double r[10];
    int j,k=0,ntg,ofsA,ofsB,ofs;
    __m256d u,v,w,x,y;
    __m256d u0,u1,u2,u3;
    double *pw=(double *)&w;
    double *px=(double *)&x;
    double *py=(double *)&y;
    while (k<n) {
        ofsA=cstrA*k+k;
        ofsB=k;
        ntg=n-k;
        if (ntg>=4) {   /* 4x4 = 4-by-4 times its transpose */
            b[0]=B[ofsB]; b[1]=B[ofsB+cstrB]; b[2]=B[ofsB+2*cstrB]; b[3]=B[ofsB+3*cstrB];
            b[4]=B[ofsB+1]; b[5]=B[ofsB+cstrB+1]; b[6]=B[ofsB+2*cstrB+1]; b[7]=B[ofsB+3*cstrB+1];
            b[8]=B[ofsB+2]; b[9]=B[ofsB+cstrB+2]; b[10]=B[ofsB+2*cstrB+2]; b[11]=B[ofsB+3*cstrB+2];
            b[12]=B[ofsB+3]; b[13]=B[ofsB+cstrB+3]; b[14]=B[ofsB+2*cstrB+3]; b[15]=B[ofsB+3*cstrB+3];
            /* column 1: [r0,r1,r2,r3], 2: [-,4,5,6], 3: [-,-,7,8], 4: [-,-,-,9] */

            /*r[0]=b[0]*b[0]+b[1]*b[1]+b[2]*b[2]+b[3]*b[3];
            r[1]=b[4]*b[0]+b[5]*b[1]+b[6]*b[2]+b[7]*b[3];
            r[2]=b[8]*b[0]+b[9]*b[1]+b[10]*b[2]+b[11]*b[3];
            r[3]=b[12]*b[0]+b[13]*b[1]+b[14]*b[2]+b[15]*b[3];*/

            u0 = _mm256_setr_pd(b[0],b[4],b[8],b[12]);
            v = _mm256_set1_pd(b[0]);
            w = _mm256_mul_pd(u0,v);
            u1 = _mm256_setr_pd(b[1],b[5],b[9],b[13]);
            v = _mm256_set1_pd(b[1]);
            w = _mm256_fmadd_pd(u1,v,w);
            u2 = _mm256_setr_pd(b[2],b[6],b[10],b[14]);
            v = _mm256_set1_pd(b[2]);
            w = _mm256_fmadd_pd(u2,v,w);
            u3 = _mm256_setr_pd(b[3],b[7],b[11],b[15]);
            v = _mm256_set1_pd(b[3]);
            w = _mm256_fmadd_pd(u3,v,w);

            /*r[4]=b[4]*b[4]+b[5]*b[5]+b[6]*b[6]+b[7]*b[7];
            r[5]=b[8]*b[4]+b[9]*b[5]+b[10]*b[6]+b[11]*b[7];
            r[6]=b[12]*b[4]+b[13]*b[5]+b[14]*b[6]+b[15]*b[7];*/

            v = _mm256_set1_pd(b[4]);
            x = _mm256_mul_pd(u0,v);
            v = _mm256_set1_pd(b[5]);
            x = _mm256_fmadd_pd(u1,v,x);
            v = _mm256_set1_pd(b[6]);
            x = _mm256_fmadd_pd(u2,v,x);
            v = _mm256_set1_pd(b[7]);
            x = _mm256_fmadd_pd(u3,v,x);

            /*r[7]=b[8]*b[8]+b[9]*b[9]+b[10]*b[10]+b[11]*b[11];
            r[8]=b[12]*b[8]+b[13]*b[9]+b[14]*b[10]+b[15]*b[11];
            r[9]=b[12]*b[12]+b[13]*b[13]+b[14]*b[14]+b[15]*b[15];*/

            /* TODO: get from u0,u1 etc.. no actual need to call setr again */

            u = _mm256_setr_pd(b[8],b[12],b[12],0.0);
            /*v = _mm256_setr_pd(b[8],b[8],b[12],0.0);*/
            v = _mm256_permute_pd(u,0b00001000);
            y = _mm256_mul_pd(u,v);
            u = _mm256_setr_pd(b[9],b[13],b[13],0.0);
            /*v = _mm256_setr_pd(b[9],b[9],b[13],0.0);*/
            v = _mm256_permute_pd(u,0b00001000);
            y = _mm256_fmadd_pd(u,v,y);
            u = _mm256_setr_pd(b[10],b[14],b[14],0.0);
            /*v = _mm256_setr_pd(b[10],b[10],b[14],0.0);*/
            v = _mm256_permute_pd(u,0b00001000);
            y = _mm256_fmadd_pd(u,v,y);
            u = _mm256_setr_pd(b[11],b[15],b[15],0.0);
            /*v = _mm256_setr_pd(b[11],b[11],b[15],0.0);*/
            v = _mm256_permute_pd(u,0b00001000);
            y = _mm256_fmadd_pd(u,v,y);

            A[ofsA]-=pw[0]; /* r[0] */
            A[ofsA+1]-=pw[1];
            A[ofsA+2]-=pw[2];
            A[ofsA+3]-=pw[3];
            A[ofsA+cstrA+1]-=px[1]; /* r[4] */
            A[ofsA+cstrA+2]-=px[2];
            A[ofsA+cstrA+3]-=px[3];
            A[ofsA+2*cstrA+2]-=py[0]; /* r[7] */
            A[ofsA+2*cstrA+3]-=py[1];
            A[ofsA+3*cstrA+3]-=py[2];
            if (ntg>4) {
                ofsA+=4;
                for (j=k+4;j<n;j++) {
                    ofs=j;
                    /*r[0]=B[ofs]; ofs+=cstrB;
                    r[1]=B[ofs]; ofs+=cstrB;
                    r[2]=B[ofs]; ofs+=cstrB;
                    r[3]=B[ofs];*/

                    v = _mm256_set1_pd(B[ofs]); ofs+=cstrB;
                    w = _mm256_mul_pd(u0,v);
                    v = _mm256_set1_pd(B[ofs]); ofs+=cstrB;
                    w = _mm256_fmadd_pd(u1,v,w);
                    v = _mm256_set1_pd(B[ofs]); ofs+=cstrB;
                    w = _mm256_fmadd_pd(u2,v,w);
                    v = _mm256_set1_pd(B[ofs]);
                    w = _mm256_fmadd_pd(u3,v,w);

                    /*r[4]=b[0]*r[0]+b[1]*r[1]+b[2]*r[2]+b[3]*r[3];
                    r[5]=b[4]*r[0]+b[5]*r[1]+b[6]*r[2]+b[7]*r[3];
                    r[6]=b[8]*r[0]+b[9]*r[1]+b[10]*r[2]+b[11]*r[3];
                    r[7]=b[12]*r[0]+b[13]*r[1]+b[14]*r[2]+b[15]*r[3];*/

                    /* pw[0]=r[4] ... */
                    ofs=ofsA;
                    A[ofs]-=pw[0]; ofs+=cstrA;
                    A[ofs]-=pw[1]; ofs+=cstrA;
                    A[ofs]-=pw[2]; ofs+=cstrA;
                    A[ofs]-=pw[3];
                    ofsA++;
                } 
            }
            k+=4;
        } else if (ntg==3) { /* 3x3 = result of 3-by-4 times its transpose */
            b[0]=B[ofsB]; b[1]=B[ofsB+cstrB]; b[2]=B[ofsB+2*cstrB]; b[3]=B[ofsB+3*cstrB];
            b[4]=B[ofsB+1]; b[5]=B[ofsB+cstrB+1]; b[6]=B[ofsB+2*cstrB+1]; b[7]=B[ofsB+3*cstrB+1];
            b[8]=B[ofsB+2]; b[9]=B[ofsB+cstrB+2]; b[10]=B[ofsB+2*cstrB+2]; b[11]=B[ofsB+3*cstrB+2];
            r[0]=b[0]*b[0]+b[1]*b[1]+b[2]*b[2]+b[3]*b[3];
            r[1]=b[4]*b[0]+b[5]*b[1]+b[6]*b[2]+b[7]*b[3];
            r[2]=b[8]*b[0]+b[9]*b[1]+b[10]*b[2]+b[11]*b[3];
            r[3]=b[4]*b[4]+b[5]*b[5]+b[6]*b[6]+b[7]*b[7];
            r[4]=b[8]*b[4]+b[9]*b[5]+b[10]*b[6]+b[11]*b[7];
            r[5]=b[8]*b[8]+b[9]*b[9]+b[10]*b[10]+b[11]*b[11];           
            A[ofsA]-=r[0];
            A[ofsA+1]-=r[1];
            A[ofsA+2]-=r[2];
            A[ofsA+cstrA+1]-=r[3];
            A[ofsA+cstrA+2]-=r[4];
            A[ofsA+2*cstrA+2]-=r[5];
            k+=3;
        } else if (ntg==2) { /* 2x2 = result of 2-by-4 times its transpose */
            b[0]=B[ofsB]; b[1]=B[ofsB+cstrB]; b[2]=B[ofsB+2*cstrB]; b[3]=B[ofsB+3*cstrB];
            b[4]=B[ofsB+1]; b[5]=B[ofsB+cstrB+1]; b[6]=B[ofsB+2*cstrB+1]; b[7]=B[ofsB+3*cstrB+1];
            r[0]=b[0]*b[0]+b[1]*b[1]+b[2]*b[2]+b[3]*b[3];
            r[1]=b[4]*b[0]+b[5]*b[1]+b[6]*b[2]+b[7]*b[3];
            r[2]=b[4]*b[4]+b[5]*b[5]+b[6]*b[6]+b[7]*b[7];
            A[ofsA]-=r[0];
            A[ofsA+1]-=r[1];
            A[ofsA+cstrA+1]-=r[2];
            k+=2;
        } else { /* 1x1 scalar = result of 1-by-4 times its tranpose */
            b[0]=B[ofsB]; b[1]=B[ofsB+cstrB]; b[2]=B[ofsB+2*cstrB]; b[3]=B[ofsB+3*cstrB];
            r[0]=b[0]*b[0]+b[1]*b[1]+b[2]*b[2]+b[3]*b[3];
            A[ofsA]-=r[0];
            k++;
        }
    }
}
#endif

/* Rank-4 update version of the previous Chol. routine.
 * Explicit handling of 1x1, 2x2, 3x3, and 4x4 Cholesky block factorizations and backsolves.
 * Appears to have OK speed on medium sized matrices.
 * NOTE: the dclo2,3 routines are basically "included" in this one.
 *
 * TODO: remove the if(..) structure and make it switch(..) instead for the remainder
 */
static inline int matopc_cholesky_dclo4(matopc_real *A,int n) {
	matopc_real a11,a21,a31,a41,a22,a32,a42,a33,a43,a44;
	matopc_real l11,l21,l31,l41,l22,l32,l42,l33,l43,l44;
	int i,k=0,ntg,ofs,ofsa,ofsb,ofsc,ofsd;
/*	int j,idx; */
	while (k<n) {
		ofs=n*k;
		ntg=n-k;
		if (ntg>=4) {
			ofsa=ofs+k;
			a11=A[ofsa]; a21=A[ofsa+1]; a31=A[ofsa+2]; a41=A[ofsa+3];
			ofsb=ofsa+n;
			a22=A[ofsb+1]; a32=A[ofsb+2]; a42=A[ofsb+3];
			ofsc=ofsb+n;
			a33=A[ofsc+2]; a43=A[ofsc+3];
			ofsd=ofsc+n;
			a44=A[ofsd+3];
			if (a11<=0.0) return 1;
			l11=sqrt(a11); l21=a21/l11; l31=a31/l11; l41=a41/l11; a22-=l21*l21;
			if (a22<=0.0) return 1;
			l22=sqrt(a22); l32=(a32-l31*l21)/l22; l42=(a42-l41*l21)/l22; a33-=(l31*l31+l32*l32);
			if (a33<=0.0) return 1;
			l33=sqrt(a33); l43=(a43-l41*l31-l42*l32)/l33; a44-=(l41*l41+l42*l42+l43*l43);
			if (a44<=0.0) return 1;
			l44=sqrt(a44);
			A[ofsa]=l11; A[ofsa+1]=l21; A[ofsa+2]=l31; A[ofsa+3]=l41;
			A[ofsb+1]=l22; A[ofsb+2]=l32; A[ofsb+3]=l42;
			A[ofsc+2]=l33; A[ofsc+3]=l43;
			A[ofsd+3]=l44;
			if (ntg>4) { /* done if A22 is empty */
				for (i=k+4;i<n;i++) { /* back solve in place for the 4 columns in L21 block */
					ofsa=ofs+i; ofsb=ofsa+n; ofsc=ofsb+n; ofsd=ofsc+n;
					a11=A[ofsa]; a21=A[ofsb]; a31=A[ofsc]; a41=A[ofsd];
					a11=a11/l11; a21=(a21-l21*a11)/l22; a31=(a31-l31*a11-l32*a21)/l33;
					a41=(a41-l41*a11-l42*a21-l43*a31)/l44;
					A[ofsa]=a11; A[ofsb]=a21; A[ofsc]=a31; A[ofsd]=a41;
				}
				/* rank-4 downdate L22 block (lower triangle); -=L21*L21' */
				matopc_ambbt_lo4(&A[ofs+4*n+k+4],n,&A[ofs+k+4],n,ntg-4);
		/*		for (j=k+4;j<n;j++) {
					ofsa=ofs+j; ofsb=ofsa+n; ofsc=ofsb+n; ofsd=ofsc+n;
					a11=A[ofsa]; a21=A[ofsb]; a31=A[ofsc]; a41=A[ofsd]; idx=n*j+j;
					for (i=j;i<n;i++) {
						A[idx++]-=(A[ofsa++]*a11+A[ofsb++]*a21+A[ofsc++]*a31+A[ofsd++]*a41);
					}
				}
		*/
			}
			k+=4;
		} else if (ntg==3) {
			ofsa=ofs+k; a11=A[ofsa]; a21=A[ofsa+1]; a31=A[ofsa+2];
			ofsb=ofsa+n; a22=A[ofsb+1]; a32=A[ofsb+2];
			ofsc=ofsb+n; a33=A[ofsc+2];
			if (a11<=0.0) return 1;
			l11=sqrt(a11); l21=a21/l11; l31=a31/l11; a22-=l21*l21;
			if (a22<=0.0) return 1;
			l22=sqrt(a22); l32=(a32-l31*l21)/l22; a33-=(l31*l31+l32*l32);
			if (a33<=0.0) return 1;
			l33=sqrt(a33);
			A[ofsa]=l11; A[ofsa+1]=l21; A[ofsa+2]=l31;
			A[ofsb+1]=l22; A[ofsb+2]=l32;
			A[ofsc+2]=l33;
			k+=3;
		} else if (ntg==2) {	/* explicit solve of 2x2 block */
			ofsa=ofs+k; a11=A[ofsa]; a21=A[ofsa+1];
			ofsb=ofsa+n; a22=A[ofsb+1];
			if (a11<=0.0) return 1;
			l11=sqrt(a11); l21=a21/l11; a22-=l21*l21;
			if (a22<=0.0) return 1;
			l22=sqrt(a22);
			A[ofsa]=l11; A[ofsa+1]=l21; A[ofsb+1]=l22;
			k+=2;
		} else { /* ntg=1; scalar last block */
			ofsa=ofs+k; l11=A[ofsa];
			if (l11<=0.0) return 1;
			A[ofsa]=sqrt(l11);
			k++;
		}
	}
	return 0;
}

/* NOTE: -mavx option to GCC will set __AVX__
 * The below function only supports double precision using the pd intrinsics;
 * AVX 256 bit registers.
 */
#ifdef __AVX__
static inline int matopc_cholesky_dclo4_avx(double *A,int n) {
    double a11,a21,a31,a41,a22,a32,a42,a33,a43,a44;
    double l11,l21,l31,l41,l22,l32,l42,l33,l43,l44;
    int i,k=0,ofs,ofsa,ofsb,ofsc,ofsd;
    while (n-k>=4) {
        ofs=n*k;
        ofsa=ofs+k;
        a11=A[ofsa]; a21=A[ofsa+1]; a31=A[ofsa+2]; a41=A[ofsa+3];
        ofsb=ofsa+n;
        a22=A[ofsb+1]; a32=A[ofsb+2]; a42=A[ofsb+3];
        ofsc=ofsb+n;
        a33=A[ofsc+2]; a43=A[ofsc+3];
        ofsd=ofsc+n;
        a44=A[ofsd+3];
        if (a11<=0.0) return 1;
        l11=sqrt(a11); l21=a21/l11; l31=a31/l11; l41=a41/l11; a22-=l21*l21;
        if (a22<=0.0) return 1;
        l22=sqrt(a22); l32=(a32-l31*l21)/l22; l42=(a42-l41*l21)/l22; a33-=(l31*l31+l32*l32);
        if (a33<=0.0) return 1;
        l33=sqrt(a33); l43=(a43-l41*l31-l42*l32)/l33; a44-=(l41*l41+l42*l42+l43*l43);
        if (a44<=0.0) return 1;
        l44=sqrt(a44);
        A[ofsa]=l11; A[ofsa+1]=l21; A[ofsa+2]=l31; A[ofsa+3]=l41;
        A[ofsb+1]=l22; A[ofsb+2]=l32; A[ofsb+3]=l42;
        A[ofsc+2]=l33; A[ofsc+3]=l43;
        A[ofsd+3]=l44;
        for (i=k+4;i<n;i++) { /* back solve in place for the 4 columns in L21 block */
            ofsa=ofs+i; ofsb=ofsa+n; ofsc=ofsb+n; ofsd=ofsc+n;
            a11=A[ofsa]; a21=A[ofsb]; a31=A[ofsc]; a41=A[ofsd];
            a11=a11/l11; a21=(a21-l21*a11)/l22; a31=(a31-l31*a11-l32*a21)/l33;
            a41=(a41-l41*a11-l42*a21-l43*a31)/l44;
            A[ofsa]=a11; A[ofsb]=a21; A[ofsc]=a31; A[ofsd]=a41;
        }
        /* rank-4 downdate L22 block (lower triangle); -=L21*L21' */
        matopc_ambbt_lo4_avx(&A[ofs+4*n+k+4],n,&A[ofs+k+4],n,n-k-4);
        k+=4;
    }
    ofs=n*k;
    switch (n-k) {
        case 3:
            ofsa=ofs+k; a11=A[ofsa]; a21=A[ofsa+1]; a31=A[ofsa+2];
            ofsb=ofsa+n; a22=A[ofsb+1]; a32=A[ofsb+2];
            ofsc=ofsb+n; a33=A[ofsc+2];
            if (a11<=0.0) return 1;
            l11=sqrt(a11); l21=a21/l11; l31=a31/l11; a22-=l21*l21;
            if (a22<=0.0) return 1;
            l22=sqrt(a22); l32=(a32-l31*l21)/l22; a33-=(l31*l31+l32*l32);
            if (a33<=0.0) return 1;
            l33=sqrt(a33);
            A[ofsa]=l11; A[ofsa+1]=l21; A[ofsa+2]=l31;
            A[ofsb+1]=l22; A[ofsb+2]=l32;
            A[ofsc+2]=l33;
            break;
        case 2:
            ofsa=ofs+k; a11=A[ofsa]; a21=A[ofsa+1];
            ofsb=ofsa+n; a22=A[ofsb+1];
            if (a11<=0.0) return 1;
            l11=sqrt(a11); l21=a21/l11; a22-=l21*l21;
            if (a22<=0.0) return 1;
            l22=sqrt(a22);
            A[ofsa]=l11; A[ofsa+1]=l21; A[ofsb+1]=l22;
            break;
        case 1:
            ofsa=ofs+k; l11=A[ofsa];
            if (l11<=0.0) return 1;
            A[ofsa]=sqrt(l11);
            break;
        case 0:
            break;
        default:
            return 1;
    }
    return 0;
}
#endif

/* Solution routines for the lower-plus-diagonal storage version above: solves L*x=b */
static inline void matopc_cholesky_lo_trisubst_left(matopc_real *A,int n,matopc_real *b,matopc_real *x) {
	int i,j,ofs; matopc_real s;
	for (i=0;i<n;i++) {
		s=b[i]; for (j=0,ofs=i;j<i;j++,ofs+=n) s-=A[ofs]*x[j]; /* A[n*j+i]*x[j] */
		x[i]=s/A[n*i+i];
	}
}

/* Solution routines for the lower-plus-diagonal storage version above: solves L'*x=b */
static inline void matopc_cholesky_lo_trisubst_tr_left(matopc_real *A,int n,matopc_real *b,matopc_real *x) {
	int i,j,ofs; matopc_real s;
	for (i=n-1;i>=0;i--) {
		s=b[i]; for (j=n-1,ofs=n*i+j;j>i;j--,ofs--) s-=A[ofs]*x[j]; /* A[n*i+j]*x[j] */
		x[i]=s/A[n*i+i];
	}
}

/* Solve (L*L')*x=b, in-place OK */
static inline void matopc_cholesky_lo_solve(matopc_real *A,int n,matopc_real *b,matopc_real *x) {
	matopc_cholesky_lo_trisubst_left(A,n,b,x);
	matopc_cholesky_lo_trisubst_tr_left(A,n,x,x);
}

/* Rank-1 Cholesky update; factor L stored in lower triangle of A (incl. diagonal).
 * After the call the lower triangle will instead be the factor of the perturbed matrix L*L'+x*x'.
 * Vector x will be demolished during the call.
 */
static inline void matopc_cholesky_lo_update(matopc_real *A,int n,matopc_real *x) {
    int i,j; matopc_real ell,lt11,a,b;
    for (j=0;j<n;j++) {
        ell=A[n*j+j]; lt11=sqrt(ell*ell+x[j]*x[j]);
        a=ell/lt11; b=x[j]/lt11;
        A[n*j+j]=lt11;
        for (i=j+1;i<n;i++) {
            ell=A[n*j+i];
            A[n*j+i]=a*ell+b*x[i];
            x[i]=b*ell-a*x[i];
        }
    }
}

/* Test program to mock-up an identical interface to the next function.
 * Not intended for intensive use since it comes with "peripheral costs".
 */
/*
static inline int matopc_cholesky_decompose_alt(matopc_real *A,matopc_real *p,int n,int sel) {
	matopc_symmetrize(A,n,MATOPC_UPPER);
	matopc_getdiag(matopc_temparray,A,n);
	if (sel==1) {
		if (matopc_cholesky_dclo(A,n)!=0) return 1;
	} else if (sel==2) {
		if (matopc_cholesky_dclo2(A,n)!=0) return 1;
	} else if (sel==3) {
		if (matopc_cholesky_dclo3(A,n)!=0) return 1;
	} else if (sel==4) {
		if (matopc_cholesky_dclo4(A,n)!=0) return 1;
	} else {
		return 2;
	}
	matopc_getdiag(p,A,n);
	matopc_setdiag(A,matopc_temparray,n);
	return 0;
}
*/

/* Cholesky decomposition and various forward and backward substitutions
 * using lower triangular factors L. Only upper part of A (n-by-n) will
 * be used, L will be stored in lower part and the diagonal of L will be put into p.
 */
static inline int matopc_cholesky_decompose(matopc_real *A,matopc_real *p,int n) {
	int i,j,k,c1,c2;
	matopc_real sum;
	for (i=0;i<n;i++) {
		for (j=i;j<n;j++) {
			sum=A[j*n+i]; c1=(i-1)*n+i; c2=(i-1)*n+j;
			for (k=i-1;k>=0;k--) {
				sum-=A[c1]*A[c2]; c1-=n; c2-=n;	/* sum-=A[k*n+i]*A[k*n+j]; */
			}
			if (i==j) {
				if (sum<=0.0) return 1;	/* Failure to factorize: A is not pos. def. */
				p[i]=sqrt(sum);
			} else {
				A[i*n+j]=sum/p[i];	/* A(j,i)=... */
			}
		}
	}
	return 0;
}

/* Cholesky factor rank-1 update; if L*L'=A then after calling this routine
 * L*L'=A+x*x' for the column vector x; NOTE: x will be modified during the call.
 * Lower triangle of A will be the updated factor L; vector p will be the 
 * updated diagonal of L.
 */
static inline void matopc_cholesky_update(
        matopc_real *A,matopc_real *p,int n,
        matopc_real *x) {
    int k,l,i;
    matopc_real r,c,s;
    for (k=0;k<n;k++) {
        r=sqrt(p[k]*p[k]+x[k]*x[k]);
        c=r/p[k];
        s=x[k]/p[k];
        p[k]=r; i=n*k;
        for (l=k;l<n;l++) {
            A[i+l]=(A[i+l]+s*x[l])/c;
            x[l]=c*x[l]-s*A[i+l];
        }
        /*for (l=k;l<n;l++) {
            x[l]=c*x[l]-s*A[i+l];
        }*/
    }
}

/* The Cholesky factor L is assumed to be in the lower triangle of A. The diagonal of L is assumed to be at p.
   The equation (L*L')*x=b is solved for x using forward and backward substitution. It is allowed to equate x=b
   which means that b will be overwritten by the solution x.
 */
static inline void matopc_cholesky_solve(
        matopc_real *A,matopc_real *p,int n,
        matopc_real *b,matopc_real *x) {
	int i,k,c1;
	matopc_real sum;
	for (i=0;i<n;i++) {	/* solve L*y=b store y in x */
		c1=(i-1)*n+i;
		for (sum=b[i],k=i-1;k>=0;k--) {
			sum-=A[c1]*x[k]; c1-=n;	/* sum-=a[i][k]*x[k]; */
		}
		x[i]=sum/p[i];
	}
	for (i=n-1;i>=0;i--) { /* solve L'*x=y */
		c1=i*n+(i+1);
		for (sum=x[i],k=i+1;k<n;k++) {
			sum-=A[c1]*x[k]; c1++; /* sum-=a[k][i]*x[k]; */
		}
		x[i]=sum/p[i];
	}
}

/* Solve (L*L')*X=B for matrix X (column by column), X and B is n-by-m (m columns of length n) */
static inline void matopc_cholesky_solve_matrix(
	matopc_real *A,matopc_real *p,int n,
        matopc_real *B,matopc_real *X,int m) {
	int i,c; c=0;
	for (i=0;i<m;i++) {
		matopc_cholesky_solve(A,p,n,&B[c],&X[c]);
		c+=n;
	}
}

/* L\(.) and L'\(.) operations (half of the above solver routine that is) */
/* NOTE: (.)/L and (.)/L' operations are transposes of the above */
/* Solve L*x=b where L is the lower triangle of A and the diagonal of L is p (x=L\b).
   NOTE: x and b can be identical, which means the RHS will be overwritten.   
 */
static inline void matopc_cholesky_trisubst_left(
        matopc_real *A,matopc_real *p,int n,
        matopc_real *b,matopc_real *x) {
	int i,k,c1;
	matopc_real sum;
	for (i=0;i<n;i++) {
		c1=(i-1)*n+i;
		for (sum=b[i],k=i-1;k>=0;k--) {
			sum-=A[c1]*x[k]; c1-=n;	/* sum-=a[i][k]*x[k]; */
		}
		x[i]=sum/p[i];
	}
}

/* L\(.) operation on column; but where it is assumed that the first nskip
 * elements of b are zeros; then also the first nskip of x are zeros; and
 * it is possible to shorten the for loops.
 * nskip=0 degenerates to the standard L\(.) operation.
 */
static inline void matopc_cholesky_trisubst_left_skip(
        matopc_real *A,matopc_real *p,int n,
        matopc_real *b,matopc_real *x,int nskip) {
	int i,k,c1;
	matopc_real sum;
	for (i=nskip;i<n;i++) {
		c1=(i-1)*n+i;
		for (sum=b[i],k=i-1;k>=nskip;k--) {
			sum-=A[c1]*x[k]; c1-=n;
		}
		x[i]=sum/p[i];
	}
}

/* Operation L*X=B where B is n-by-m and A,p contains the Cholesky factor L: X=L\B */
static inline void matopc_cholesky_trisubst_left_matrix(
        matopc_real *A,matopc_real *p,int n,
        matopc_real *B,matopc_real *X,int m) {
	int j,idx;
	idx=0;
	for (j=0;j<m;j++) {
		matopc_cholesky_trisubst_left(A,p,n,&B[idx],&X[idx]);
		idx+=n;
	}
}

/* Operation L*X=B' where B is m-by-n (B' is n-by-m) and A,p contains the Cholesky factor L: X=L\B' */
static inline void matopc_cholesky_trisubst_left_tr_matrix(
        matopc_real *A,matopc_real *p,int n,
        matopc_real *B,matopc_real *X,int m,
        matopc_real *tmp) {
	int i,j,idx,ofs;
	if (tmp==NULL) tmp=matopc_temparray;
	ofs=0;
	for (j=0;j<m;j++) {
		for (i=0,idx=j;i<n;i++,idx+=m) tmp[i]=B[idx];
		matopc_cholesky_trisubst_left(A,p,n,tmp,&X[ofs]);
		ofs+=n;
	}
}

/* Same as above but the equation L'*x=b will be solved instead (x=L'\b).*/
static inline void matopc_cholesky_trisubst_tr_left(
        matopc_real *A,matopc_real *p,int n,
        matopc_real *b,matopc_real *x) {
	int i,k,c1;
	matopc_real sum;
	for (i=n-1;i>=0;i--) {
		c1=i*n+(i+1);
		for (sum=b[i],k=i+1;k<n;k++) {
			sum-=A[c1]*x[k]; c1++; /* sum-=a[k][i]*x[k]; */
		}
		x[i]=sum/p[i];
	}
}

/* Solve L'*X=B where B is n-by-m and A,p contains the Cholesky factor L: X=L'\B */
static inline void matopc_cholesky_trisubst_tr_left_matrix(
        matopc_real *A,matopc_real *p,int n,
        matopc_real *B,matopc_real *X,int m) {
	int j,idx;
	idx=0;
	for (j=0;j<m;j++) {
		matopc_cholesky_trisubst_tr_left(A,p,n,&B[idx],&X[idx]);
		idx+=n;
	}
}

/* Solve X*L=B where L is n-by-n, X and B are m-by-n: X=B/L, X and B can be equated */
static inline void matopc_cholesky_trisubst_right_matrix(
        matopc_real *A,matopc_real *p,int n,
        matopc_real *B,matopc_real *X,int m) {
	int i,j,r1;
	for (j=0;j<m;j++) {
		/* This inner loop is equivalent to L'\(.) on a column, but here operates instead on a row */
		r1=j; for (i=0;i<n;i++) { matopc_temparray[i]=B[r1]; r1+=m; }
		matopc_cholesky_trisubst_tr_left(A,p,n,matopc_temparray,matopc_temparray);
		r1=j; for (i=0;i<n;i++) { X[r1]=matopc_temparray[i]; r1+=m; }
	}
}

/* Solve X*L'=B where L is n-by-n, X and B are m-by-n: X=B/L', X and B can be equated */
static inline void matopc_cholesky_trisubst_tr_right_matrix(
        matopc_real *A,matopc_real *p,int n,
        matopc_real *B,matopc_real *X,int m) {
	int i,j,r1;
	for (j=0;j<m;j++) {
		/* This inner loop is equivalent to L\(.) on a column, but here operates instead on a row */
		r1=j; for (i=0;i<n;i++) { matopc_temparray[i]=B[r1]; r1+=m; }
		matopc_cholesky_trisubst_left(A,p,n,matopc_temparray,matopc_temparray);
		r1=j; for (i=0;i<n;i++) { X[r1]=matopc_temparray[i]; r1+=m; }
	}
}

/* Below are "generic" block-based routines that do not have any inner
 * loop unrolling; but explicitly loads smaller panels into memory; may be
 * significantly better at avoiding cache misses by selecting an
 * appropriate block-size b.
 *
 * These routines typically utilize a column-stride parameter that can be
 * different from the number of rows; more versatile that way.
 */
 
static inline void matopc_blk_copy(
				matopc_real *B,int m,int n,
				matopc_real *A,int cstrA) {
	int i,j,ofsa=0,ofsb=0;
	for (j=0;j<n;j++) {
		ofsa=j*cstrA; for (i=0;i<m;i++) B[ofsb++]=A[ofsa++];
	}
}

static inline void matopc_blk_op(
				matopc_real *C,int cstrC,
				matopc_real *B,int m,int n,
				int op) {
	int i,j,ofsC,ofsB=0;
	if (op==0) {
		for (j=0;j<n;j++) {
			ofsC=j*cstrC; for (i=0;i<m;i++) C[ofsC++]=B[ofsB++];
		}
	} else if (op>0) {
		for (j=0;j<n;j++) {
			ofsC=j*cstrC; for (i=0;i<m;i++) C[ofsC++]+=B[ofsB++];
		}
	} else {
		for (j=0;j<n;j++) {
			ofsC=j*cstrC; for (i=0;i<m;i++) C[ofsC++]-=B[ofsB++];
		}
	}
}

static inline void matopc_blk_4x4_op(
				matopc_real *C,int cstrC,
				matopc_real *cc,int mcmax,int ncmax,
				int op) {
	int i,j,ofsC;
	if (mcmax<4 || ncmax<4) { /* limited 4x4 op */
		if (op==0) {
			for (j=0;j<ncmax;j++) for (i=0;i<mcmax;i++) C[j*cstrC+i]=cc[j*4+i];
		} else if (op>0) {
			for (j=0;j<ncmax;j++) for (i=0;i<mcmax;i++) C[j*cstrC+i]+=cc[j*4+i];
		} else {
			for (j=0;j<ncmax;j++) for (i=0;i<mcmax;i++) C[j*cstrC+i]-=cc[j*4+i];
		}
	} else { /* unrolled 4x4 op */
		ofsC=0;
		if (op==0) {
			/*for (j=0;j<4;j++) for (i=0;i<4;i++) C[j*cstrC+i]=cc[j*4+i];*/
			C[ofsC]=cc[0]; C[ofsC+1]=cc[1]; C[ofsC+2]=cc[2]; C[ofsC+3]=cc[3]; ofsC+=cstrC;
			C[ofsC]=cc[4]; C[ofsC+1]=cc[5]; C[ofsC+2]=cc[6]; C[ofsC+3]=cc[7]; ofsC+=cstrC;
			C[ofsC]=cc[8]; C[ofsC+1]=cc[9]; C[ofsC+2]=cc[10]; C[ofsC+3]=cc[11]; ofsC+=cstrC;
			C[ofsC]=cc[12]; C[ofsC+1]=cc[13]; C[ofsC+2]=cc[14]; C[ofsC+3]=cc[15];
		} else if (op>0) {
			/*for (j=0;j<4;j++) for (i=0;i<4;i++) C[j*cstrC+i]+=cc[j*4+i];*/
			C[ofsC]+=cc[0]; C[ofsC+1]+=cc[1]; C[ofsC+2]+=cc[2]; C[ofsC+3]+=cc[3]; ofsC+=cstrC;
			C[ofsC]+=cc[4]; C[ofsC+1]+=cc[5]; C[ofsC+2]+=cc[6]; C[ofsC+3]+=cc[7]; ofsC+=cstrC;
			C[ofsC]+=cc[8]; C[ofsC+1]+=cc[9]; C[ofsC+2]+=cc[10]; C[ofsC+3]+=cc[11]; ofsC+=cstrC;
			C[ofsC]+=cc[12]; C[ofsC+1]+=cc[13]; C[ofsC+2]+=cc[14]; C[ofsC+3]+=cc[15];
		} else {
			/*for (j=0;j<4;j++) for (i=0;i<4;i++) C[j*cstrC+i]-=cc[j*4+i];*/
			C[ofsC]-=cc[0]; C[ofsC+1]-=cc[1]; C[ofsC+2]-=cc[2]; C[ofsC+3]-=cc[3]; ofsC-=cstrC;
			C[ofsC]-=cc[4]; C[ofsC+1]-=cc[5]; C[ofsC+2]-=cc[6]; C[ofsC+3]-=cc[7]; ofsC-=cstrC;
			C[ofsC]-=cc[8]; C[ofsC+1]-=cc[9]; C[ofsC+2]-=cc[10]; C[ofsC+3]-=cc[11]; ofsC-=cstrC;
			C[ofsC]-=cc[12]; C[ofsC+1]-=cc[13]; C[ofsC+2]-=cc[14]; C[ofsC+3]-=cc[15];
		}
	}
}

/* Symmetric ("half") version of the above subprogram */
static inline void matopc_blk_4x4_sym_op(
			matopc_real *C,int cstrC,
			matopc_real *cc,int mcmax,
			int which,int op) {
	int i,j; /*,ofsC;*/
	if (which==MATOPC_LOWER) {
		if (op==0) {
			for (j=0;j<mcmax;j++) for (i=j;i<mcmax;i++) C[j*cstrC+i]=cc[j*4+i];
		} else if (op>0) {
			for (j=0;j<mcmax;j++) for (i=j;i<mcmax;i++) C[j*cstrC+i]+=cc[j*4+i];
		} else {
			for (j=0;j<mcmax;j++) for (i=j;i<mcmax;i++) C[j*cstrC+i]-=cc[j*4+i];
		}
		/* TODO: unroll the normal unlimited case */
	} else {
		if (op==0) {
			for (j=0;j<mcmax;j++) for (i=0;i<=j;i++) C[j*cstrC+i]=cc[j*4+i];
		} else if (op>0) {
			for (j=0;j<mcmax;j++) for (i=0;i<=j;i++) C[j*cstrC+i]+=cc[j*4+i];
		} else {
			for (j=0;j<mcmax;j++) for (i=0;i<=j;i++) C[j*cstrC+i]-=cc[j*4+i];
		}
		/* TODO: unroll the normal unlimited case */
	}
}

/* Pack q 4-by-1 rows of panel of A as the q 4-by-1 columns of aa */
static inline void matopc_blk_pack4x1(
				matopc_real *aa,int q,
				matopc_real *A,int cstrA,
				int maxcols) {
	int i,j,k,ofsA=0,ofsaa=0;
	if (maxcols<4) { /* apply zero-padding for unused space */
		for (i=0;i<q;i++) {
			for (j=0;j<maxcols;j++) aa[ofsaa+j]=A[ofsA+j*cstrA];
			for (j=maxcols;j<4;j++) aa[ofsaa+j]=0.0;
			ofsaa+=4;
			ofsA++;
		}
	} else {
		for (i=0,ofsA=0;i<q;i++,ofsA++) {
			k=ofsA;
			aa[ofsaa++]=A[k]; k+=cstrA;
			aa[ofsaa++]=A[k]; k+=cstrA;
			aa[ofsaa++]=A[k]; k+=cstrA;
			aa[ofsaa++]=A[k];
/*			ofsaa+=4;
			ofsA++;*/
		}
	}
}

/* aa is a set of q 4-vectors, bb is a set of q 4 vectors
 * calculate the 4-by-4 matrix cc <- sum_j(a_j*b_j') 
 * by accumulation of q rank-1 matrices.
 */
static inline void matopc_blk_4x4_cpabt(
				matopc_real *cc,
				matopc_real *aa,
				matopc_real *bb,
				int q) {
	matopc_real a1,a2,a3,a4;
	matopc_real b1,b2,b3,b4;
	int i,j;
/*	for (i=0;i<4*4;i++) cc[i]=0.0; */
	for (i=0,j=0;i<q;i++,j+=4) {
		a1=aa[j]; a2=aa[j+1]; a3=aa[j+2]; a4=aa[j+3];
		b1=bb[j]; b2=bb[j+1]; b3=bb[j+2]; b4=bb[j+3];
		cc[0]+=a1*b1;
		cc[1]+=a2*b1;
		cc[2]+=a3*b1;
		cc[3]+=a4*b1;
		cc[4]+=a1*b2;
		cc[5]+=a2*b2;
		cc[6]+=a3*b2;
		cc[7]+=a4*b2;
		cc[8]+=a1*b3;
		cc[9]+=a2*b3;
		cc[10]+=a3*b3;
		cc[11]+=a4*b3;
		cc[12]+=a1*b4;
		cc[13]+=a2*b4;
		cc[14]+=a3*b4;
		cc[15]+=a4*b4;
	}
}

/* symmetric version; only maintains upper or lower part of cc;
 * which is MATOPC_UPPER or MATOPC_LOWER */
static inline void matopc_blk_4x4_cpaat(
				matopc_real *cc,
				matopc_real *aa,
				int q,int which) {
	matopc_real a1,a2,a3,a4;
	int i,j;
	if (which==MATOPC_UPPER) {
		for (i=0,j=0;i<q;i++,j+=4) { /* upper part of 4x4 block only */
			a1=aa[j]; a2=aa[j+1]; a3=aa[j+2]; a4=aa[j+3];
			cc[0]+=a1*a1;
			cc[4]+=a1*a2;
			cc[5]+=a2*a2;
			cc[8]+=a1*a3;
			cc[9]+=a2*a3;
			cc[10]+=a3*a3;
			cc[12]+=a1*a4;
			cc[13]+=a2*a4;
			cc[14]+=a3*a4;
			cc[15]+=a4*a4;
		}
	} else {
		for (i=0,j=0;i<q;i++,j+=4) { /* lower part of 4x4 block only */
			a1=aa[j]; a2=aa[j+1]; a3=aa[j+2]; a4=aa[j+3];
			cc[0]+=a1*a1;
			cc[1]+=a2*a1;
			cc[2]+=a3*a1;
			cc[3]+=a4*a1;
			cc[5]+=a2*a2;
			cc[6]+=a3*a2;
			cc[7]+=a4*a2;
			cc[10]+=a3*a3;
			cc[11]+=a4*a3;
			cc[15]+=a4*a4;
		}
	}
}

/* weighted q times rank-1 4x4 accumulation;
 * used as subprogram in C <- A'*diag(w)*A
 */
static inline void matopc_blk_4x4_cpabtw(
				matopc_real *cc,
				matopc_real *aa,
				matopc_real *bb,
				matopc_real *w,
				int q) {
	matopc_real a1,a2,a3,a4;
	matopc_real b1,b2,b3,b4;
	matopc_real d;
	int i,j;
	for (i=0,j=0;i<q;i++,j+=4) {
		d=w[i];
		a1=aa[j]; a2=aa[j+1]; a3=aa[j+2]; a4=aa[j+3];
		b1=bb[j]; b2=bb[j+1]; b3=bb[j+2]; b4=bb[j+3];
		cc[0]+=a1*b1*d;
		cc[1]+=a2*b1*d;
		cc[2]+=a3*b1*d;
		cc[3]+=a4*b1*d;
		cc[4]+=a1*b2*d;
		cc[5]+=a2*b2*d;
		cc[6]+=a3*b2*d;
		cc[7]+=a4*b2*d;
		cc[8]+=a1*b3*d;
		cc[9]+=a2*b3*d;
		cc[10]+=a3*b3*d;
		cc[11]+=a4*b3*d;
		cc[12]+=a1*b4*d;
		cc[13]+=a2*b4*d;
		cc[14]+=a3*b4*d;
		cc[15]+=a4*b4*d;
	}
}

/* weighted on-diagonal block version of the above */
static inline void matopc_blk_4x4_cpaatw(
				matopc_real *cc,
				matopc_real *aa,
				matopc_real *w,
				int q,int which) {
	matopc_real d,a1,a2,a3,a4;
	int i,j;
	if (which==MATOPC_UPPER) {
		for (i=0,j=0;i<q;i++,j+=4) { /* upper part of 4x4 block only */
			d=w[i]; a1=aa[j]; a2=aa[j+1]; a3=aa[j+2]; a4=aa[j+3];
			cc[0]+=a1*a1*d;
			cc[4]+=a1*a2*d;
			cc[5]+=a2*a2*d;
			cc[8]+=a1*a3*d;
			cc[9]+=a2*a3*d;
			cc[10]+=a3*a3*d;
			cc[12]+=a1*a4*d;
			cc[13]+=a2*a4*d;
			cc[14]+=a3*a4*d;
			cc[15]+=a4*a4*d;
		}
	} else {
		for (i=0,j=0;i<q;i++,j+=4) { /* lower part of 4x4 block only */
			d=w[i]; a1=aa[j]; a2=aa[j+1]; a3=aa[j+2]; a4=aa[j+3];
			cc[0]+=a1*a1*d;
			cc[1]+=a2*a1*d;
			cc[2]+=a3*a1*d;
			cc[3]+=a4*a1*d;
			cc[5]+=a2*a2*d;
			cc[6]+=a3*a2*d;
			cc[7]+=a4*a2*d;
			cc[10]+=a3*a3*d;
			cc[11]+=a4*a3*d;
			cc[15]+=a4*a4*d;
		}
	}
}

#define MAX_BLK_PANEL_LENGTH 256
static double __aa[4*MAX_BLK_PANEL_LENGTH];
static double __bb[4*MAX_BLK_PANEL_LENGTH];
static double __cc[4*4];

/* Fourth version uses fixed-size 4 panels and unrolled inner loop and zero-padding
 * if needed. This one is typically faster than a naive triple loop for general usage.
 */
static inline void matopc_blk_atb_4x4__(
				matopc_real *C,int cstrC,
				matopc_real *A,int cstrA,int m,
				matopc_real *B,int cstrB,int n,
				int q,int op) {
	if (q>MAX_BLK_PANEL_LENGTH) return; /* emergency protection */
	int i=0,j,k,di,dj,ntgi,ntgj;
	while (i<m) {
		ntgi=m-i;
		if (ntgi>=4) di=4; else di=ntgi;
		matopc_blk_pack4x1(__aa,q,&A[i*cstrA],cstrA,di);
		j=0;
		while (j<n) {
			ntgj=n-j;
			if (ntgj>=4) dj=4; else dj=ntgj;
			matopc_blk_pack4x1(__bb,q,&B[j*cstrB],cstrB,dj);
			for (k=0;k<4*4;k++) __cc[k]=0.0;
			matopc_blk_4x4_cpabt(__cc,__aa,__bb,q);
			matopc_blk_4x4_op(&C[j*cstrC+i],cstrC,__cc,di,dj,op);
			j+=dj;
		}
		i+=di;
	}
}

/* user interface; handles larger mults with several passes if needed */
static inline void matopc_blk_atb_4x4(
				matopc_real *C,int cstrC,
				matopc_real *A,int cstrA,int m,
				matopc_real *B,int cstrB,int n,
				int q,int op) {
	int ntgr,r=0,dr;
	if (q>MAX_BLK_PANEL_LENGTH) {
		while (r<q) {
			ntgr=q-r;
			if (ntgr>MAX_BLK_PANEL_LENGTH) dr=MAX_BLK_PANEL_LENGTH; else dr=ntgr;
			matopc_blk_atb_4x4__(C,cstrC,&A[r],cstrA,m,&B[r],cstrB,n,dr,op);
			if (op==0 && r==0) op=1; /* if op is assign then assign on first pass; and subsequent must accumulate */
			r+=dr; /* bitwise equivalence to triple loop is lost by doing this split though */
		}
	} else {
		matopc_blk_atb_4x4__(C,cstrC,A,cstrA,m,B,cstrB,n,q,op);
	}
}

/* C <- C+A'*A or C<-C-A'*A or C<-A'*A
 * A is q-by-m so C is m-by-m, op = {<0,0,>0} for downdate/assign/update
 * if q is larger than the static panel buffer size then
 * several passes over C are needed.
 */
static inline void matopc_blk_ata_4x4__(
				matopc_real *C,int cstrC,
				matopc_real *A,int cstrA,int m,
				int q,int op,int which) {
	if (q>MAX_BLK_PANEL_LENGTH) return; /* emergency protection */
	int i=0,j,k,di,dj,ntgi,ntgj;
	while (i<m) {
		ntgi=m-i;
		if (ntgi>=4) di=4; else di=ntgi;
		matopc_blk_pack4x1(__aa,q,&A[i*cstrA],cstrA,di);
		if (which==MATOPC_LOWER) {
			j=0;
			while (j<=i) { /* lower triangle */
				for (k=0;k<4*4;k++) __cc[k]=0.0;
				ntgj=i-j;
				if (ntgj>=4) dj=4; else dj=ntgj;
				if (i==j) {	/* symmetric block on diagonal */
					dj=di;
					matopc_blk_4x4_cpaat(__cc,__aa,q,MATOPC_LOWER);
					matopc_blk_4x4_sym_op(&C[j*cstrC+i],cstrC,__cc,di,MATOPC_LOWER,op);
				} else { /* off-diagonal */
					matopc_blk_pack4x1(__bb,q,&A[j*cstrA],cstrA,dj);
					matopc_blk_4x4_cpabt(__cc,__aa,__bb,q);
					matopc_blk_4x4_op(&C[j*cstrC+i],cstrC,__cc,di,dj,op);
				}
				j+=dj;
			}
		} else {
			j=i;
			while (j<m) {	/* upper triangle */
				for (k=0;k<4*4;k++) __cc[k]=0.0;
				ntgj=m-j;
				if (ntgj>=4) dj=4; else dj=ntgj;
				if (i==j) {	/* symmetric block on diagonal */
					matopc_blk_4x4_cpaat(__cc,__aa,q,MATOPC_UPPER);
					matopc_blk_4x4_sym_op(&C[j*cstrC+i],cstrC,__cc,di,MATOPC_UPPER,op);
				} else { /* off-diagonal */
					matopc_blk_pack4x1(__bb,q,&A[j*cstrA],cstrA,dj);
					matopc_blk_4x4_cpabt(__cc,__aa,__bb,q);
					matopc_blk_4x4_op(&C[j*cstrC+i],cstrC,__cc,di,dj,op);
				}
				j+=dj;
			}
		}
		i+=di;
	}
}

/* user interface; handles larger mults with several passes if needed;
 * same template as atb function above.
 */
static inline void matopc_blk_ata_4x4(
				matopc_real *C,int cstrC,
				matopc_real *A,int cstrA,int m,
				int q,int op,int which) {
	int ntgr,r=0,dr;
	if (q>MAX_BLK_PANEL_LENGTH) {
		while (r<q) {
			ntgr=q-r;
			if (ntgr>MAX_BLK_PANEL_LENGTH) dr=MAX_BLK_PANEL_LENGTH; else dr=ntgr;
			matopc_blk_ata_4x4__(C,cstrC,&A[r],cstrA,m,dr,op,which);
			if (op==0 && r==0) op=1;
			r+=dr;
		}
	} else { /* small: fits in single call */
		matopc_blk_ata_4x4__(C,cstrC,A,cstrA,m,q,op,which);
	}
}

/* C <- (C+,C-,0+) A'*diag(W)*A routine
 * Almost exactly like the A'*A routine but with one extra argument W.
 * Uses the inner 4x4 update loop including weights W.
 */
static inline void matopc_blk_atwa_4x4__(
				matopc_real *C,int cstrC,
				matopc_real *A,int cstrA,int m,
				int q,matopc_real *W,int op,int which) {
	if (q>MAX_BLK_PANEL_LENGTH) return; /* emergency protection */
	int i=0,j,k,di,dj,ntgi,ntgj;
	while (i<m) {
		ntgi=m-i;
		if (ntgi>=4) di=4; else di=ntgi;
		matopc_blk_pack4x1(__aa,q,&A[i*cstrA],cstrA,di);
		if (which==MATOPC_LOWER) {
			j=0;
			while (j<=i) { /* lower triangle */
				for (k=0;k<4*4;k++) __cc[k]=0.0;
				ntgj=i-j;
				if (ntgj>=4) dj=4; else dj=ntgj;
				if (i==j) {	/* symmetric block on diagonal */
					dj=di;
					matopc_blk_4x4_cpaatw(__cc,__aa,W,q,MATOPC_LOWER);
					matopc_blk_4x4_sym_op(&C[j*cstrC+i],cstrC,__cc,di,MATOPC_LOWER,op);
				} else { /* off-diagonal */
					matopc_blk_pack4x1(__bb,q,&A[j*cstrA],cstrA,dj);
					matopc_blk_4x4_cpabtw(__cc,__aa,__bb,W,q);
					matopc_blk_4x4_op(&C[j*cstrC+i],cstrC,__cc,di,dj,op);
				}
				j+=dj;
			}
		} else {
			j=i;
			while (j<m) {	/* upper triangle */
				for (k=0;k<4*4;k++) __cc[k]=0.0;
				ntgj=m-j;
				if (ntgj>=4) dj=4; else dj=ntgj;
				if (i==j) {	/* symmetric block on diagonal */
					matopc_blk_4x4_cpaatw(__cc,__aa,W,q,MATOPC_UPPER);
					matopc_blk_4x4_sym_op(&C[j*cstrC+i],cstrC,__cc,di,MATOPC_UPPER,op);
				} else { /* off-diagonal */
					matopc_blk_pack4x1(__bb,q,&A[j*cstrA],cstrA,dj);
					matopc_blk_4x4_cpabtw(__cc,__aa,__bb,W,q);
					matopc_blk_4x4_op(&C[j*cstrC+i],cstrC,__cc,di,dj,op);
				}
				j+=dj;
			}
		}
		i+=di;
	}
}

/* User interface for A'*diag(W)*A update/downdate/assign 
 * It is only a little bit speedier than the naive triple loop apparently.
 */
static inline void matopc_blk_atwa_4x4(
				matopc_real *C,int cstrC,
				matopc_real *A,int cstrA,int m,
				int q,matopc_real *W,int op,int which) {
	int ntgr,r=0,dr;
	if (q>MAX_BLK_PANEL_LENGTH) {
		while (r<q) {
			ntgr=q-r;
			if (ntgr>MAX_BLK_PANEL_LENGTH) dr=MAX_BLK_PANEL_LENGTH; else dr=ntgr;
			matopc_blk_atwa_4x4__(C,cstrC,&A[r],cstrA,m,dr,&W[r],op,which);
			if (op==0 && r==0) op=1;
			r+=dr;
		}
	} else { /* small: fits in single call */
		matopc_blk_atwa_4x4__(C,cstrC,A,cstrA,m,q,W,op,which);
	}
}

/* Basic compressed column/row storage (CCS/CRS) routines follow below.
 * These are useful for exploitation of sparsity structures that are used
 * repeatedly (even though the actual matrix may still be stored densely).
 *
 * NOTE: CCS and CRS on the same matrix is redundant in data but can
 * allow quicker access to elements.
 *
 * The CCS of a sparse A will make A'*(.) be efficient since row indices
 * are easily accessed as a function of column indices. This class of
 * operations include A'*A.
 * The CRS of a sparse A will make A*(.) be efficient since column 
 * indices for the nonzeros are accessed as a function of row indices.
 */

#ifdef __MATRIXOPSC_SPARSE_TESTDATA__
/* test matrix; nnz=19  */
static matopc_real matopc_ccs_testmatrix[]=
{
    10,3,0,3,0,0,
    0, 9,7,0,8,4,
    0, 0,8,8,0,0,
    0, 0,7,7,9,0,
    -2,0,0,5,9,2,
    0, 3,0,0,13,-1
};
#endif

/* Create three CCS arrays:
 *      - colptr must have room for n+1 elements
 *      - val must have room for nnz elements
 *      - rowind must have room for nnz elements
 * colptr[n] will equal nnz after the call
 * for j=0..n-1, if colptr[j]==colptr[j+1] then col j is empty
 * otherwise colptr[j]<=k<colptr[j+1] for indices k for col j
 * where indices k are into val and rowind arrays.
 */
static inline void matopc_create_ccs(
        matopc_real *A,int m,int n,
        matopc_real *val,int *rowind,int *colptr) {
    int c,r,e,k;
    matopc_real v;
    e=0; k=0;
    for (c=0;c<n;c++) {
        colptr[c]=k;
        for (r=0;r<m;r++) {
            v=A[e++];
            if (v!=0.0) {
                val[k]=v;
                rowind[k]=r;
                k++;
            }
        }
    }
    colptr[c]=k;
}

/* Same as above but row-based (NOTE: matrix A is still column major)
 *      - colptr must have room for m+1 elements
 *      - val must have room for nnz elements
 *      - rowind must have room for nnz elements
 * rowptr[m] will equal nnz after the call
 * for j=0..n-1, if rowptr[i]==rowptr[i+1] then row i is empty
 * otherwise rowptr[i]<=k<rowptr[i+1] for indices k for row i
 * where indices k are into val and colind arrays.
 */
static inline void matopc_create_crs(
        matopc_real *A,int m,int n,
        matopc_real *val,int *colind,int *rowptr) {
    int c,r,e,k;
    matopc_real v;
    k=0;
    for (r=0;r<m;r++) {
        rowptr[r]=k; e=r;
        for (c=0;c<n;c++) {
            v=A[e]; e+=m;
            if (v!=0.0) {
                val[k]=v;
                colind[k]=c;
                k++;
            }
        }
    }
    rowptr[r]=k;
}

/* Create CRS index/array from an existing CCS index/array.
 * May be more efficient since the zero elements are already discarded.
 * Hence no need to re-traverse the full matrix again.
 * It is assumed that the arrays are pre-allocated (and large enough).
 */
static inline void matopc_ccs2crs(
        int m,int n,
        matopc_real *ccsval,int *rowind,int *colptr,
        matopc_real *crsval,int *colind,int *rowptr) {
    /*int nnz=colptr[n];*/
    int c,r,e,k;
    k=0;
    for (r=0;r<m;r++) { /* for every row */
        rowptr[r]=k;
        for (c=0;c<n;c++) { /* search along the column index for nonzeros */
            for (e=colptr[c];e<colptr[c+1];e++) {
                if (rowind[e]==r) {
                    crsval[k]=ccsval[e];
                    colind[k]=c;
                    k++;
                    break; /* e-loop early exit */
                }
            }
        }
    }
    rowptr[r]=k;
}

/* Basic CRS matrix-vector multiplication.
 * y <- a*y+b*A*x, where the CRS representation of A m-by-n is provided.
 */
static inline void matopc_crs_ypax(
        matopc_real *y,matopc_real a,matopc_real b,
        int m,int n,
        matopc_real *crsval,int *colind,int *rowptr,
        matopc_real *x) {
	matopc_real s;
	int r,k;
	if (a==0.0 && b==0.0) {
		matopc_zeros(y,m,1);
	} else if (a==0.0 && b==1.0) {	/* common case y <- A*x */
		for (r=0;r<m;r++) {
            if (rowptr[r]!=rowptr[r+1]) {
                s=0.0;
                for (k=rowptr[r];k<rowptr[r+1];k++)
                    s+=crsval[k]*x[colind[k]];
                y[r]=s;
            } else {
                y[r]=0.0;
            }
        }
	} else if (a==1.0 && b==0.0) {
		/* copy y <- y (do nothing here) */
	} else if (a==1.0 && b==1.0) {	/* accumulation y <- y+A*x */
		for (r=0;r<m;r++) {
            if (rowptr[r]!=rowptr[r+1]) {
                s=0.0;
                for (k=rowptr[r];k<rowptr[r+1];k++)
                    s+=crsval[k]*x[colind[k]];
                y[r]+=s;
            }
        }
	} else {	/* general case y <- a*y+b*A*x */
		for (r=0;r<m;r++) {
            if (rowptr[r]!=rowptr[r+1]) {
                s=0.0;
                for (k=rowptr[r];k<rowptr[r+1];k++)
                    s+=crsval[k]*x[colind[k]];
                y[r]=a*y[r]+b*s;
            } else { /* row is empty A*x; does not contribute here */
                y[r]*=a;
            }
        }
	}
}

/* Basic CCS transposed-matrix-vector multiplication.
 * y <- a*y+b*A'*x, where the CCS representation of A m-by-n is provided.
 * So y is n-by-1 and x is m-by-1; a,b scalars
 */
static inline void matopc_ccs_ypatx(
        matopc_real *y,matopc_real a,matopc_real b,
        int m,int n,
        matopc_real *ccsval,int *rowind,int *colptr,
        matopc_real *x) {
	matopc_real s;
	int c,k;
	if (a==0.0 && b==0.0) {
		matopc_zeros(y,n,1);
	} else if (a==0.0 && b==1.0) {	/* common case y <- A*x */
        for (c=0;c<n;c++) {
            s=0.0;
            if (colptr[c]!=colptr[c+1]) {
                for (k=colptr[c];k<colptr[c+1];k++)
                    s+=ccsval[k]*x[rowind[k]];
            }
            y[c]=s;
        }
	} else if (a==1.0 && b==0.0) {
		/* copy y <- y (do nothing here) */
	} else if (a==1.0 && b==1.0) {	/* accumulation y <- y+A*x */
        for (c=0;c<n;c++) {
            if (colptr[c]!=colptr[c+1]) {
                s=0.0;
                for (k=colptr[c];k<colptr[c+1];k++)
                    s+=ccsval[k]*x[rowind[k]];
                y[c]+=s;
            }
        }
	} else {	/* general case y <- a*y+b*A*x */
		for (c=0;c<n;c++) {
            if (colptr[c]!=colptr[c+1]) {
                s=0.0;
                for (k=colptr[c];k<colptr[c+1];k++)
                    s+=ccsval[k]*x[rowind[k]];
                y[c]=a*y[c]+b*s;
            } else { /* row is empty A*x; does not contribute here */
                y[c]*=a;
            }
        }
	}
}

/* Operation L*X=B' where B is m-by-n (B' is n-by-m)
 * and A,p contains the Cholesky factor L: X=L\B'.
 * The CRS of B is used; and the triangular solve for
 * each row-to-column is skipping any initial zeros
 * for the rows of B; which shortens the computation.
 */
static inline void matopc_crs_cholesky_trisubst_left_tr_matrix(
        matopc_real *A,matopc_real *p,int n,
        matopc_real *X,int m,
        matopc_real *crsval,int *colind,int *rowptr,
        matopc_real *tmp) {
	int i,j,ofs,nj;
	if (tmp==NULL) tmp=matopc_temparray;
	ofs=0;
	for (j=0;j<m;j++) {
		nj=rowptr[j+1]-rowptr[j];
		if (nj==0) {
			for (i=0;i<n;i++) X[ofs+i]=0.0;
		} else {
			for (i=0;i<n;i++) tmp[i]=0.0;
			for (i=rowptr[j];i<rowptr[j+1];i++) tmp[colind[i]]=crsval[i];
			matopc_cholesky_trisubst_left_skip(A,p,n,tmp,tmp,colind[rowptr[j]]);
			for (i=0;i<n;i++) X[ofs+i]=tmp[i];
		}
		ofs+=n;
	}
}

/* Aux. function to intersect two sorted (ascending) integer sets.
 * None of the input sets can be empty. If the intersection is empty
 * zero will be written to *ndest and *dest will be untouched.
 * Input sets cannot have duplicate entries.
 */
static inline void matopc_intersect_sorted_sets(
        int *set1,int n1,int *set2,int n2,
        int *dest,int *ndest) {
    int i1=0,i2=0,i3=0;
    int e1,e2;
    while (i1<n1 && i2<n2) {
        e1=set1[i1];
        e2=set2[i2];
        if (e1==e2) {
            dest[i3++]=e1; i1++; i2++;
        } else if (e1<e2) {
            i1++;
        } else {
            i2++;
        }
    }
    *ndest=i3;
}

/* Similar to the above function but accumulates a weighted intersection 
 * but does not return any intersected indices. It does provide the number
 * of terms that was added up.
 */
static inline matopc_real matopc_accum_wghtd_intrsct(
        int *set1,int n1,matopc_real *val1,
        int *set2,int n2,matopc_real *val2,
        matopc_real *w,int *nsum) {
    matopc_real s=0.0;
    int i1=0,i2=0,i3=0;
    int e1,e2;
    while (i1<n1 && i2<n2) {
        e1=set1[i1];
        e2=set2[i2];
        if (e1==e2) {
            s+=val1[i1]*val2[i2]*w[e1];
            i1++; i2++; i3++;
        } else if (e1<e2) {
            i1++;
        } else {
            i2++;
        }
    }
    *nsum=i3;
    return s;
}

/* Symmetric update C <- c*C+A'*diag(d)*A using the CCS representation of A.
 * A is m-by-n so d is of length m and C is n-by-n; c is a scalar. Code
 * branches by checking the cases c=0, c=1, and general c.
 * Only updates the upper triangle (incl. diagonal) of C;
 * and it is assumed that C is dense.
 */
static inline void matopc_ccs_cpatda(
        matopc_real *C,matopc_real c,int m,int n,
        matopc_real *ccsval,int *rowind,int *colptr,
        matopc_real *d) {
    int i,j,ni,nj,nx,idx;
    matopc_real sx;
    if (c==0.0) { /* C <- A'*diag(d)*A */
        for (i=0;i<n;i++) {
            ni=colptr[i+1]-colptr[i];
            if (ni!=0) {
                for (j=i;j<n;j++) {
                    idx=n*j+i;
                    nj=colptr[j+1]-colptr[j];
                    if (nj!=0) {
                        /* columns i and j of A happen to be nonzero both; there might be an intersection */
                        sx=matopc_accum_wghtd_intrsct(
                                &rowind[colptr[i]],ni,&ccsval[colptr[i]],
                                &rowind[colptr[j]],nj,&ccsval[colptr[j]],
                                d,&nx);
                        C[idx]=sx; /* no intersection: need to write zero anyway */
                    } else {
                        C[idx]=0.0;
                    }
                }
            } else {
                /* entire row i of A'*D*A will be zero; due to zero column i of A */
                for (j=i;j<n;j++) C[n*j+i]=0.0;
            }
        }
    } else if (c==1.0) { /* accumulate C <- C+A'*diag(d)*A */
        for (i=0;i<n;i++) {
            ni=colptr[i+1]-colptr[i];
            if (ni!=0) {
                for (j=i;j<n;j++) {
                    idx=n*j+i;
                    nj=colptr[j+1]-colptr[j];
                    if (nj!=0) {
                        sx=matopc_accum_wghtd_intrsct(
                                &rowind[colptr[i]],ni,&ccsval[colptr[i]],
                                &rowind[colptr[j]],nj,&ccsval[colptr[j]],
                                d,&nx);
                        if (nx>0) {
                            C[idx]+=sx;
                        }
                    }
                }
            }
        }
    } else { /* generic c: C <- c*C+A'*diag(d)*A */
        for (i=0;i<n;i++) {
            ni=colptr[i+1]-colptr[i];
            if (ni!=0) {
                for (j=i;j<n;j++) {
                    idx=n*j+i;
                    nj=colptr[j+1]-colptr[j];
                    if (nj!=0) {
                        /* columns i and j of A happen to be nonzero both; there might be an intersection */
                        sx=matopc_accum_wghtd_intrsct(
                                &rowind[colptr[i]],ni,&ccsval[colptr[i]],
                                &rowind[colptr[j]],nj,&ccsval[colptr[j]],
                                d,&nx);
                        if (nx>0) {
                            C[idx]=c*C[idx]+sx;
                        } else {
                            C[idx]*=c;
                        }
                    } else {
                        /* column j of A is zero so (i,j) gets nothing from A'*D*A */
                        C[idx]*=c;
                    }
                }
            } else {
                /* entire row i of A'*D*A will be zero; due to zero column i of A */
                for (j=i;j<n;j++) C[n*j+i]*=c;
            }
        }
    }
}

/* Special version of the previous routine.
 * Only updates lower triangle of C (inc. diagonal).
 * Assumes c=1.0 (so c is not an argument): C <- C + A'*diag(d)*A
 */
static inline void matopc_ccs_cpatda_lo(
        matopc_real *C,int m,int n,
        matopc_real *ccsval,int *rowind,int *colptr,
        matopc_real *d) {
	int i,j,ni,nj,nx,idx;
	matopc_real sx;
	for (i=0;i<n;i++) {
		ni=colptr[i+1]-colptr[i];
		if (ni!=0) {
			for (j=0;j<=i;j++) {
				idx=n*j+i;
        nj=colptr[j+1]-colptr[j];
        if (nj!=0) {
					sx=matopc_accum_wghtd_intrsct(
          			&rowind[colptr[i]],ni,&ccsval[colptr[i]],
                &rowind[colptr[j]],nj,&ccsval[colptr[j]],
                d,&nx);
          if (nx>0) {
          	C[idx]+=sx;
          }
        }
      }
    }
  }
}

/* Create/precompute arrays that enable efficient matrix evaluation
 * of the type A'*D*A where D is diagonal, based on the CCS form of A.
 * where A is m-by-n: (rowind, colptr). Note that m is never referenced
 * so is not part of the arguments. The product is n-by-n; and all
 * summation indices will be bounded by m but this is hidden.
 */
static inline int matopc_create_atda_index(
        int n,int *rowind,int *colptr,
        int *pr,int *pc,int *pe,int *rcx) {
    int r,c,nc,nr,nrc;
    int nume=0;
    int nx=0;   /* total number of intersected row indices */
    for (c=0;c<n;c++) {
        nc=colptr[c+1]-colptr[c];
        if (nc!=0) { /* skip entire scan over rows if column is empty */
            for (r=0;r<=c;r++) {
                nr=colptr[r+1]-colptr[r];
                if (nr!=0) {
                    /* intersect two nonzero columns; result may be nonzero */
                    matopc_intersect_sorted_sets(
                            &rowind[colptr[c]],nc,
                            &rowind[colptr[r]],nr,
                            &rcx[nx],&nrc);
                    if (nrc!=0) { /* nonzero intersection */
                        pr[nume]=r;
                        pc[nume]=c;
                        pe[nume]=nx; /* start index into rcrows */
                        nx+=nrc;     /* end index stored in next slot */ 
                        nume++;
                    }
                }
            }
        }
    }
    pe[nume]=nx;    /* end index for last nonzero */
    return nume;    /* number of possible nonzeros in A'*D*A */
}

/* M += A'*diag(d)*A using the pre-index created by the above function
 * NOTE: only updates the upper triangle of M.
 */
static inline void matopc_indexed_update_atda(
        matopc_real *M,matopc_real *A,matopc_real *d,int m, int n,
        int ne,int *pr,int *pc,int *pe,int *rcx) {
    int ll,qq,r,c,dst,kk;
    matopc_real *pi,*pj;
    matopc_real s;
    for (ll=0;ll<ne;ll++) {
        s=0.0; r=pr[ll]; c=pc[ll];
        pi=&A[m*r]; pj=&A[m*c];
        for (qq=pe[ll];qq<pe[ll+1];qq++) {
            kk=rcx[qq];
            s+=pi[kk]*pj[kk]*d[kk]; /* pay attention to the ordering here! */
        }
        dst=n*c+r;
        M[dst]+=s;
        /*if (r!=c) {
            dst=n*r+c;
            M[dst]+=s;    
        }*/
    }
}

/* TODO: create a linearized index given m also?
 * If A is constant and only d changes the A terms can also be
 * pre-multiplied once and reused..
 */

#endif
