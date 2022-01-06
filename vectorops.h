/*
 * vectorops.h
 *
 * Assortment of utility functions for operations with vectors
 * (arrays of doubles)
 *
 *
 * Protocol: all utility functions have the prefix vecop_
 *           the first two arguments are
 *           (1) destination pointer (if applicable)
 *           (2) vector length
 *
 */

#ifndef __VECTOROPS_H__
#define __VECTOROPS_H__

#ifndef __VECTOROPS_REAL_TYPE__
typedef double vecop_real;
#endif

/* set all components of dst to zero. */
static inline void vecop_zeros(vecop_real *dst,int n) {
	memset(dst,0,n*sizeof(vecop_real));
}

/* set all components of dst to one. */
static inline void vecop_ones(vecop_real *dst,int n) {
	int i; for (i=0;i<n;i++) dst[i]=1.0;
}

/* set all components of dst to a scalar s */
static inline void vecop_assign(vecop_real *dst,int n,vecop_real s) {
	int i; for (i=0;i<n;i++) dst[i]=s;
}

/* make dst=src */
static inline void vecop_copy(vecop_real *dst,int n,vecop_real *src) {
	if (dst!=src) memcpy(dst,src,sizeof(vecop_real)*n);
}

/* make dst=-src, dst=src is OK. */
static inline void vecop_neg_copy(vecop_real *dst,int n,vecop_real *src) {
	int i; for (i=0;i<n;i++) dst[i]=-src[i];
}

#ifdef __MT19937AR_H__
/* Define a randomized initialization routine if MT is defined */
static inline void vecop_rand(vecop_real *dst,int n,vecop_real a,vecop_real b) {
	int i; for (i=0;i<n;i++) dst[i]=(vecop_real)(a+(b-a)*genrand_real3());
}
#endif

/* dst=x+y */
static inline void vecop_add(vecop_real *dst,int n,vecop_real *x,vecop_real *y) {
	int i; for (i=0;i<n;i++) dst[i]=x[i]+y[i];
}

/* dst+=x */
static inline void vecop_addx(vecop_real *dst,int n,vecop_real *x) {
	int i; for (i=0;i<n;i++) dst[i]+=x[i];
}

/* add the same scalar "a" to each element of dst */
static inline void vecop_adda(vecop_real *dst,int n,vecop_real a) {
	int i; for (i=0;i<n;i++) dst[i]+=a;
}

/* dst+=a*x, a scalar */
static inline void vecop_addax(vecop_real *dst,int n,vecop_real a,vecop_real *x) {
	int i; for (i=0;i<n;i++) dst[i]+=a*x[i];
}

/* dst=x-y */
static inline void vecop_sub(vecop_real *dst,int n,vecop_real *x,vecop_real *y) {
	int i; for (i=0;i<n;i++) dst[i]=x[i]-y[i];
}

/* dst-=x */
static inline void vecop_subx(vecop_real *dst,int n,vecop_real *x) {
	int i; for (i=0;i<n;i++) dst[i]-=x[i];
}

/* dst=x.*y, elementwise multiplication */
static inline void vecop_mul(vecop_real *dst,int n,vecop_real *x,vecop_real *y) {
	int i; for (i=0;i<n;i++) dst[i]=x[i]*y[i];
}

/* dst*=x */
static inline void vecop_mulx(vecop_real *dst,int n,vecop_real *x) {
	int i; for (i=0;i<n;i++) dst[i]*=x[i];
}

/* dst=x./y, elementwise division */
static inline void vecop_div(vecop_real *dst,int n,vecop_real *x,vecop_real *y) {
	int i; for (i=0;i<n;i++) dst[i]=x[i]/y[i];
}

/* Makes the multiply-accumulate (macc) update operation dst=a*dst+b*x for scalar a and b, and vector x */
static inline void vecop_macc(vecop_real *dst,int n,vecop_real a,vecop_real *x,vecop_real b) {
	int i;
	if (a==0.0) {
		if (b==0.0) {
			vecop_zeros(dst,n);
		} else if (b==1.0) { /* dst=x; just copy */
			vecop_copy(dst,n,x);
		} else { /* dst=b*x */
			for (i=0;i<n;i++) dst[i]=b*x[i];
		}
	} else if (a==1.0) {
		if (b==0.0) { /* dst=dst, nothing to do ... */
		} else if (b==1.0) { /* dst=dst+x; just add x */
			for (i=0;i<n;i++) dst[i]+=x[i];
		} else { /* dst=dst+b*x */
			for (i=0;i<n;i++) dst[i]+=b*x[i];
		}	
	} else { /* arbitrary a, dst=a*dst+b*x */
		if (b==0.0) {
			for (i=0;i<n;i++) dst[i]*=a;
		} else if (b==1.0) {
			for (i=0;i<n;i++) dst[i]=a*dst[i]+x[i];
		} else {
			for (i=0;i<n;i++) dst[i]=a*dst[i]+b*x[i];
		}
	}
}

/* Makes the multiply-accumulate (macc) update operation dst=a*dst+b.*x for scalar a and vectors b,x */
static inline void vecop_maccv(vecop_real *dst,int n,vecop_real a,vecop_real *x,vecop_real *b) {
	int i;
	if (a==0.0) {	/* dst=b.*x */
		for (i=0;i<n;i++) dst[i]=x[i]*b[i];
	} else if (a==1.0) {	/* dst=dst+b.*x */
		for (i=0;i<n;i++) dst[i]+=x[i]*b[i];
	} else {	/* dst=a*dst+b.*x, general case */
		for (i=0;i<n;i++) dst[i]=a*dst[i]+x[i]*b[i];
	}
}

/* Makes the multiply-accumulate (macc) update operation dst=a.*dst+b.*x for vectors a,b,x */
static inline void vecop_maccvv(vecop_real *dst,int n,vecop_real *a,vecop_real *x,vecop_real *b) {
	int i; for (i=0;i<n;i++) dst[i]=a[i]*dst[i]+x[i]*b[i];
}

/* sum all elements; return sum(x) */
static inline vecop_real vecop_sum(int n,vecop_real *x) {
	vecop_real s=0.0; int i; for (i=0;i<n;i++) s+=x[i];
    return s;
}

/* scalar product x'*y=sum(x.*y); x,y column vectors */
static inline vecop_real vecop_dot(int n,vecop_real *x,vecop_real *y) {
	vecop_real dot=0.0; int i; for (i=0;i<n;i++) dot+=x[i]*y[i];
	return dot;
}

/* norm(typ) of a vector x of length n; typ=2 (2-norm, default) 1 (1-norm) 0 (inf-norm) */
static inline vecop_real vecop_norm(vecop_real *x,int n,int typ) {
	vecop_real sum=0.0; int i;
	if (typ==0) {	/* inf-norm = max_i(|x[i]|) */
		for (i=0;i<n;i++) if (fabs(x[i])>sum) sum=fabs(x[i]);
		return sum;
	} else if (typ==1){		/* 1-norm */
		for (i=0;i<n;i++) sum+=fabs(x[i]);
		return sum;
	} else {	/* 2-norm is default */
		for (i=0;i<n;i++) sum+=x[i]*x[i];
		return (vecop_real)sqrt(sum);
	}
}

/* maximum absolute elementwise difference between two vectors u,v */
static inline vecop_real vecop_max_abs_diff(vecop_real *u,vecop_real *v,int n) {
    vecop_real d=0.0,tmp; int i;
    for (i=0;i<n;i++) {
        tmp=fabs(u[i]-v[i]);
        if (tmp>d) d=tmp;
    }
    return tmp;
}

#endif
