/*
  Remake of qpmpclti2f.c MEX source code as Python C extension.
*/

// FIXME: basic check of solver scaling with n for tripleint system in test.py
//        set a "complete" docstring (from 2e matlab program header)
//        develop better Python test programs (masses, F-16)

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h"

#undef __DEVELOPMENT_TEXT_OUTPUT__ 
#define __CLUMSY_ASSERTIONS__
#define __COMPILE_WITH_INTERNAL_TICTOC__ 

#include "vectorops.h"
#include "matrixopsc.h"

#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
#include "fastclock.h"
#endif

#include "multisolver.h"

#include <stdbool.h>

/*** UTILITIES ***/

// All dict keys below are required
typedef struct problemInputObjects {
  int n;         // "n",
  PyObject* A;   // "A",
  PyObject* B;   // "B",
  PyObject* C;   // "C",
  PyObject* D;   // "D",
  PyObject* Qx;  // "Qx",
  PyObject* W;   // "W",
  PyObject* R;   // "R",
  PyObject* F1;  // "F1",
  PyObject* F2;  // "F2",
  PyObject* f3;  // "f3",
  PyObject* w;   // "w",
  PyObject* r;   // "r",
  PyObject* x;   // "x",
  PyObject* Qxn; // "Qxn",
  PyObject* Wn;  // "Wn",
  PyObject* sc;  // "sc"
  int xreturn;
  int ureturn;
  int sreturn;
} problemInputObjects;

/* These types are used to decide the type of calculation needed 
 * when using the cost matrices W,R,Q,[Wn,Qn]
 */
#define TYP_UNDEF  -1
#define TYP_SCALAR  0
#define TYP_VECTOR  1
#define TYP_MATRIX  2
#define TYP_MATRIXT 3   /* transposed matrix type */

int aux_read_square_matrix(PyArrayObject* ao,
                           int n,
                           int *typ,
                           double *sclr,
                           double **ptr)
{
  const int M = PyArray_DIM(ao, 0);
  const int N = PyArray_DIM(ao, 1);
  int retval = 1;
  if (M == 1 && N == 1) {
    *typ = TYP_SCALAR; 
    *sclr = *((double *) PyArray_DATA(ao)); 
    *ptr = sclr;
  } else if ((M == n && N == 1) ||
             (M == 1 && N == n)) {
    /* interpret the n numbers as the diagonal in a matrix */
    *typ = TYP_VECTOR; 
    *sclr = -1;
    *ptr = (double *) PyArray_DATA(ao);
  } else if (M == n && N == n) {
    /* full matrix n-by-n */
    *typ = TYP_MATRIX; 
    *sclr = -1; 
    *ptr = (double *) PyArray_DATA(ao);
  } else {
    *typ = TYP_UNDEF; 
    *sclr = 0; 
    *ptr = NULL;
    retval = 0;
  }
  return retval;
}

int aux_read_signal_matrix(PyArrayObject* ao,
                           int n,
                           int nt,
                           int *typ,
                           double *sclr,
                           double **ptr) 
{
  const int M = PyArray_DIM(ao, 0);
  const int N = PyArray_DIM(ao, 1);
  int retval = 1;
  if (M == 1 && N == 1) {
    /* Check first if scalar */
    *typ = TYP_SCALAR; 
    *sclr = *((double *) PyArray_DATA(ao)); 
    *ptr = sclr;
  } else if ((M == n && N == 1) ||
             (M == 1 && N == n)) {
    /* n-vector, row or column; assumed to be constant over nt timesteps */
    *typ = TYP_VECTOR; 
    *sclr = -1; 
    *ptr = (double *) PyArray_DATA(ao);
  } else if (M == n && N == nt) {
    /* full matrix n-by-nt */
    *typ = TYP_MATRIX; 
    *sclr = -1; 
    *ptr = (double *) PyArray_DATA(ao);
  } else if (M == nt && N == n) {
    /* full matrix nt-by-n (transposed, non-transposed has precedence) */
    *typ = TYP_MATRIXT; 
    *sclr = -1; 
    *ptr = (double *) PyArray_DATA(ao);
  } else {
    *typ = TYP_UNDEF; 
    *sclr = 0; 
    *ptr = NULL;
    retval = 0;
  }
  return retval;
}

/*
 * Aux. sub-program to create symmetric cost-matrices.
 */

void aux_compute_sym_cost_matrix(
        int nd,int nx,int nu,int ny,double *pQstg,
        int typQ,double sQ,double *pQ,
        int typR,double sR,double *pR,
        int typW,double sW,double *pW,
        double *pC,double *pD)
{    
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

bool np_is_empty(PyArrayObject* a) {
  return (PyArray_SIZE(a) == 0);
}

bool np_is_scalar(PyArrayObject* a) {
  return (PyArray_SIZE(a) == 1);
}

bool np_is_matrix(PyArrayObject* a) {
  const int nrows = PyArray_DIM(a, 0);
  const int ncols = PyArray_DIM(a, 1);
  return (nrows > 1 && ncols > 1);
}

// assumes ndim == 2 confirmed already; true if scalar
bool np_is_vector(PyArrayObject* a, 
                  bool scalar_is_true)
{
  const int nrows = PyArray_DIM(a, 0);
  const int ncols = PyArray_DIM(a, 1);
  if (scalar_is_true && nrows == 1 && ncols == 1) return true;
  return (nrows == 1 && ncols > 1) || (nrows > 1 && ncols == 1);
}

// Return NULL if no key in dict, or it is there but not a numpy array.
// Optionally require a specific number of array dimensions.
// Optionally issue warning if keyname exists but is incompatible.
// Return array object as Fortran layout (possibly converted).
PyObject* 
get_numpy_array(PyObject* dict, 
                const char* keyname, 
                int requireDims,
                bool warnIfIncompatible)
{
  PyObject* item = PyDict_GetItemString(dict, keyname);
  if (item == NULL) return NULL;
  if (!PyArray_Check(item)) {
    if (warnIfIncompatible)
      printf("WARNING: key \"%s\" exists but is not a Numpy array. It will be ignored.\n", 
             keyname);
    return NULL;
  }
  if (requireDims > 0) {
    if (PyArray_NDIM((PyArrayObject *) item) != requireDims) {
      if (warnIfIncompatible)
        printf("WARNING: key \"%s\" exists and is a Numpy array but has ndim != %i. It will be ignored.\n", 
               keyname, 
               requireDims);
      return NULL;
    }
  }
  PyObject *arr = PyArray_FROM_OTF(item, 
                                   NPY_DOUBLE, 
                                   NPY_ARRAY_IN_ARRAY|NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_ALIGNED);
  return arr;
}

int get_integer_from_dict(PyObject* dict, 
                          const char* keyname,
                          bool warnIfIncompatible)
{
  PyObject* item = PyDict_GetItemString(dict, keyname);
  if (item == NULL) return 0;
  if (!PyLong_Check(item)) {
    if (warnIfIncompatible)
      printf("WARNING: key \"%s\" exists but is not a a Python integer. It will be ignored.\n", 
             keyname);
    return 0;
  }
  return ((int) PyLong_AsLong(item));
}

// grab numpy array object, requiring shape defined for 2 dimensions, and Fortran memory layout
void loadProblemInputs(PyObject* dict, 
                       problemInputObjects* pIO) 
{
  const bool warnIfIncompatible = true;  // This helps detecting input data mistakes
  const int requiredNdim = 2;

  memset(pIO, 0, sizeof(problemInputObjects));

  pIO->n       = get_integer_from_dict(dict, "n",       warnIfIncompatible);
  pIO->xreturn = get_integer_from_dict(dict, "xreturn", warnIfIncompatible);
  pIO->ureturn = get_integer_from_dict(dict, "ureturn", warnIfIncompatible);
  pIO->sreturn = get_integer_from_dict(dict, "sreturn", warnIfIncompatible);

  pIO->A   = get_numpy_array(dict, "A",   requiredNdim, warnIfIncompatible); 
  pIO->B   = get_numpy_array(dict, "B",   requiredNdim, warnIfIncompatible);
  pIO->C   = get_numpy_array(dict, "C",   requiredNdim, warnIfIncompatible);
  pIO->D   = get_numpy_array(dict, "D",   requiredNdim, warnIfIncompatible);
  pIO->Qx  = get_numpy_array(dict, "Qx",  requiredNdim, warnIfIncompatible);
  pIO->W   = get_numpy_array(dict, "W",   requiredNdim, warnIfIncompatible);
  pIO->R   = get_numpy_array(dict, "R",   requiredNdim, warnIfIncompatible);
  pIO->F1  = get_numpy_array(dict, "F1",  requiredNdim, warnIfIncompatible);
  pIO->F2  = get_numpy_array(dict, "F2",  requiredNdim, warnIfIncompatible);
  pIO->f3  = get_numpy_array(dict, "f3",  requiredNdim, warnIfIncompatible);
  pIO->w   = get_numpy_array(dict, "w",   requiredNdim, warnIfIncompatible);
  pIO->r   = get_numpy_array(dict, "r",   requiredNdim, warnIfIncompatible);
  pIO->x   = get_numpy_array(dict, "x",   requiredNdim, warnIfIncompatible);
  pIO->Qxn = get_numpy_array(dict, "Qxn", requiredNdim, warnIfIncompatible);
  pIO->Wn  = get_numpy_array(dict, "Wn",  requiredNdim, warnIfIncompatible);
  pIO->sc  = get_numpy_array(dict, "sc",  requiredNdim, warnIfIncompatible);
}

// All strict matrices (both m,n > 1) are required to have Fortran memory layout (if m or n is 1, it doesn't matter)
#define CHECK_FORTRAN(a) \
  if (a != NULL && np_is_matrix((PyArrayObject *) a) && !PyArray_ISFORTRAN((PyArrayObject *) a)) return false;

bool allFortranProblemInputs(const problemInputObjects* pIO) {
  CHECK_FORTRAN(pIO->A)
  CHECK_FORTRAN(pIO->B)
  CHECK_FORTRAN(pIO->C)
  CHECK_FORTRAN(pIO->D)
  CHECK_FORTRAN(pIO->Qx)
  CHECK_FORTRAN(pIO->W)
  CHECK_FORTRAN(pIO->R)
  CHECK_FORTRAN(pIO->F1)
  CHECK_FORTRAN(pIO->F2)
  CHECK_FORTRAN(pIO->f3)
  CHECK_FORTRAN(pIO->w)
  CHECK_FORTRAN(pIO->r)
  CHECK_FORTRAN(pIO->x)
  CHECK_FORTRAN(pIO->Qxn)
  CHECK_FORTRAN(pIO->Wn)
  CHECK_FORTRAN(pIO->sc)
  return true;
}

void offloadProblemInputs(problemInputObjects* pIO) {
  Py_XDECREF(pIO->A);
  Py_XDECREF(pIO->B);
  Py_XDECREF(pIO->C);
  Py_XDECREF(pIO->D);
  Py_XDECREF(pIO->Qx);
  Py_XDECREF(pIO->W);
  Py_XDECREF(pIO->R);
  Py_XDECREF(pIO->F1);
  Py_XDECREF(pIO->F2);
  Py_XDECREF(pIO->f3);
  Py_XDECREF(pIO->w);
  Py_XDECREF(pIO->r);
  Py_XDECREF(pIO->x);
  Py_XDECREF(pIO->Qxn);
  Py_XDECREF(pIO->Wn);
  Py_XDECREF(pIO->sc);
  memset(pIO, 0, sizeof(problemInputObjects));
}

void print_array_layout(PyArrayObject* a) {
  const int nelems = PyArray_SIZE(a);
  const double* e = PyArray_DATA(a);
  printf("layout: ");
  for (int i = 0; i < nelems; i++) {
    printf("%f ", e[i]);
  }
  printf("\n");
}

#define GRAB_DICT_INTEGER(strname, lhsname) \
  if ((o = PyDict_GetItemString(dict, strname)) != NULL) { \
    if (!PyLong_Check(o)) return false; \
    lhsname = (int) PyLong_AsLong(o); \
  } \

#define GRAB_DICT_DOUBLE(strname, lhsname) \
  if ((o = PyDict_GetItemString(dict, strname)) != NULL) { \
    if (!PyFloat_Check(o)) return false; \
    lhsname = PyFloat_AsDouble(o); \
  } \

// setup options struct based on Python dict (if dict is NULL, qpo will be default)
bool assign_options_struct_from_dict(qpoptStruct* qpo,
                                     PyObject* dict)
{
  setup_qpopt_defaults(qpo);
  if (dict == NULL) return true;
  PyObject* o = NULL;
  GRAB_DICT_INTEGER("verbosity", qpo->verbosity)
  GRAB_DICT_INTEGER("maxiters", qpo->maxiters)
  GRAB_DICT_INTEGER("refinement", qpo->refinement)
  GRAB_DICT_INTEGER("expl_sparse", qpo->expl_sparse)
  GRAB_DICT_INTEGER("chol_update", qpo->chol_update)
  GRAB_DICT_INTEGER("blas_suite", qpo->blas_suite)
  GRAB_DICT_DOUBLE("eta", qpo->eta)
  GRAB_DICT_DOUBLE("ep", qpo->ep)
  return true;
}

void print_options_struct(const qpoptStruct* qpo) {
  printf("verbosity   = %i\n", qpo->verbosity);
  printf("maxiters    = %i\n", qpo->maxiters);
  printf("eta         = %f\n", qpo->eta);
  printf("ep          = %f\n", qpo->ep);
  printf("refinement  = %i\n", qpo->refinement);
  printf("expl_sparse = %i\n", qpo->expl_sparse);
  printf("chol_update = %i\n", qpo->chol_update);
  printf("blas_suite  = %i\n", qpo->blas_suite);
}

// Assume arr has 2 dimensions; and contiguous; want to work with it as Fortran layout
void enforce_fortran_property(PyArrayObject* arr) {
  PyArray_CLEARFLAGS(arr, NPY_ARRAY_CARRAY);
  PyArray_ENABLEFLAGS(arr, NPY_ARRAY_FARRAY);
  const npy_intp sz = PyArray_ITEMSIZE(arr);
  npy_intp* strides = PyArray_STRIDES(arr);
  strides[0] = sz;
  strides[1] = sz * PyArray_DIM(arr, 0);
  return;
}

PyObject* createReturnDict(int retcode,
                           int problemClass,
                           const qpretStruct* qpRet,
                           int n,
                           int nu,
                           int numReturnStepsU,
                           int nx,
                           int numReturnStepsX,
                           int ns,
                           int numReturnStepsS,
                           const double* pS)
{
  const char *prblm_class_names[3] = {"qp-eq", "qp-eq-ineq", "qp-eq-ineq-slack"};
  PyObject* newDict = PyDict_New();
  if (newDict == NULL) return newDict;
  PyDict_SetItemString(newDict, "isconverged", PyLong_FromLong(retcode == 0 ? 1 : 0));
  PyDict_SetItemString(newDict, "cholfail", PyLong_FromLong(retcode < 0 ? 1 : 0));
  if (qpRet == NULL) return newDict;
  PyDict_SetItemString(newDict, "iters", PyLong_FromLong(qpRet->iters));
  PyDict_SetItemString(newDict, "fxopt", PyFloat_FromDouble(qpRet->fxopt));
  PyDict_SetItemString(newDict, "fxofs", PyFloat_FromDouble(qpRet->fxofs));
  const npy_intp inftuple_dims[] = {4};
  PyObject* inftuple = PyArray_SimpleNew(1, inftuple_dims, NPY_DOUBLE);
  memcpy((double *) PyArray_DATA((PyArrayObject *) inftuple), qpRet->inftuple, 4 * sizeof(double));
  PyDict_SetItemString(newDict, "inftuple", inftuple);
  PyDict_SetItemString(newDict, "qpclass", PyUnicode_FromString(prblm_class_names[problemClass]));
  if (retcode == 0) {
    const int nd = nx + nu + ns;
    const double* px = qpRet->x;

    if (numReturnStepsU > 0) {
      if (numReturnStepsU > n + 1) numReturnStepsU = n + 1; /* clip if needed */
      const npy_intp utraj_dims[] = {numReturnStepsU, nu};
      PyObject* utraj_object = PyArray_SimpleNew(2, utraj_dims, NPY_DOUBLE);
      enforce_fortran_property((PyArrayObject *) utraj_object);
      int kk = nx; 
      double* pdd = (double *) PyArray_DATA((PyArrayObject *) utraj_object);
      for (int ll = 0; ll < numReturnStepsU; ll++) {
        for (int mm = 0; mm < nu; mm++)
          pdd[mm * numReturnStepsU + ll] = px[kk + mm];
        kk += nd;
      }
      PyDict_SetItemString(newDict, "utraj", utraj_object);
    }

    if (numReturnStepsX > 0) {
      if (numReturnStepsX > n + 1) numReturnStepsX = n + 1; /* clip if needed */
      const npy_intp xtraj_dims[] = {numReturnStepsX, nx};
      PyObject* xtraj_object = PyArray_SimpleNew(2, xtraj_dims, NPY_DOUBLE);
      enforce_fortran_property((PyArrayObject *) xtraj_object);
      int kk = 0; 
      double* pdd = (double *) PyArray_DATA((PyArrayObject *) xtraj_object);
      for (int ll = 0; ll < numReturnStepsX; ll++) {
        for (int mm = 0; mm < nx; mm++)
          pdd[mm * numReturnStepsX + ll] = px[kk + mm];
        kk += nd;
      }
      PyDict_SetItemString(newDict, "xtraj", xtraj_object);
    }

    if (ns > 0 && numReturnStepsS > 0) {
      if (numReturnStepsS > n + 1) numReturnStepsS = n + 1; /* clip if needed */
      const npy_intp straj_dims[] = {numReturnStepsS, ns};
      PyObject* straj_object = PyArray_SimpleNew(2, straj_dims, NPY_DOUBLE);
      enforce_fortran_property((PyArrayObject *) straj_object);
      int kk = nx + nu;
      double* pdd = (double *) PyArray_DATA((PyArrayObject *) straj_object);
      for (int ll = 0; ll < numReturnStepsS; ll++) {
        for (int mm = 0; mm < ns; mm++)
          pdd[mm * numReturnStepsS + ll] = px[kk + mm];
        kk += nd;
      }
      PyDict_SetItemString(newDict, "straj", straj_object);
    }

    /* Evaluate the part of the cost that comes from the slack terms only;
       Return it in the report field "fxoft"; it is very easy to compute
       due to the implicit diagonal structure of the S matrix.
     */
    if (ns > 0 && pS != NULL) {
      double dd = 0.0;
      int kk = nx + nu;
      for (int qq = 0; qq < n + 1; qq++) {
        const double* pdd = (const double *) &(px[kk]);
        for (int ll = 0; ll < ns; ll++)
          dd += pdd[ll] * pS[ll] * pdd[ll];
        kk += nd;
      }
      PyDict_SetItemString(newDict, "fxoft", PyFloat_FromDouble(dd));
    }
  }

  return newDict;
}

/*** MODULE ERRORS OBJECT ***/

static PyObject *ModuleError;

#define ERRORMESSAGE(msg) \
  { PyErr_SetString(ModuleError, msg); \
    goto offload_and_return_null; } \

/*** EXPOSED FUNCTIONS ***/

// Usage: result = qpmpclti2f(problem, options)
static PyObject*
rempc_qpmpclti2f(PyObject *self, 
                 PyObject *args)
{
  #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
  fclk_timespec _tic1, _tic2;
  fclk_timespec _toc1, _toc2;
  #endif
    
  #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
  fclk_timestamp(&_tic1);
  #endif

  problemInputObjects P;
  qpdatStruct qpDat;
  qpoptStruct qpOpt;
  qpretStruct qpRet;

  memset(&qpDat, 0, sizeof(qpdatStruct));
  memset(&qpRet, 0, sizeof(qpretStruct));

  int auxbufsz = 0, bufofs = 0;
  double *pauxbuf = NULL;
  stageStruct *pStages = NULL;

  PyObject* problem_dict = NULL;
  PyObject* options_dict = NULL;
  if (!PyArg_ParseTuple(args, "O!O!", 
                        &PyDict_Type, &problem_dict, 
                        &PyDict_Type, &options_dict))
    return NULL;

  if (!assign_options_struct_from_dict(&qpOpt, options_dict)) {
    PyErr_SetString(ModuleError, "Failed to process options dict (2nd argument)");
    return NULL;
  }

  if (qpOpt.verbosity > 1)
    print_options_struct(&qpOpt);

  loadProblemInputs(problem_dict, &P);

  if (!allFortranProblemInputs(&P))
    ERRORMESSAGE("All available input arrays are not Fortran")

  // TBD: if super-verbose, make an explicit test of the strides of all matrices; are they exactly as expected for Fortran layout?

  if (P.n + 1 < __MULTISOLVER_MINIMUM_STAGES)
    ERRORMESSAGE("Horizon is too short.")

  int nx, nu, ny, nq, ns = 0, nd, ni;
  double *pA, *pB, *pC, *pD;

  int hasOutput = 0;            /* is C!=0 ? */
  int hasDirectTerm = 0;        /* is D!=0 ? */
  int hasF1 = 0, hasF2 = 0;
  int hasInequalities = 0;      /* is F1!=0 or F2!=0 ? */
  int hasSlackCost = 0;         /* slack variable extension required? ns>0? */
  int hasTerminalQ = 0;         /* Qn exists? */
  int hasTerminalW = 0;         /* Wn exists? */
  int prblmClass = -1;

  double *pW, *pR, *pQ, *pS;     /* Cost term matrices */
  int typW, typR, typQ; /*,typS;*/
  double sW, sR, sQ; /*,sS;*/         /* Only used if typX = TYP_SCALAR */

  double *pF1, *pF2;

  double *pw, *pr, *px, *pf3, *psc=NULL;	/* vectors & vector signals */
  int typw, typr, typf3;
  double sw, sr, sf3;  

  double *pWn, *pQn;       /* used only with special terminal costs */
  int typWn, typQn;
  double sWn, sQn; //, q0tmp;

  /* These are constructed aux. data matrices */
  double *pJ = NULL, *pCstg = NULL, *pDstg = NULL, *pQstg = NULL;
  double *pCstg0 = NULL, *pDstg1 = NULL, *pQNstg = NULL;
  double *pCC1, *pCC2;

  /* Meta-data storage for sparse J and D (if needed) */
  sparseMatrix spJay;
  sparseMatrix spDee;
  spJay.buf = NULL; /* mark the sparse structs as unused */
	spDee.buf = NULL;

  /* These are constructed aux. data vectors */
  double *pvecd = NULL, *pvecq = NULL, *pvecf = NULL, *pvecq0 = NULL, *pvecr = NULL, *prtmp = NULL;

  // define pA and nx
  if (P.A == NULL)
    ERRORMESSAGE("No system matrix A")
  nx = PyArray_DIM((PyArrayObject *) P.A, 0);
  if (nx != PyArray_DIM((PyArrayObject *) P.A, 1))
    ERRORMESSAGE("System matrix A must be square")
  pA = (double *) PyArray_DATA((PyArrayObject *) P.A);

  // define pB and nu
  if (P.B == NULL)
    ERRORMESSAGE("No input matrix B")
  if (nx != PyArray_DIM((PyArrayObject *) P.B, 0))
    ERRORMESSAGE("B and A must have same number of rows")
  nu = PyArray_DIM((PyArrayObject *) P.B, 1);
  pB = (double *) PyArray_DATA((PyArrayObject *) P.B);

  // setup pC and ny (if given)
  if (P.C != NULL) {
    if (nx != PyArray_DIM((PyArrayObject *) P.C, 1))
      ERRORMESSAGE("C and A must have same number of columns")
    ny = PyArray_DIM((PyArrayObject *) P.C, 0);
    pC = (double *) PyArray_DATA((PyArrayObject *) P.C);
    hasOutput = 1;
  } else {
    pC = NULL;
    ny = 0;
    hasOutput = 0;
  }

  // setup pD when applicable
  if (P.D != NULL && hasOutput != 0) {
    if (ny != PyArray_DIM((PyArrayObject *) P.D, 0))
      ERRORMESSAGE("D and C must have same number of rows")
    if (nu != PyArray_DIM((PyArrayObject *) P.D, 1))
      ERRORMESSAGE("D and B must have same number of columns")
    pD = (double *) PyArray_DATA((PyArrayObject *) P.D);
    hasDirectTerm = 1;
  } else {
    hasDirectTerm = 0;
    pD = NULL;
  }

  if (P.x == NULL)
    ERRORMESSAGE("State vector x must be provided")
  if (!np_is_vector((PyArrayObject *) P.x, true))
    ERRORMESSAGE("x must be a vector (or scalar)")
  if (PyArray_SIZE((PyArrayObject *) P.x) != nx)
    ERRORMESSAGE("x does not have the correct number of elements")
  px = (double *) PyArray_DATA((PyArrayObject *) P.x);

  /* Get nq, number of linear inequalities per stage */
  if (P.F1 != NULL) {
    if (nx != PyArray_DIM((PyArrayObject *) P.F1, 1))
      ERRORMESSAGE("F1 has inconsistent dimensions");
    nq = PyArray_DIM((PyArrayObject *) P.F1, 0);
    pF1 = (double *) PyArray_DATA((PyArrayObject *) P.F1);
    hasF1 = 0; /* set to 1 only if nonzero */
    for (int qq = 0; qq < nq * nx; qq++) {
      if (pF1[qq] != 0.0) { hasF1 = 1; break; }
    }
  } else {
    hasF1 = 0;
    nq = 0;
    pF1 = NULL;
  }

  if (P.F2 != NULL) {
    if (nu != PyArray_DIM((PyArrayObject *) P.F2, 1))
      ERRORMESSAGE("F2 has inconsistent dimensions")
    if (hasF1) {
      if (PyArray_DIM((PyArrayObject *) P.F2, 0) != nq)
        ERRORMESSAGE("F2 row-count does not match that of F1")
    } else {
      nq = PyArray_DIM((PyArrayObject *) P.F2, 0);
    }
    pF2 = (double *) PyArray_DATA((PyArrayObject *) P.F2);
    hasF2 = 0;    /* set to 1 only if nonzero */
    for (int qq = 0; qq < nq * nu; qq++) {
      if (pF2[qq] != 0.0) { hasF2 = 1; break; }
    }
  } else {
    hasF2 = 0;
    pF2 = NULL;
  }

  /* If either of F1 or F2 is nonzero; there are inequalities present */
  hasInequalities = ((hasF1 != 0 || hasF2 != 0) ? 1 : 0);

  /* If hasInequalitites, then check whether constraint softening is requested; update ns accordingly */
  if (hasInequalities > 0 && P.sc != NULL) {
    const int sc_rows = PyArray_DIM((PyArrayObject *) P.sc, 0);
    const int sc_cols = PyArray_DIM((PyArrayObject *) P.sc, 1);
    /* Make sure sc is nq-by-1 or 1-by-nq; both are OK */
	  if (!(((sc_rows == nq && sc_cols == 1)) || (sc_rows == 1 && sc_cols == nq)))
      ERRORMESSAGE("Soft/slack cost vector length must have nq elements")
    /* next extract the implied slack vector dimension (maximum nq, minimum 0) */
    psc = (double *) PyArray_DATA((PyArrayObject *) P.sc);
    for (int qq = 0; qq < nq; qq++)
      if (psc[qq] > 0.0) ns++;
      /* ... it is enough to just find the value of ns here;
         then populate S and J later when req. storage is allocated
       */
    if (ns > 0) hasSlackCost = 1;
  }

  /* nd = dimension of stage-variable */
  nd = nx + nu + ns;	/* also add ns, if ns>0 */
  ni = nq + ns;

  const int n = P.n;

  /* Since the dimensions are known at this point; allocate aux. space */
  auxbufsz = (n+1)*nx+(n+1)*nd+(n+1)*ni+ni*nd+nx*nd+nx*nd
             +2*nx*nd+2*nx*nd+nd*nd+nd*nd+(n+1)+(n+1)*ny+ny
             +nd*(nd+1)+nd*(nd+1)+ns;

  pauxbuf = malloc(auxbufsz * sizeof(double));
  if (pauxbuf == NULL)
    ERRORMESSAGE("Failed to allocate aux. memory block")

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
    
  if (bufofs != auxbufsz) {
    printf("WARNING: auxbufsz=%i, bufofs=%i\n", auxbufsz, bufofs);
  }

  if (hasInequalities > 0) {
    if (P.f3 == NULL)
      ERRORMESSAGE("f3 missing")

    /* Check/read f3 nq-by-(n+1) data */
    if (aux_read_signal_matrix((PyArrayObject *) P.f3, nq, n + 1, &typf3, &sf3, &pf3) != 1)
      ERRORMESSAGE("f3 has inconsistent size")
      /* NOTE: J may have significant sparsity.
       * Typically there will only be a single element per row which is
       * nonzero in J=[F1,F2] so this is important to handle efficiently.
       */
    if (hasSlackCost > 0) {
      matopc_zeros(pJ, ni, nd);
      if (hasF1 > 0) matopc_sub_assign(pJ, ni, nd, 0, 0, pF1, nq, nx, +1);
      if (hasF2 > 0) matopc_sub_assign(pJ, ni, nd, 0, nx, pF2, nq, nu, +1);
      /* ... assemble the rest ... indices for psc[.]>0.0: (nq+qq,nx+nu+qq)=-1*/
      for (int ll = 0, qq = 0; qq < nq; qq++) {
        if (psc[qq] > 0.0) {
          pJ[ni * (nx + nu + ll) + (qq)] = -1.0;
          pJ[ni * (nx + nu + ll) + (nq + ll)] = -1.0;
          pS[ll] = psc[qq];
          ll++;
        }
      }
    } else {
      /* Create J by copying submatrices into proper patches .. */
      matopc_zeros(pJ, nq, nd);
      if (hasF1 > 0) matopc_sub_assign(pJ, nq, nd, 0, 0, pF1, nq, nx, +1);
      if (hasF2 > 0) matopc_sub_assign(pJ, nq, nd, 0, nx, pF2, nq, nu, +1);
    }
  }

  /* Always check w (part of evolution equation); data should be
   * a matrix of size nx-by-n ideally; but could also be scalar, vector
   * or a transposed matrix.
   */
  if (P.w == NULL)
    ERRORMESSAGE("w missing")
  if (aux_read_signal_matrix((PyArrayObject *) P.w, nx, n, &typw, &sw, &pw) != 1)
    ERRORMESSAGE("w has inconsistent size")

  /* Check W if hasOutput */
  if (hasOutput > 0) {
    if (P.W == NULL)
      ERRORMESSAGE("W missing")
    if (aux_read_square_matrix((PyArrayObject *) P.W, ny, &typW, &sW, &pW) != 1)
      ERRORMESSAGE("W has inconsistent size")
    
    if (P.r == NULL)
      ERRORMESSAGE("r missing")
    /* Field r should ideally be a ny-by-(n+1) matrix.
     * It can also be a ny-vector interpreted as repeated n+1 times.
     * Or it can be a scalar; assumed to fill up the full matrix.
     * A transposed matrix (n+1)-by-ny is also allowed; if r does not
     * fit any of the above sizes. 
     */
    if (aux_read_signal_matrix((PyArrayObject *) P.r, ny, n + 1, &typr, &sr, &pr) != 1)
      ERRORMESSAGE("r has inconsistent size")
    
    /* Check whether special terminal cost matrix Wn is present among the arguments */
    if (P.Wn != NULL) {
      if (aux_read_square_matrix((PyArrayObject *) P.Wn, ny, &typWn, &sWn, &pWn) != 1)
        ERRORMESSAGE("Wn has inconsistent size")
      hasTerminalW = 1;
    }
  }

  /* Check Qx always; must not be undefined */
  if (P.Qx == NULL)
    ERRORMESSAGE("Qx missing")
  if (aux_read_square_matrix((PyArrayObject *) P.Qx, nx, &typQ, &sQ, &pQ) != 1)
    ERRORMESSAGE("Qx has inconsistent size")
    
  /* Check R always; must not be undefined */
  if (P.R == NULL)
    ERRORMESSAGE("R missing")
  if (aux_read_square_matrix((PyArrayObject *) P.R, nu, &typR, &sR, &pR) != 1)
    ERRORMESSAGE("R has inconsistent size")
    
  /* Special terminal cost matrix Qxn exists? */
  if (P.Qxn != NULL) {
    if (aux_read_square_matrix((PyArrayObject *) P.Qxn, nx, &typQn, &sQn, &pQn) != 1)
      ERRORMESSAGE("Qxn has inconsistent size")
    hasTerminalQ = 1;
  }

  /* Setup big d vector */
  matopc_copy(pvecd, px, nx, 1);
  if (typw == TYP_SCALAR) {
    for (int qq = nx; qq < nx * (n + 1); qq++) pvecd[qq] = sw;
  } else if (typw == TYP_VECTOR) {
    for (int qq = 0, ll = 0; qq < n; qq++, ll += nx) {
      matopc_copy(&pvecd[nx + ll], pw, nx, 1);
    }
  } else if (typw == TYP_MATRIX) {
    /*for (qq=0,ll=0;qq<n;qq++,ll+=nx) { matopc_copy(&pvecd[nx+ll],&pw[ll],nx,1); }*/
    matopc_copy(&pvecd[nx], pw, nx, n);
  } else {
    /*matopc_zeros(pvecd,nx*(n+1),1);
      mexErrMsgTxt("TYP_MATRIXT not yet supported @ d.");*/
    matopc_copy_transpose(&pvecd[nx], pw, n, nx);
  }
  /* aux_print_array(pvecd,nx*(n+1),1); */
    
  /* Setup big q vector (also known as vector h).
   * Start by expanding/rearranging to obtain a stacked vector of r.
   */
  matopc_zeros(pvecq, nd * (n + 1), 1);
  matopc_zeros(pvecq0, (n + 1), 1);
  if (hasOutput > 0) {
    /* Setup a full-size reference stacked vector */ 
    if (typr == TYP_SCALAR) {
      /* Treat scalar sr as the vector ones(ny,1)*sr at each stage. */
      for (int qq = 0; qq < ny * (n + 1); qq++) pvecr[qq] = sr;
    } else if (typr == TYP_VECTOR) {
      /* Use vector ny-by-1 pr for each stage. */
      for (int qq = 0, ll = 0; qq < n + 1; qq++, ll += ny) {
        matopc_copy(&pvecr[ll], pr, ny, 1);
      }
    } else if (typr == TYP_MATRIX) {
      /* Each column in matrix ny-by-(n+1) pr is a unique reference vector */
      matopc_copy(pvecr, pr, ny, n + 1);
    } else {
      /* Each row in matrix (n+1)-by-ny pr is a (transposed) unique reference vector */
      matopc_copy_transpose(pvecr, pr, n + 1, ny);
    }
        
    /* Save last reference vector if it needs to be used for modification of last stage below */
    if (hasTerminalW > 0) {
      matopc_copy(prtmp, &pvecr[n * ny], ny, 1); /* copy last ny-vector r(n) of pvecr to prtmp */
    }

    /* Apply in-place transformation r(i) <- -2*W*r(i) on the buffer pvecr */
    /* Evaluate q0(i)=r(i)'*W*r(i) for each stage i=0..n; before the inplace op. (!) */
    if (typW == TYP_SCALAR) {
      int kk = 0;
      for (int qq = 0; qq < n + 1; qq++) {
        double dd = 0.0;
        for (int ll = 0; ll < ny; ll++)
          dd += sW * pvecr[kk + ll] * pvecr[kk + ll];
        pvecq0[qq] = dd;
        kk += ny;
      }
      const double dd = -2.0 * sW;
      for (int qq = 0; qq < ny * (n + 1); qq++) pvecr[qq] *= dd;
    } else if (typW == TYP_VECTOR) {
      int kk = 0;
      for (int qq = 0; qq < n + 1; qq++) {
        double dd = 0.0;
        for (int ll = 0; ll < ny; ll++)
          dd += pW[ll] * pvecr[kk + ll] * pvecr[kk + ll];
        pvecq0[qq] = dd;
        kk += ny;
      }
      kk = 0;
      for (int qq = 0; qq < n + 1; qq++) {
        for (int ll = 0; ll < ny; ll++) pvecr[kk++] *= (-2.0 * pW[ll]);
      }
    } else { /* TYP_MATRIX */
      if (ny > MATOPC_TEMPARRAYSIZE)
        ERRORMESSAGE("ERROR: ny>MATOPC_TEMPARRAYSIZE")
      int kk = 0;
      for (int qq = 0; qq < n + 1; qq++) {
        pvecq0[qq] = matopc_xtax_sym(pW, ny, &pvecr[kk]);
        /* Multiply by general ny-by-ny matrix W */
        matopc_inplace_scaled_ax(&pvecr[kk], pW, ny, ny, -2.0);
        /* TODO: actually never access lower triangle of W */
        /* ATTN: uses static temp. array of fixed size internally */
        kk += ny;
      }
    }
        
    /* Need modification of linear cost term and offset for last stage if Wn exists */
    if (hasTerminalW > 0) {
      double q0tmp = 0.0;
      /* evaluate q0n <- r(n)'*Wn*r(n) and r(n) <- -2*Wn*r(n) for usage later */
      if (typWn == TYP_SCALAR) {
        for (int ll = 0; ll < ny; ll++)
          q0tmp += prtmp[ll] * prtmp[ll];
        q0tmp *= sWn;
        const double dd = -2.0 * sWn;
        for (int ll = 0; ll < ny; ll++)
          prtmp[ll] *= dd;
      } else if (typWn == TYP_VECTOR) {
        for (int ll = 0; ll < ny; ll++)
          q0tmp += pWn[ll] * prtmp[ll] * prtmp[ll];
        for (int ll = 0; ll < ny; ll++)
          prtmp[ll] *= (-2.0 * pWn[ll]);
      } else { /* TYP_MATRIX */
        if (ny > MATOPC_TEMPARRAYSIZE)
          ERRORMESSAGE("ERROR: ny>MATOPC_TEMPARRAYSIZE")
        q0tmp = matopc_xtax_sym(pWn, ny, prtmp);
        matopc_inplace_scaled_ax(prtmp, pWn, ny, ny, -2.0);
        /* TODO: symmetrized version that never accessed lower part */
      }
      /* overwrite last vector in pvecr and last element in pvecq0 */
      pvecq0[n] = q0tmp;
      matopc_copy(&pvecr[n * ny], prtmp, ny, 1);
    }
        
    /* Apply C' and D' transformations to obtain q(i) from W*r(i) vectors */
    if (hasDirectTerm > 0) { /* q(i) = -2*[C';D']*W*r(i) */
      int kk = 0; int ll = 0;
      for (int qq = 0; qq < n + 1; qq++) {
        matopc_atx(&pvecq[kk], pC, ny, nx, &pvecr[ll]);
        matopc_atx(&pvecq[kk + nx], pD, ny, nu, &pvecr[ll]);
        kk += nd; ll += ny;
      }
    } else { /* q(i) = -2*[C';zeros(nu,ny)]*W*r(i) */
      int kk = 0; int ll = 0;
      for (int qq = 0; qq < n + 1; qq++) {
        matopc_atx(&pvecq[kk], pC, ny, nx, &pvecr[ll]);
        kk += nd; ll += ny;
      }
    }

    /*mexPrintf("[pvecq0-new]\n"); 
    aux_print_array(pvecq0,n+1,1);*/    
  } /* end if (hasOutput) */

  /* NOTE: pvecf must be padded correctly when ns>0 */
  /* Setup big f vector; length of pvecf is ni*(n+1), with ni=nq+ns, where ns might be zero */
  if (hasInequalities > 0) {
    matopc_zeros(pvecf, ni * (n + 1), 1);
    if (typf3 == TYP_SCALAR) {
      if (ns>0) {
	      for (int qq = 0, ll = 0; qq < n + 1; qq++, ll += ni) {
	        vecop_assign(&pvecf[ll], nq, sf3);
  	    }
      } else {
        /*for (qq=0;qq<nq*(n+1);qq++) pvecf[qq]=sf3;*/
        vecop_assign(&pvecf[0], nq * (n + 1), sf3);
      }
    } else if (typf3 == TYP_VECTOR) {
      for (int qq = 0, ll = 0; qq < n + 1; qq++, ll += ni) {
        matopc_copy(&pvecf[ll], pf3, nq, 1);
      }
    } else if (typf3 == TYP_MATRIX) {
      for (int qq = 0, ll = 0, kk = 0; qq < n + 1; qq++, ll += ni, kk += nq) {
        matopc_copy(&pvecf[ll], &pf3[kk], nq, 1);
      }
    } else {
      /*mexErrMsgTxt("TYP_MATRIXT not yet supported @ f.");*/
      /* NOTE: should read in as if pf3 is a column-major (n+1)-by-nq matrix */
      for (int qq = 0, ll = 0; qq < n + 1; qq++, ll += ni) {
        for (int kk = 0; kk < nq; kk++)
          pvecf[ll + kk] = pf3[qq + kk * (n + 1)];
      }
    }
  }

  /* Cstg=[-A,-B] */
  matopc_sub_assign(pCstg,nx,nd,0,0,pA,nx,nx,-1);
  matopc_sub_assign(pCstg,nx,nd,0,nx,pB,nx,nu,-1);
  if (ns > 0) matopc_sub_assign_zeros(pCstg,nx,nd,0,nx+nu,nx,ns); /* [-A,-B,0] needed here */
    
  /* Dstg=[eye(nx),zeros(nx,nu)] */
  matopc_zeros(pDstg,nx,nd);
  matopc_sub_assign_diag_scalar(pDstg,nx,nd,0,0,1.0,nx,-1);
    
  /* C0=[I,0;-A,-B] */
  matopc_zeros(pCstg0,2*nx,nd);
  matopc_sub_assign_diag_scalar(pCstg0,2*nx,nd,0,0,1.0,nx,-1);
  matopc_sub_assign(pCstg0,2*nx,nd,nx,0,pCstg,nx,nd,+1); /* Do not change sign again! */
    
  /* D1=[0,0;I,0] */
  matopc_zeros(pDstg1,2*nx,nd);
  matopc_sub_assign_diag_scalar(pDstg1,2*nx,nd,nx,0,1.0,nx,-1);
    
  /* If there are no slack variables;
   * the cost matrix is square symmetric with side = nx+nu.
   * If there are slack avariables the cost matrix has side = nx+nu+ns.
   * This dimension is stored in stage size: nd.
   */

  aux_compute_sym_cost_matrix(nd,nx,nu,ny,pQstg,typQ,sQ,pQ,typR,sR,pR,typW,sW,pW,pC,pD);
    
  /* Setup special stage cost matrix for last stage as needed */
  if (hasTerminalQ > 0 && hasTerminalW == 0) {
    /* create a Qn cost matrix which only need a new upper left block */
    aux_compute_sym_cost_matrix(nd,nx,nu,ny,pQNstg,typQn,sQn,pQn,typR,sR,pR,typW,sW,pW,pC,pD);
  } else if (hasTerminalQ == 0 && hasTerminalW > 0) {
    /* use previous/common Q but need to recreate the rest of Qn */
    aux_compute_sym_cost_matrix(nd,nx,nu,ny,pQNstg,typQ,sQ,pQ,typR,sR,pR,typWn,sWn,pWn,pC,pD);
  } else if (hasTerminalQ > 0 && hasTerminalW > 0) {
    /* both Qn and Wn are special; new stage cost from */
    aux_compute_sym_cost_matrix(nd,nx,nu,ny,pQNstg,typQn,sQn,pQn,typR,sR,pR,typWn,sWn,pWn,pC,pD);
  }
    
  /* May need to insert the slack cost on the bottom-right part of the diagonal */
  if (ns > 0) {
    /* insert diag(ps)*2.0 */
    matopc_sub_assign_scaled_diag_vector(pQstg,nd,nd,nx+nu,nx+nu,pS,ns,-1,2.0);
    if (hasTerminalQ > 0 || hasTerminalW > 0) {
    	/* pQNstg update required too */
    	matopc_sub_assign_scaled_diag_vector(pQNstg,nd,nd,nx+nu,nx+nu,pS,ns,-1,2.0);
    }
  }

  /*
  if (P.A != NULL) {
    printf("isfortran(P.A) = %i\n", PyArray_ISFORTRAN((PyArrayObject *) P.A));
    print_array_layout((PyArrayObject *) P.A);
  }
  if (P.B != NULL) {
    printf("isfortran(P.B) = %i\n", PyArray_ISFORTRAN((PyArrayObject *) P.B));
    print_array_layout((PyArrayObject *) P.B);
  }*/

  if (qpOpt.verbosity > 0) {
    printf("mpc horizon: 0..%i.\n", P.n);
    printf("system: (states,inputs,outputs):[hasout,dterm] = (%i,%i,%i):[%i,%i].\n", nx, nu, ny, hasOutput, hasDirectTerm);
    printf("inequalities: (hasF1,hasF2,nq,ns) = (%i,%i,%i,%i).\n", hasF1, hasF2, nq, ns);
  }

  /* Now create sparse block representations; if requested
   * Note that this may fail e.g. if matrices are deemed too dense; let the failure
   * be silent; the rest of the code runs through anyway.
   */
  if (qpOpt.expl_sparse > 0) {
    if (hasInequalities > 0) {
    	sparseMatrixCreate(&spJay, pJ, ni, nd); /* was nq,nd */
    }
    sparseMatrixCreate(&spDee, pDstg, nx, nd);
  }

  /* Set the default return variable options: full control trajectory
   * but nothing for the state trajectory.
   */
  int numReturnStepsU = n + 1;
  int numReturnStepsX = 0;
  int numReturnStepsS = 0;

  if (P.ureturn > 0) numReturnStepsU = P.ureturn;
  if (P.xreturn > 0) numReturnStepsX = P.xreturn;
  if (P.sreturn > 0) numReturnStepsS = P.sreturn;

  /* Allocate stage structure and initialize the pointers */
  pStages = (stageStruct *) malloc((n + 1) * sizeof(stageStruct));
  if (pStages == NULL)
    ERRORMESSAGE("Stage struct array memory allocation failure")

  /* Initialize the stage struct array */
  int ll = 0, kk = 0, mm = 0;
  for (int qq = 0; qq <= n; qq++) {
    stageStruct* thisStage = &pStages[qq];
    thisStage->idx = qq;
    thisStage->nd = nd;   /* in general: nd=nx+nu+ns */
    thisStage->niq = ni; /* nq+ns = ni (was ...=nq) */
    if (qq == 0) {
      /* Stage 0 includes initial condition equality constraint */
      thisStage->neq = 2 * nx;
    } else if (qq == n) {
      thisStage->neq = 0;
    } else {
      thisStage->neq = nx;
    }
    /* Setup data pointers */
    thisStage->ptrQ = pQstg;
    thisStage->ptrJ = pJ; /* ... */
    thisStage->ptrq = &pvecq[qq * nd];
    thisStage->ptrq0 = &pvecq0[qq];
    thisStage->ptrf = &pvecf[qq * ni]; /* was qq*nq*/
    thisStage->ptrd = &pvecd[ll];
    ll += thisStage->neq;
    kk += thisStage->niq;
    mm += (thisStage->nd) * (thisStage->nd) + (thisStage->nd);
    thisStage->ptrC = NULL;  /* C does not exist for stage n */
    if (qq == 0) {
      thisStage->ptrC = pCstg0;
    } else if (qq < n) {
      thisStage->ptrC = pCstg;
    }
    thisStage->ptrD = NULL; /* D does not exist for stage 0 */
    if (qq == 1) {
      thisStage->ptrD = pDstg1;
      thisStage->ptrDsp = NULL; /* TODO: special sparse matrix object for first stage (may not be worth it though) */
    } else if (qq > 1) {
      thisStage->ptrD = pDstg;
      thisStage->ptrDsp = (spDee.buf == NULL ? NULL : &spDee);
    }
    thisStage->ptrL = NULL; /* must be initialized separately, when needed */

    /*mexPrintf("stage=%i: (neq=%i)\n",pStages[qq].idx,pStages[qq].neq);*/

    /* J is shared for every stage in the present formulation */
    thisStage->ptrJsp = (spJay.buf == NULL ? NULL : &spJay);
  }

  #ifdef __CLUMSY_ASSERTIONS__
  if (kk != ni * (n + 1) || ll != nx * (n + 1))
    printf("Eq. or ineq. sizing error(s)\n");
  #endif

  /* Modify matrix pointer for the last stage if there is a special terminal cost matrix */
  if (hasTerminalQ > 0 || hasTerminalW > 0)
    pStages[n].ptrQ = pQNstg;

  // PERHAPS init before metadata assignment?
  // ...

  /* Setup problem meta data structure (collection of pointers mostly) */
  qpDat.ndec = (n + 1) * nd;
  qpDat.neq = (n + 1) * nx;
  qpDat.niq = (n + 1) * ni; /* (was nq*(n+1)) */
  qpDat.ph = pvecq;
  qpDat.pf = pvecf;
  qpDat.pd = pvecd;
  qpDat.nstg = n + 1;         /* qq=0..n; there are n+1 stages */
  qpDat.pstg = pStages;

  if (!msqp_pdipm_init(&qpDat, qpOpt.verbosity))
    ERRORMESSAGE("Failed to initialize working vectors memory buffer")
    
  /* Setup memory required for block Cholesky matrix factorizations.
   * Both programs (with or without inequalities) use the same block
   * Cholesky factorization program and can be initialized the same way.
   * But the simpler program does use less working memory (not adjusted for).
   */
  if (!InitializePrblmStruct(&qpDat, 0x001|0x002|0x004, qpOpt.verbosity))
    ERRORMESSAGE("Failed to initialize Cholesky working memory buffer(s)")

  #ifdef __CLUMSY_ASSERTIONS__
  if (qpDat.neq != qpDat.netot || qpDat.ndec != qpDat.ndtot)
    printf("Eq. or ineq. sizing error(s)\n");
  if (mm != qpDat.blkphsz)
    printf("Phi blk buffer sizing mismatch\n");
  #endif

  int retcode = -1;
  if (hasInequalities > 0) {
    /* Full IPM required; equality and inequality constrained QP */
    if (qpOpt.chol_update > 0) {
      /* TODO: pre-factor the only two unique cost matrices (before modification) */
      /* pCC1, pCC2 to be prepared with factorizations of pQstg and pQNstg (if it exists) */
      /* then the stage meta-data should be updated accordingly... ptrL field */
      retcode = CreateCholeskyCache(&qpDat, pCC1, pCC2, nd);
      if (retcode != 0)
        ERRORMESSAGE("Fatal block Cholesky factorization failure")
    }

    prblmClass = (ns > 0 ? 2 : 1);
        
    /* Slack extension or not; it is the same code invokation here */
    #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
		fclk_timestamp(&_tic2);
		#endif

		retcode = msqp_pdipm_solve(&qpDat, &qpOpt, &qpRet); /* Call main PDIPM algorithm */

		#ifdef __COMPILE_WITH_INTERNAL_TICTOC__
		fclk_timestamp(&_toc2);
		#endif
        
    if (retcode != 0 && qpOpt.verbosity > 0)
      printf("WARNING: main PDIPM solver did not converge\n");
        
  } else {
    /* No need to use the full IPM iteration; equality-constrained QP only */    
    prblmClass = 0;
        
    /* Always pre-factor the stage cost for the simplified solver.
     * The simplified solver will force itself to use the Cholesky cache always.
     * This is possible since the factors never need to be "updated".
     */
    if (qpOpt.chol_update > 0) {
      retcode = CreateCholeskyCache(&qpDat, pCC1, pCC2, nd);
      if (retcode != 0)
        ERRORMESSAGE("Fatal block Cholesky factorization failure")
    }

    #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_tic2);
    #endif

    retcode = msqp_solve_niq(&qpDat, &qpOpt, &qpRet);  /* call KKT solver code */
    
    #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
    fclk_timestamp(&_toc2);
    #endif
        
    if (retcode != 0 && qpOpt.verbosity > 0)
      printf("WARNING: main PDIPM solver did not converge.\n");
  }

  /* retcode == 0 implies converged solution;
     retcode == 1 implies not converged after max iters;
     retcode < 0 implies Cholesky error
   */

  PyObject* returnDict = createReturnDict(retcode,
                                          prblmClass,
                                          &qpRet,
                                          n,
                                          nu,
                                          numReturnStepsU,
                                          nx,
                                          numReturnStepsX,
                                          ns,
                                          numReturnStepsS,
                                          pS);

  FreePrblmStruct(&qpDat, qpOpt.verbosity);
  msqp_pdipm_free(&qpDat, qpOpt.verbosity);
  sparseMatrixDestroy(&spJay);
  sparseMatrixDestroy(&spDee);
  if (pStages != NULL) free(pStages);
  if (pauxbuf != NULL) free(pauxbuf);
  offloadProblemInputs(&P);

  #ifdef __COMPILE_WITH_INTERNAL_TICTOC__
  fclk_timestamp(&_toc1);
  PyDict_SetItemString(returnDict, 
                       "totalclock", 
                       PyFloat_FromDouble(fclk_delta_timestamps(&_tic1, &_toc1)));
  PyDict_SetItemString(returnDict, 
                       "solveclock", 
                       PyFloat_FromDouble(fclk_delta_timestamps(&_tic2, &_toc2)));
  fclk_get_resolution(&_tic1);
  PyDict_SetItemString(returnDict, 
                       "clockresol", 
                       PyFloat_FromDouble(fclk_time(&_tic1)));
  if (qpRet.cholytime >= 0.0) { /* set to -1 if unused */
    PyDict_SetItemString(returnDict, 
                         "cholyclock", 
                         PyFloat_FromDouble(qpRet.cholytime));
  }
  #endif

  return returnDict;

offload_and_return_null:
  msqp_pdipm_free(&qpDat, qpOpt.verbosity);
  sparseMatrixDestroy(&spJay);
  sparseMatrixDestroy(&spDee);
  if (pStages != NULL) free(pStages);
  if (pauxbuf != NULL) free(pauxbuf);
  offloadProblemInputs(&P);
  return NULL;
}

static PyObject*
rempc_options_qpmpclti2f(PyObject *self)
{
  qpoptStruct qpo;
  setup_qpopt_defaults(&qpo);

  PyObject* newDict = Py_BuildValue("{s:i,s:i,s:d,s:d,s:i,s:i,s:i,s:i}",
                                    "verbosity", qpo.verbosity, 
                                    "maxiters", qpo.maxiters,
                                    "ep", qpo.ep,
                                    "eta", qpo.eta,
                                    "expl_sparse", qpo.expl_sparse,
                                    "chol_update", qpo.chol_update,
                                    "blas_suite", qpo.blas_suite,
                                    "refinement", qpo.refinement);

  return newDict;
}

/*** METHODS TABLE ***/

static PyMethodDef rempc_methods[] = {
    {"qpmpclti2f", (PyCFunction) rempc_qpmpclti2f, METH_VARARGS,
     PyDoc_STR("Basic MPC solver for LTI system.")},
    {"options_qpmpclti2f", (PyCFunction) rempc_options_qpmpclti2f, METH_NOARGS,
     PyDoc_STR("Obtain default options for basic MPC solver.")},
    {NULL, NULL, 0, NULL}  /* end-of-table */
};

/*** MODULE INITIALIZATION ***/

PyDoc_STRVAR(docstring, "TBA.");

static struct PyModuleDef rempc_module = {
    PyModuleDef_HEAD_INIT,
    "rempc",   /* name of module */
    docstring, /* module documentation, may be NULL */
    -1,        /* size of per-interpreter state of the module,
                  or -1 if the module keeps state in global variables. */
    rempc_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit_rempc(void)
{
  PyObject* m = PyModule_Create(&rempc_module);

  if (m == NULL)
    return m;

  ModuleError = PyErr_NewException("rempc.error", NULL, NULL);
  Py_XINCREF(ModuleError);

  if (PyModule_AddObject(m, "error", ModuleError) < 0) {
    Py_XDECREF(ModuleError);
    Py_CLEAR(ModuleError);
    Py_DECREF(m);
    return NULL;
  }

  import_array();

  return m;
}
