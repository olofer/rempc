/*
  Remake of qpmpclti2f.c MEX source code as Python C extension.
*/

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h"

#undef __DEVELOPMENT_TEXT_OUTPUT__ 
#undef __CLUMSY_ASSERTIONS__
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
// Return array object as Fortran layout (possibly converted).
PyObject* 
get_numpy_array(PyObject* dict, 
                const char* keyname, 
                int requireDims)
{
  PyObject* item = PyDict_GetItemString(dict, keyname);
  if (item == NULL) return NULL;
  if (!PyArray_Check(item)) return NULL;
  if (requireDims > 0)
    if (PyArray_NDIM((PyArrayObject *) item) != requireDims)
      return NULL;
  PyObject *arr = PyArray_FROM_OTF(item, 
                                   NPY_DOUBLE, 
                                   NPY_ARRAY_IN_ARRAY|NPY_ARRAY_F_CONTIGUOUS|NPY_ARRAY_ALIGNED);
  return arr;
}

int get_integer_from_dict(PyObject* dict, 
                          const char* keyname)
{
  PyObject* item = PyDict_GetItemString(dict, keyname);
  if (item == NULL) return 0;
  if (!PyLong_Check(item)) return 0;
  return ((int) PyLong_AsLong(item));
}

// grab numpy array object, requiring shape defined for 2 dimensions, and Fortran memory layout
void loadProblemInputs(PyObject* dict, problemInputObjects* pIO) {
  memset(pIO, 0, sizeof(problemInputObjects));
  pIO->n   = get_integer_from_dict(dict, "n");  // set to 0 if failing to read integer object by key name
  pIO->A   = get_numpy_array(dict, "A", 2);     // pointers are set to NULL if failure to read by key name
  pIO->B   = get_numpy_array(dict, "B", 2);
  pIO->C   = get_numpy_array(dict, "C", 2);
  pIO->D   = get_numpy_array(dict, "D", 2);
  pIO->Qx  = get_numpy_array(dict, "Qx", 2);
  pIO->W   = get_numpy_array(dict, "W", 2);
  pIO->R   = get_numpy_array(dict, "R", 2);
  pIO->F1  = get_numpy_array(dict, "F1", 2);
  pIO->F2  = get_numpy_array(dict, "F2", 2);
  pIO->f3  = get_numpy_array(dict, "f3", 2);
  pIO->w   = get_numpy_array(dict, "w", 2);
  pIO->r   = get_numpy_array(dict, "r", 2);
  pIO->x   = get_numpy_array(dict, "x", 2);
  pIO->Qxn = get_numpy_array(dict, "Qxn", 2);
  pIO->Wn  = get_numpy_array(dict, "Wn", 2);
  pIO->sc  = get_numpy_array(dict, "sc", 2);
  return;
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
  problemInputObjects P;
  qpdatStruct qpDat;
  qpoptStruct qpOpt;
  qpretStruct qpRet;

  memset(&qpDat, 0, sizeof(qpdatStruct));
  memset(&qpRet, 0, sizeof(qpretStruct));

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

  int nx, nu, ny, nq, ns, nd, ni;
  const double *pA, *pB, *pC, *pD; // "const" will prob. break

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
  double sWn, sQn, q0tmp;  

  /* These are constructed aux. data matrices */
  double *pJ = NULL, *pCstg = NULL, *pDstg = NULL, *pQstg = NULL;
  double *pCstg0 = NULL, *pDstg1 = NULL, *pQNstg = NULL;
  double *pCC1, *pCC2;

  /* These are constructed aux. data vectors */
  double *pvecd = NULL, *pvecq = NULL, *pvecf = NULL, *pvecq0 = NULL, *pvecr = NULL, *prtmp = NULL;

  // define pA and nx
  if (P.A == NULL)
    ERRORMESSAGE("No system matrix A")
  nx = PyArray_DIM((PyArrayObject *) P.A, 0);
  if (nx != PyArray_DIM((PyArrayObject *) P.A, 1))
    ERRORMESSAGE("System matrix A must be square")
  pA = (const double *) PyArray_DATA((PyArrayObject *) P.A);

  // define pB and nu
  if (P.B == NULL)
    ERRORMESSAGE("No input matrix B")
  if (nx != PyArray_DIM((PyArrayObject *) P.B, 0))
    ERRORMESSAGE("B and A must have same number of rows")
  nu = PyArray_DIM((PyArrayObject *) P.B, 1);
  pB = (const double *) PyArray_DATA((PyArrayObject *) P.B);

  // setup pC and ny (if given)
  if (P.C != NULL) {
    if (nx != PyArray_DIM((PyArrayObject *) P.C, 1))
      ERRORMESSAGE("C and A must have same number of columns")
    ny = PyArray_DIM((PyArrayObject *) P.C, 0);
    pC = (const double *) PyArray_DATA((PyArrayObject *) P.C);
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
    pD = (const double *) PyArray_DATA((PyArrayObject *) P.D);
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

  // TODO: process F1, F2, and cost matrices, and signal vectors/matrices

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

  // TODO: continue direct port the MEX code qpmpclti2f.c using Python/numpy "equivalents" ...

  offloadProblemInputs(&P);
  Py_RETURN_NONE;

offload_and_return_null:
  offloadProblemInputs(&P);
  return NULL;
}

/*
static PyObject*
rempc_funktion2(PyObject *self, 
                PyObject *args,
                PyObject *kwds)
{
  Py_RETURN_NONE;
}
*/

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
    /*{"funktion2", (PyCFunction) rempc_funktion2, METH_VARARGS|METH_KEYWORDS,
     PyDoc_STR("Testfunktion 2: returns None.")},*/
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
