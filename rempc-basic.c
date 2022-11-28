#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h"

#undef __DEVELOPMENT_TEXT_OUTPUT__ 
#undef __CLUMSY_ASSERTIONS__
#undef __COMPILE_WITH_INTERNAL_TICTOC__ 

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

// ...
// TODO: sanity check routine; double check Fortran property and such ...
// ...

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

bool np_is_empty(PyArrayObject* a) {
  return (PyArray_SIZE(a) == 0);
}

bool np_is_scalar(PyArrayObject* a) {
  return (PyArray_SIZE(a) == 1);
}

// assumes ndim == 2 confirmed already; false if scalar
bool np_is_vector(PyArrayObject* a) {
  const int nrows = PyArray_DIM(a, 0);
  const int ncols = PyArray_DIM(a, 1);
  return (nrows == 1 && ncols > 1) || (nrows > 1 && ncols == 1);
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

/*** MODULE ERRORS OBJECT ***/

static PyObject *ModuleError;

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

  //printf("maxiters = %i\n", qpOpt.maxiters);
  //printf("eta      = %f\n", qpOpt.eta);

  loadProblemInputs(problem_dict, &P);

  printf("P.n = %i\n", P.n);

  if (P.A != NULL) {
    printf("isfortran(P.A) = %i\n", PyArray_ISFORTRAN((PyArrayObject *) P.A));
    print_array_layout((PyArrayObject *) P.A);
  }

  if (P.B != NULL) {
    printf("isfortran(P.B) = %i\n", PyArray_ISFORTRAN((PyArrayObject *) P.B));
    print_array_layout((PyArrayObject *) P.B);
  }

  // TODO: port the MEX code qpmpclti2f.c using Python/numpy "equivalents" ...

  offloadProblemInputs(&P);
  Py_RETURN_NONE;
}

static PyObject*
rempc_funktion2(PyObject *self, 
                PyObject *args,
                PyObject *kwds)
{
    // TODO: basic function with keyword arguments
    Py_RETURN_NONE;
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
    {"funktion2", (PyCFunction) rempc_funktion2, METH_VARARGS|METH_KEYWORDS,
     PyDoc_STR("Testfunktion 2: returns None.")},
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
