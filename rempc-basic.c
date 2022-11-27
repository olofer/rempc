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

/*** MODULE ERRORS OBJECT ***/

static PyObject *ModuleError;

/*** EXPOSED FUNCTIONS ***/

// Usage: result = qpmpclti2f(problem, options)
static PyObject*
rempc_qpmpclti2f(PyObject *self, 
                 PyObject *args)
{
  PyObject* problem_dict = NULL;
  PyObject* options_dict = NULL;
  if (!PyArg_ParseTuple(args, "O!O!", 
                        &PyDict_Type, &problem_dict, 
                        &PyDict_Type, &options_dict))
    return NULL;

  PyObject* A = get_numpy_array(problem_dict, "A", 2);
  PyObject* B = get_numpy_array(problem_dict, "B", 2);

  if (A != NULL) {
    printf("isfortran(A) = %i\n", PyArray_ISFORTRAN((PyArrayObject *) A));
    print_array_layout((PyArrayObject *) A);
  }

  if (B != NULL) {
    printf("isfortran(B) = %i\n", PyArray_ISFORTRAN((PyArrayObject *) B));
    print_array_layout((PyArrayObject *) B);
  }

  // TODO: port the MEX code qpmpclti2f.c using Python/numpy "equivalents" ...

  Py_XDECREF(A);
  Py_XDECREF(B);

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
rempc_funktion3(PyObject *self)
{
    // TODO: basic function with no arguments
    Py_RETURN_NONE;
}

/*** METHODS TABLE ***/

static PyMethodDef rempc_methods[] = {
    {"qpmpclti2f", (PyCFunction) rempc_qpmpclti2f, METH_VARARGS,
     PyDoc_STR("Basic MPC solver for LTI system.")},
    {"funktion2", (PyCFunction) rempc_funktion2, METH_VARARGS|METH_KEYWORDS,
     PyDoc_STR("Testfunktion 2: returns None.")},
    {"funktion3", (PyCFunction) rempc_funktion3, METH_NOARGS,
     PyDoc_STR("Testfunktion 3: returns None.")},
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
