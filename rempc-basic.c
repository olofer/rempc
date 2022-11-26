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

/*** MODULE ERRORS OBJECT ***/

static PyObject *ModuleError;

/*** EXPOSED FUNCTIONS ***/

static PyObject*
mynew_funktion1(PyObject *self, 
                PyObject *args)
{
    // TODO: basic function with positional arguments
    Py_RETURN_NONE;
}

static PyObject*
mynew_funktion2(PyObject *self, 
                PyObject *args,
                PyObject *kwds)
{
    // TODO: basic function with keyword arguments
    Py_RETURN_NONE;
}

static PyObject*
mynew_funktion3(PyObject *self)
{
    // TODO: basic function with no arguments
    Py_RETURN_NONE;
}

/*** METHODS TABLE ***/

static PyMethodDef rempc_methods[] = {
    {"funktion1", (PyCFunction) mynew_funktion1, METH_VARARGS,
     PyDoc_STR("Testfunktion 1: returns None.")},
    {"funktion2", (PyCFunction) mynew_funktion2, METH_VARARGS|METH_KEYWORDS, // cannot have METH_KEYWORDS by itself.
     PyDoc_STR("Testfunktion 2: returns None.")},
    {"funktion3", (PyCFunction) mynew_funktion3, METH_NOARGS,
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
