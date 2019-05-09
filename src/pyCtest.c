#include <stdio.h>
#include <Python.h>

#if PY_MAJOR_VERSION >= 3
#  define PyString_FromString PyUnicode_FromString
#endif

int pycatch(const char* err)
{
    fprintf(stderr, " Error! %s\n", err);
    fflush(stderr);
    return 1;
}

int main()
{
    PyObject *pyplotname    = NULL,
             *pyplotmodule  = NULL,
             *pltfigurename = NULL,
             *pyplot_title  = NULL,
             *py_title_args = NULL,
             *py_str  = NULL,
             *title         = NULL;

    Py_Initialize();

    PyObject *sys = PyImport_ImportModule("sys");
    PyObject *path = PyObject_GetAttrString(sys, "path");
    PyList_Append(path, PyString_FromString("."));

    pyplotname = PyString_FromString("matplotlib.pyplot");
    pyplotmodule = PyImport_Import(pyplotname);
    Py_XDECREF(pyplotname);
    if (!pyplotmodule)
        return pycatch("Could not load module 'matplotlib.pyplot'.");

    pyplot_title = PyObject_GetAttrString(pyplotmodule, "title");
    if (!PyFunction_Check(pyplot_title))
        return pycatch("Could not find 'title'.");

    py_title_args = PyTuple_New(1);
    py_str = PyString_FromString("My Plot");
    PyTuple_SetItem(py_title_args, 0, py_str);

    title = PyObject_CallObject(pyplot_title, py_title_args);
    Py_XDECREF(py_str);

    if (!title)
        return pycatch("Call to title() failed.");

    Py_Finalize();

    printf("Success.\n");
    return 0;
}
