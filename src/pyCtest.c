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
             *pyplot_figure = NULL,
             *py_empty_args = NULL,
             *fig = NULL;

    Py_Initialize();

    pyplotname = PyString_FromString("matplotlib.pyplot");
    pyplotmodule = PyImport_Import(pyplotname);
    Py_XDECREF(pyplotname);
    if (!pyplotmodule)
        return pycatch("Could not load module 'matplotlib.pyplot'.");

    pyplot_figure = PyObject_GetAttrString(pyplotmodule, "figure");
    if (!PyFunction_Check(pyplot_figure))
        return pycatch("Could not find 'figure'.");

    py_empty_args = PyTuple_New(0);

    fig = PyObject_CallObject(pyplot_figure, py_empty_args);

    if (!fig)
        return pycatch("Call to figure() failed.");

    Py_Finalize();

    printf("Success.\n");
    return 0;
}
