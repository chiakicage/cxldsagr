#define PY_SSIZE_T_CLEAN
#include <Python.h>

// The caller owns an exact list snapshot. Keep the GIL throughout this scan.
static PyObject* nonnegative_builtin_ints(PyObject*, PyObject* snapshot) {
    if (!PyList_CheckExact(snapshot)) {
        PyErr_SetString(PyExc_TypeError, "expected an exact list snapshot");
        return nullptr;
    }
    const Py_ssize_t count = PyList_GET_SIZE(snapshot);
    if (count == 0) {
        Py_RETURN_FALSE;
    }
    bool negative = false;
    for (Py_ssize_t index = 0; index < count; ++index) {
        PyObject* value = PyList_GET_ITEM(snapshot, index);
        if (!PyLong_CheckExact(value)) {
            Py_RETURN_FALSE;
        }
        int overflow = 0;
        const long long decoded = PyLong_AsLongLongAndOverflow(value, &overflow);
        if (decoded == -1 && PyErr_Occurred()) {
            return nullptr;
        }
        // Positive arbitrary-size ints stay valid; later packing owns overflow.
        // Finish the type scan even after seeing a negative integer.
        negative |= overflow < 0 || (overflow == 0 && decoded < 0);
    }
    return PyBool_FromLong(!negative);
}

static PyMethodDef methods[] = {
    {"nonnegative_builtin_ints", nonnegative_builtin_ints, METH_O,
     "Check an owned list for exact nonnegative builtin integers."},
    {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_cxldsagr_token_validation", nullptr, -1, methods,
    nullptr, nullptr, nullptr, nullptr,
};

PyMODINIT_FUNC PyInit__cxldsagr_token_validation() {
    return PyModule_Create(&module);
}
