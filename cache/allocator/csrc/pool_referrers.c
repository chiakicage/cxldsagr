/* External, exact-CPython-3.12.13 pool-filter feasibility prototype. */
#define PY_SSIZE_T_CLEAN
#define Py_BUILD_CORE_MODULE
#include <Python.h>
#include "internal/pycore_interp.h"
#include "internal/pycore_frame.h"
#include "internal/pycore_runtime.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

PyMODINIT_FUNC PyInit_gc(void);
PyMODINIT_FUNC PyInit__signal(void);

typedef struct {
    PyObject *gc_module, *originals[4], *keys[4], *subclasscheck_key;
    PyObject *gc_key, *filtered_entry;
    PyObject *primitive_keys[4], *primitive_values[4], *pool_types_key;
    PyObject *helper, *helper_code, *pool_types, *pool_types_code;
    PyObject *signal_module, *signal_getter, *default_int_handler, *signal_numbers;
    PyCFunction getsignal_fn;
    int busy;
    Py_ssize_t visits, candidates, traversals, cache_hits;
    unsigned long long filtered_calls, unfiltered_calls, fallback_calls, audit_errors;
    const char *route;
    const char *context_reason;
} State;

static const char *names[4] = {"get_objects", "get_referrers", "get_stats", "get_freeze_count"};
static const char *primitive_names[4] = {"any", "issubclass", "type", "id"};

static int collecting(void) { return PyInterpreterState_Get()->gc.collecting; }

static int identities_match(State *state) {
    if (!PyModule_CheckExact(state->gc_module)) return 0;
    PyObject *dict = PyModule_GetDict(state->gc_module);
    for (int i = 0; i < 4; i++)
        if (PyDict_GetItemWithError(dict, state->keys[i]) != state->originals[i]) return 0;
    return 1;
}

static PyObject *effective_lookup(PyObject *globals, PyObject *builtins, PyObject *key) {
    if (!PyDict_CheckExact(globals) || !PyDict_CheckExact(builtins)) return NULL;
    PyObject *value = PyDict_GetItemWithError(globals, key);
    return value ? value : PyDict_GetItemWithError(builtins, key);
}

static int helper_lookups_match(State *state) {
    _PyInterpreterFrame *frame = PyThreadState_Get()->cframe->current_frame;
    if (!state->helper || !frame || frame->owner == FRAME_OWNED_BY_CSTACK ||
        frame->f_funcobj != state->helper || (PyObject *)frame->f_code != state->helper_code ||
        !PyFunction_Check(state->pool_types) ||
        PyFunction_GET_CODE(state->pool_types) != state->pool_types_code) return 0;
    if (effective_lookup(frame->f_globals, frame->f_builtins, state->pool_types_key) != state->pool_types) return 0;
    if (effective_lookup(frame->f_globals, frame->f_builtins, state->gc_key) != state->gc_module) return 0;
    PyFunctionObject *pool_types = (PyFunctionObject *)state->pool_types;
    for (int i = 0; i < 4; i++) {
        if (effective_lookup(frame->f_globals, frame->f_builtins, state->primitive_keys[i]) != state->primitive_values[i]) return 0;
        if (i >= 2 && effective_lookup(pool_types->func_globals, pool_types->func_builtins,
                                      state->primitive_keys[i]) != state->primitive_values[i]) return 0;
    }
    return 1;
}

static int observation_or_pending_work(void) {
    PyThreadState *thread = PyThreadState_Get();
    PyInterpreterState *interp = thread->interp;
    if (thread->c_tracefunc || thread->c_profilefunc) return 1;
    /* pycore_ceval.h uses NULL for the normal default evaluator. */
    if (interp->eval_frame && interp->eval_frame != _PyEval_EvalFrameDefault) return 2;
    for (int event = 0; event < _PY_MONITORING_UNGROUPED_EVENTS; event++)
        if (interp->monitors.tools[event]) return 3;
    /* Conservatively include registered but currently inactive callbacks/tools.
       This covers local-only monitoring without walking or allocating frames. */
    for (int tool = 0; tool < PY_MONITORING_TOOL_IDS; tool++) {
        if (interp->monitoring_tool_names[tool]) return 3;
        for (int event = 0; event < _PY_MONITORING_EVENTS; event++)
            if (interp->monitoring_callables[tool][event]) return 3;
    }
    struct _pending_calls *local = &interp->ceval.pending;
    struct _pending_calls *main = &_PyRuntime.ceval.pending_mainthread;
    return (local->busy || main->busy ||
           _Py_atomic_load_relaxed(&local->calls_to_do) ||
           _Py_atomic_load_relaxed(&main->calls_to_do) ||
           _Py_atomic_load_relaxed(&_PyRuntime.ceval.signals_pending)) ? 4 : 0;
}

static int standard_signal_handlers(State *state) {
    if (!PyModule_CheckExact(state->signal_module)) return 0;
    PyObject *dict = PyModule_GetDict(state->signal_module);
    PyObject *current = PyDict_GetItemString(dict, "getsignal");
    if (current != state->signal_getter) return 0;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(state->signal_numbers); i++) {
        PyObject *handler = state->getsignal_fn(state->signal_module,
                                             PyTuple_GET_ITEM(state->signal_numbers, i));
        if (!handler) return -1;
        int allowed = handler == state->default_int_handler;
        if (PyLong_CheckExact(handler)) {
            long value = PyLong_AsLong(handler);
            allowed = value == 0 || value == 1;
        }
        Py_DECREF(handler);
        if (!allowed) return 0;
    }
    return 1;
}

static int execution_context_supported(State *state) {
    state->context_reason = "supported";
    if (!helper_lookups_match(state)) { state->context_reason = "helper_lookups"; return 0; }
    int observed = observation_or_pending_work();
    const char *reasons[] = {"supported", "trace_profile", "frame_evaluator", "monitoring", "pending_work"};
    if (observed) { state->context_reason = reasons[observed]; return 0; }
    int standard = standard_signal_handlers(state);
    if (standard == 0) state->context_reason = "signal_handler";
    return standard;
}

static int standard_subclasscheck(State *state, PyTypeObject *kind) {
    /* Lookup through real type MRO/dicts, with no Python attribute dispatch. */
    PyObject *actual = _PyType_Lookup(Py_TYPE(kind), state->subclasscheck_key);
    PyObject *standard = _PyType_Lookup(&PyType_Type, state->subclasscheck_key);
    return actual == standard && standard != NULL;
}

static int supported_args(State *state, PyObject *args) {
    if (!PyTuple_GET_SIZE(args)) return 0;
    PyObject *kind = PyTuple_GET_ITEM(args, 0);
    if (!PyType_Check(kind) || !standard_subclasscheck(state, (PyTypeObject *)kind)) return 0;
    if (!(((PyTypeObject *)kind)->tp_flags & Py_TPFLAGS_HAVE_GC)) return 0;
    for (Py_ssize_t i = 1; i < PyTuple_GET_SIZE(args); i++) {
        PyObject *child = PyTuple_GET_ITEM(args, i);
        if (!PyType_Check(child) || !PyType_IsSubtype((PyTypeObject *)child, (PyTypeObject *)kind)) return 0;
    }
    return 1;
}

static PyObject *fallback(State *state, PyObject *args, const char *route) {
    state->fallback_calls++;
    state->route = route;
    /* resolve_referrers already bound the original builtin at the original
       Python lookup position. Preserve that target even if callbacks changed
       a later module/global lookup. Other targets never enter this function. */
    return PyObject_Call(state->originals[1], args, NULL);
}

static PyObject *resolve_referrers(PyObject *module, PyObject *target) {
    State *state = PyModule_GetState(module);
    /* Return custom wrappers unchanged. Keep *kinds argument expansion at its
       original Python position, with no extra target-containing args tuple. */
    return Py_NewRef(target == state->originals[1] ? state->filtered_entry : target);
}

static int refers_to_original_kind(PyObject *object, void *args) {
    PyObject *kinds = args;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(kinds); i++)
        if (object == PyTuple_GET_ITEM(kinds, i)) return 1;
    return 0;
}

typedef struct { PyTypeObject *kind; int match; } TypeEntry;
#define TYPE_CACHE_SIZE 512

static int scan_ordinary(State *state, PyObject *args, PyObject *result, int filter) {
    PyTypeObject *pool_type = (PyTypeObject *)PyTuple_GET_ITEM(args, 0);
    TypeEntry cache[TYPE_CACHE_SIZE] = {{NULL, 0}};
    struct _gc_runtime_state *gcstate = &PyInterpreterState_Get()->gc;
    for (int generation = 0; generation < NUM_GENERATIONS; generation++) {
        PyGC_Head *head = &gcstate->generations[generation].head;
        for (PyGC_Head *gc = _PyGCHead_NEXT(head); gc != head; gc = _PyGCHead_NEXT(gc)) {
            PyObject *object = (PyObject *)((char *)gc + sizeof(PyGC_Head));
            state->visits++;
            if (object == args || object == result) continue;
            PyTypeObject *kind = Py_TYPE(object);
            if (filter) {
                size_t slot = ((uintptr_t)kind >> 4) & (TYPE_CACHE_SIZE - 1);
                TypeEntry *entry = &cache[slot];
                if (entry->kind != kind) {
                    entry->kind = kind;
                    entry->match = PyType_IsSubtype(kind, pool_type);
                } else {
                    state->cache_hits++;
                }
                if (!entry->match) continue;
                state->candidates++;
            }
            state->traversals++;
            /* Keep the original get_referrers argument-membership semantics:
               a new subclass absent from args is not detected until retry. */
            if (kind->tp_traverse(object, refers_to_original_kind, args)) {
                if (PyList_Append(result, object) < 0) return -1;
            }
        }
    }
    return 0;
}

static PyObject *pool_referrers(PyObject *module, PyObject *args) {
    State *state = PyModule_GetState(module);
    state->visits = state->candidates = state->traversals = state->cache_hits = 0;
    if (state->busy) return fallback(state, args, "fallback_reentrant_before_audit");
    if (collecting()) return fallback(state, args, "fallback_collection_before_audit");
    if (!identities_match(state)) return fallback(state, args, "fallback_identity_before_audit");
    int context_ok = execution_context_supported(state);
    if (context_ok < 0) return NULL;
    if (!context_ok) return fallback(state, args, "fallback_context_before_audit");
    if (!supported_args(state, args)) return fallback(state, args, "fallback_type_before_audit");
    state->busy = 1;
    /* Same event shape and order as gc_get_referrers in pinned gcmodule.c. */
    if (PySys_Audit("gc.get_referrers", "(O)", args) < 0) {
        state->audit_errors++;
        state->busy = 0;
        state->route = "audit_error";
        return NULL;
    }
    PyObject *result = PyList_New(0);
    if (!result) {
        state->busy = 0;
        state->route = "result_allocation_error";
        return NULL;
    }
    /* An audit/GC callback may change the metaclass or hierarchy. Continue the
       already-started builtin call with an unfiltered copy of its original
       scan; do not call the audited fallback or emit another event. Changes
       to gc attributes do not replace a builtin call already in progress. */
    context_ok = execution_context_supported(state);
    if (context_ok < 0) { state->busy = 0; Py_DECREF(result); return NULL; }
    int filter = identities_match(state) && context_ok && supported_args(state, args);
    state->route = filter ? "filtered_ordinary" : "unfiltered_after_callback";
    if (filter) state->filtered_calls++; else state->unfiltered_calls++;
    int error = scan_ordinary(state, args, result, filter);
    state->busy = 0;
    if (error) { Py_DECREF(result); return NULL; }
    return result;
}

static PyObject *configure_helper(PyObject *module, PyObject *args) {
    State *state = PyModule_GetState(module);
    PyObject *helper, *pool_types;
    if (!PyArg_ParseTuple(args, "OO:configure_helper", &helper, &pool_types)) return NULL;
    if (!PyFunction_Check(helper) || !PyFunction_Check(pool_types) || state->helper) {
        PyErr_SetString(PyExc_ValueError, "configure once with exact Python functions");
        return NULL;
    }
    state->helper = Py_NewRef(helper);
    state->helper_code = Py_NewRef(PyFunction_GET_CODE(helper));
    state->pool_types = Py_NewRef(pool_types);
    state->pool_types_code = Py_NewRef(PyFunction_GET_CODE(pool_types));
    Py_RETURN_NONE;
}

static PyObject *last_scan(PyObject *module, PyObject *unused) {
    (void)unused;
    State *state = PyModule_GetState(module);
    return Py_BuildValue("{s:s,s:n,s:n,s:n,s:n,s:i,s:s}",
                         "route", state->route ? state->route : "not_run",
                         "visits", state->visits, "candidates", state->candidates,
                         "tp_traverse_calls", state->traversals,
                         "type_cache_hits", state->cache_hits,
                         "active_collection", collecting(),
                         "context_reason", state->context_reason ? state->context_reason : "not_checked");
}

static PyObject *build_info(PyObject *module, PyObject *unused) {
    (void)module; (void)unused;
    return Py_BuildValue("{s:s,s:n,s:n,s:n,s:n,s:i}",
                         "header_version", PY_VERSION,
                         "gc_offset", (Py_ssize_t)offsetof(PyInterpreterState, gc),
                         "generations_offset", (Py_ssize_t)offsetof(PyInterpreterState, gc.generations),
                         "permanent_offset", (Py_ssize_t)offsetof(PyInterpreterState, gc.permanent_generation),
                         "collecting_offset", (Py_ssize_t)offsetof(PyInterpreterState, gc.collecting),
                         "num_generations", NUM_GENERATIONS);
}

static PyObject *counters(PyObject *module, PyObject *unused) {
    (void)unused;
    State *state = PyModule_GetState(module);
    return Py_BuildValue("{s:K,s:K,s:K,s:K}", "filtered_calls", state->filtered_calls,
                         "unfiltered_calls", state->unfiltered_calls,
                         "fallback_calls", state->fallback_calls,
                         "audit_errors", state->audit_errors);
}

static int module_traverse(PyObject *module, visitproc visit, void *arg) {
    State *state = PyModule_GetState(module);
    Py_VISIT(state->gc_module);
    Py_VISIT(state->gc_key); Py_VISIT(state->filtered_entry);
    Py_VISIT(state->subclasscheck_key);
    Py_VISIT(state->pool_types_key);
    Py_VISIT(state->helper); Py_VISIT(state->helper_code);
    Py_VISIT(state->pool_types); Py_VISIT(state->pool_types_code);
    Py_VISIT(state->signal_module); Py_VISIT(state->signal_getter);
    Py_VISIT(state->default_int_handler); Py_VISIT(state->signal_numbers);
    for (int i = 0; i < 4; i++) { Py_VISIT(state->primitive_keys[i]); Py_VISIT(state->primitive_values[i]); }
    for (int i = 0; i < 4; i++) { Py_VISIT(state->keys[i]); Py_VISIT(state->originals[i]); }
    return 0;
}

static int module_clear(PyObject *module) {
    State *state = PyModule_GetState(module);
    Py_CLEAR(state->gc_module);
    Py_CLEAR(state->gc_key); Py_CLEAR(state->filtered_entry);
    Py_CLEAR(state->subclasscheck_key);
    Py_CLEAR(state->pool_types_key);
    Py_CLEAR(state->helper); Py_CLEAR(state->helper_code);
    Py_CLEAR(state->pool_types); Py_CLEAR(state->pool_types_code);
    Py_CLEAR(state->signal_module); Py_CLEAR(state->signal_getter);
    Py_CLEAR(state->default_int_handler); Py_CLEAR(state->signal_numbers);
    for (int i = 0; i < 4; i++) { Py_CLEAR(state->primitive_keys[i]); Py_CLEAR(state->primitive_values[i]); }
    for (int i = 0; i < 4; i++) { Py_CLEAR(state->keys[i]); Py_CLEAR(state->originals[i]); }
    return 0;
}

static PyMethodDef methods[] = {
    {"pool_referrers", pool_referrers, METH_VARARGS, "Filter only for the unchanged pool-helper contract."},
    {"resolve_referrers", resolve_referrers, METH_O, NULL},
    {"last_scan", last_scan, METH_NOARGS, NULL},
    {"build_info", build_info, METH_NOARGS, NULL},
    {"configure_helper", configure_helper, METH_VARARGS, NULL},
    {"counters", counters, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef definition = {
    PyModuleDef_HEAD_INIT, "_cxldsagr_nosa_pool_referrers", NULL, sizeof(State), methods,
    NULL, module_traverse, module_clear, NULL
};

PyMODINIT_FUNC PyInit__cxldsagr_nosa_pool_referrers(void) {
    if (PY_VERSION_HEX != 0x030c0df0 || strncmp(Py_GetVersion(), "3.12.13 ", 8) ||
        NUM_GENERATIONS != 3 || offsetof(PyInterpreterState, gc.generations) != 0x88 ||
        offsetof(PyInterpreterState, gc.permanent_generation) != 0xd8) {
        PyErr_SetString(PyExc_ImportError, "unsupported CPython private GC layout");
        return NULL;
    }
    PyObject *module = PyModule_Create(&definition);
    if (!module) return NULL;
    State *state = PyModule_GetState(module);
    state->gc_module = PyImport_ImportModule("gc");
    state->gc_key = PyUnicode_InternFromString("gc");
    state->filtered_entry = PyObject_GetAttrString(module, "pool_referrers");
    state->subclasscheck_key = PyUnicode_InternFromString("__subclasscheck__");
    if (!state->gc_module || !state->subclasscheck_key || !state->gc_key || !state->filtered_entry) goto error;
    if (!PyModule_CheckExact(state->gc_module)) {
        PyErr_SetString(PyExc_ImportError, "gc must be the exact builtin module");
        goto error;
    }
    PyModuleDef *gc_definition = (PyModuleDef *)PyInit_gc();
    for (int i = 0; i < 4; i++) {
        state->keys[i] = PyUnicode_InternFromString(names[i]);
        state->originals[i] = PyObject_GetAttrString(state->gc_module, names[i]);
        if (!state->keys[i] || !state->originals[i]) goto error;
        PyMethodDef *method = gc_definition->m_methods;
        while (method->ml_name && strcmp(method->ml_name, names[i])) method++;
        if (!method->ml_name || !PyCFunction_Check(state->originals[i]) ||
            PyCFunction_GetFunction(state->originals[i]) != method->ml_meth ||
            PyCFunction_GetFlags(state->originals[i]) != method->ml_flags) {
            PyErr_SetString(PyExc_ImportError, "gc callable differs from pinned module definition");
            goto error;
        }
    }
    state->pool_types_key = PyUnicode_InternFromString("_python_pool_types");
    if (!state->pool_types_key) goto error;
    PyObject *builtins_copy = PyInterpreterState_Get()->builtins_copy;
    if (!builtins_copy || !PyDict_CheckExact(builtins_copy)) {
        PyErr_SetString(PyExc_ImportError, "original builtin inventory unavailable"); goto error;
    }
    for (int i = 0; i < 4; i++) {
        state->primitive_keys[i] = PyUnicode_InternFromString(primitive_names[i]);
        if (!state->primitive_keys[i]) goto error;
        PyObject *value = PyDict_GetItemWithError(builtins_copy, state->primitive_keys[i]);
        if (!value) { PyErr_SetString(PyExc_ImportError, "original builtin missing"); goto error; }
        state->primitive_values[i] = Py_NewRef(value);
    }
    state->signal_module = PyImport_ImportModule("_signal");
    if (!state->signal_module) goto error;
    if (!PyModule_CheckExact(state->signal_module)) {
        PyErr_SetString(PyExc_ImportError, "_signal must be exact builtin module"); goto error;
    }
    PyModuleDef *signal_definition = (PyModuleDef *)PyInit__signal();
    const char *signal_names[] = {"getsignal", "default_int_handler", "valid_signals"};
    PyObject *signal_functions[3] = {NULL, NULL, NULL};
    for (int i = 0; i < 3; i++) {
        PyMethodDef *method = signal_definition->m_methods;
        while (method->ml_name && strcmp(method->ml_name, signal_names[i])) method++;
        PyObject *function = PyObject_GetAttrString(state->signal_module, signal_names[i]);
        if (!function || !method->ml_name || !PyCFunction_Check(function) ||
            PyCFunction_GetFunction(function) != method->ml_meth ||
            PyCFunction_GetFlags(function) != method->ml_flags) {
            Py_XDECREF(function);
            for (int j = 0; j < i; j++) Py_DECREF(signal_functions[j]);
            PyErr_SetString(PyExc_ImportError, "signal builtin identity unsupported"); goto error;
        }
        signal_functions[i] = function;
    }
    state->signal_getter = signal_functions[0];
    state->default_int_handler = signal_functions[1];
    if (PyCFunction_GetFlags(state->signal_getter) != METH_O) {
        Py_DECREF(signal_functions[2]);
        PyErr_SetString(PyExc_ImportError, "unexpected getsignal convention"); goto error;
    }
    state->getsignal_fn = PyCFunction_GetFunction(state->signal_getter);
    PyObject *valid_signals = PyObject_CallNoArgs(signal_functions[2]);
    Py_DECREF(signal_functions[2]);
    if (!valid_signals) goto error;
    state->signal_numbers = PySequence_Tuple(valid_signals);
    Py_DECREF(valid_signals);
    if (!state->signal_numbers) goto error;
    return module;
error:
    Py_DECREF(module);
    return NULL;
}
