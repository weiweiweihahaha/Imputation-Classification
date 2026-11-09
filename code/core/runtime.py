import multiprocessing as mp
import os


OPENMP_SAFE_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "KMP_INIT_AT_FORK": "FALSE",
}

TENSORFLOW_SAFE_ENV = {
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_USE_LEGACY_KERAS": "1",
    "WRAPT_DISABLE_EXTENSIONS": "true",
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
}


_POSITIVE_INT_ENV_KEYS = {
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
}


def _is_valid_positive_int_string(value):
    try:
        return int(str(value).strip()) > 0
    except Exception:
        return False


def _sanitize_thread_env(env):
    for key, value in OPENMP_SAFE_ENV.items():
        current = env.get(key)
        if key in _POSITIVE_INT_ENV_KEYS:
            if not _is_valid_positive_int_string(current):
                env[key] = value
        elif not current:
            env[key] = value
    return env


def configure_runtime_for_stability():
    _sanitize_thread_env(os.environ)
    for key, value in TENSORFLOW_SAFE_ENV.items():
        os.environ.setdefault(key, value)

    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        # Another start method may already be active in the current process.
        pass


def build_safe_subprocess_env(extra_env=None):
    env = os.environ.copy()
    _sanitize_thread_env(env)
    for key, value in TENSORFLOW_SAFE_ENV.items():
        env.setdefault(key, value)
    if extra_env:
        env.update(extra_env)
        _sanitize_thread_env(env)
        for key, value in TENSORFLOW_SAFE_ENV.items():
            env.setdefault(key, value)
    return env
