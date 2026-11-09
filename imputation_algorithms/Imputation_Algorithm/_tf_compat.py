import logging
import os
import warnings


# Keep TensorFlow's legacy-v1 noise from drowning out experiment progress.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# TensorFlow's oneDNN CPU kernels have been a common source of hard crashes
# in our batch-imputation jobs, especially for legacy v1-style graphs such as
# Keep the default runtime conservative unless the caller overrides it.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")


def quiet_tensorflow_logging(tf_module):
    warnings.filterwarnings(
        "ignore",
        message=".*disable_resource_variables.*",
    )

    logger = None
    try:
        logger = tf_module.get_logger()
    except Exception:
        logger = None

    if logger is not None:
        logger.setLevel(logging.ERROR)

    compat_logging = getattr(getattr(tf_module, "compat", None), "v1", None)
    compat_logging = getattr(compat_logging, "logging", None)
    if compat_logging is not None:
        try:
            compat_logging.set_verbosity(compat_logging.ERROR)
        except Exception:
            pass


def disable_v2_behavior_quietly(tf_module):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*disable_resource_variables.*",
        )
        tf_module.disable_v2_behavior()
