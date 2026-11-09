import os


SUPPORTED_COMPUTE_BACKENDS = ("cpu", "gpu", "auto")


def normalize_compute_backend(value):
    backend = str(value or "cpu").strip().lower()
    if backend not in SUPPORTED_COMPUTE_BACKENDS:
        raise ValueError(
            f"Unsupported compute backend: {value}. "
            f"Expected one of {SUPPORTED_COMPUTE_BACKENDS}."
        )
    return backend


def initialize_compute_backend(args):
    requested = normalize_compute_backend(getattr(args, "compute_backend", "cpu"))
    selected = "cpu"

    if requested in {"gpu", "auto"}:
        gpu_device = int(getattr(args, "gpu_device", 0))
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)

        try:
            import cuml
            import cuml.accel

            cuml.accel.install()
            if hasattr(cuml, "set_global_output_type"):
                cuml.set_global_output_type("cupy")
            selected = "gpu"
        except Exception as exc:
            if requested == "gpu":
                raise RuntimeError(
                    "GPU compute backend was requested, but RAPIDS cuML acceleration "
                    "could not be enabled. Install a compatible RAPIDS build or switch "
                    "to --compute_backend cpu."
                ) from exc

    setattr(args, "compute_backend_requested", requested)
    setattr(args, "selected_compute_backend", selected)
    return selected


def enforce_safe_gpu_parallelism(args):
    backend = normalize_compute_backend(
        getattr(args, "selected_compute_backend", getattr(args, "compute_backend", "cpu"))
    )
    notes = []
    if backend != "gpu":
        return notes

    if hasattr(args, "tuning_n_jobs") and int(getattr(args, "tuning_n_jobs", 1)) != 1:
        args.tuning_n_jobs = 1
        notes.append("forcing tuning_n_jobs=1 for a single-GPU classification run")

    if hasattr(args, "parallel_jobs") and int(getattr(args, "parallel_jobs", 1)) != 1:
        args.parallel_jobs = 1
        notes.append("forcing parallel_jobs=1 to avoid multiple workers contending for one GPU")

    return notes
