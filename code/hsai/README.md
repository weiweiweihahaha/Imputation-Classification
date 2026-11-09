# HSR: Hard-Sample Refine

HSR is a method-agnostic refinement layer applied after a framework imputer.
It adjusts only missing cells using an anchored hard-sample objective and an
internal validation guard.

The implementation is in `hs_refine.py`; the full experiment entry point is:

```bash
python -m code.hsai.experiment_refine
```

The implementation uses the project's required scikit-learn dependency.
