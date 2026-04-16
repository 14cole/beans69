from fmm_helmholtz_2d import FMMOperator
print("C loaded" if FMMOperator._load_c_lib() else "Python fallback")