def check_morpheus_import():
    try:
        import morpheus

        _ = morpheus._version

        return True
    except ImportError:
        return False


def check_cuda_driver():
    try:
        import cupy

        import cudf

        _ = cupy.cuda.runtime.driverGetVersion()
        _ = cudf.DataFrame({"a": [1, 2, 3]})
        return True
    except Exception as e:
        print(f"\nError: {e}\n", flush=True)
        return False


MORPHEUS_IMPORT_OK = check_morpheus_import()
CUDA_DRIVER_OK = check_cuda_driver()
