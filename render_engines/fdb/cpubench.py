import numpy as np
import time
import os

# Matrix size
N = 1024*32  # You can try 2048 or 4096 if you have enough RAM

# Set number of threads (for MKL/OpenBLAS, if used)

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())  # Or set manually
print(f"Using {os.cpu_count()} threads")

# Initialize matrices
A = np.ones((N, N), dtype=np.float64)
B = np.ones((N, N), dtype=np.float64)

# Time matrix multiplication
start = time.time()
C = np.dot(A, B)
end = time.time()

# Compute GFLOPs: 2 * N^3 / time
flops = 2.0 * N**3
gflops = flops / ((end - start) * 1e9)

print(f"Time: {end - start:.3f} seconds")
print(f"Performance: {gflops:.2f} GFLOPs")
