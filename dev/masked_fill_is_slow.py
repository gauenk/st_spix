import time
import torch as th

# Simulate data and masks
data = th.randn(5, 3, 100, 100)
mask = th.randn(5, 3, 100, 100) > 0
data1 = th.randn(5, 3, 100, 100)
mask1 = th.randn(5, 3, 100, 100) > 0
data1.masked_fill_(mask1, 1)
data.masked_fill_(mask, -1)

# Synchronize before starting timer if using CUDA
# if th.cuda.is_available():
th.cuda.synchronize()

# Start the timer
start_time = time.perf_counter()

# Perform the operation
data.masked_fill_(mask, -1)

# Synchronize after operation to ensure all GPU tasks are done
# if th.cuda.is_available():
th.cuda.synchronize()

# End the timer
end_time = time.perf_counter()

# Calculate elapsed time
elapsed_time = end_time - start_time

# Output the elapsed time
print(f"Elapsed time: {elapsed_time:.6f} seconds")
