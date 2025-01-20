
import math
core_num = 12
min_workers = core_num * 2 + 1
total_gpu = 15.0 * 1024.0 * 1024.0 
val_used_gpu = 2.38 * 1024.0 * 1024.0 
real_used_gpu = 7.93 * 1024.0 * 1024.0
min_workers = min(min_workers, total_gpu / val_used_gpu)
min_workers = math.floor(min_workers)
if min_workers % 2 == 0:
    min_workers -= 1
press_workers = max(min_workers, 1)

print(press_workers, press_workers * val_used_gpu, real_used_gpu)
miss_gpu_size = (real_used_gpu - press_workers * val_used_gpu) / (1024.0 * 1024.0)
print(miss_gpu_size)

# 27%~41%/963.50M;23%~31%/963.50M;9%~39%/963.50M
# 100%~100%/7.93G;75%~82%/7.93G;100%~100%/7.93G