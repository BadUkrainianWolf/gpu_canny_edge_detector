from numba import cuda
import numba as nb
import numpy as np
from timeit import default_timer as timer
from threads_and_blocks_calculator import compute_threads_and_blocks

@cuda.jit
def RGBToBWKernel(source, destination, offset):
    height = source.shape[1]
    width = source.shape[0]
    x,y = cuda.grid(2)
    if (x<width and y<height) :
        r_x= (x + offset) % width
        r_y= (y+offset) % height
        destination[r_x, r_y]=np.uint8(0.3 * source[r_x, r_y, 0] + 0.59 * source[r_x, r_y, 1] + 0.11 * source[r_x, r_y, 2])

def bw_kernel(imagetab, threadsperblockInt=(8, 8)):
    s_image = cuda.to_device(imagetab)
    d_image = cuda.device_array((imagetab.shape[0],imagetab.shape[1],3),dtype = np.uint8)
    threadsperblock, blockspergrid = compute_threads_and_blocks(imagetab, threadsperblockInt)

    for off in range(33,1,-1):
        runs = 6
        result =np.zeros(runs, dtype=np.float32)
        for i in range(runs):
            start = timer()
            RGBToBWKernel[blockspergrid, threadsperblock](s_image, d_image,off) 
            cuda.synchronize()
            dt = timer() - start
            result[i]=dt
                    
    output = d_image.copy_to_host()
    cuda.synchronize()
    return output
    
# if len(sys.argv) < 3:
#     print("Usage: ", sys.argv[0]," <inputFile> <outputFile>")
#     quit(-1)
    
# inputFile = sys.argv[1]
# outputFile=sys.argv[2]


# im = Image.open(inputFile)
# imagetab = np.array(im)

# output = bw_kernel(imagetab)
# m = Image.fromarray(output)
# m.save(outputFile)