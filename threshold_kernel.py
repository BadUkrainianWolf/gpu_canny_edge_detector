from numba import cuda
from threads_and_blocks_calculator import compute_threads_and_blocks

LOW = 51
HIGH = 102

@cuda.jit
def threshold_kernel(source, destination):
    x, y = cuda.grid(2)
#     This scheme is used to determine the two neighboring 
# pixels that are used in the local maximum testing. Afterwards, 
# the center pixel is classified based on its strength relative to the 
# two predefined thresholds - non-edge pixels are brought to 0, 
# weak pixels are brought to 127, and strong pixels are brought to 
# 255. These results are placed within an intermediate shared tile 
# (of identical dimensions to the first shared tile), as a search is 
# subsequently performed around each weak pixel for any 
# immediately-connected strong pixels so they may be 
# reclassified as either strong or non-edge pixels. 

    if x < source.shape[0] and y < source.shape[1]:
        if source.ndim == 2:  # 2D image
            if source[x, y] >= HIGH:
                destination[x, y] = 255
            elif source[x, y] < LOW:
                destination[x, y] = 0
            else:
                destination[x, y] = 127
        elif source.ndim == 3:  # 3D image (RGB)
            for c in range(source.shape[2]):  # Iterate over color channels
                if source[x, y, c] >= HIGH:
                    destination[x, y, c] = 255
                elif source[x, y, c] < LOW:
                    destination[x, y, c] = 0
                else:
                    destination[x, y, c] = 127

def threshold(image, threadsperblockInt=(8, 8)):
    s_image = cuda.to_device(image)
    d_image = cuda.device_array_like(s_image)
    threadsperblock, blockspergrid = compute_threads_and_blocks(image, threadsperblockInt)
    threshold_kernel[blockspergrid, threadsperblock](s_image, d_image)
    output = d_image.copy_to_host()

    cuda.synchronize()

    return output

# if len(sys.argv) < 3:
#    print("Usage: ", sys.argv[0], " <inputFile> <outputFile>")
#    sys.exit(-1)

# inputFile = sys.argv[1]
# outputFile = sys.argv[2]

# image_source = load_image(inputFile)

# image_output = sobel_filter(image_source)

# image_output_save = threshold(image_output)

# m = Image.fromarray(image_output_save)
# m.save(outputFile)