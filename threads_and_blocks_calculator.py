import math
import sys
import numpy as np
from PIL import Image

DEFAULT_THREADS_PER_BLOCK = (8, 8)

def compute_threads_and_blocks(imagetab, threadsperblock=(8, 8)):
    width, height = imagetab.shape[:2]
    blockspergrid_x = math.ceil(width / threadsperblock[0])
    blockspergrid_y = math.ceil(height / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    return threadsperblock,blockspergrid
    
def load_image(inputFile):
    im = Image.open(inputFile)
    imagetab = np.array(im)
    return imagetab
