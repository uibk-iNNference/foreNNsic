import numpy as np
from PIL import Image
from glob import glob
import os.path
from os import makedirs


def main(output_dir: str):
    boundary_paths = glob('boundaries/*.npy')
    for path in boundary_paths:
        if 'batch' in path:
            continue
        matrix = np.load(path)[0]
        matrix = (matrix * 255).astype(np.int8)
        if matrix.shape[2] == 1:
            matrix = matrix[:, :, 0]
            mode = "L"
        else:
            mode = "RGB"
        im = Image.fromarray(matrix, mode=mode)

        basename = os.path.basename(path)
        without_ext = os.path.splitext(basename)[0]

        makedirs(output_dir, exist_ok=True)
        im.convert("RGB").save(os.path.join(output_dir, f"{without_ext}.png"))


if __name__ == "__main__":
    main('/tmp/adv_images')
