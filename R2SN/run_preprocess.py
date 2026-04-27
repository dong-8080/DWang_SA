import ants
import numpy as np
from tqdm import tqdm
import os

def reg_run(input_img, out_dir, atlas):
    print(input_img)
    image = ants.image_read(input_img)
    base_name = os.path.basename(input_img)

    imagedenoise = ants.denoise_image(image, ants.get_mask(image))
    image_n4 = ants.n4_bias_field_correction(imagedenoise)

    out = ants.registration(atlas, image_n4, type_of_transform='SyN')
    reg_img = out['warpedmovout']
    ants.image_write(reg_img, os.path.join(out_dir, base_name))

if __name__ == "__main__":

    atlas_dir = ants.image_read(r'./MNI152_T1_1mm.nii')

    basepath = r'/home/wd/xuanwu/raw'
    outputpath = r"/home/wd/xuanwu/processed"
    for fname in tqdm(os.listdir(basepath)):
        input_img_path = os.path.join(basepath, fname)

        reg_run(input_img_path, outputpath, atlas_dir)