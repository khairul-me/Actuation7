"""
need 1min for 800x1400
"""
from spectral import *
from  pathlib import Path
import cv2 as cv
import numpy as np
from modeling_spectral_data import find_nearest

use_ref1 = False

root_dir = r'E:\PHD\Blueberry\Blueberry\Blueberry_Pretest'
names = [ 'Blueberry_Pretest_Sample000_05112022', 'Blueberry_Pretest_Dark_05112022', 'Blueberry_Pretest_White_05112022']

path = Path(root_dir) / f'{names[0]}.bil'.__str__()
path_head = Path(root_dir) / f'{names[0]}.bil.hdr'.__str__()

ref_path = Path(root_dir) / f'{names[1]}.bil'.__str__()
ref_path_head = Path(root_dir) / f'{names[1]}.bil.hdr'.__str__()

ref1_path = Path(root_dir) / f'{names[2]}.bil'.__str__()
ref1_path_head = Path(root_dir) / f'{names[2]}.bil.hdr'.__str__()


root_dir = r'E:\PHD\Blueberry\Blueberry\Blueberry Scans for Destructive Testing 06132022'
root_dir = Path(root_dir)
good_dir = root_dir / 'Good Blueberry Scans'
bad_dir = root_dir / 'Bad Blueberry Scans'

#names = ['Good Blueberry 1-42', 'Good Blueberry 43-84']
names = ['Bad Blueberry 1-42', 'Bad Blueberry 43-84', 'Bad Blueberry 85-126']

data_dir = bad_dir

path = Path(data_dir) / f'{names[0]}.bil'.__str__()
path_head = Path(data_dir) / f'{names[0]}.bil.hdr'.__str__()


ref_names = ['WhiteReference']
ref_path = Path(root_dir) / f'{ref_names[0]}.bil'.__str__()
ref_path_head = Path(root_dir) / f'{ref_names[0]}.bil.hdr'.__str__()


from spectral.io import envi
"""
    Returns:

        :class:`spectral.SpyFile` or :class:`spectral.io.envi.SpectralLibrary`
        object.
"""
data = envi.open(path_head, path)
ref_data = envi.open(ref_path_head, ref_path)
if use_ref1:
    ref1_data = envi.open(ref1_path_head, ref1_path)
else:
    ref1_data = None

import numpy as np
from glob import glob
import matplotlib
matplotlib.use('TkAgg')
#%matplotlib inline
from PIL import Image
from matplotlib import pyplot as plt
from felzenszwalb_segmentation_revised import segment
import random
import numpy as np

from  pathlib import Path
wavelengths = data.metadata['wavelength']
wavelengths = [float(x) for x in wavelengths]
save_dir= r'E:\PHD\Blueberry\Blueberry\one_sample_wavelength_change'
save_dir = Path(save_dir)

save_path = save_dir/'origin1'
save_path = save_path / f'*.png'

save_path1 = save_dir/'segment1'
save_path1 = save_path1 / f'*.png'

image_files = glob(save_path.__str__())
image_files1 = glob(save_path1.__str__())
image_files = sorted(image_files, key=lambda x: float(Path(x).name.split('_')[0]))
image_files1 = sorted(image_files1, key=lambda x: float(Path(x).name.split('_')[0]))
im_num = len(image_files)
im_num1 = len(image_files1)

norm = plt.Normalize(1,4)
cmap = plt.cm.RdYlGn
np.random.seed(10)
c = np.random.randint(1,5,size=15)
first_time = True

target_wavelength = 900
im_band, im_idx = find_nearest(wavelengths, target_wavelength)

def compare_im_with_process(event):
    global first_time, im_idx
    if True:
    #for i_im in range(im_num):
    #    if i_im <= 364:
    #        continue
        if first_time is False:
            if event.key == 'a':
                im_idx -= 1
            if event.key == 'd':
                im_idx += 1
        i_im = im_idx
        image = np.array(Image.open(image_files[i_im]))
        #segmented_image = np.array(Image.open(image_files1[i_im]))
        segmented_image = segment(image, 0.2, 400, 50)
        segmented_image = segmented_image.astype(np.uint8)

        fig = plt.figure(0, figsize=(6, 6))
        fig.clear()
        ax1 = fig.add_subplot(1, 2, 1)
        handle1 = plt.imshow(image)
        ax2 = fig.add_subplot(1, 2, 2)
        handle2= plt.imshow(segmented_image)
        ax1.set_title(wavelengths[i_im])

        annot1 = ax1.annotate("", xy=(0,0), xytext=(40,40),textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot1.set_visible(False)

        annot2 = ax2.annotate("", xy=(0,0), xytext=(40,40),textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot2.set_visible(False)

        def on_move(event):
            if event.inaxes:
                x, y  = event.xdata, event.ydata
                annot1.xy = [x, y]
                x = round(x)
                y = round(y)
                text = f"{x}, {y}, {image[y, x]}"
                annot1.set_text(text)
                annot1.get_bbox_patch().set_facecolor(cmap(norm(c[0])))
                annot1.get_bbox_patch().set_alpha(0.4)
                annot1.set_visible(True)

                annot2.xy = [x, y]
                text = f"{x}, {y}, {segmented_image[y, x]}"
                annot2.set_text(text)
                annot2.get_bbox_patch().set_facecolor(cmap(norm(c[0])))
                annot2.get_bbox_patch().set_alpha(0.4)
                annot2.set_visible(True)
                fig.canvas.draw_idle()
            else:
                annot1.set_visible(False)
                annot2.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_move)
        if first_time is True:
            cid = fig.canvas.mpl_connect('key_press_event', compare_im_with_process)
        if first_time is True:
            plt.show()
            first_time = False
        else:
            plt.show(block=False)
compare_im_with_process(0)