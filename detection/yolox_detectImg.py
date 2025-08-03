import argparse
import os
import time
from loguru import logger

import cv2

import torch

try:
    from yolox.data.data_augment import ValTransform
    from yolox.data.datasets import COCO_CLASSES
    from yolox.exp import get_exp
    from yolox.utils import fuse_model, get_model_info, postprocess, vis
except Exception as e:
    print(e)

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        #pdb.set_trace()
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, t0, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", t0)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


exp = get_exp("/media/agfoodsensinglab/512ssd/WeedGUIProject/DCW-main/YOLOX-main/exps/example/custom/custom/yolox_l_weed_2021.py", None)

cls_names = COCO_CLASSES
trt_file = None
decoder = None
device = "gpu"
fp16 = False
legacy = False
experiment_name=None
save_result=True

exp.test_conf = 0.25
exp.nmsthre = 0.45
exp.test_size = (640, 640)

model = exp.get_model()
logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

if device == "gpu":
    model.cuda()
    if fp16:
        model.half()  # to FP16
model.eval()

ckpt_file="/media/agfoodsensinglab/512ssd/WeedGUIProject/DCW-main/YOLOX-main/best_ckpt.pth"
logger.info("loading checkpoint")
ckpt = torch.load(ckpt_file, map_location="cpu")
# load the model state dict
model.load_state_dict(ckpt["model"])
logger.info("loaded checkpoint done.")

predictor = Predictor(
    model, exp, COCO_CLASSES, trt_file, decoder,
    device, fp16, legacy,
)

if not experiment_name:
    experiment_name = exp.exp_name

file_name = os.path.join(exp.output_dir, experiment_name)
os.makedirs(file_name, exist_ok=True)

vis_folder = None
if save_result:
    vis_folder = os.path.join(file_name, "vis_res")
    os.makedirs(vis_folder, exist_ok=True)

t0 = time.localtime()
path="/media/agfoodsensinglab/512ssd/WeedGUIProject/DCW-main/YOLOX-main/example1.jpg"
image_demo(predictor, vis_folder, path, t0, save_result)

frame=cv2.imread(path)

outputs, img_info = predictor.inference(frame)

print(1)