#!/usr/bin/env python3

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg_openvino_model     # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov5s-seg_paddle_model       # PaddlePaddle
"""
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from yolov5.msg import YoloResult, YoloResultList

import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams, letterbox
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, scale_image
from utils.torch_utils import select_device, smart_inference_mode

from threading import Thread
import numpy as np

class LoadTopic:
    # load ros topic image stream
    def __init__(self, sources='/camera/color/image_raw', img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = 'topic'
        self.sub_img=rospy.Subscriber(sources, Image, self.image_cb)
        self.bridge = CvBridge()
        self.sources = sources
        self.img_size = img_size
        self.stride = stride
        self.vid_stride = vid_stride  # video frame-rate stride
        self.imgs = []

        # check for common shapes
        self.rect = None
        self.auto = None
        self.transforms = None
        if not self.rect:
            LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def image_cb(self,Image): 
        try:
            self.imgs=[self.bridge.imgmsg_to_cv2(Image,"bgr8")]
            
        except CvBridgeError as e:
            print(e)
            self.img=None
            return
        except Exception as e:
            print(e)
            self.img=None
            return
        
    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if rospy.is_shutdown():
            raise StopIteration

        while len(self.imgs)==0:
            rospy.logwarn("no image!")
            rospy.sleep(1)
        
        im0 = self.imgs.copy()
        if self.transforms:
            im = np.stack([self.transforms(x) for x in im0])  # transforms
        else:
            im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])  # resize
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous

        return self.sources, im, im0, None, ''

    def __len__(self):
        return 0

@smart_inference_mode()
def run(
    weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
    source='/camera/color/image_raw',  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    retina_masks=False,
):
    
    if type(imgsz)==int:
        imgsz = (imgsz, imgsz)
    classes=None  # filter by class: --class 0, or --class 0 2 3
    print("wei", weights)
    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadTopic(sources=source, img_size=imgsz, stride=stride, auto=True)
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f'{i}: '

            s += '%gx%g ' % im.shape[2:]  # print string

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # mask
                # mask_np = masks.detach().cpu().numpy()
                # mask_np = scale_image(mask_np[0].shape, mask_np[0], im0.shape)
                
                # Mask plotting
                annotator.masks(masks,
                                colors=[colors(x, True) for x in det[:, 5]],
                                im_gpu=None if retina_masks else im[i])

                # pub text
                segments = reversed(masks2segments(masks))
                segments = [scale_segments(im.shape[2:], x, im0.shape, normalize=True) for x in segments]

                # Yolo Result List
                yrl = []
                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)

                    # pub Yolo Result
                    yr = YoloResult()
                    yr.cls = int(cls)
                    print(segments[j].shape, type(segments[j]))
                    ss = segments[j].T
                    print(ss.shape)
                    yr.x=ss[0]
                    yr.y=ss[1]
                    yrl.append(yr)
                
                pub_res.publish(YoloResultList(yrl))
            
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def params():
    opt = {
        "weights"       :    rospy.get_param("/yolov5/weights"),
        "source"        :    rospy.get_param("/yolov5/source"),
        "data"          :    rospy.get_param("/yolov5/data"),
        "imgsz"         :    rospy.get_param("/yolov5/imgsz"),
        "conf_thres"    :    rospy.get_param("/yolov5/conf_thres"),
        "iou_thres"     :    rospy.get_param("/yolov5/iou_thres"),
        "max_det"       :    rospy.get_param("/yolov5/max_det"),
        "device"        :    rospy.get_param("/yolov5/device"),
        "view_img"      :    rospy.get_param("/yolov5/view_img"),
        "agnostic_nms"  :    rospy.get_param("/yolov5/agnostic_nms"),
        "augment"       :    rospy.get_param("/yolov5/augment"),
        "visualize"     :    rospy.get_param("/yolov5/visualize"),
        "update"        :    rospy.get_param("/yolov5/update"),
        "line_thickness":    rospy.get_param("/yolov5/line_thickness"),
        "hide_labels"   :    rospy.get_param("/yolov5/hide_labels"),
        "hide_conf"     :    rospy.get_param("/yolov5/hide_conf"),
        "half"          :    rospy.get_param("/yolov5/half"),
        "dnn"           :    rospy.get_param("/yolov5/dnn"),
        "retina_masks"  :    rospy.get_param("/yolov5/retina_masks"),
    }
    print_args(opt)
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop', 'opencv-python'))
    run(**opt)

if __name__ == "__main__":
    node_name = os.path.basename(sys.argv[0]).split('.')[0]
    rospy.init_node(node_name)
    pub_res=rospy.Publisher('/yolov5/result', YoloResultList, queue_size=10)
    
    opt=params()
    main(opt)
    