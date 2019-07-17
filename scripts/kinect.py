# coding: utf-8

# An example using startStreams

import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame

try:
    from pylibfreenect2 import OpenGLPacketPipeline

    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline

        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline

        pipeline = CpuPacketPipeline()

import os
import argparse
import torch

sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.utils import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict


def main(hrnet_c, hrnet_j, hrnet_weights, hrnet_joints_set, single_person,
         max_batch_size, disable_vidgear, device):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available() and True:
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    print(device)

    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        multiperson=not single_person,
        max_batch_size=max_batch_size,
        device=device
    )

    print("Packet pipeline:", type(pipeline).__name__)

    enable_rgb = True
    enable_depth = True

    fn = Freenect2()
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        print("No device connected!")
        sys.exit(1)

    serial = fn.getDeviceSerialNumber(0)
    device = fn.openDevice(serial, pipeline=pipeline)

    types = 0
    if enable_rgb:
        types |= FrameType.Color
    if enable_depth:
        types |= (FrameType.Ir | FrameType.Depth)
    listener = SyncMultiFrameListener(types)

    # Register listeners
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)

    if enable_rgb and enable_depth:
        device.start()
    else:
        device.startStreams(rgb=enable_rgb, depth=enable_depth)

    # NOTE: must be called after device.start()
    if enable_depth:
        registration = Registration(device.getIrCameraParams(),
                                    device.getColorCameraParams())

    undistorted = Frame(512, 424, 4)
    registered = Frame(512, 424, 4)

    while True:
        frames = listener.waitForNewFrame()

        if enable_rgb:
            color = frames["color"]
        if enable_depth:
            ir = frames["ir"]
            depth = frames["depth"]

        if enable_rgb and enable_depth:
            registration.apply(color, depth, undistorted, registered)
        elif enable_depth:
            registration.undistortDepth(depth, undistorted)

        if enable_depth:
            cv2.imshow("ir", ir.asarray() / 65535.)
            cv2.imshow("depth", depth.asarray() / 4500.)
            # cv2.imshow("undistorted", undistorted.asarray(np.float32) / 4500.)

        if enable_rgb and enable_depth:
            cv2.imshow("registered", registered.asarray(np.uint8))

        # color = cv2.resize(color.asarray()[:, :, :3], (int(1920 / 3), int(1080 / 3)))
        color = registered.asarray(np.uint8)[:, :, :3]
        color = np.ascontiguousarray(color, dtype=np.uint8)
        pts = model.predict(color)

        undistorted_nor = undistorted.asarray(np.float32) / 4500.
        undistorted_image = cv2.cvtColor(undistorted_nor, cv2.COLOR_GRAY2BGR)
        for i, pt in enumerate(pts):
            color = draw_points_and_skeleton(color, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=i,
                                             joints_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                             joints_palette_samples=10)
            for j, point in enumerate(pt):
                if point[2]>0.5 and point[0] < undistorted_image.shape[0] and point[1] < undistorted_image.shape[1]:
                    if j == 9:
                        left_wrist_depth = undistorted_nor[int(point[0]), int(point[1])]
                        print('left_wrist_depth',left_wrist_depth)
                        undistorted_image = cv2.circle(undistorted_image, (int(point[1]), int(point[0])), 30, (255, 255, 0),-1)

                    if j == 10:
                        right_wrist_depth = undistorted_nor[int(point[0]), int(point[1])]
                        print('right_wrist_depth', right_wrist_depth)
                        undistorted_image = cv2.circle(undistorted_image, (int(point[1]), int(point[0])), 30, (255, 0, 255),-1)

        if enable_rgb:
            cv2.imshow("color", color)
            cv2.imshow("undistorted", undistorted_image)

        listener.release(frames)

        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            break

    device.stop()
    device.close()

    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels", type=int, default=48)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="./weights/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--hrnet_joints_set",
                        help="use the specified set of joints ('coco' and 'mpii' are currently supported)",
                        type=str, default="coco")
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--disable_vidgear",
                        help="disable vidgear (which is used for slightly better realtime performance)",
                        action="store_true")  # see https://pypi.org/project/vidgear/
    parser.add_argument("--device", help="device to be used (default: cuda, if available)", type=str, default=None)
    args = parser.parse_args()
    main(**args.__dict__)
