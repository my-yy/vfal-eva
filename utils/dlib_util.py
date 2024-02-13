import dlib
import skimage.draw
import numpy as np
import cv2
import math
# ======== 人脸矫正技术
from utils import faceBlendCommon

face_detector = dlib.get_frontal_face_detector()

root_path = "/ssd2/7_论文数据/23_reconstructing_faces_from_voices/dlib-models/"
landmark68_predictor = dlib.shape_predictor(root_path + 'shape_predictor_68_face_landmarks.dat')
landmark5_detector = dlib.shape_predictor(root_path + "shape_predictor_5_face_landmarks.dat")


def crop_frames(origin_frames):
    croped_frames = []
    for frame in origin_frames:
        try:
            landmark = get_landmark(frame)
            output = crop_image(frame, landmark)
            croped_frames.append(output)
        except Exception as e:
            print("error:", e)
    assert len(croped_frames) > 0, "未提取到frame"
    return croped_frames


def get_landmark(img):
    ans = face_detector(img)
    assert len(ans) > 0, "未检测到人脸"
    rect = ans[0]
    sp = landmark68_predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])
    return landmarks


def crop_image(img, landmarks):
    # 并不是进行裁切提取，而是将需要的部分给复制了一份出来
    outline = landmarks[[*range(17), *range(26, 16, -1)]]
    # [[x,y],[x,y].... ]

    Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
    cropped_img = np.zeros(img.shape, dtype=np.uint8)
    cropped_img[Y, X] = img[Y, X]
    return cropped_img


# 利用5点对齐，从图片中截取出面部图片，含有人脸旋转矫正功能，-。-可以不用自己写的了-。-
def face_crop_and_alignment(image, out_height=224, out_width=224):
    points = faceBlendCommon.getLandmarks(face_detector, landmark5_detector, image)
    landmarks = np.array(points)
    assert len(landmarks) > 0

    # 由[0,255]区间变为[0,1]
    image = np.float32(image) / 255.0
    normalized_image, normalized_landmarks = faceBlendCommon.normalizeImagesAndLandmarks((out_height, out_width), image, landmarks)
    # 变回[0,255]区间：
    normalized_image = np.uint8(normalized_image * 255)
    return normalized_image


# ======== landmark稳定技术：

def crop_frames_with_stabled_landmark(origin_frames):
    landmarks = get_stabled_landmarks(origin_frames)
    ans = []
    for frame, landmark in zip(origin_frames, landmarks):
        if landmark is not None:
            output = crop_image(frame, landmark)
            ans.append(output)
    assert len(ans) > 0, "未提取到frame"
    return ans


def get_stabled_landmarks(frames):
    stable_points_list = []
    eyeDistance = None
    for step in range(0, len(frames)):
        # 1.检测人脸框
        imDlib = cv2.cvtColor(frames[step], cv2.COLOR_BGR2RGB)
        faces = face_detector(imDlib, 0)

        if len(faces) == 0:
            print("未检测到人脸")
            stable_points_list.append(None)
            continue

        # 2.关键点检测
        newRect = faces[0]
        landmark_point_list = landmark68_predictor(imDlib, newRect).parts()
        if eyeDistance is None:
            eyeDistance = interEyeDistance(landmark_point_list)

        landmark_float_matrix2d = np.array([(p.x, p.y) for p in landmark_point_list], dtype=np.float32)

        # 3.稳定点：
        if step == 0:
            # 对于第一帧，并没可以用来进行稳定的信息
            stable_points_list.append(landmark_float_matrix2d)
            continue

        last_frame = to_grave(frames[step - 1])
        this_frame = to_grave(frames[step])
        last_stable_point = stable_points_list[step - 1]

        if last_stable_point is None:
            # 无法进行stable运算：
            this_stable_points_matrix = landmark_float_matrix2d
        else:
            this_stable_points_matrix = calc_stable_points(last_frame, this_frame, last_stable_point, landmark_float_matrix2d, eyeDistance)

        stable_points_list.append(this_stable_points_matrix)
    return stable_points_list


def interEyeDistance(predict):
    leftEyeLeftCorner = (predict[36].x, predict[36].y)
    rightEyeRightCorner = (predict[45].x, predict[45].y)
    distance = cv2.norm(np.array(rightEyeRightCorner) - np.array(leftEyeLeftCorner))
    distance = int(distance)
    return distance


def to_grave(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def calc_stable_points(imGrayPrev, imGray, landmarks_pre, landmarks_this, eyeDistance):
    # 计算参数
    if eyeDistance > 100:
        dotRadius = 3
    else:
        dotRadius = 2
    sigma = eyeDistance * eyeDistance / 400
    s = dotRadius * int(eyeDistance / 4) + 1
    lk_params = dict(winSize=(s, s), maxLevel=5, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))

    # 基于光流得到的点
    points_optical_flow, _, _ = cv2.calcOpticalFlowPyrLK(imGrayPrev, imGray, landmarks_pre, landmarks_this, **lk_params)

    ans = []
    for i in range(len(landmarks_this)):
        point_now = landmarks_pre[i]
        point_last = landmarks_pre[i]
        point_flo = points_optical_flow[i]

        # 计算两帧各个点之间的预测差值距离
        distance = cv2.norm(point_now - point_last)

        # 距离越大则光流结果越不可靠，因此更应该相信当前的预测值（alpha越小，1-alpha越大，当前比重越大）
        weight = math.exp(-distance * distance / sigma)

        # 组合权重
        final_point_value = (1 - weight) * point_now + weight * point_flo

        ans.append(final_point_value)
    return np.array(ans, np.float32)
