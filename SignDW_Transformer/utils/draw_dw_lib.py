import math
import numpy as np
import matplotlib
import cv2
import sys
import os
import _pickle as cPickle
import gzip
import subprocess
import torch
import colorsys
from typing import List, Dict, Any, Optional, Tuple


eps = 0.01

RENDER_STYLES = {
    "bold": {"line_scale": 3.0, "point_scale": 2.0},
    "thin": {"line_scale": 1.0, "point_scale": 1.0},
}


def get_render_style(style: str = "bold") -> Dict[str, float]:
    """Return named DWPose rendering style parameters."""
    if style not in RENDER_STYLES:
        raise ValueError(f"Unknown render style '{style}'. Choose from: {sorted(RENDER_STYLES)}")
    return RENDER_STYLES[style]


def filter_pose_by_confidence(
    pose_data: Dict[str, Any],
    conf_threshold: float = 0.6,
) -> Dict[str, Any]:
    """Filter low-confidence joints before rendering."""
    filtered = {}
    for key, value in pose_data.items():
        if isinstance(value, np.ndarray):
            filtered[key] = value.copy()
        else:
            filtered[key] = value

    bodies = filtered.get("bodies")
    body_scores = filtered.get("body_scores")
    if bodies is not None:
        bodies = np.array(bodies, copy=True)
        min_valid = 1e-6
        coord_mask = (bodies[:, 0] > min_valid) & (bodies[:, 1] > min_valid)

        conf_mask = None
        if body_scores is not None:
            score_vec = np.asarray(body_scores).reshape(-1).astype(float)
            conf_mask = score_vec < conf_threshold
            if conf_mask.shape[0] < bodies.shape[0]:
                conf_mask = np.pad(
                    conf_mask,
                    (0, bodies.shape[0] - conf_mask.shape[0]),
                    constant_values=False,
                )
            elif conf_mask.shape[0] > bodies.shape[0]:
                conf_mask = conf_mask[: bodies.shape[0]]

        valid_mask = coord_mask if conf_mask is None else coord_mask & (~conf_mask)
        bodies[~valid_mask, :] = 0
        filtered["bodies"] = bodies

        if body_scores is not None:
            subset = np.array(body_scores, copy=True)
            if subset.ndim == 1:
                subset = subset.reshape(1, -1)
        else:
            subset = np.arange(bodies.shape[0], dtype=float).reshape(1, -1)

        if subset.shape[1] < bodies.shape[0]:
            subset = np.pad(
                subset,
                ((0, 0), (0, bodies.shape[0] - subset.shape[1])),
                constant_values=-1,
            )
        elif subset.shape[1] > bodies.shape[0]:
            subset = subset[:, : bodies.shape[0]]

        subset[:, ~valid_mask] = -1
        filtered["body_scores"] = subset

    hands = filtered.get("hands")
    hand_scores = filtered.get("hands_scores")
    if hands is not None and hand_scores is not None:
        hands = np.array(hands, copy=True)
        scores = np.array(hand_scores)
        if hands.ndim == 3 and scores.ndim in (2, 3):
            for h in range(hands.shape[0]):
                cur_scores = scores[h] if scores.ndim == 2 else scores[h]
                mask = (cur_scores < conf_threshold) | (cur_scores <= 0)
                hands[h][mask, :] = 0
        elif hands.ndim == 2 and scores.ndim in (1, 2):
            cur_scores = scores if scores.ndim == 1 else scores.reshape(-1)
            mask = (cur_scores < conf_threshold) | (cur_scores <= 0)
            hands[mask, :] = 0
        filtered["hands"] = hands

    faces = filtered.get("faces")
    face_scores = filtered.get("faces_scores")
    if faces is not None and face_scores is not None:
        faces = np.array(faces, copy=True)
        scores = np.array(face_scores)
        if faces.ndim == 3 and scores.ndim in (2, 3):
            for f in range(faces.shape[0]):
                cur_scores = scores[f] if scores.ndim == 2 else scores[f]
                mask = (cur_scores < conf_threshold) | (cur_scores <= 0)
                faces[f][mask, :] = 0
        elif faces.ndim == 2 and scores.ndim in (1, 2):
            cur_scores = scores if scores.ndim == 1 else scores.reshape(-1)
            mask = (cur_scores < conf_threshold) | (cur_scores <= 0)
            faces[mask, :] = 0
        filtered["faces"] = faces

    return filtered

def alpha_blend_color(color, alpha):
    """blend color according to point conf
    """
    return [int(c * alpha) for c in color]

def draw_bodypose(canvas, candidate, subset, score, transparent=False, line_scale=1.0, point_scale=1.0):
    """Draw body pose on canvas
    Args:
        canvas: numpy array canvas to draw on
        candidate: pose candidate
        subset: pose subset
        score: confidence scores
        transparent: whether to use transparent background
    Returns:
        canvas: drawn canvas
    """
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = max(1, int(round(4 * line_scale)))

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    # Add alpha channel if transparent
    if transparent:
        colors = [color + [255] for color in colors]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            conf = score[n][np.array(limbSeq[i]) - 1]
            if conf[0] < 0.3 or conf[1] < 0.3:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            if transparent:
                color = colors[i][:-1] + [int(255 * conf[0] * conf[1])]  # Adjust alpha based on confidence
            else:
                color = colors[i]
            cv2.fillConvexPoly(canvas, polygon, color)

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = score[n][i]
            x = int(x * W)
            y = int(y * H)
            if transparent:
                color = colors[i][:-1] + [int(255 * conf)]  # Adjust alpha based on confidence
            else:
                color = colors[i]
            cv2.circle(canvas, (int(x), int(y)), max(1, int(round(4 * point_scale))), color, thickness=-1)

    return canvas

def draw_handpose(canvas, all_hand_peaks, all_hand_scores, transparent=False, line_scale=1.0, point_scale=1.0):
    """Draw hand pose on canvas"""
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks, scores in zip(all_hand_peaks, all_hand_scores):
        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            score = scores[e[0]] * scores[e[1]]
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                color = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                if transparent:
                    color = tuple(int(c * 255) for c in color) + (int(score * 255),)
                else:
                    color = tuple(int(c * score * 255) for c in color)
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=max(1, int(round(2 * line_scale))))

        for i, keypoint in enumerate(peaks):
            x, y = keypoint
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                if transparent:
                    color = (0, 0, 0, int(scores[i] * 255))
                else:
                    color = (0, 0, int(scores[i] * 255))
                cv2.circle(canvas, (x, y), max(1, int(round(4 * point_scale))), color, thickness=-1)
    return canvas

def draw_facepose(canvas, all_lmks, all_scores, transparent=False, point_scale=1.0):
    """Draw face pose on canvas"""
    H, W, C = canvas.shape
    for lmks, scores in zip(all_lmks, all_scores):
        for lmk, score in zip(lmks, scores):
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                if transparent:
                    color = (255, 255, 255, int(score * 255))  # White with alpha
                else:
                    conf = int(score * 255)
                    color = (conf, conf, conf)  # Original grayscale
                cv2.circle(canvas, (x, y), max(1, int(round(3 * point_scale))), color, thickness=-1)
    return canvas

def draw_pose(
    pose,
    H,
    W,
    include_body=True,
    include_hand=True,
    include_face=True,
    ref_w=2160,
    transparent=False,
    line_scale=1.0,
    point_scale=1.0,
):
    """vis dwpose outputs with optional transparent background

    Args:
        pose (Dict): DWposeDetector outputs - 支持新的person_id格式和旧格式
        H (int): height
        W (int): width
        include_body (bool): whether to draw body keypoints
        include_hand (bool): whether to draw hand keypoints
        include_face (bool): whether to draw face keypoints
        ref_w (int, optional): reference width. Defaults to 2160.
        transparent (bool, optional): whether to use transparent background. Defaults to False.

    Returns:
        np.ndarray: image pixel value in RGBA mode if transparent=True, otherwise RGB mode
    """
    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1

    # Create canvas - now with alpha channel if transparent
    if transparent:
        canvas = np.zeros(shape=(int(H*sr), int(W*sr), 4), dtype=np.uint8)
    else:
        canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)

    # 检查是否是新的person_id数据格式
    if 'num_persons' in pose and pose['num_persons'] > 0:
        # 使用新的多人数据结构
        processed_data = process_pose_data(pose, H, W)
        bodies = processed_data['bodies']
        faces = processed_data['faces']
        hands = processed_data['hands']
        candidate = bodies['candidate']
        subset = bodies['subset']
        
        if include_body:
            canvas = draw_bodypose(
                canvas,
                candidate,
                subset,
                score=bodies['score'],
                transparent=transparent,
                line_scale=line_scale,
                point_scale=point_scale,
            )

        if include_hand:
            canvas = draw_handpose(
                canvas,
                hands,
                processed_data['hands_score'],
                transparent=transparent,
                line_scale=line_scale,
                point_scale=point_scale,
            )

        if include_face:
            canvas = draw_facepose(
                canvas,
                faces,
                processed_data['faces_score'],
                transparent=transparent,
                point_scale=point_scale,
            )
    
    else:
        # 兼容旧的数据格式 - 作为备选方案
        try:
            processed_data = process_pose_data(pose, H, W)
            bodies = processed_data['bodies']
            faces = processed_data['faces']
            hands = processed_data['hands']
            candidate = bodies['candidate']
            subset = bodies['subset']

            if include_body:
                canvas = draw_bodypose(
                    canvas,
                    candidate,
                    subset,
                    score=bodies['score'],
                    transparent=transparent,
                    line_scale=line_scale,
                    point_scale=point_scale,
                )

            if include_hand:
                canvas = draw_handpose(
                    canvas,
                    hands,
                    processed_data['hands_score'],
                    transparent=transparent,
                    line_scale=line_scale,
                    point_scale=point_scale,
                )

            if include_face:
                canvas = draw_facepose(
                    canvas,
                    faces,
                    processed_data['faces_score'],
                    transparent=transparent,
                    point_scale=point_scale,
                )
        except Exception as e:
            print(f"绘制旧格式数据失败: {str(e)}")
            # 返回空画布
            pass

    if transparent:
        return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGRA2RGBA).transpose(2, 0, 1)
    else:
        return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

def process_pose_data(pose_data: Dict[str, Any], height: int, width: int) -> Dict[str, Any]:
    """
    处理姿势数据，完全支持新的person_id数据结构
    """
    processed_data = {}
    
    # 确保使用新的数据结构
    if 'num_persons' in pose_data and pose_data['num_persons'] > 0:
        num_persons = pose_data['num_persons']
        
        # 收集所有人的关键点数据
        all_bodies = []
        all_body_scores = []
        all_hands = []
        all_hand_scores = []
        all_faces = []
        all_face_scores = []
        
        for person_id in range(num_persons):
            person_key = f'person_{person_id}'
            if person_key in pose_data:
                person_data = pose_data[person_key]
                all_bodies.append(person_data['body_keypoints'])
                all_body_scores.append(person_data['body_scores'])
                all_hands.extend([person_data['left_hand_keypoints'], person_data['right_hand_keypoints']])
                all_hand_scores.extend([person_data['left_hand_scores'], person_data['right_hand_scores']])
                all_faces.append(person_data['face_keypoints'])
                all_face_scores.append(person_data['face_scores'])
        
        # 合并所有人的数据
        if all_bodies:
            bodies = np.vstack(all_bodies)
            body_scores = np.array(all_body_scores)
            
            # 创建subset - 为每个人创建独立的subset行
            subset = []
            for person_id in range(num_persons):
                person_subset = list(range(person_id * 18, (person_id + 1) * 18))
                subset.append(person_subset)
            subset = np.array(subset)
            
            # 创建scores - 基于body_scores中的有效性
            scores = np.ones_like(body_scores)
            for i in range(num_persons):
                for j in range(18):
                    if body_scores[i, j] < 0:  # 如果body_scores为负数，认为无效
                        scores[i, j] = 0.0
                    else:
                        scores[i, j] = 1.0
        else:
            bodies = np.array([])
            subset = np.array([[]])
            scores = np.array([[]])
        
        hands = np.array(all_hands) if all_hands else np.array([])
        hand_scores = np.array(all_hand_scores) if all_hand_scores else np.array([])
        faces = np.array(all_faces) if all_faces else np.array([])
        face_scores = np.array(all_face_scores) if all_face_scores else np.array([])
        
    else:
        # 兼容旧的单人数据格式
        raw_bodies = np.array(pose_data.get('bodies', []), dtype=np.float32)
        raw_body_scores = np.array(pose_data.get('body_scores', []), dtype=np.float32)
        raw_hands = np.array(pose_data.get('hands', []), dtype=np.float32)
        raw_hand_scores = np.array(pose_data.get('hands_scores', []), dtype=np.float32)
        raw_faces = np.array(pose_data.get('faces', []), dtype=np.float32)
        raw_face_scores = np.array(pose_data.get('faces_scores', []), dtype=np.float32)

        if raw_bodies.size > 0:
            bodies = raw_bodies.reshape(-1, 2)
            subset = np.arange(bodies.shape[0], dtype=np.int32)[None, :]

            if raw_body_scores.size > 0:
                scores = raw_body_scores.reshape(1, -1)
            else:
                scores = np.ones((1, bodies.shape[0]), dtype=np.float32)
        else:
            bodies = np.array([])
            subset = np.array([[]])
            scores = np.array([[]])

        hands = raw_hands if raw_hands.size > 0 else np.array([])
        hand_scores = raw_hand_scores if raw_hand_scores.size > 0 else np.array([])
        faces = raw_faces if raw_faces.size > 0 else np.array([])
        face_scores = raw_face_scores if raw_face_scores.size > 0 else np.array([])
    
    processed_data['bodies'] = {
        'candidate': bodies,
        'subset': subset,
        'score': scores
    }
    
    processed_data['hands'] = hands
    processed_data['hands_score'] = hand_scores
    
    processed_data['faces'] = faces
    processed_data['faces_score'] = face_scores
    
    return processed_data
