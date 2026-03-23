from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import dlib
import numpy as np
from skimage import transform as trans

DEFAULT_RESOLUTION = 256
DEFAULT_SCALE = 1.3


@dataclass(slots=True)
class AlignedFaceResult:
    image: np.ndarray
    landmarks: np.ndarray | None
    mask: np.ndarray | None = None


def build_face_detector() -> dlib.fhog_object_detector:
    return dlib.get_frontal_face_detector()


def build_shape_predictor(predictor_path: str | Path) -> dlib.shape_predictor:
    path = Path(predictor_path)
    if not path.exists():
        raise FileNotFoundError(f"Predictor path does not exist: {path}")
    return dlib.shape_predictor(str(path))


def shape_to_np(shape: dlib.full_object_detection, dtype: str = "int") -> np.ndarray:
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for index in range(shape.num_parts):
        coords[index] = (shape.part(index).x, shape.part(index).y)
    return coords


def get_keypts(
    image: np.ndarray,
    face: dlib.rectangle,
    predictor: dlib.shape_predictor,
) -> np.ndarray:
    shape = predictor(image, face)
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    return np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)


def _img_align_crop(
    img: np.ndarray,
    landmark: np.ndarray,
    outsize: tuple[int, int],
    scale: float = DEFAULT_SCALE,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    target_size = [112, 112]
    dst = np.array(
        [
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    if target_size[1] == 112:
        dst[:, 0] += 8.0

    dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
    dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]
    target_size = list(outsize)

    margin_rate = scale - 1.0
    x_margin = target_size[0] * margin_rate / 2.0
    y_margin = target_size[1] * margin_rate / 2.0

    dst[:, 0] += x_margin
    dst[:, 1] += y_margin
    dst[:, 0] *= target_size[0] / (target_size[0] + 2.0 * x_margin)
    dst[:, 1] *= target_size[1] / (target_size[1] + 2.0 * y_margin)

    src = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    matrix = tform.params[0:2, :]

    aligned = cv2.warpAffine(img, matrix, (target_size[1], target_size[0]))
    aligned = cv2.resize(aligned, (outsize[1], outsize[0]))

    if mask is None:
        return aligned, None

    aligned_mask = cv2.warpAffine(mask, matrix, (target_size[1], target_size[0]))
    aligned_mask = cv2.resize(aligned_mask, (outsize[1], outsize[0]))
    return aligned, aligned_mask


def select_face_like_source(faces: dlib.rectangles, rgb: np.ndarray) -> dlib.rectangle:
    del rgb
    return max(faces, key=lambda rect: rect.width() * rect.height())


def extract_aligned_face_dlib(
    face_detector: dlib.fhog_object_detector,
    predictor: dlib.shape_predictor,
    image: np.ndarray,
    res: int = DEFAULT_RESOLUTION,
    mask: np.ndarray | None = None,
    scale: float = DEFAULT_SCALE,
    verify_aligned_face: bool = True,
) -> AlignedFaceResult | None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb, 1)
    if len(faces) == 0:
        return None

    face = select_face_like_source(faces, rgb)
    landmarks5 = get_keypts(rgb, face, predictor)
    cropped_face, cropped_mask = _img_align_crop(
        rgb,
        landmarks5,
        outsize=(res, res),
        scale=scale,
        mask=mask,
    )
    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)

    aligned_landmarks = None
    if verify_aligned_face:
        face_align = face_detector(cropped_face, 1)
        if len(face_align) == 0:
            return None
        landmark = predictor(cropped_face, face_align[0])
        aligned_landmarks = shape_to_np(landmark)

    return AlignedFaceResult(
        image=cropped_face,
        landmarks=aligned_landmarks,
        mask=cropped_mask,
    )
