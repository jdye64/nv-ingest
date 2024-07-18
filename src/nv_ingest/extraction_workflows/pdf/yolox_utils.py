import cv2
import numpy as np
import torch
import torchvision


def postprocess_model_prediction(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    prediction = torch.from_numpy(prediction.copy())
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)  # noqa: E203

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def postprocess_results(results, original_image_shapes, min_score=0.0):
    """
    For each item (==image) in results, computes annotations in the form

     {"table": [[0.0107, 0.0859, 0.7537, 0.1219, 0.9861], ...],
      "figure": [...],
      "title": [...]
      }
    where each list of 5 floats represents a bounding box in the format [x1, y1, x2, y2, confidence]

    Keep only bboxes with high enough confidence.
    """
    labels = ["table", "chart", "title"]
    annotation_dict = {label: [] for label in labels}
    out = []

    for original_image_shape, result in zip(original_image_shapes, results):
        if result is None:
            continue
        try:
            result = result.cpu().numpy()
            scores = result[:, 4] * result[:, 5]
            result = result[scores > min_score]

            # ratio is used when image was padded
            ratio = min(1024 / original_image_shape[0], 1024 / original_image_shape[1])
            bboxes = result[:, :4] / ratio

            bboxes[:, [0, 2]] /= original_image_shape[1]
            bboxes[:, [1, 3]] /= original_image_shape[0]
            bboxes = np.clip(bboxes, 0.0, 1.0)

            label_idxs = result[:, 6]
            scores = scores[scores > min_score]
        except Exception as e:
            raise ValueError(f"Error in postprocessing {result.shape} and {original_image_shape}: {e}")
        # bboxes are in format [x_min, y_min, x_max, y_max]
        for j in range(len(bboxes)):
            label = labels[int(label_idxs[j])]
            bbox = bboxes[j]
            score = scores[j]

            # additional preprocessing for tables: extend the upper bounds to capture titles if any.
            if label == "table":
                height = bbox[3] - bbox[1]
                bbox[1] = (bbox[1] - height * 0.2).clip(0.0, 1.0)

            annotation_dict[label].append([round(float(x), 4) for x in np.concatenate((bbox, [score]))])

        out.append(annotation_dict)

    return out


def resize_image(image, target_img_size):
    w, h, _ = np.array(image.shape)

    if target_img_size is not None:  # Resize + Pad
        r = min(target_img_size[0] / w, target_img_size[1] / h)
        image = cv2.resize(
            image,
            (int(h * r), int(w * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        image = np.pad(
            image,
            ((0, target_img_size[0] - image.shape[0]), (0, target_img_size[1] - image.shape[1]), (0, 0)),
            mode="constant",
            constant_values=114,
        )
    return image
