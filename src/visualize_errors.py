from __future__ import annotations
import cv2
import numpy as np
from matching import match_boxes


def draw_error_map(image: np.ndarray, gt_boxes, pred_boxes, iou_threshold=0.5):
    matches, misses, false_pos = match_boxes(np.asarray(gt_boxes), np.asarray(pred_boxes), iou_threshold)
    out = image.copy()
    for g, p, _ in matches:
        x1,y1,x2,y2 = map(int, pred_boxes[p]); cv2.rectangle(out,(x1,y1),(x2,y2),(0,200,0),2)
    for i in misses:
        x1,y1,x2,y2 = map(int, gt_boxes[i]); cv2.rectangle(out,(x1,y1),(x2,y2),(0,0,255),3)
        cv2.putText(out,"MISS",(x1,max(y1-5,10)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),2)
    for i in false_pos:
        x1,y1,x2,y2 = map(int, pred_boxes[i]); cv2.rectangle(out,(x1,y1),(x2,y2),(255,0,0),3)
        cv2.putText(out,"FP",(x1,max(y1-5,10)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0),2)
    cv2.putText(out, f"GT={len(gt_boxes)} Pred={len(pred_boxes)} Error={abs(len(gt_boxes)-len(pred_boxes))}",
                (15,30), cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),3)
    return out, {"matched":len(matches), "missed":len(misses), "false_positive":len(false_pos)}
