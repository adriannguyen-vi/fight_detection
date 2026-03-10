import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) of two bounding boxes."""
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def point_in_box(pt, box):
    """Checks if a point (x, y) is inside a bounding box (x1, y1, x2, y2)."""
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


# ==========================================
# 2. INDIVIDUAL RULE CLASSES
# ==========================================
class GrapplingRule:
    """Rule 1: Detects Wrestling/Grappling via IoU and erratic motion."""
    def __init__(self, iou_thresh=0.3, var_thresh=50.0, min_history_frames=15):
        self.iou_thresh = iou_thresh
        self.var_thresh = var_thresh
        self.min_history = min_history_frames

    def evaluate(self, id_A, id_B, track_A, track_B):
        box_A, box_B = track_A['boxes'][-1], track_B['boxes'][-1]
        
        if calculate_iou(box_A, box_B) > self.iou_thresh:
            if len(track_A['centers']) >= self.min_history:
                centers_A = np.array(track_A['centers'])
                var_A = np.var(centers_A[:, 0]) + np.var(centers_A[:, 1])
                
                if var_A > self.var_thresh:
                    return True, f"RULE 1 (GRAPPLE): IDs {id_A} & {id_B}"
        return False, None


class StrikeRule:
    """Rule 2: Detects fast punches/kicks landing inside another person's bounding box."""
    def __init__(self, vel_thresh=30.0):
        self.vel_thresh = vel_thresh
        # YOLO Indices: Wrists (9, 10), Ankles (15, 16)
        self.strike_joints = [9, 10, 15, 16]

    def evaluate(self, id_A, id_B, track_A, track_B):
        if len(track_A['kps']) < 2:
            return False, None

        box_B = track_B['boxes'][-1]
        kps_A = track_A['kps'][-1]
        prev_kps_A = track_A['kps'][-2]

        for joint in self.strike_joints:
            # Ensure joint is detected in both frames
            if kps_A[joint][0] != 0 and prev_kps_A[joint][0] != 0:
                velocity = np.linalg.norm(kps_A[joint] - prev_kps_A[joint])
                
                if velocity > self.vel_thresh and point_in_box(kps_A[joint], box_B):
                    return True, f"RULE 2 (STRIKE): {id_A} hit {id_B}"
        return False, None


class SevereViolenceRule:
    """Rule 3: Detects if a person has fallen to the ground following a fight."""
    def __init__(self, fall_ratio=1.2):
        self.fall_ratio = fall_ratio

    def evaluate(self, tid, track, recently_fought):
        if not recently_fought:
            return False, None

        box = track['boxes'][-1]
        width = box[2] - box[0]
        height = box[3] - box[1]
        aspect_ratio = width / height if height > 0 else 0
        
        if aspect_ratio > self.fall_ratio:
            return True, f"🚨 RULE 3 (SEVERE): ID {tid} KNOCKED DOWN! 🚨"
        return False, None


# ==========================================
# 3. CORE ENGINE
# ==========================================
class ViolenceRulesEngine:
    def __init__(self, fps=30):
        self.fps = fps
        self.history_frames = int(fps * 1.5)
        self.tracks = {} 
        self.recent_fight_timers = {} 
        
        # Instantiate the Rules
        self.rule_grappling = GrapplingRule(iou_thresh=0.3, var_thresh=50.0, min_history_frames=fps//2)
        self.rule_striking = StrikeRule(vel_thresh=30.0)
        self.rule_severe = SevereViolenceRule(fall_ratio=1.2)

    def _update_memory(self, results):
        """Updates bounding box and pose memory for all detected people."""
        current_ids = []
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            keypoints = results[0].keypoints.xy.cpu().numpy()
            
            for box, track_id, kps in zip(boxes, track_ids, keypoints):
                current_ids.append(track_id)
                center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                
                if track_id not in self.tracks:
                    self.tracks[track_id] = {
                        'boxes': deque(maxlen=self.history_frames),
                        'centers': deque(maxlen=self.history_frames),
                        'kps': deque(maxlen=self.history_frames)
                    }
                
                self.tracks[track_id]['boxes'].append(box)
                self.tracks[track_id]['centers'].append((center_x, center_y))
                self.tracks[track_id]['kps'].append(kps)
                
                # Decay fight timers
                if self.recent_fight_timers.get(track_id, 0) > 0:
                    self.recent_fight_timers[track_id] -= 1

        # Cleanup lost tracks
        for tid in list(self.tracks.keys()):
            if tid not in current_ids:
                del self.tracks[tid]
                
        return current_ids

    def update_and_evaluate(self, results):
        alerts = []
        current_ids = self._update_memory(results)

        # Evaluate Multi-Person Rules (Rules 1 & 2)
        for i in range(len(current_ids)):
            for j in range(i + 1, len(current_ids)):
                id_A, id_B = current_ids[i], current_ids[j]
                track_A, track_B = self.tracks[id_A], self.tracks[id_B]
                
                # Check Rule 1
                is_grapple, msg1 = self.rule_grappling.evaluate(id_A, id_B, track_A, track_B)
                if is_grapple:
                    alerts.append(msg1)
                    self.recent_fight_timers[id_A] = self.recent_fight_timers[id_B] = self.fps * 3

                # Check Rule 2
                is_strike, msg2 = self.rule_striking.evaluate(id_A, id_B, track_A, track_B)
                if is_strike:
                    alerts.append(msg2)
                    self.recent_fight_timers[id_A] = self.recent_fight_timers[id_B] = self.fps * 3

        # Evaluate Single-Person Rules (Rule 3)
        for tid in current_ids:
            recently_fought = self.recent_fight_timers.get(tid, 0) > 0
            is_severe, msg3 = self.rule_severe.evaluate(tid, self.tracks[tid], recently_fought)
            if is_severe:
                alerts.append(msg3)
                    
        return alerts

# ==========================================
# 4. EXECUTION
# ==========================================
def run_demo(video_path, output_path):
    print("Loading yolov26x-Pose model...")
    model = YOLO('/home/adrian/fight_detection/yolo26l-pose.pt') # Using standard nano model name
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    rules_engine = ViolenceRulesEngine(fps=fps)
    
    print(f"Processing video: {video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
        annotated_frame = results[0].plot()
        
        alerts = rules_engine.update_and_evaluate(results)
        if alerts:
            print(alerts)
        
        y_offset = 50
        for alert in alerts:
            color = (0, 0, 255) if "SEVERE" in alert else (0, 165, 255)
            cv2.putText(annotated_frame, alert, (50, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
            y_offset += 40
            
        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Demo saved to {output_path}")

if __name__ == "__main__":
    INPUT_VIDEO = "/home/adrian/fight_detection/data/fight-detection-surv-dataset/train/NonViolence/nofi008.mp4" 
    OUTPUT_VIDEO = "rules_demo_output.mp4"
    run_demo(INPUT_VIDEO, OUTPUT_VIDEO)