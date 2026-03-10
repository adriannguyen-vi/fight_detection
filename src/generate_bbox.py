import os
import json
import cv2
import argparse
from detect_track import BaseDetectAndTrack
from utils import draw_tracks, CLASS_NAMES
from tqdm import tqdm 
VIDEO_EXT = set(["mp4", "avi", "mpeg"])
TARGET_SHAPE = (1280, 704) # (width, height)
WRITE_VIDEO = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--list_dir", nargs="+", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    print("Going to load onnx file: ", args.onnx_path)
    print("Going to process label for all dirs:", args.list_dir)
    tracker = BaseDetectAndTrack(
        onnx_path=args.onnx_path,
        conf_thres=0.45, 
        iou_thres=0.35,
        target_shape=TARGET_SHAPE      
    )
    print("Initialized detector and tracker. ")
    
    for data_dir in args.list_dir:
        bbox_file = os.path.join(data_dir, "person_bbox.json")
        bbox_data = {} # 1 json file for whole folder
        
        files = [file for file in os.listdir(data_dir) if file.split(".")[-1] in VIDEO_EXT]
        with tqdm(total=len(files), desc=f"Processing {data_dir}") as pbar:
            for file in files:
                video_path = os.path.join(data_dir, file)
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                original_shape = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                tracker.reset()
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                bbox_data[file] = {} # Initialize dict for current video
                
                writer = None
                if WRITE_VIDEO:
                    out_dir = "/home/adrian/fight_detection/debug_vid"
                    os.makedirs(out_dir, exist_ok=True) # Ensure directory exists
                    output_video_path = os.path.join(out_dir, file)
                    writer = cv2.VideoWriter(
                        output_video_path, 
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        original_shape
                    )
                    
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if WRITE_VIDEO:
                        visualized_frame = cv2.resize(frame, TARGET_SHAPE)
                        
                    tracks = tracker.infer(frame)
                    tracks = [x for x in tracks if x[-1] == CLASS_NAMES.index("person")]
                    
                    # ---- Requirement 1 & 2: Normalize and Save BBoxes ----
                    frame_bboxes = []
                    for track in tracks:
                        x1, y1, x2, y2 = track[:4]
                        
                        # Normalize to 0-1 scaling
                        norm_x1 = max(0.0, min(1.0, float(x1) / TARGET_SHAPE[0]))
                        norm_y1 = max(0.0, min(1.0, float(y1) / TARGET_SHAPE[1]))
                        norm_x2 = max(0.0, min(1.0, float(x2) / TARGET_SHAPE[0]))
                        norm_y2 = max(0.0, min(1.0, float(y2) / TARGET_SHAPE[1]))
                        
                        # Store normalized coords (and track_id if available for tracking context)
                        if len(track) >= 5:
                            track_id = int(track[4])
                            frame_bboxes.append([norm_x1, norm_y1, norm_x2, norm_y2, track_id])
                        else:
                            frame_bboxes.append([norm_x1, norm_y1, norm_x2, norm_y2])
                            
                    bbox_data[file][str(frame_idx)] = frame_bboxes

                    if WRITE_VIDEO:
                        draw_frame = draw_tracks(visualized_frame, tracks)
                        
                        draw_frame = cv2.resize(draw_frame, original_shape)
                        writer.write(draw_frame)
                        
                    frame_idx += 1
                    
                # Clean up after each video
                cap.release()
                if writer is not None:
                    writer.release()
                pbar.set_postfix({"vid": file, "progress": ((frame_idx+1)*1.0)/(total_frames)})
                pbar.update(1)

        # Save JSON file for the current data_dir
        with open(bbox_file, "w") as f:
            json.dump(bbox_data, f, indent=4)
        print(f"Saved normalized bounding boxes to: {bbox_file}")

if __name__ == "__main__":
    main()