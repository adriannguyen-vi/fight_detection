import cv2

CLASS_NAMES = [
    "car",
    "van",
    "bus",
    "truck",
    "motorcycle",
    "person",
    "face",
    "wheel",
    "excavator",
    "concrete_mixer",
    "bulldozer/backhoe/loader",
    "roller",
    "boom_lift",
    "scissor_lift",
    "mobile_crane",
    "machinery",
]

def draw_tracks(frame, tracks):
    for x1, y1, x2, y2, trk_id, cls_id in tracks:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        label = CLASS_NAMES[int(cls_id)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"{label}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )
    return frame
