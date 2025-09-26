from ultralytics import YOLO
import cv2
import numpy as np
import os
import time

class YOLODetector:
    def __init__(self, model_path="yolov8n-face.pt", device="cpu"):
        """
        Initialize YOLO detector dengan face tracking

        Args:
            model_path (str): Path to YOLO model
            device (str): Device to run on ('cpu' or 'cuda')
        """
        self.model = YOLO(model_path)
        self.device = device

        self.tracked_faces = {}
        self.next_face_id = 0
        self.tracking_threshold = 35  # Dinaikkan untuk lebih stabil
        self.max_missed_frames = 5    # Dinaikkan agar bounding box tidak hilang cepat

    def detect_faces_from_frame(self, frame):
        """
        Detect faces dengan STABLE TRACKING untuk eliminate "lari-lari" bounding boxes
        Args:
            frame: Input frame from camera
        Returns:
            List of STABLE detections dengan smooth bounding boxes
        """
        try:
            detections = []

            results = self.model(
                frame, 
                verbose=False, 
                conf=0.2,   # Diturunkan dari 0.25 untuk deteksi lebih awal
                iou=0.4,    
                imgsz=416,  
                max_det=5   # Dinaikkan kembali dari 3 ke 5 untuk capture lebih banyak
            )
            raw_detections = self._process_results(results, frame, scale_factor=1.0)

            valid_detections = self._smart_face_filter(raw_detections, frame)

            stable_detections = self._track_and_smooth_faces(valid_detections, frame)

            return stable_detections

        except Exception as e:
            print(f"Error in face detection: {e}")
            return []

    def _process_results(self, results, frame, scale_factor=1.0):
        """
        Process YOLO Face Detection results and extract face crops
        """
        detections = []


        for box in results[0].boxes:
            if box is not None:
                confidence = float(box.conf[0])

                if confidence > 0.15:  # Diturunkan dari 0.2 untuk deteksi lebih awal
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor)

                    if x1 < x2 and y1 < y2:
                        face_width = x2 - x1
                        face_height = y2 - y1

                        min_face_size = 40  # Dinaikkan dari 30 untuk kualitas lebih baik
                        max_face_size = min(frame.shape[1], frame.shape[0]) // 2

                        aspect_ratio = face_width / face_height
                        min_aspect = 0.7   # Dipersempit dari 0.6
                        max_aspect = 1.5   # Dipersempit dari 1.8

                        if (min_face_size <= face_width <= max_face_size and 
                            min_face_size <= face_height <= max_face_size and
                            min_aspect <= aspect_ratio <= max_aspect):

                            # Margin yang lebih konservatif untuk presisi lebih baik
                            margin_x = int(face_width * 0.08)  # Dikurangi dari 0.1
                            margin_y = int(face_height * 0.08) # Dikurangi dari 0.1

                            face_x1 = max(0, x1 - margin_x)
                            face_y1 = max(0, y1 - margin_y)
                            face_x2 = min(frame.shape[1], x2 + margin_x)
                            face_y2 = min(frame.shape[0], y2 + margin_y)

                            try:
                                face_crop = frame[face_y1:face_y2, face_x1:face_x2]

                                if face_crop.size > 0 and face_crop.shape[0] > 30 and face_crop.shape[1] > 30:  # Dinaikkan dari 20
                                    detections.append({
                                        "bbox": [face_x1, face_y1, face_x2, face_y2],
                                        "confidence": confidence,
                                        "face_crop": face_crop,
                                        "class_name": "face"
                                    })
                            except Exception as e:
                                print(f"Error cropping face: {e}")
                        else:
                            # print(f"🚫 Filtered detection: size={face_width}x{face_height}, aspect={aspect_ratio:.2f}, conf={confidence:.2f}")  # Commented untuk clean log
                            continue

        return detections

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on frame
        Args:
            frame: Input frame
            detections: List of detections
        Returns:
            Frame with drawn detections
        """
        annotated_frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            class_name = detection.get("class_name", "face")

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return annotated_frame

    def _apply_nms(self, detections, iou_threshold=0.3):
        """
        Apply Non-Maximum Suppression untuk remove overlapping detections
        Args:
            detections: List of detections
            iou_threshold: IoU threshold untuk NMS
        Returns:
            Filtered list of detections
        """
        if len(detections) <= 1:
            return detections

        boxes = []
        scores = []

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            boxes.append([x1, y1, x2, y2])
            scores.append(detection["confidence"])

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                                  score_threshold=0.1, nms_threshold=iou_threshold)

        filtered_detections = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                filtered_detections.append(detections[i])

        print(f"🔧 NMS: {len(detections)} -> {len(filtered_detections)} detections")
        return filtered_detections

    def _filter_face_detections(self, detections, frame):
        """
        Filter detections untuk hanya bounding box yang valid sebagai wajah
        Args:
            detections: List of detections
            frame: Original frame untuk size validation
        Returns:
            Filtered list of valid face detections
        """
        valid_detections = []
        frame_height, frame_width = frame.shape[:2]

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]

            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            area = width * height

            min_face_size = 30
            max_face_size = min(frame_width, frame_height) * 0.8
            min_aspect_ratio = 0.6
            max_aspect_ratio = 1.6

            is_valid = (
                width >= min_face_size and 
                height >= min_face_size and
                width <= max_face_size and 
                height <= max_face_size and
                min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and
                x1 >= 0 and y1 >= 0 and x2 <= frame_width and y2 <= frame_height
            )

            if is_valid:
                valid_detections.append(detection)
                print(f"✅ Valid face: {width:.0f}x{height:.0f}, aspect: {aspect_ratio:.2f}, conf: {detection['confidence']:.3f}")
            else:
                print(f"❌ Invalid face: {width:.0f}x{height:.0f}, aspect: {aspect_ratio:.2f}, conf: {detection['confidence']:.3f}")

        print(f"🔧 Face filter: {len(detections)} -> {len(valid_detections)} valid faces")
        return valid_detections

    def _smart_face_filter(self, detections, frame):
        """
        SMART filtering untuk wajah dengan analisis yang lebih detail
        Args:
            detections: List of detections
            frame: Original frame
        Returns:
            Filtered list of detections yang benar-benar wajah
        """
        smart_detections = []
        frame_height, frame_width = frame.shape[:2]

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]

            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            area = width * height

            min_face_size = 30  # Diturunkan kembali dari 35 untuk deteksi awal
            max_face_size = min(frame_width, frame_height) * 0.7  # Dinaikkan kembali
            ideal_aspect_ratio_min = 0.6  # Diperlebar kembali untuk toleransi
            ideal_aspect_ratio_max = 1.6  # Diperlebar kembali
            min_confidence = 0.15  # Diturunkan dari 0.2 untuk deteksi awal

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            edge_margin = 0.02  # Dikurangi dari 0.05 untuk lebih toleran

            too_close_to_edge = (
                center_x < frame_width * edge_margin or 
                center_x > frame_width * (1 - edge_margin) or
                center_y < frame_height * edge_margin or 
                center_y > frame_height * (1 - edge_margin)
            )

            is_smart_face = (
                width >= min_face_size and 
                height >= min_face_size and
                width <= max_face_size and 
                height <= max_face_size and
                ideal_aspect_ratio_min <= aspect_ratio <= ideal_aspect_ratio_max and
                confidence >= min_confidence and
                not too_close_to_edge and
                x1 >= 0 and y1 >= 0 and x2 <= frame_width and y2 <= frame_height
            )

            if is_smart_face:
                distance_from_center = abs(center_x - frame_width/2) + abs(center_y - frame_height/2)
                center_bonus = 1.0 - (distance_from_center / (frame_width + frame_height))

                detection["smart_score"] = confidence + (center_bonus * 0.1)
                smart_detections.append(detection)
                # print(f"✅ SMART face: {width:.0f}x{height:.0f}, AR: {aspect_ratio:.2f}, conf: {confidence:.3f}, score: {detection['smart_score']:.3f}")  # Commented untuk speed
            # else:
                # print(f"❌ REJECTED: {width:.0f}x{height:.0f}, AR: {aspect_ratio:.2f}, conf: {confidence:.3f}, edge: {too_close_to_edge}")  # Commented untuk speed

        smart_detections.sort(key=lambda x: x.get("smart_score", 0), reverse=True)

        print(f"🧠 SMART filter: {len(detections)} -> {len(smart_detections)} smart faces")
        return smart_detections

    def _track_and_smooth_faces(self, detections, frame):
        """
        Track faces across frames dan smooth bounding boxes untuk eliminate "lari-lari"
        Args:
            detections: Current frame detections
            frame: Current frame
        Returns:
            List of tracked and smoothed detections
        """
        current_time = time.time()

        for face_id in list(self.tracked_faces.keys()):
            self.tracked_faces[face_id]['missed_frames'] += 1

            if self.tracked_faces[face_id]['missed_frames'] > self.max_missed_frames:
                del self.tracked_faces[face_id]
                print(f"🗑️ Removed face {face_id} (lost tracking)")

        matched_detections = []
        unmatched_detections = list(detections)

        for face_id, tracked_face in self.tracked_faces.items():
            best_match = None
            best_distance = float('inf')

            for detection in unmatched_detections:
                distance = self._calculate_bbox_distance(tracked_face['bbox'], detection['bbox'])

                if distance < self.tracking_threshold and distance < best_distance:
                    best_match = detection
                    best_distance = distance

            if best_match:
                smoothed_bbox = self._smooth_bbox(tracked_face['bbox'], best_match['bbox'])

                self.tracked_faces[face_id].update({
                    'bbox': smoothed_bbox,
                    'confidence': best_match['confidence'],
                    'missed_frames': 0,
                    'last_seen': current_time
                })

                smoothed_detection = {
                    'bbox': smoothed_bbox,
                    'confidence': best_match['confidence'],
                    'face_id': face_id,
                    'tracked': True
                }
                matched_detections.append(smoothed_detection)
                unmatched_detections.remove(best_match)

                print(f"📍 Tracked face {face_id}: conf={best_match['confidence']:.3f}, distance={best_distance:.1f}")

        for detection in unmatched_detections:
            face_id = self.next_face_id
            self.next_face_id += 1

            self.tracked_faces[face_id] = {
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'missed_frames': 0,
                'last_seen': current_time,
                'created_at': current_time
            }

            new_detection = {
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'face_id': face_id,
                'tracked': False  # New face, not tracked yet
            }
            matched_detections.append(new_detection)

            print(f"🆕 New face {face_id}: conf={detection['confidence']:.3f}")

        print(f"📊 Tracking: {len(matched_detections)} faces ({len(self.tracked_faces)} tracked)")
        return matched_detections

    def _calculate_bbox_distance(self, bbox1, bbox2):
        """
        Calculate distance antara dua bounding boxes (center distance)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        center1_x = (x1_1 + x2_1) / 2
        center1_y = (y1_1 + y2_1) / 2
        center2_x = (x1_2 + x2_2) / 2
        center2_y = (y1_2 + y2_2) / 2

        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        return distance

    def _smooth_bbox(self, old_bbox, new_bbox, alpha=0.6):
        """
        STABLE smoothing untuk mencegah flicker bounding box
        Args:
            old_bbox: Previous bounding box
            new_bbox: New detected bounding box  
            alpha: Smoothing factor - DINAIKKAN dari 0.3 ke 0.6 untuk stabilitas
        Returns:
            Smoothed bounding box
        """
        smoothed_bbox = []
        for i in range(4):
            smoothed_value = alpha * old_bbox[i] + (1 - alpha) * new_bbox[i]
            smoothed_bbox.append(int(smoothed_value))

        return smoothed_bbox