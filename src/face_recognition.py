import cv2
import numpy as np
from deepface import DeepFace
import pandas as pd
from collections import defaultdict
import os
import time
from statistics import median, mode


class FaceRecognitionSystem:
    def __init__(self, db_path, detector_backend="retinaface", model_name="Facenet", distance_metric="cosine"):
        self.db_path = db_path
        self.detector_backend = detector_backend
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.known_faces = {}
        self.face_records = defaultdict(lambda: {
            'name': 'Unknown',
            'face_img': None,
            'gender': [],
            'age': [],
            'emotion': defaultdict(list),
            'race': defaultdict(list),
            'first_appearance': None,
            'last_appearance': None,
            'appearance_count': 0,
            'confidence_scores': []
        })

    def load_known_faces(self):
        """Tải các khuôn mặt đã biết từ thư mục database"""
        try:
            for person_name in os.listdir(self.db_path):
                person_path = os.path.join(self.db_path, person_name)
                if os.path.isdir(person_path):
                    for img_file in os.listdir(person_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(person_path, img_file)
                            try:
                                embedding = DeepFace.represent(
                                    img_path=img_path,
                                    model_name=self.model_name,
                                    detector_backend=self.detector_backend,
                                    enforce_detection=False
                                )

                                if embedding and len(embedding) > 0:
                                    if person_name not in self.known_faces:
                                        self.known_faces[person_name] = []
                                    self.known_faces[person_name].append(embedding[0]["embedding"])
                                else:
                                    print(f"Không thể trích xuất đặc điểm từ ảnh {img_path}")
                            except Exception as e:
                                print(f"Không thể xử lý ảnh {img_path}: {str(e)}")
            print(f"Đã tải {len(self.known_faces)} người từ database")
        except Exception as e:
            print(f"Lỗi khi tải database: {str(e)}")

    def enhanced_face_analysis(self, face_img):
        """Phân tích khuôn mặt nâng cao"""
        try:
            # Kiểm tra kiểu dữ liệu đầu vào
            if isinstance(face_img, str):
                # Nếu là đường dẫn, đọc ảnh
                if os.path.exists(face_img):
                    face_img = cv2.imread(face_img)
                else:
                    print(f"Đường dẫn ảnh không tồn tại: {face_img}")
                    return None
            elif not isinstance(face_img, np.ndarray):
                print(f"Định dạng ảnh không hỗ trợ: {type(face_img)}")
                return None

            # Kiểm tra kích thước ảnh
            if face_img.size == 0:
                print("Ảnh rỗng")
                return None

            analysis = DeepFace.analyze(
                img_path=face_img,  # Truyền trực tiếp image array
                actions=["emotion", "age", "gender", "race"],
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True
            )

            if not analysis:
                return None

            face_info = analysis[0]
            enhanced_info = self._get_enhanced_analysis(face_img, face_info)
            return enhanced_info

        except Exception as e:
            print(f"Lỗi khi phân tích khuôn mặt nâng cao: {str(e)}")
            return None

    def _get_enhanced_analysis(self, face_img, base_info):
        """Cải thiện độ chính xác của phân tích"""
        enhanced_info = base_info.copy()

        # 1. Xử lý tuổi
        enhanced_info['age'] = self._refine_age_estimation(face_img, base_info['age'])

        # 2. Xử lý cảm xúc
        enhanced_info['emotion_details'] = base_info.get('emotion', {})
        enhanced_info = self._refine_emotion_analysis(enhanced_info)

        # 3. Xử lý chủng tộc
        enhanced_info['race_details'] = base_info.get('race', {})
        enhanced_info = self._refine_race_analysis(enhanced_info)

        # 4. Tính toán độ tin cậy tổng thể
        enhanced_info['analysis_confidence'] = self._calculate_overall_confidence(face_img, enhanced_info)

        return enhanced_info

    def _refine_age_estimation(self, face_img, estimated_age):
        """Tinh chỉnh ước tính tuổi"""
        height, width = face_img.shape[:2]
        resolution_factor = min(height, width) / 100.0

        if resolution_factor < 0.8:
            age_variance = 5
        elif resolution_factor < 1.2:
            age_variance = 3
        else:
            age_variance = 2

        refined_age = max(1, int(estimated_age))
        return refined_age

    def _refine_emotion_analysis(self, face_info):
        """Tinh chỉnh phân tích cảm xúc"""
        emotion_details = face_info.get('emotion_details', {})

        if emotion_details:
            scores = list(emotion_details.values())
            max_score = max(scores) if scores else 0

            if max_score < 30:
                face_info['emotion_confidence'] = 'low'
            elif max_score < 60:
                face_info['emotion_confidence'] = 'medium'
            else:
                face_info['emotion_confidence'] = 'high'

            if face_info['emotion_confidence'] == 'low' and max_score < 25:
                face_info['dominant_emotion'] = 'uncertain'

        return face_info

    def _refine_race_analysis(self, face_info):
        """Tinh chỉnh phân tích chủng tộc"""
        race_details = face_info.get('race_details', {})

        if race_details:
            scores = list(race_details.values())
            max_score = max(scores) if scores else 0

            if max_score < 40:
                face_info['race_confidence'] = 'low'
            elif max_score < 70:
                face_info['race_confidence'] = 'medium'
            else:
                face_info['race_confidence'] = 'high'

            if face_info['race_confidence'] == 'low' and max_score < 30:
                face_info['dominant_race'] = 'uncertain'

        return face_info

    def _calculate_overall_confidence(self, face_img, face_info):
        """Tính toán độ tin cậy tổng thể"""
        confidence_factors = []

        if 'emotion_confidence' in face_info:
            conf_map = {'low': 0.3, 'medium': 0.7, 'high': 0.9}
            confidence_factors.append(conf_map.get(face_info['emotion_confidence'], 0.5))

        if 'race_confidence' in face_info:
            conf_map = {'low': 0.3, 'medium': 0.7, 'high': 0.9}
            confidence_factors.append(conf_map.get(face_info['race_confidence'], 0.5))

        # Confidence dựa trên chất lượng ảnh
        height, width = face_img.shape[:2]
        resolution_confidence = min(1.0, min(height, width) / 200.0)
        confidence_factors.append(resolution_confidence)

        return np.mean(confidence_factors) if confidence_factors else 0.5

    def recognize_face(self, face_img):
        """Nhận diện khuôn mặt và trả về thông tin"""
        try:
            # Sử dụng phân tích nâng cao
            analysis = self.enhanced_face_analysis(face_img)

            if not analysis:
                return None

            # Nhận diện người từ database
            recognition_results = DeepFace.find(
                img_path=face_img,
                db_path=self.db_path,
                detector_backend=self.detector_backend,
                model_name=self.model_name,
                distance_metric=self.distance_metric,
                enforce_detection=False,
                silent=True
            )

            # Xác định danh tính
            identity = "Unknown"
            confidence = 0.0

            if recognition_results and not recognition_results[0].empty:
                result = recognition_results[0].iloc[0]

                distance_value = None
                if 'distance' in result:
                    distance_value = result['distance']
                elif self.distance_metric in result:
                    distance_value = result[self.distance_metric]

                # Điều chỉnh ngưỡng cho từng model
                threshold_map = {
                    "VGG-Face": 0.6,
                    "Facenet": 0.4,  # Giảm ngưỡng cho Facenet
                    "Facenet512": 0.3,
                    "OpenFace": 0.1,
                    "DeepFace": 0.23,
                    "DeepID": 0.015,
                    "ArcFace": 0.68,
                    "SFace": 0.593
                }

                threshold = threshold_map.get(self.model_name, 0.4)

                if distance_value is not None and distance_value < threshold:
                    identity_path = result['identity']
                    identity = os.path.basename(os.path.dirname(identity_path))
                    confidence = 1 - distance_value
                    print(f"Đã nhận diện: {identity} với độ tương đồng: {confidence:.4f}, distance: {distance_value:.4f}")
                else:
                    print(f"Distance quá cao: {distance_value:.4f} (ngưỡng: {threshold})")

            analysis['identity'] = identity
            return analysis

        except Exception as e:
            print(f"Lỗi khi nhận diện khuôn mặt: {str(e)}")
            return None

    def process_video(self, video_path, output_path=None):
        """Xử lý video và nhận diện khuôn mặt"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Không thể mở video")
            return

        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        face_track_id = 0
        face_tracking = {}

        print("Bắt đầu xử lý video...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Kết thúc video")
                break

            frame_count += 1
            print(f"Đang xử lý frame {frame_count}")

            try:
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=self.detector_backend,
                    enforce_detection=False
                )

                current_frame_faces = []

                for i, face_data in enumerate(faces):
                    if 'confidence' in face_data and face_data['confidence'] > 0.5:
                        facial_area = face_data['facial_area']
                        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

                        x, y, w, h = int(x), int(y), int(w), int(h)
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, frame.shape[1] - x)
                        h = min(h, frame.shape[0] - y)

                        if w <= 0 or h <= 0:
                            continue

                        face_img = frame[y:y + h, x:x + w]

                        if face_img.size == 0:
                            continue

                        face_info = self.recognize_face(face_img)

                        if face_info:
                            face_id = self._get_face_id(x, y, w, h, face_tracking)

                            if face_id is None:
                                face_track_id += 1
                                face_id = face_track_id

                            face_tracking[face_id] = {
                                'x': x, 'y': y, 'w': w, 'h': h,
                                'info': face_info,
                                'last_seen': frame_count
                            }

                            self._update_face_record(face_id, face_info, face_img, frame_count)
                            self._draw_face_info(frame, x, y, w, h, face_info)
                            current_frame_faces.append(face_id)
                        else:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                self._cleanup_old_faces(face_tracking, current_frame_faces, frame_count)

            except Exception as e:
                print(f"Lỗi khi xử lý frame {frame_count}: {str(e)}")

            if output_path:
                out.write(frame)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

        print("Đã xử lý xong video")

    def _get_face_id(self, x, y, w, h, face_tracking):
        """Tìm ID khuôn mặt dựa trên vị trí"""
        for face_id, data in face_tracking.items():
            prev_x, prev_y, prev_w, prev_h = data['x'], data['y'], data['w'], data['h']

            x_left = max(x, prev_x)
            y_top = max(y, prev_y)
            x_right = min(x + w, prev_x + prev_w)
            y_bottom = min(y + h, prev_y + prev_h)

            if x_right < x_left or y_bottom < y_top:
                continue

            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            union_area = w * h + prev_w * prev_h - intersection_area
            iou = intersection_area / union_area if union_area > 0 else 0

            if iou > 0.3:
                return face_id

        return None

    def _update_face_record(self, face_id, face_info, face_img, frame_count):
        """Cập nhật bản ghi khuôn mặt"""
        identity = face_info['identity']
        record = self.face_records[identity]

        if face_info['identity'] != 'Unknown':
            record['name'] = face_info['identity']

        if record['face_img'] is None or face_info.get('analysis_confidence', 0) > 0.7:
            record['face_img'] = face_img.copy()

        record['gender'].append(face_info['dominant_gender'])
        record['age'].append(face_info['age'])

        if 'emotion_details' in face_info:
            for emotion, score in face_info['emotion_details'].items():
                record['emotion'][emotion].append(score)

        if 'race_details' in face_info:
            for race, score in face_info['race_details'].items():
                record['race'][race].append(score)

        record['confidence_scores'].append(face_info.get('analysis_confidence', 0.5))

        if record['first_appearance'] is None:
            record['first_appearance'] = frame_count
        record['last_appearance'] = frame_count
        record['appearance_count'] += 1

    def _draw_face_info(self, frame, x, y, w, h, face_info):
        """Vẽ bounding box và thông tin"""
        frame_height, frame_width = frame.shape[:2]

        base_scale = min(frame_width, frame_height) / 1000.0
        font_scale = max(0.3, base_scale * 0.8)
        thickness = max(1, int(base_scale * 1.5))
        line_height = max(15, int(base_scale * 20))

        confidence = face_info.get('analysis_confidence', 0.5)
        if confidence > 0.7:
            color = (0, 255, 0)
        elif confidence > 0.4:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        if face_info['identity'] == 'Unknown':
            color = (0, 0, 255)

        box_thickness = max(2, int(base_scale * 2.5))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, box_thickness)

        confidence_text = f"({confidence * 100:.0f}%)" if 'analysis_confidence' in face_info else ""
        label = f"{face_info['identity']} {confidence_text}"

        info_lines = [
            f"Gender: {face_info['dominant_gender']}",
            f"Age: {face_info['age']}",
            f"Emotion: {face_info['dominant_emotion']}",
            f"Race: {face_info['dominant_race']}"
        ]

        if 'emotion_confidence' in face_info:
            info_lines.append(f"Emo Conf: {face_info['emotion_confidence']}")
        if 'race_confidence' in face_info:
            info_lines.append(f"Race Conf: {face_info['race_confidence']}")

        label_y_offset = max(5, int(base_scale * 10))
        info_y_offset = max(10, int(base_scale * 15))

        cv2.putText(frame, label, (x, y - label_y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        for i, line in enumerate(info_lines):
            y_position = y + h + info_y_offset + i * line_height
            cv2.putText(frame, line, (x, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, color, max(1, thickness - 1))

    def _cleanup_old_faces(self, face_tracking, current_faces, frame_count):
        """Xóa các khuôn mặt không còn xuất hiện"""
        to_remove = []
        for face_id, data in face_tracking.items():
            if face_id not in current_faces and frame_count - data['last_seen'] > 30:
                to_remove.append(face_id)

        for face_id in to_remove:
            del face_tracking[face_id]

    def generate_summary(self):
        """Tạo báo cáo tổng hợp"""
        print("\n=== BÁO CÁO TỔNG HỢP CÁC KHUÔN MẶT TRONG VIDEO ===")

        for identity, record in self.face_records.items():
            if identity == 'Unknown' and record['appearance_count'] == 0:
                continue

            print(f"\n--- {identity} ---")
            print(f"Số lần xuất hiện: {record['appearance_count']}")
            print(f"Frame đầu tiên: {record['first_appearance']}")
            print(f"Frame cuối cùng: {record['last_appearance']}")

            if record['gender']:
                try:
                    dominant_gender = mode(record['gender'])
                    print(f"Giới tính: {dominant_gender}")
                except:
                    print(f"Giới tính: {record['gender'][0]}")

            if record['age']:
                median_age = median(record['age'])
                age_range = f"từ {min(record['age'])} đến {max(record['age'])}"
                print(f"Tuổi (trung vị): {median_age} {age_range}")

            if record['emotion']:
                print("Phân tích cảm xúc chi tiết:")
                for emotion, scores in record['emotion'].items():
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        if avg_score > 10:
                            print(f"  - {emotion}: {avg_score:.1f}%")

            if record['race']:
                print("Phân tích chủng tộc chi tiết:")
                for race, scores in record['race'].items():
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        if avg_score > 10:
                            print(f"  - {race}: {avg_score:.1f}%")

            if record['confidence_scores']:
                avg_confidence = sum(record['confidence_scores']) / len(record['confidence_scores'])
                print(f"Độ tin cậy trung bình: {avg_confidence * 100:.1f}%")

            if record['face_img'] is not None:
                os.makedirs("output_faces", exist_ok=True)
                safe_name = "".join(c for c in identity if c.isalnum() or c in (' ', '-', '_')).rstrip()
                output_path = f"output_faces/{safe_name}.jpg"
                cv2.imwrite(output_path, record['face_img'])
                print(f"Ảnh đại diện đã lưu: {output_path}")


def check_dataset_structure(db_path):
    """Kiểm tra cấu trúc dataset và in thông tin"""
    print("=== KIỂM TRA CẤU TRÚC DATASET ===")

    if not os.path.exists(db_path):
        print(f"Lỗi: Đường dẫn {db_path} không tồn tại")
        return False

    print(f"Đường dẫn dataset: {os.path.abspath(db_path)}")

    person_count = 0
    image_count = 0

    for person_name in os.listdir(db_path):
        person_path = os.path.join(db_path, person_name)
        if os.path.isdir(person_path):
            person_count += 1
            person_images = 0

            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    person_images += 1
                    image_count += 1

            print(f"- {person_name}: {person_images} ảnh")

    print(f"Tổng số người: {person_count}")
    print(f"Tổng số ảnh: {image_count}")

    return person_count > 0 and image_count > 0


if __name__ == "__main__":
    db_path = "../dataset/training"

    if not check_dataset_structure(db_path):
        print("Cấu trúc dataset không hợp lệ. Vui lòng kiểm tra lại.")
        exit()

    fr_system = FaceRecognitionSystem(db_path=db_path)
    fr_system.load_known_faces()

    test_img_path = "../dataset/test/joe1.jpg"
    if os.path.exists(test_img_path):
        print("\n=== KIỂM TRA NHẬN DIỆN TRÊN ẢNH TĨNH ===")
        test_result = fr_system.recognize_face(test_img_path)
        if test_result:
            print(f"Kết quả nhận diện: {test_result['identity']}")
        else:
            print("Không thể nhận diện trên ảnh tĩnh")
    else:
        print(f"Ảnh test không tồn tại: {test_img_path}")

    video_path = "../resource/video/putin.mp4"
    if os.path.exists(video_path):
        fr_system.process_video(
            video_path=video_path,
            output_path="output_video.mp4"
        )
    else:
        print(f"Video không tồn tại: {video_path}")

    fr_system.generate_summary()
