from flask import Flask, Response, render_template, jsonify, request, redirect, url_for
import cv2
import numpy as np
import threading
import time
from datetime import datetime
import os
import sys
from queue import Queue

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
from database import Database

from app.face_recognition import FaceRecognition
from app.yolo_detector import YOLODetector

app = Flask(__name__)

camera = None
latest_frame = None
is_running = False
current_session = None
database = None
local_camera_window = True  # Otomatis tampil
mirror_camera = True  # Otomatis mirror

frame_queue = Queue(maxsize=1)
processing_active = False
last_detections = []
last_recognition_results = {}

face_recognizer = None
yolo_detector = None

def init_components():
    """Initialize REAL AI components - YOLO + ARCFACE"""
    global database, face_recognizer, yolo_detector

    try:
        database = Database()
        print("✅ Database connected!")

        if database and database.connection:
            test_dosens = database.get_dosen_all()
            test_subjects = database.get_mata_kuliah_all()
            print(f"🔥 INIT TEST: {len(test_dosens)} dosens: {test_dosens}")
            print(f"🔥 INIT TEST: {len(test_subjects)} subjects: {test_subjects}")
        else:
            print("❌ Database connection failed in init")
            database = None

    except Exception as e:
        print(f"❌ Database error: {e}")
        database = None

    try:
        print("🔄 Loading YOLO detector...")
        yolo_detector = YOLODetector()
        print("✅ YOLO detector loaded successfully!")
    except Exception as e:
        print(f"❌ YOLO error: {e}")
        yolo_detector = None

    try:
        print("🔄 Loading ArcFace recognition...")
        face_recognizer = FaceRecognition()
        print("✅ ArcFace recognition loaded successfully!")
    except Exception as e:
        print(f"❌ ArcFace error: {e}")
        face_recognizer = None

def camera_loop():
    """ULTRA SMOOTH camera loop dengan threaded AI processing dan LOCAL DISPLAY"""
    global latest_frame, is_running, camera, frame_queue, processing_active, local_camera_window, mirror_camera

    try:
        for camera_index in [0, 1]:
            camera = cv2.VideoCapture(camera_index)
            if camera.isOpened():
                ret, test_frame = camera.read()
                if ret and test_frame is not None:
                    print(f"📷 Using camera index {camera_index}")
                    break
                else:
                    camera.release()
            else:
                if camera:
                    camera.release()

        if not camera or not camera.isOpened():
            print("❌ Cannot open any camera!")
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        print("📷 ULTRA SMOOTH CAMERA started!")

        if local_camera_window:
            cv2.namedWindow('Smart Attendance - Live Camera', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Smart Attendance - Live Camera', 800, 600)

        frame_count = 0

        if not processing_active:
            processing_thread = threading.Thread(target=ai_processing_loop, daemon=True)
            processing_thread.start()

        while is_running:
            ret, frame = camera.read()
            if not ret:
                continue

            if mirror_camera:
                frame = cv2.flip(frame, 1)

            frame_with_overlay = draw_detection_overlay(frame.copy())
            latest_frame = frame_with_overlay.copy()

            if local_camera_window:
                display_frame = cv2.resize(frame_with_overlay, (800, 600))

                if current_session:
                    info_text = f"Session: {current_session['dosen']} - {current_session['subject']}"
                    cv2.putText(display_frame, info_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    attendance_count = len([r for r in last_recognition_results.values() if r.get('recorded', False)])
                    count_text = f"Attendance: {attendance_count} students"
                    cv2.putText(display_frame, count_text, (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.imshow('Smart Attendance - Live Camera', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # ESC key
                    print("📷 Local camera window closed by user")
                    break

            frame_count += 1
            if frame_count % 15 == 0:
                if frame_queue.empty():
                    frame_queue.put(frame.copy())

            time.sleep(0.01)

    except Exception as e:
        print(f"❌ Camera error: {e}")
    finally:
        if camera:
            camera.release()
        if local_camera_window:
            cv2.destroyAllWindows()
        print("📷 Camera stopped")

def ai_processing_loop():
    """ULTRA FAST AI processing untuk pintu masuk real-time dengan STABILITY"""
    global frame_queue, processing_active, last_detections, last_recognition_results
    global face_recognizer, yolo_detector, current_session, database

    processing_active = True
    recorded_students = set()
    frame_skip_counter = 0

    face_memory = {}
    stable_results = {}

    print("🤖 ULTRA FAST AI Processing thread started with STABILITY!")

    while is_running:
        try:
            if not frame_queue.empty():
                frame = frame_queue.get_nowait()

                # ULTRA FAST processing - prioritas speed
                print("🔍 FAST AI Processing...")  # Mengurangi print

                if yolo_detector:
                    try:
                        detections = yolo_detector.detect_faces_from_frame(frame)
                        # print(f"🎯 YOLO detected {len(detections)} faces")  # Commented untuk speed

                        last_detections = detections.copy()

                        current_face_ids = set()

                        for i, detection in enumerate(detections):
                            bbox = detection["bbox"]
                            face_id = detection.get("face_id", i)
                            tracked = detection.get("tracked", False)
                            x1, y1, x2, y2 = bbox

                            current_face_ids.add(face_id)

                            if face_id not in last_recognition_results:
                                # print(f"🆕 Creating new face record for ID: {face_id}")  # Commented untuk speed
                                recognition_result = {
                                    'bbox': [x1, y1, x2, y2],
                                    'face_id': face_id,
                                    'tracked': tracked,
                                    'name': 'Unknown',  # Langsung Unknown, tidak ada "Detecting..."
                                    'confidence': 0.0,
                                    'recorded': False,
                                    'is_fake': False,
                                    'spoof_reason': '',
                                    'visible': True  # Force visible dari awal
                                }
                                last_recognition_results[face_id] = recognition_result
                            else:
                                # print(f"🔄 Updating existing face record for ID: {face_id}, current recorded status: {last_recognition_results[face_id]['recorded']}")  # Commented untuk speed
                                last_recognition_results[face_id]['bbox'] = [x1, y1, x2, y2]
                                last_recognition_results[face_id]['tracked'] = tracked

                            # Padding yang lebih konservatif
                            padding = 3  # Dikurangi dari 5
                            x1_pad = max(0, x1 - padding)
                            y1_pad = max(0, y1 - padding)
                            x2_pad = min(frame.shape[1], x2 + padding)
                            y2_pad = min(frame.shape[0], y2 + padding)

                            face_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

                            # Quality validation untuk face crop
                            if (face_crop.size > 0 and 
                                face_crop.shape[0] >= 40 and face_crop.shape[1] >= 40 and  # Minimal 40x40
                                face_crop.shape[0] <= 300 and face_crop.shape[1] <= 300):  # Maksimal 300x300
                                should_process = True
                                if face_id in face_memory:
                                    # Check jika sudah stop processing (sudah tercatat)
                                    if face_memory[face_id].get('stop_processing', False):
                                        should_process = False
                                        # print(f"⏭️ SKIP Face {face_id} - sudah tercatat, tidak diproses lagi")  # Commented untuk clean log
                                    else:
                                        last_process_time = face_memory[face_id].get('last_time', 0)
                                        stability_count = face_memory[face_id].get('stability_count', 0)

                                        # Proses lebih agresif untuk startup cepat
                                        if stability_count >= 1 and (time.time() - last_process_time) < 0.8:  # Dikurangi drastis
                                            should_process = False

                                if should_process and face_recognizer:
                                    try:
                                        name, confidence, is_fake, spoof_reason = face_recognizer.recognize_face(
                                            face_crop, threshold=0.3  # Disesuaikan dari 0.35 untuk balance
                                        )

                                        if face_id not in face_memory:
                                            face_memory[face_id] = {
                                                'results': [],
                                                'stability_count': 0,
                                                'last_time': time.time()
                                            }

                                        face_memory[face_id]['results'].append({
                                            'name': name,
                                            'confidence': confidence,
                                            'is_fake': is_fake,
                                            'spoof_reason': spoof_reason
                                        })

                                        if len(face_memory[face_id]['results']) > 2:  # Dikurangi dari 3 ke 2
                                            face_memory[face_id]['results'] = face_memory[face_id]['results'][-2:]

                                        results = face_memory[face_id]['results']
                                        if len(results) >= 1:  # Dikurangi dari 2 ke 1 untuk decision super cepat
                                            name_votes = {}
                                            fake_votes = 0
                                            total_votes = len(results)

                                            for result in results:
                                                if result['is_fake']:
                                                    fake_votes += 1
                                                elif result['name']:
                                                    name_votes[result['name']] = name_votes.get(result['name'], 0) + 1

                                            if fake_votes >= 1:  # Dikurangi dari 2 ke 1 untuk decision cepat
                                                stable_name = 'FAKE'
                                                stable_is_fake = True
                                                stable_confidence = sum(r['confidence'] for r in results) / len(results)
                                                stable_spoof_reason = results[-1]['spoof_reason']
                                            elif name_votes:
                                                stable_name = max(name_votes, key=name_votes.get)
                                                if name_votes[stable_name] >= 1:  # Dikurangi dari 2 ke 1
                                                    stable_is_fake = False
                                                    stable_confidence = sum(r['confidence'] for r in results if r['name'] == stable_name) / name_votes[stable_name]
                                                    stable_spoof_reason = ''
                                                    face_memory[face_id]['stability_count'] += 1
                                                    
                                                    # LANGSUNG RECORD saat terkenali - tidak perlu menunggu biru
                                                    if stable_confidence > 0.32 and not last_recognition_results[face_id].get('recorded', False):
                                                        print(f"🟢➡️📝 HIJAU LANGSUNG RECORD: {stable_name} ({stable_confidence:.3f})")
                                                        
                                                        # Check if not already recorded
                                                        if stable_name not in recorded_students:
                                                            try:
                                                                success = database.add_attendance(
                                                                    current_session['session_id'], 
                                                                    stable_name, 
                                                                    float(stable_confidence)
                                                                )
                                                                if success:
                                                                    recorded_students.add(stable_name)
                                                                    last_recognition_results[face_id]['recorded'] = True
                                                                    # STOP PROCESSING untuk face ini - mencegah double process
                                                                    face_memory[face_id]['stop_processing'] = True
                                                                    print(f"✅ LANGSUNG TERCATAT: {stable_name} - Confidence: {stable_confidence:.3f}")
                                                                    print(f"🛑 STOP PROCESSING untuk Face {face_id} - sudah tercatat!")
                                                                    print(f"📊 IMMEDIATE UPDATE: Total recorded = {len(recorded_students)}")
                                                                    
                                                                    # Force immediate database check
                                                                    time.sleep(0.2)  # Brief pause untuk memastikan database ready
                                                                else:
                                                                    print(f"❌ Failed to save attendance for {stable_name}")
                                                            except Exception as db_error:
                                                                print(f"❌ DB error: {db_error}")
                                                        else:
                                                            print(f"⚠️ {stable_name} sudah tercatat sebelumnya")
                                                else:
                                                    stable_name = 'Unknown'
                                                    stable_is_fake = False
                                                    stable_confidence = 0.0
                                                    stable_spoof_reason = ''
                                            else:
                                                stable_name = 'Unknown'
                                                stable_is_fake = False
                                                stable_confidence = 0.0
                                                stable_spoof_reason = ''

                                            last_recognition_results[face_id].update({
                                                'name': stable_name,
                                                'confidence': stable_confidence,
                                                'is_fake': stable_is_fake,
                                                'spoof_reason': stable_spoof_reason
                                            })

                                            if (stable_name not in ['Unknown', 'FAKE'] and 
                                                not stable_is_fake and 
                                                stable_confidence >= 0.35 and
                                                current_session and 
                                                stable_name not in recorded_students and 
                                                database and
                                                face_memory[face_id]['stability_count'] >= 3):

                                                try:
                                                    print(f"🔄 Attempting to record attendance for {stable_name}")
                                                    print(f"   Session ID: {current_session['session_id']}")
                                                    print(f"   Confidence: {stable_confidence:.3f}")
                                                    print(f"   Already recorded students: {list(recorded_students)}")
                                                    
                                                    success = database.add_attendance(
                                                        current_session['session_id'], 
                                                        stable_name, 
                                                        float(stable_confidence)
                                                    )
                                                    if success:
                                                        recorded_students.add(stable_name)
                                                        last_recognition_results[face_id]['recorded'] = True
                                                        print(f"🎉 ATTENDANCE RECORDED! (ID:{face_id}): {stable_name} - Confidence: {stable_confidence:.3f}")
                                                        print(f"📊 Total recorded students now: {len(recorded_students)}")
                                                        print(f"📝 Recorded students list: {list(recorded_students)}")
                                                        print(f"🔵 Face {face_id} should now be BLUE!")
                                                    else:
                                                        print(f"❌ Failed to save attendance for {stable_name}")
                                                except Exception as db_error:
                                                    print(f"❌ DB error: {db_error}")

                                        face_memory[face_id]['last_time'] = time.time()

                                    except Exception as e:
                                        print(f"❌ ArcFace error: {e}")

                        faces_to_remove = []
                        for face_id in list(last_recognition_results.keys()):
                            if face_id not in current_face_ids:
                                faces_to_remove.append(face_id)

                        for face_id in faces_to_remove:
                            del last_recognition_results[face_id]
                            if face_id in face_memory:
                                del face_memory[face_id]

                    except Exception as e:
                        print(f"❌ YOLO error: {e}")

            time.sleep(0.1)

        except Exception as e:
            print(f"❌ AI processing error: {e}")
            time.sleep(0.1)

    processing_active = False
    print("🤖 AI Processing stopped")

def draw_detection_overlay(frame):
    """Draw STABLE detection overlay dengan tracking info"""
    global last_recognition_results

    # FIX: Create copy to avoid "dictionary changed size during iteration"
    results_copy = dict(last_recognition_results)

    for i, result in results_copy.items():
        bbox = result['bbox']
        x1, y1, x2, y2 = bbox
        name = result.get('name', 'Unknown')
        confidence = result.get('confidence', 0)
        recorded = result.get('recorded', False)
        face_id = result.get('face_id', i)
        tracked = result.get('tracked', False)

        # Log status tapi tetap tampilkan bounding box
        if recorded:
            # Tampilkan dengan warna berbeda untuk yang sudah tercatat
            color = (128, 128, 128)  # Abu-abu untuk sudah tercatat
            label = f"{name} ✓ TERCATAT"
            # print(f"🔵 Face {face_id} sudah tercatat - TAMPIL ABU-ABU")  # Commented untuk clean log
        elif name == "FAKE":
            color = (0, 0, 255)    # MERAH untuk FAKE dalam BGR
            label = f"FAKE"
        elif name != 'Unknown':
            color = (0, 255, 0)    # HIJAU untuk terdeteksi dalam BGR
            label = f"{name}"
        else:
            color = (0, 255, 255)  # KUNING untuk unknown dalam BGR
            label = f"Unknown"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

def generate_frames():
    """Generate frames untuk streaming"""
    global latest_frame

    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    """Main page dengan data dosen dan mata kuliah"""
    dosens = []
    subjects = []
    error = None

    try:
        if database and database.connection:
            dosens = database.get_dosen_all()
            subjects = database.get_mata_kuliah_all()
            print(f"🔥 DEBUG: Loaded {len(dosens)} dosens: {dosens}")
            print(f"🔥 DEBUG: Loaded {len(subjects)} subjects: {subjects}")
        else:
            error = "Database tidak terhubung"
            print("❌ Database tidak terhubung")
    except Exception as e:
        error = f"Gagal mengambil data dari database: {e}"
        print(f"❌ Error loading data: {e}")

    return render_template('index.html', dosens=dosens, subjects=subjects, error=error)

@app.route('/camera')
@app.route('/rekap')
def camera():
    """Halaman rekap - gabungan camera + tabel rekap - PERBAIKAN INFORMASI SESI & ATTENDANCE"""
    global current_session, database
    
    try:
        session_to_use = None
        session_id = None
        
        print("🔍 DEBUGGING CAMERA PAGE:")
        print(f"   Current session: {current_session}")
        print(f"   Database connection: {database is not None}")

        # Priority 1: Gunakan current_session jika ada
        if current_session and 'session_id' in current_session:
            session_to_use = current_session
            session_id = current_session['session_id']
            print(f"✅ Using current session: {session_id}")
            print(f"   Session details: {session_to_use}")
            
        # Priority 2: Cari dari database jika current_session kosong
        elif database and database.connection:
            try:
                print("🔍 Searching for active session in database...")
                database.cursor.execute("SELECT * FROM sessions WHERE status = 'active' ORDER BY id DESC LIMIT 1")
                latest_session = database.cursor.fetchone()

                if latest_session:
                    session_id = latest_session[0]
                    session_to_use = {
                        'session_id': session_id,
                        'dosen': latest_session[1] if latest_session[1] else 'Unknown Dosen',
                        'subject': latest_session[2] if latest_session[2] else 'Unknown Subject',
                        'start_time': str(latest_session[4]) if latest_session[4] else datetime.now().strftime('%H:%M:%S')
                    }
                    print(f"✅ Found active session from database: {session_id}")
                    print(f"   Session details: {session_to_use}")
                    
                    # Update current_session dengan data dari database
                    current_session = session_to_use
                else:
                    print("❌ No active session found in database")
            except Exception as db_error:
                print(f"❌ Database session check error: {db_error}")

        # Jika tidak ada session sama sekali
        if not session_to_use or not session_id:
            print("❌ No session available, showing error")
            return render_template('camera.html', 
                                 session=None, 
                                 attendance_data=[], 
                                 error="Tidak ada sesi aktif. Silakan buat sesi baru dari halaman input.")

        # Ambil data attendance untuk tabel rekap dengan debugging lengkap
        print(f"🔄 Fetching attendance for session {session_id}...")
        attendance_raw = database.get_session_attendance(session_id) if database else []
        print(f"🔍 Raw attendance data: {len(attendance_raw)} records")
        
        for i, record in enumerate(attendance_raw):
            print(f"   Raw record {i+1}: {record}")

        # Format data attendance untuk template dengan error handling
        attendance_formatted = []
        for i, record in enumerate(attendance_raw):
            try:
                formatted_record = {
                    'student_name': record.get('student_name', record.get('nama_mahasiswa', 'Unknown Student')),
                    'nim': record.get('nim', 'N/A'),
                    'confidence': float(record.get('confidence', 0)),
                    'waktu_hadir': record.get('waktu_hadir', ''),
                    'timestamp': record.get('timestamp', record.get('created_at', datetime.now())),
                    'status': record.get('status', 'hadir')
                }
                
                # Format waktu_hadir jika berupa datetime object
                if hasattr(formatted_record['waktu_hadir'], 'strftime'):
                    formatted_record['waktu_hadir'] = formatted_record['waktu_hadir'].strftime('%H:%M:%S')
                elif not formatted_record['waktu_hadir']:
                    formatted_record['waktu_hadir'] = 'Just now'
                
                attendance_formatted.append(formatted_record)
                print(f"   Formatted record {i+1}: {formatted_record}")
                
            except Exception as format_error:
                print(f"❌ Error formatting record {i+1}: {format_error}")
                continue

        # Pastikan session memiliki start_time
        if 'start_time' not in session_to_use or not session_to_use['start_time']:
            session_to_use['start_time'] = datetime.now().strftime('%H:%M:%S')

        print(f"📊 FINAL DATA FOR TEMPLATE:")
        print(f"   Session: {session_to_use}")
        print(f"   Attendance count: {len(attendance_formatted)}")
        print(f"   Attendance data sample: {attendance_formatted[:2] if attendance_formatted else 'No data'}")

        return render_template('camera.html',
                             session=session_to_use,
                             attendance_data=attendance_formatted,
                             error=None)

    except Exception as e:
        print(f"❌ Camera page error: {e}")
        import traceback
        traceback.print_exc()
        
        return render_template('camera.html', 
                             session=None, 
                             attendance_data=[], 
                             error=f"Terjadi kesalahan: {str(e)}")

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_dosen')
def get_dosen():
    """Get dosen list"""
    try:
        if database:
            dosens = database.get_dosen_all()
            return jsonify({'success': True, 'dosens': dosens})
        return jsonify({'success': False, 'error': 'Database not connected'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_subjects')
def get_subjects():
    """Get subjects list"""
    try:
        if database:
            subjects = database.get_mata_kuliah_all()
            return jsonify({'success': True, 'subjects': subjects})
        return jsonify({'success': False, 'error': 'Database not connected'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/start_session', methods=['POST'])
def start_session():
    """Start attendance session - PURE PYTHON NO JS! dengan LOCAL CAMERA DISPLAY"""
    global current_session, is_running, local_camera_window, mirror_camera

    try:
        dosen_name = request.form.get('dosen')
        subject = request.form.get('subject')
        
        # Otomatis aktifkan kamera dan mirror tanpa perlu checkbox
        local_camera_window = True  # Selalu tampilkan kamera
        mirror_camera = True  # Selalu mirror

        print(f"🔥 DEBUG: Form data - Dosen: {dosen_name}, Subject: {subject}")
        print(f"📷 Local camera window: ENABLED (otomatis)")
        print(f"🪞 Mirror camera: ENABLED (otomatis)")

        if not dosen_name or not subject:
            print(f"❌ Missing data: dosen={dosen_name}, subject={subject}")
            return redirect(url_for('index'))

        if database:
            session_id = database.create_session(dosen_name, subject)
            if session_id:
                current_session = {
                    'session_id': session_id,
                    'dosen': dosen_name,
                    'subject': subject,
                    'start_time': datetime.now().strftime('%H:%M:%S')
                }
                print(f"✅ Session created: ID={session_id}, Dosen={dosen_name}, Subject={subject}")

                if not is_running:
                    is_running = True
                    camera_thread = threading.Thread(target=camera_loop, daemon=True)
                    camera_thread.start()

                    print("📷 Starting LOCAL CAMERA WINDOW (otomatis)...")

                return redirect(url_for('camera'))
            else:
                print("❌ Failed to create session")
                return redirect(url_for('index'))
        else:
            print("❌ Database not available - cannot create session")
            return redirect(url_for('index'))

    except Exception as e:
        print(f"❌ Start session error: {e}")
        return redirect(url_for('index'))

@app.route('/download_csv')
def download_csv():
    """Download attendance sebagai CSV file - PURE PYTHON"""
    try:
        global current_session, database
        
        if not current_session or not database:
            print("❌ No session or database for CSV download")
            return redirect(url_for('camera'))

        session_id = current_session.get('session_id')
        if not session_id:
            print("❌ No session_id for CSV download")
            return redirect(url_for('camera'))

        # Ambil data attendance
        attendance_data = database.get_session_attendance(session_id)
        
        if not attendance_data:
            print("❌ No attendance data for CSV download")
            return redirect(url_for('camera'))

        # Generate CSV content - PURE PYTHON
        import io
        from flask import make_response
        
        output = io.StringIO()
        
        # Header CSV
        output.write("No,Nama Mahasiswa,NIM,Confidence,Waktu Hadir,Tanggal,Status\n")
        
        # Data rows
        for i, record in enumerate(attendance_data, 1):
            nama = record.get('student_name', record.get('nama_mahasiswa', 'Unknown'))
            nim = record.get('nim', 'N/A')
            confidence = record.get('confidence', 0) * 100  # Convert to percentage
            waktu = record.get('waktu_hadir', 'Unknown')
            tanggal = record.get('timestamp', record.get('created_at', 'Unknown'))
            status = record.get('status', 'Hadir')
            
            # Format tanggal
            if hasattr(tanggal, 'strftime'):
                tanggal_str = tanggal.strftime('%Y-%m-%d')
            else:
                tanggal_str = str(tanggal)
            
            # Format waktu
            if hasattr(waktu, 'strftime'):
                waktu_str = waktu.strftime('%H:%M:%S')
            else:
                waktu_str = str(waktu)
            
            output.write(f"{i},\"{nama}\",{nim},{confidence:.1f}%,{waktu_str},{tanggal_str},{status}\n")
        
        # Generate response
        csv_content = output.getvalue()
        output.close()
        
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=attendance_session_{session_id}.csv'
        
        print(f"✅ CSV downloaded: {len(attendance_data)} records")
        return response

    except Exception as e:
        print(f"❌ CSV download error: {e}")
        return redirect(url_for('camera'))

@app.route('/selesai', methods=['POST'])
def selesai():
    """SELESAI - Stop semua proses dan kembali ke halaman input"""
    global current_session, is_running, local_camera_window, mirror_camera, recorded_students, face_memory, last_recognition_results

    try:
        print("🛑 SELESAI - Stopping all processes...")
        
        # Stop semua proses
        is_running = False
        local_camera_window = False
        
        # Clear semua memory dan cache
        recorded_students.clear()
        face_memory.clear()
        last_recognition_results.clear()
        
        # Clear session
        current_session = None
        
        # Close semua OpenCV windows
        cv2.destroyAllWindows()
        
        # Brief pause untuk cleanup
        time.sleep(1)
        
        print("✅ SELESAI - All processes stopped, returning to input page")
        return redirect(url_for('index'))

    except Exception as e:
        print(f"❌ Selesai error: {e}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    print("🚀 Starting ULTRA SMOOTH Face Recognition App...")
    init_components()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
