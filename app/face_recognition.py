import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import time
import threading
from queue import Queue

class ArcFaceRecognition:
    def __init__(self, model_path="models/arcface_model.onnx"):
        """
        Initialize ArcFace recognition system with CACHING for performance
        Args:
            model_path: Path to ArcFace ONNX model
        """
        self.model_path = model_path
        self.net = None
        self.face_database = {}
        self.embeddings_file = "models/embeddings.pkl"

        self.embedding_cache = {}
        self.cache_max_size = 50

        self.load_model()

        self.load_embeddings()
        self.net = None
        self.face_database = {}
        self.embeddings_file = "models/embeddings.pkl"

        self.embedding_cache = {}
        self.cache_max_size = 50

        self.load_model()

        self.load_embeddings()

    def load_model(self):
        """Load ONLY ArcFace ONNX model - NO FALLBACK"""
        try:
            if os.path.exists(self.model_path):
                import onnxruntime as ort
                self.net = ort.InferenceSession(self.model_path)
                print("✅ ArcFace ONNX model loaded successfully!")
            else:
                raise FileNotFoundError(f"ArcFace ONNX model not found at {self.model_path}")
        except Exception as e:
            print(f"❌ CRITICAL: ArcFace model loading failed: {e}")
            print("� NO FALLBACK - ArcFace ONNX model is REQUIRED!")
            raise e

    def preprocess_face(self, face_image):
        """
        ENHANCED preprocessing untuk ArcFace model dengan quality control
        Args:
            face_image: Cropped face image
        Returns:
            Preprocessed face array for ArcFace [1, 3, 112, 112]
        """
        if face_image is None or face_image.size == 0:
            return None
            
        try:
            # Quality check: pastikan face image minimal 32x32 
            h, w = face_image.shape[:2]
            if h < 32 or w < 32:
                print(f"⚠️ Face image too small: {face_image.shape}")
                return None
            
            # Aspect ratio check: jika terlalu ekstrem, mungkin bukan wajah yang baik
            aspect_ratio = max(h, w) / min(h, w)
            if aspect_ratio > 3.0:
                print(f"⚠️ Extreme aspect ratio: {aspect_ratio:.2f}")
                # return None  # Comment untuk sementara, tapi warning
                
            # ENHANCED: Resize dengan padding untuk maintain aspect ratio
            # Buat square image dulu
            max_dim = max(h, w)
            square_image = np.zeros((max_dim, max_dim, 3), dtype=face_image.dtype)
            
            # Center the face in square
            y_offset = (max_dim - h) // 2
            x_offset = (max_dim - w) // 2
            square_image[y_offset:y_offset+h, x_offset:x_offset+w] = face_image
            
            # Resize to 112x112 dengan interpolasi terbaik
            face_resized = cv2.resize(square_image, (112, 112), interpolation=cv2.INTER_LANCZOS4)
            
            # OPTIONAL: Light histogram equalization untuk better contrast
            # Convert to YUV, equalize Y channel, convert back
            if len(face_resized.shape) == 3:
                yuv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                face_resized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            # Light denoising untuk reduce noise
            face_resized = cv2.bilateralFilter(face_resized, 3, 10, 10)

            # Normalisasi ke [0, 1]
            face_normalized = face_resized.astype(np.float32) / 255.0

            # Convert ke RGB (ArcFace trained dengan RGB)
            face_rgb = cv2.cvtColor(face_normalized, cv2.COLOR_BGR2RGB)

            # OPTIONAL: Mean subtraction (uncomment jika hasil buruk)
            # mean = np.array([0.5, 0.5, 0.5])
            # std = np.array([0.5, 0.5, 0.5])
            # face_rgb = (face_rgb - mean) / std

            # Transpose ke format CHW (Channel-Height-Width)
            face_transposed = np.transpose(face_rgb, (2, 0, 1))

            # Add batch dimension [1, 3, 112, 112]
            face_batch = np.expand_dims(face_transposed, axis=0)

            return face_batch
            
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return None

            # Add batch dimension
            face_batch = np.expand_dims(face_transposed, axis=0)

            return face_batch
        return None

    def extract_embedding(self, face_image):
        """
        Extract face embedding using ONLY ArcFace ONNX model
        Args:
            face_image: Cropped face image
        Returns:
            512-dimensional ArcFace embedding vector
        """
        try:
            if isinstance(self.net, str):
                raise ValueError("ArcFace ONNX model is required - no fallback allowed!")

            preprocessed = self.preprocess_face(face_image)
            if preprocessed is None:
                return None

            input_name = self.net.get_inputs()[0].name
            outputs = self.net.run(None, {input_name: preprocessed})
            embedding = outputs[0][0]

            embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            print(f"❌ ArcFace embedding extraction error: {e}")
            return None

    def add_person(self, name, face_image):
        """
        Add person to face database
        Args:
            name: Person's name
            face_image: Cropped face image
        """
        embedding = self.extract_embedding(face_image)
        if embedding is not None:
            if name not in self.face_database:
                self.face_database[name] = []
            self.face_database[name].append(embedding)
            print(f"✅ Added {name} to database (total embeddings: {len(self.face_database[name])})")

            self.save_embeddings()
            return True
        else:
            print(f"❌ Failed to extract embedding for {name}")
            return False

    def recognize_face(self, face_image, threshold=0.25):  # Naikkan dari 0.18 ke 0.25
        """
        ULTRA STRICT ArcFace recognition untuk mengurangi false positive drastis
        Args:
            face_image: Cropped face image
            threshold: Recognition threshold (STRICT)
        Returns:
            Tuple of (name, confidence) or (None, 0)
        """
        if len(self.face_database) == 0:
            print("❌ No face database loaded!")
            return None, 0

        query_embedding = self.extract_embedding(face_image)
        if query_embedding is None:
            print("❌ Failed to extract embedding from face")
            return None, 0

        best_match = None
        best_confidence = 0
        all_similarities = []

        from sklearn.metrics.pairwise import cosine_similarity

        for name, stored_embeddings in self.face_database.items():
            if isinstance(stored_embeddings, list):
                similarities = []
                for stored_embedding in stored_embeddings:
                    # ULTRA STRICT: Hanya cosine similarity, no combination
                    cosine_sim = cosine_similarity([query_embedding], [stored_embedding])[0][0]
                    similarities.append(cosine_sim)
                
                if not similarities:
                    continue
                
                # ULTRA STRICT: Gunakan rata-rata untuk konsistensi, bukan max
                max_similarity = max(similarities)
                avg_similarity = sum(similarities) / len(similarities)
                min_similarity = min(similarities)
                
                # CONSISTENCY CHECK: Jika range terlalu besar, ada masalah
                similarity_range = max_similarity - min_similarity
                consistency_penalty = min(similarity_range * 2.5, 0.4)  # NAIKKAN penalty
                
                # ULTRA STRICT: Prioritas konsistensi daripada peak performance
                if len(similarities) >= 4:
                    # Buang outlier: gunakan interquartile mean
                    sorted_sims = sorted(similarities)
                    q1_idx = len(sorted_sims) // 4
                    q3_idx = 3 * len(sorted_sims) // 4
                    iqr_sims = sorted_sims[q1_idx:q3_idx+1] if q3_idx > q1_idx else sorted_sims
                    iqr_avg = sum(iqr_sims) / len(iqr_sims)
                    final_confidence = (iqr_avg * 0.7) + (avg_similarity * 0.3)  # Prioritas IQR
                elif len(similarities) >= 3:
                    # Median-based untuk mengurangi outlier impact
                    sorted_sims = sorted(similarities)
                    median_similarity = sorted_sims[len(sorted_sims)//2]
                    final_confidence = (median_similarity * 0.6) + (avg_similarity * 0.4)
                else:
                    # STRICT: Dengan sedikit embeddings, harus sangat konsisten
                    final_confidence = avg_similarity * 0.8  # LEBIH KONSERVATIF
                
                # APPLY STRICT consistency penalty
                final_confidence = max(0, final_confidence - consistency_penalty)
                
                # ADDITIONAL CHECK: Jika ada outlier yang terlalu rendah, turunkan confidence
                if len(similarities) > 1 and min_similarity < 0.12:
                    outlier_penalty = (0.12 - min_similarity) * 0.5
                    final_confidence = max(0, final_confidence - outlier_penalty)
                    print(f"⚠️ {name}: Low outlier detected, penalty={outlier_penalty:.3f}")
                
                all_similarities.append((name, final_confidence, max_similarity, avg_similarity, consistency_penalty))
            else:
                # Single embedding - lebih ketat
                cosine_sim = cosine_similarity([query_embedding], [stored_embeddings])[0][0]
                # STRICT: Single embedding mendapat penalty karena tidak ada validasi
                single_penalty = 0.05  # Penalty untuk single embedding
                final_confidence = max(0, cosine_sim - single_penalty)
                all_similarities.append((name, final_confidence, final_confidence, final_confidence, single_penalty))

            if final_confidence > best_confidence:
                best_match = name
                best_confidence = final_confidence

        # ULTRA STRICT: Gap analysis dengan threshold yang sangat tinggi
        if len(all_similarities) > 1:
            sorted_matches = sorted(all_similarities, key=lambda x: x[1], reverse=True)
            first_score = sorted_matches[0][1]
            second_score = sorted_matches[1][1] if len(sorted_matches) > 1 else 0
            
            gap = first_score - second_score
            
            # STRICT: Gap requirement yang tinggi untuk semua level
            if first_score > 0.35:
                min_gap = 0.06  # High confidence, tapi tetap perlu gap besar
            elif first_score > 0.28:
                min_gap = 0.08  # Medium confidence, gap besar
            else:
                min_gap = 0.12  # Low confidence, gap sangat besar
                
            if gap < min_gap:
                print(f"⚠️ STRICT: Ambiguous match rejected: gap={gap:.3f} (need {min_gap:.3f})")
                best_confidence *= 0.5  # BESAR penalty untuk ambiguous result
                
            # ADDITIONAL: Bahkan untuk confidence tinggi, cek second place
            if first_score > 0.30 and second_score > 0.25:
                print(f"⚠️ STRICT: Multiple high confidence matches - reducing confidence")
                best_confidence *= 0.8
                
        # Enhanced debug output dengan strict analysis
        if all_similarities:
            sorted_similarities = sorted(all_similarities, key=lambda x: x[1], reverse=True)
            print(f"🔍 STRICT Recognition analysis:")
            for i, (name, final_conf, max_conf, avg_conf, penalty) in enumerate(sorted_similarities[:3]):
                marker = "🎯" if name == best_match else "  "
                status = "✓" if final_conf >= threshold else "✗"
                print(f"  {marker} {name}: {status} final={final_conf:.3f} (max={max_conf:.3f}, avg={avg_conf:.3f}, penalty={penalty:.3f})")
            
        # ULTRA STRICT threshold check
        effective_threshold = threshold
        
        # ADDITIONAL STRICT CHECKS
        if best_confidence > threshold:
            # Extra validation untuk recognition
            confidence_level = "REJECTED"
            
            if best_confidence > 0.40:
                confidence_level = "ULTRA HIGH"
            elif best_confidence > 0.32:
                confidence_level = "HIGH"
            elif best_confidence > 0.28:
                confidence_level = "MODERATE"
            elif best_confidence > 0.25:
                confidence_level = "LOW - WATCH"
            
            # STRICT: Tambahan check untuk confidence rendah
            if best_confidence < 0.30:
                print(f"⚠️ {confidence_level} CONFIDENCE: {best_match} ({best_confidence:.3f}) - NEEDS VERIFICATION")
                # Mungkin perlu second validation di masa depan
            else:
                print(f"✅ {confidence_level} CONFIDENCE: {best_match} ({best_confidence:.3f})")
                
            return best_match, best_confidence
        else:
            if best_match:
                print(f"❌ REJECTED: {best_match} below strict threshold {threshold:.3f} (got: {best_confidence:.3f})")
            else:
                print(f"❌ No match found above strict threshold {threshold:.3f}")
            return None, 0

    def get_face_embedding(self, face_image):
        """
        Get face embedding for given image
        Args:
            face_image: Cropped face image
        Returns:
            Face embedding vector or None if failed
        """
        return self.extract_embedding(face_image)

    def save_embeddings(self):
        """Save face embeddings to file"""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.face_database, f)
            print(f"💾 Embeddings saved to {self.embeddings_file}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")

    def load_embeddings(self):
        """Load face embeddings from file"""
        try:
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'rb') as f:
                    self.face_database = pickle.load(f)
                print(f"📂 Loaded {len(self.face_database)} faces from {self.embeddings_file}")

                print("🔍 EMBEDDING ANALYSIS:")
                for name, emb_list in self.face_database.items():
                    if isinstance(emb_list, list) and len(emb_list) > 0:
                        first_emb = emb_list[0]
                        print(f"✅ {name}: {len(emb_list)} embeddings")
                        print(f"📏 Dimension: {len(first_emb)}")
                        print(f"🔬 First 5 values: {first_emb[:5]}")

                        if len(first_emb) == 512:
                            print(f"✅ {name}: ArcFace format (512D) ✓")
                        elif len(first_emb) == 128:
                            print(f"⚠️  {name}: face_recognition format (128D) - MIXED DATABASE!")
                        elif len(first_emb) == 4096:
                            print(f"❌ {name}: Simple histogram format (4096D) - CORRUPTED!")
                        else:
                            print(f"❓ {name}: Unknown format ({len(first_emb)}D)")
                    else:
                        print(f"❌ {name}: Invalid embedding format")
                print("=" * 50)
            else:
                print("📁 No existing embeddings found, starting fresh")
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            self.face_database = {}

    def list_people(self):
        """List all people in database"""
        if len(self.face_database) == 0:
            print("📭 No people in database")
        else:
            print(f"👥 People in database ({len(self.face_database)}):")
            for name in self.face_database.keys():
                print(f"  - {name}")

    def remove_person(self, name):
        """Remove person from database"""
        if name in self.face_database:
            del self.face_database[name]
            self.save_embeddings()
            print(f"🗑️  Removed {name} from database")
            return True
        else:
            print(f"❌ {name} not found in database")
            return False

def test_arcface():
    """Test ArcFace recognition system"""
    print("🧪 Testing ArcFace Recognition System...")

    arcface = ArcFaceRecognition()

    print("\n📊 Database status:")
    arcface.list_people()

    print("\n✅ ArcFace system ready!")
    return arcface

if __name__ == "__main__":
    test_arcface()


class FaceRecognition:
    """Main Face Recognition class with anti-spoofing integration"""

    def __init__(self):
        """Initialize face recognition with HIGH PERFORMANCE optimizations"""
        # print("🔄 Initializing face recognition...")  # Commented untuk speed

        self.recognition_cache = {}
        self.last_recognition_time = {}
        self.cache_duration = 1.0  # Dikurangi dari 2.0 ke 1.0 untuk response lebih cepat

        print("🛡️ ENHANCED ANTI-SPOOFING ENABLED - FOKUS B&W DETECTION!")
        try:    
            from .anti_spoofing import AntiSpoofing
            self.anti_spoofing = AntiSpoofing()
            print("✅ Enhanced anti-spoofing loaded - ultra-aggressive B&W detection")
        except Exception as e:
            print(f"❌ Failed to load enhanced anti-spoofing: {e}")
            self.anti_spoofing = None

        try:
            self.arcface = ArcFaceRecognition()
            # print("✅ Using ArcFace ONNX model (512 dimensions)")  # Commented untuk speed
        except Exception as e:
            print(f"❌ Error initializing ArcFace: {e}")
            self.arcface = None

        self.load_database()

    def load_database(self):
        """Load face embeddings database"""
        if self.arcface:
            people_count = len(self.arcface.face_database)
            if people_count > 0:
                print(f"📂 Loaded {people_count} faces from models/embeddings.pkl")
                print("🔄 Database compatible with arcface method")
                print(f"✅ System ready - Database: {people_count} people")
            else:
                print("📂 No faces found in database")
        else:
            print("❌ Face recognition not available")

    def recognize_face(self, face_image, threshold=0.3):
        """Face recognition method dengan anti-spoofing integration - ULTRA STRICT"""
        if self.arcface:
            is_fake = False
            spoof_reason = ""

            print(f"🔍 STARTING STRICT RECOGNITION: Face shape={face_image.shape if face_image is not None else 'None'}")

            if self.anti_spoofing:
                try:
                    print("🛡️ CALLING ANTI-SPOOFING...")
                    is_fake, fake_confidence, spoof_reason = self.anti_spoofing.detect_fake(face_image)
                    print(f"🛡️ ANTI-SPOOFING RESULT: is_fake={is_fake}, confidence={fake_confidence:.3f}, reason='{spoof_reason}'")

                    # STRICT FAKE DETECTION: Block jika ada indikasi B&W/fake
                    if is_fake and fake_confidence > 0.8:  # Strict threshold untuk fake
                        print("🚫🚫🚫 BLACK & WHITE / FAKE DETECTED - BLOCKING RECOGNITION")
                        return "FAKE", fake_confidence, is_fake, spoof_reason
                    elif is_fake and fake_confidence > 0.6:
                        print("⚠️ SUSPICIOUS CONTENT - PROCEEDING WITH EXTRA CAUTION")
                        # Naikkan threshold ArcFace untuk suspicious content
                        threshold = min(threshold + 0.05, 0.4)  # Tambah difficulty
                    else:
                        print("✅ CONTENT APPROVED - PROCEEDING TO RECOGNITION")
                        is_fake = False
                except Exception as e:
                    print(f"❌ Anti-spoofing error: {e}")
                    print("⚠️ ANTI-SPOOFING ERROR - PROCEEDING WITH HIGH THRESHOLD")
                    threshold = min(threshold + 0.03, 0.35)  # Extra caution bila error
            
            print(f"🔍 PROCEEDING TO ARCFACE with threshold={threshold:.3f}...")
            
            # ULTRA STRICT ArcFace recognition
            name, confidence = self.arcface.recognize_face(face_image, threshold)
            print(f"🔍 ARCFACE RESULT: name='{name}', confidence={confidence:.3f}")

            # ULTRA STRICT VALIDATION: Triple-tier confidence system
            ultra_strict_threshold = 0.35  # Ultra high confidence required
            strict_threshold = 0.30       # High confidence required  
            moderate_threshold = 0.25     # Moderate confidence required
            
            if name and confidence > ultra_strict_threshold:  # ULTRA HIGH CONFIDENCE
                is_fake = False
                spoof_reason = "Real face - ULTRA HIGH confidence recognition"
                print(f"✅ ULTRA HIGH CONFIDENCE: {name} ({confidence:.3f}) - FULLY TRUSTED")
                return name, confidence, is_fake, spoof_reason
                
            elif name and confidence > strict_threshold:  # HIGH CONFIDENCE
                # ADDITIONAL VALIDATION untuk high confidence
                print(f"🔍 HIGH CONFIDENCE VALIDATION: {name} ({confidence:.3f})")
                
                # Re-extract untuk double check
                validation_embedding = self.arcface.extract_embedding(face_image)
                if validation_embedding is not None and name in self.arcface.face_database:
                    stored_embeddings = self.arcface.face_database[name]
                    if isinstance(stored_embeddings, list) and len(stored_embeddings) > 1:
                        from sklearn.metrics.pairwise import cosine_similarity
                        val_similarities = []
                        for stored_emb in stored_embeddings:
                            val_sim = cosine_similarity([validation_embedding], [stored_emb])[0][0]
                            val_similarities.append(val_sim)
                        
                        val_avg = sum(val_similarities) / len(val_similarities)
                        val_consistency = 1.0 - (max(val_similarities) - min(val_similarities))
                        
                        # Strict validation requirements
                        if val_avg > 0.28 and val_consistency > 0.85:
                            is_fake = False
                            spoof_reason = "Real face - HIGH confidence validated"
                            print(f"✅ HIGH CONFIDENCE VALIDATED: {name} ({confidence:.3f}, val_avg={val_avg:.3f})")
                            return name, confidence, is_fake, spoof_reason
                        else:
                            print(f"⚠️ VALIDATION FAILED: {name} - val_avg={val_avg:.3f}, consistency={val_consistency:.3f}")
                            return None, 0, False, "Failed validation - not reliable"
                
                # Jika tidak bisa divalidasi tapi confidence tinggi
                is_fake = False
                spoof_reason = "Real face - HIGH confidence recognition"
                print(f"✅ HIGH CONFIDENCE ACCEPTED: {name} ({confidence:.3f})")
                return name, confidence, is_fake, spoof_reason
                
            elif name and confidence > moderate_threshold:  # MODERATE CONFIDENCE - EXTRA STRICT
                print(f"⚠️ MODERATE CONFIDENCE: {name} ({confidence:.3f}) - REQUIRES STRICT VALIDATION")
                
                # EXTRA STRICT validation untuk moderate confidence
                validation_embedding = self.arcface.extract_embedding(face_image)
                if validation_embedding is None:
                    print("❌ Cannot extract validation embedding")
                    return None, 0, False, "Validation failed"
                    
                if name not in self.arcface.face_database:
                    print("❌ Name not in database during validation")
                    return None, 0, False, "Database inconsistency"
                
                stored_embeddings = self.arcface.face_database[name]
                if isinstance(stored_embeddings, list):
                    from sklearn.metrics.pairwise import cosine_similarity
                    val_similarities = []
                    for stored_emb in stored_embeddings:
                        val_sim = cosine_similarity([validation_embedding], [stored_emb])[0][0]
                        val_similarities.append(val_sim)
                    
                    val_max = max(val_similarities)
                    val_avg = sum(val_similarities) / len(val_similarities)
                    val_std = np.std(val_similarities)
                    
                    # EXTRA STRICT requirements untuk moderate confidence
                    if val_avg > 0.27 and val_max > 0.32 and val_std < 0.08:
                        is_fake = False
                        spoof_reason = "Real face - moderate confidence with strict validation"
                        print(f"✅ MODERATE CONFIDENCE STRICT VALIDATION PASSED: {name}")
                        print(f"   Validation: avg={val_avg:.3f}, max={val_max:.3f}, std={val_std:.3f}")
                        return name, val_avg, is_fake, spoof_reason  # Use validation average
                    else:
                        print(f"❌ STRICT VALIDATION FAILED: avg={val_avg:.3f}, max={val_max:.3f}, std={val_std:.3f}")
                        return None, 0, False, "Failed strict validation"
                else:
                    # Single embedding - lebih strict lagi
                    val_sim = cosine_similarity([validation_embedding], [stored_embeddings])[0][0]
                    if val_sim > 0.30:  # Higher threshold untuk single embedding
                        is_fake = False
                        spoof_reason = "Real face - moderate confidence single validation"
                        print(f"✅ SINGLE EMBEDDING VALIDATION PASSED: {name} ({val_sim:.3f})")
                        return name, val_sim, is_fake, spoof_reason
                    else:
                        print(f"❌ SINGLE EMBEDDING VALIDATION FAILED: {val_sim:.3f}")
                        return None, 0, False, "Single embedding validation failed"
                        
            else:  # BELOW MODERATE THRESHOLD
                if name:
                    print(f"❌ CONFIDENCE TOO LOW: {name} ({confidence:.3f}) below threshold {moderate_threshold:.3f}")
                else:
                    print(f"❌ NO MATCH FOUND above threshold {threshold:.3f}")
                    
                # Check untuk suspicious fake detection
                if is_fake and fake_confidence > 0.5:
                    print(f"🚫 ADDITIONAL: Potential fake detected ({fake_confidence:.3f})")
                    return "FAKE", fake_confidence, is_fake, spoof_reason

            return None, 0, False, "No reliable recognition"
        else:
            print("❌ NO ARCFACE DETECTOR AVAILABLE!")
            return None, 0, False, ""

    def recognize_face_with_antispoofing(self, face_roi):
        """OPTIMIZED face recognition WITH anti-spoofing and SMART CACHING"""
        current_time = time.time()

        #     if current_time - cached_result['timestamp'] < self.cache_duration:
        #         return cached_result['name'], cached_result['confidence'], cached_result['antispoofing_result']

        result = {
            'recognized': False,
            'name': 'Unknown',
            'confidence': 0.0,
            'antispoofing_result': {
                'is_live': True,
                'confidence': 1.0,
                'score': '5/5',
                'is_fake': False  # Add fake detection flag
            }
        }

        try:
            if self.anti_spoofing:
                # print("🛡️ Running AGGRESSIVE anti-spoofing detection...")  # Commented untuk speed
                is_fake, fake_confidence, spoof_reason = self.anti_spoofing.detect_fake(face_roi)

                result['antispoofing_result'] = {
                    'is_live': not is_fake,
                    'confidence': 1.0 - fake_confidence,
                    'score': f"{int((1.0 - fake_confidence) * 5)}/5",
                    'details': spoof_reason,
                    'is_fake': is_fake
                }

                if is_fake and fake_confidence > 0.85:
                    print(f"🚫 KERTAS/FAKE CONFIRMED! Reason: {spoof_reason}")
                    print(f"🔍 Confidence: {fake_confidence:.3f}")
                    print("⚠️ SISTEM MENOLAK - TIDAK DICATAT KE DATABASE!")
                    result['name'] = 'FAKE'
                    result['confidence'] = 0.0
                    result['recognized'] = False
                    return result['name'], result['confidence'], result['antispoofing_result']
                elif is_fake and fake_confidence > 0.7:
                    print(f"⚠️ SUSPICIOUS: Kemungkinan fake ({fake_confidence:.3f})")
                    print("🔍 Lanjut recognition tapi dengan hati-hati...")
                else:
                    print(f"✅ Live face confirmed! Confidence: {1.0 - fake_confidence:.3f}")

            else:
                print("⚠️ Anti-spoofing detector not available")

            if self.arcface:
                embedding = self.arcface.extract_embedding(face_roi)

                if embedding is not None:
                    best_match = None
                    best_similarity = 0.0
                    threshold = 0.18  # Threshold optimal untuk recognition mudah
                    
                    all_matches = []  # For debugging and optimization

                    for name, stored_embeddings in self.arcface.face_database.items():
                        similarities = []
                        if isinstance(stored_embeddings, list):
                            for stored_embedding in stored_embeddings:
                                sim = cosine_similarity([embedding], [stored_embedding])[0][0]
                                similarities.append(sim)
                            
                            # Use BEST similarity + average for robust matching
                            max_similarity = max(similarities) if similarities else 0.0
                            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
                            
                            # Weighted combination: prioritize best match
                            final_similarity = (max_similarity * 0.7) + (avg_similarity * 0.3)
                            all_matches.append((name, final_similarity, max_similarity, avg_similarity))
                        else:
                            final_similarity = cosine_similarity([embedding], [stored_embeddings])[0][0]
                            all_matches.append((name, final_similarity, final_similarity, final_similarity))

                        if final_similarity > best_similarity:
                            best_similarity = final_similarity
                            best_match = name

                    if best_similarity > threshold and best_match:
                        result['recognized'] = True
                        result['name'] = best_match
                        result['confidence'] = best_similarity

                        print(f"✅ RECOGNIZED: {best_match} (confidence: {best_similarity:.3f})")
                        
                        # Enhanced debug output for tuning
                        sorted_matches = sorted(all_matches, key=lambda x: x[1], reverse=True)
                        print("📊 Top 3 matches:")
                        for i, (name, final_sim, max_sim, avg_sim) in enumerate(sorted_matches[:3]):
                            marker = "🎯" if name == best_match else "  "
                            print(f"  {marker} {name}: final={final_sim:.3f} (max={max_sim:.3f}, avg={avg_sim:.3f})")
                        
                        # Check if recognition is strong enough
                        if best_similarity > 0.25:
                            print(f"🟢 STRONG MATCH: {best_match} - High confidence recognition")
                        elif best_similarity > 0.20:
                            print(f"🟡 GOOD MATCH: {best_match} - Acceptable recognition")
                        else:
                            print(f"🟠 WEAK MATCH: {best_match} - Low confidence recognition")
                    else:
                        # print(f"❌ No match found (best: {best_similarity:.3f}, threshold: {threshold})")  # Commented untuk speed
                        # print("📊 All similarity scores (average):")  # Commented untuk speed
                        pass

        except Exception as e:
            print(f"❌ Error in face recognition: {e}")

        # if face_hash and result['name'] != 'Unknown':
        #         'name': result['name'],
        #         'confidence': result['confidence'],
        #         'antispoofing_result': result['antispoofing_result'],
        #         'timestamp': current_time
        #                        key=lambda k: self.recognition_cache[k]['timestamp'])

        return result['name'], result['confidence'], result['antispoofing_result']

    def add_person_to_database(self, name, face_images):
        """Add person to face database"""
        if self.arcface:
            return self.arcface.add_person(name, face_images)
        return False

    def get_embedding(self, face_image):
        """Extract embedding from face image"""
        if self.arcface:
            return self.arcface.get_face_embedding(face_image)
        return None

    def list_people(self):
        """List all people in database"""
        if self.arcface:
            return self.arcface.list_people()
        return []

    def remove_person(self, name):
        """Remove person from database"""
        if self.arcface:
            return self.arcface.remove_person(name)
        return False