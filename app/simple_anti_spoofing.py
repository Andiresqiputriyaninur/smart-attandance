import cv2
import numpy as np

class SimpleAntiSpoofing:
    """
    SIMPLIFIED Anti-Spoofing - Prioritas ArcFace Recognition!
    Hanya deteksi paper/foto yang SANGAT JELAS saja
    """
    def __init__(self):
        print("🛡️ SIMPLE AntiSpoofing initialized - PRIORITAS ARCFACE!")

    def detect_fake(self, face_image):
        """
        SIMPLIFIED detection - hanya blokir paper/foto yang SANGAT jelas
        
        Args:
            face_image: numpy array dari cropped face

        Returns:
            tuple: (is_fake: bool, confidence: float, reason: str)
        """
        try:
            if face_image is None or face_image.size == 0:
                return False, 0.0, "Invalid image"

            # Convert ke grayscale untuk analisis
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
            else:
                gray = face_image
                hsv = None

            paper_indicators = 0
            reasons = []
            
            # 1. Deteksi kertas putih yang sangat jelas
            white_pixels = np.sum(gray > 230) / gray.size
            if white_pixels > 0.45:  # Threshold tinggi - hanya kertas sangat putih
                paper_indicators += 3
                reasons.append(f"Kertas putih terdeteksi ({white_pixels:.2f})")

            # 2. Deteksi saturasi sangat rendah (kertas B&W)
            if hsv is not None:
                saturation = hsv[:,:,1]
                avg_saturation = np.mean(saturation)
                if avg_saturation < 8:  # Hanya yang SANGAT rendah
                    paper_indicators += 2
                    reasons.append(f"Saturasi sangat rendah ({avg_saturation:.1f})")

            # 3. Deteksi tekstur datar (kertas)
            laplacian_var = np.var(cv2.Laplacian(gray, cv2.CV_64F))
            if laplacian_var < 25:  # Threshold tinggi - hanya surface sangat datar
                paper_indicators += 2
                reasons.append(f"Surface sangat datar ({laplacian_var:.1f})")
            
            # 4. Deteksi edge pattern artificial
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            if edge_density < 0.02:  # Sangat sedikit edge = foto smooth
                paper_indicators += 1
                reasons.append(f"Edge pattern artificial ({edge_density:.3f})")

            # DECISION: Hanya blokir jika ada banyak indikator kertas yang JELAS
            is_fake = paper_indicators >= 4  # Threshold tinggi - butuh bukti kuat
            confidence = min(paper_indicators / 6.0, 1.0)  # Normalize
            
            if is_fake:
                main_reason = f"🚫 KERTAS DETECTED! Score: {paper_indicators}/6"
                print(f"🚨 PAPER DETECTED: {main_reason}")
                print(f"   📋 Reasons: {', '.join(reasons)}")
            else:
                main_reason = f"✅ REAL FACE - Score: {paper_indicators}/6 (LANJUT KE ARCFACE)"
                print(f"✅ REAL FACE: {main_reason}")

            return is_fake, confidence, main_reason

        except Exception as e:
            print(f"❌ Anti-spoofing error: {e}")
            return False, 0.0, f"Detection failed: {e}"

def detect_fake(frame, face_coords):
    """Legacy function untuk backward compatibility"""
    try:
        if frame is None or face_coords is None:
            return False
            
        x, y, w, h = face_coords
        face_roi = frame[y:y+h, x:x+w]
        
        anti_spoof = SimpleAntiSpoofing()
        is_fake, confidence, reason = anti_spoof.detect_fake(face_roi)
        
        return is_fake
        
    except Exception as e:
        print(f"❌ Legacy detect_fake error: {e}")
        return False

if __name__ == "__main__":
    anti_spoof = SimpleAntiSpoofing()
    print("✅ SIMPLE Anti-spoofing ready! PRIORITAS ARCFACE!")
