#!/usr/bin/env python3
"""
Enhanced Anti-Spoofing System for Smart Attendance
Focus: Ultra-aggressive B&W detection dengan proteksi wajah asli
"""

import cv2
import numpy as np

class AntiSpoofing:
    def __init__(self):
        print("🔒 Enhanced Anti-Spoofing System initialized")
        print("   🎯 Focus: Ultra-aggressive B&W detection")
        print("   🛡️  Protection: Real face preservation")
    
    def detect_fake(self, face_image):
        """
        Deteksi foto fake dengan fokus utama pada B&W detection
        """
        try:
            if face_image is None or face_image.size == 0:
                return False, 0.0, "Invalid image"

            # Convert ke berbagai color space
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
            else:
                gray = face_image
                hsv = None
                lab = None

            fake_score = 0
            reasons = []
            
            # SUPER PRIORITY: INSTANT B&W DETECTION - PALING ATAS
            if hsv is not None and len(face_image.shape) == 3:
                saturation = hsv[:,:,1]
                b, g, r = cv2.split(face_image)
                
                # PROTEKSI WAJAH BERWARNA: Cek dulu apakah ini gambar berwarna yang valid
                avg_sat = np.mean(saturation)
                sat_std = np.std(saturation)
                
                # EMERGENCY COLOR PROTECTION FIRST - paling prioritas
                if avg_sat > 60 and sat_std > 30:  # Clearly color image
                    print(f"🛡️ EMERGENCY: Obviously color image (sat:{avg_sat:.1f}, std:{sat_std:.1f}) - FORCE REAL")
                    return False, 0.0, "✅ WAJAH ASLI - Obviously color image"
                
                # Jika saturasi tinggi dan ada variasi, ini kemungkinan wajah berwarna asli  
                if avg_sat > 25 and sat_std > 15:  # Color image threshold
                    print(f"🛡️ PROTEKSI: Detected color variation (sat: {avg_sat:.1f}, std: {sat_std:.1f}) - NOT B&W")
                    # Skip instant B&W detection untuk gambar berwarna yang baik
                else:
                    # INSTANT B&W DETECTION - jika kondisi ini terpenuhi, langsung fake!
                    
                    # Check 1: Zero/near-zero saturation dominance (MORE AGGRESSIVE)
                    zero_sat_pixels = np.sum(saturation == 0) / saturation.size
                    near_zero_sat_pixels = np.sum(saturation < 1) / saturation.size
                    ultra_low_sat_pixels = np.sum(saturation < 2) / saturation.size
                    very_low_sat_pixels = np.sum(saturation < 3) / saturation.size
                    
                    if zero_sat_pixels > 0.6:  # 60% pixel saturasi nol - more aggressive
                        return True, 1.0, "🚫 INSTANT B&W: 60% pixel zero saturation!"
                    
                    if near_zero_sat_pixels > 0.8:  # 80% pixel hampir nol saturasi - more aggressive
                        return True, 1.0, "🚫 INSTANT B&W: 80% pixel near-zero saturation!"
                    
                    if ultra_low_sat_pixels > 0.85:  # 85% pixel < 2 saturasi
                        return True, 1.0, f"🚫 INSTANT B&W: 85% pixel ultra-low saturation ({ultra_low_sat_pixels:.3f})!"
                    
                    if very_low_sat_pixels > 0.9:  # 90% pixel < 3 saturasi
                        return True, 1.0, f"🚫 INSTANT B&W: 90% pixel very-low saturation ({very_low_sat_pixels:.3f})!"
                    
                    # Check 2: RGB perfect correlation + low saturation (MORE AGGRESSIVE)
                    corr_rg = np.corrcoef(r.flatten(), g.flatten())[0,1]
                    corr_rb = np.corrcoef(r.flatten(), b.flatten())[0,1]
                    corr_gb = np.corrcoef(g.flatten(), b.flatten())[0,1]
                    avg_corr = (corr_rg + corr_rb + corr_gb) / 3
                    min_corr = min(corr_rg, corr_rb, corr_gb)
                    
                    if avg_corr > 0.95 and avg_sat < 12:  # More aggressive correlation threshold
                        return True, 1.0, f"🚫 INSTANT B&W: High RGB correlation ({avg_corr:.3f}) + low saturation ({avg_sat:.1f})!"
                    
                    if min_corr > 0.97 and avg_sat < 15:  # All correlations ultra high
                        return True, 1.0, f"🚫 INSTANT B&W: All RGB correlations ultra high ({min_corr:.3f}) + low saturation ({avg_sat:.1f})!"
                    
                    # Check 3: RGB means almost identical + low saturation (MORE AGGRESSIVE)
                    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
                    max_mean_diff = max(abs(mean_r-mean_g), abs(mean_r-mean_b), abs(mean_g-mean_b))
                    avg_mean_diff = (abs(mean_r-mean_g) + abs(mean_r-mean_b) + abs(mean_g-mean_b)) / 3
                    
                    if max_mean_diff < 3 and avg_sat < 12:  # More aggressive threshold
                        return True, 1.0, f"🚫 INSTANT B&W: RGB means nearly identical ({max_mean_diff:.1f}) + low saturation ({avg_sat:.1f})!"
                    
                    if avg_mean_diff < 2 and avg_sat < 8:  # Average difference sangat kecil
                        return True, 1.0, f"🚫 INSTANT B&W: RGB means avg diff tiny ({avg_mean_diff:.1f}) + very low saturation ({avg_sat:.1f})!"
                    
                    # Check 4: LAB a,b channels ultra neutral (MORE AGGRESSIVE)
                    if lab is not None:
                        a_channel = lab[:,:,1]
                        b_channel = lab[:,:,2]
                        a_neutral = np.sum(np.abs(a_channel - 128) < 4) / a_channel.size  # More aggressive
                        b_neutral = np.sum(np.abs(b_channel - 128) < 4) / b_channel.size
                        a_ultra_neutral = np.sum(np.abs(a_channel - 128) < 2) / a_channel.size
                        b_ultra_neutral = np.sum(np.abs(b_channel - 128) < 2) / b_channel.size
                        
                        if a_ultra_neutral > 0.75 and b_ultra_neutral > 0.75:  # More aggressive
                            return True, 1.0, f"🚫 INSTANT B&W: LAB ultra perfect neutral! (a:{a_ultra_neutral:.3f}, b:{b_ultra_neutral:.3f})"
                        
                        if a_neutral > 0.85 and b_neutral > 0.85:  # More aggressive
                            return True, 1.0, f"🚫 INSTANT B&W: LAB very neutral! (a:{a_neutral:.3f}, b:{b_neutral:.3f})"
                    
                    # Check 5: Saturasi standard deviation ultra rendah (MORE AGGRESSIVE)
                    if sat_std < 5 and avg_sat < 15:  # More aggressive
                        return True, 1.0, f"🚫 INSTANT B&W: Saturation STD ultra low ({sat_std:.1f}) + low avg ({avg_sat:.1f})!"
                    
                    if sat_std < 3 and avg_sat < 18:  # STD sangat rendah
                        return True, 1.0, f"🚫 INSTANT B&W: Saturation STD extremely low ({sat_std:.1f}) + moderate avg ({avg_sat:.1f})!"
                    
                    # Check 6: RGB channel standard deviation similarity (MORE AGGRESSIVE)
                    std_r, std_g, std_b = np.std(r), np.std(g), np.std(b)
                    std_max_diff = max(abs(std_r-std_g), abs(std_r-std_b), abs(std_g-std_b))
                    
                    if std_max_diff < 2 and avg_sat < 12:  # More aggressive
                        return True, 1.0, f"🚫 INSTANT B&W: RGB STD nearly identical ({std_max_diff:.1f}) + low saturation ({avg_sat:.1f})!"
                    
                    # Check 7: Histogram analysis for B&W detection (MORE AGGRESSIVE)
                    gray_temp = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    hist = cv2.calcHist([gray_temp], [0], None, [256], [0, 256]).flatten()
                    # Cek distribusi intensitas yang sangat concentrated di area gray
                    gray_region = np.sum(hist[60:180])  # Area gray
                    total_hist = np.sum(hist)
                    gray_dominance = gray_region / (total_hist + 1e-6)
                    
                    if gray_dominance > 0.8 and avg_sat < 10:  # More aggressive
                        return True, 1.0, f"🚫 INSTANT B&W: Gray dominance ({gray_dominance:.3f}) + low saturation ({avg_sat:.1f})!"
            
            # ENHANCED B&W DETECTION WITH SCORING SYSTEM
            if hsv is not None:
                saturation = hsv[:,:,1]
                avg_saturation = np.mean(saturation)
                very_low_sat = np.sum(saturation < 6) / saturation.size
                extremely_low_sat = np.sum(saturation < 3) / saturation.size
                ultra_low_sat = np.sum(saturation < 1) / saturation.size
                
                # B&W DETECTION dengan score tinggi
                if avg_saturation < 3:
                    fake_score += 200  # Ultra high score untuk B&W pasti
                    reasons.append(f"🚫 B&W PASTI: Avg saturation ultra low ({avg_saturation:.1f})")
                elif avg_saturation < 5:
                    fake_score += 150
                    reasons.append(f"🚫 B&W SANGAT JELAS: Avg saturation very low ({avg_saturation:.1f})")
                elif avg_saturation < 8:
                    fake_score += 120
                    reasons.append(f"🚫 B&W JELAS: Avg saturation low ({avg_saturation:.1f})")
                elif avg_saturation < 12:
                    fake_score += 100
                    reasons.append(f"🚫 B&W TERDETEKSI: Avg saturation moderate ({avg_saturation:.1f})")
                
                # Additional saturation checks
                if ultra_low_sat > 0.9:
                    fake_score += 150
                    reasons.append(f"🚫 B&W MURNI: 90% pixels ultra-low saturation ({ultra_low_sat:.3f})")
                elif extremely_low_sat > 0.85:
                    fake_score += 120
                    reasons.append(f"🚫 B&W EKSTREM: 85% pixels extremely low saturation ({extremely_low_sat:.3f})")
                elif very_low_sat > 0.8:
                    fake_score += 100
                    reasons.append(f"🚫 B&W DETECTED: 80% pixels very low saturation ({very_low_sat:.3f})")
            
            # RGB CORRELATION ANALYSIS
            if len(face_image.shape) == 3:
                b, g, r = cv2.split(face_image)
                
                # Korelasi antar channel
                corr_rg = np.corrcoef(r.flatten(), g.flatten())[0,1]
                corr_rb = np.corrcoef(r.flatten(), b.flatten())[0,1]
                corr_gb = np.corrcoef(g.flatten(), b.flatten())[0,1]
                avg_corr = (corr_rg + corr_rb + corr_gb) / 3
                
                if avg_corr > 0.98:
                    fake_score += 150
                    reasons.append(f"🚫 RGB KORELASI PERFECT: B&W pasti ({avg_corr:.4f})")
                elif avg_corr > 0.95:
                    fake_score += 120
                    reasons.append(f"🚫 RGB KORELASI TINGGI: B&W sangat mungkin ({avg_corr:.4f})")
                elif avg_corr > 0.92:
                    fake_score += 100
                    reasons.append(f"🚫 RGB KORELASI SUSPECT: B&W mungkin ({avg_corr:.4f})")
                
                # Mean difference analysis
                mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
                max_mean_diff = max(abs(mean_r-mean_g), abs(mean_r-mean_b), abs(mean_g-mean_b))
                
                if max_mean_diff < 1:
                    fake_score += 150
                    reasons.append(f"🚫 RGB MEANS IDENTIK: B&W pasti ({max_mean_diff:.1f})")
                elif max_mean_diff < 3:
                    fake_score += 120
                    reasons.append(f"🚫 RGB MEANS SANGAT MIRIP: B&W mungkin ({max_mean_diff:.1f})")
                elif max_mean_diff < 5:
                    fake_score += 100
                    reasons.append(f"🚫 RGB MEANS MIRIP: B&W suspect ({max_mean_diff:.1f})")
            
            # LAB COLOR SPACE ANALYSIS
            if lab is not None:
                a_channel = lab[:,:,1]
                b_channel = lab[:,:,2]
                
                a_neutrality = np.sum(np.abs(a_channel - 128) < 5) / a_channel.size
                b_neutrality = np.sum(np.abs(b_channel - 128) < 5) / b_channel.size
                
                if a_neutrality > 0.9 and b_neutrality > 0.9:
                    fake_score += 150
                    reasons.append(f"🚫 LAB PERFECT NEUTRAL: B&W pasti (a:{a_neutrality:.3f}, b:{b_neutrality:.3f})")
                elif a_neutrality > 0.85 and b_neutrality > 0.85:
                    fake_score += 120
                    reasons.append(f"🚫 LAB SANGAT NEUTRAL: B&W mungkin (a:{a_neutrality:.3f}, b:{b_neutrality:.3f})")
                elif a_neutrality > 0.8 or b_neutrality > 0.8:
                    fake_score += 100
                    reasons.append(f"🚫 LAB NEUTRAL: B&W suspect (a:{a_neutrality:.3f}, b:{b_neutrality:.3f})")
            
            # HISTOGRAM ANALYSIS
            if gray is not None:
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
                
                # Gray area dominance
                gray_region = np.sum(hist[50:200])
                total_hist = np.sum(hist)
                gray_ratio = gray_region / (total_hist + 1e-6)
                
                if gray_ratio > 0.9:
                    fake_score += 120
                    reasons.append(f"🚫 GRAY DOMINANCE EKSTREM: B&W pasti ({gray_ratio:.3f})")
                elif gray_ratio > 0.85:
                    fake_score += 100
                    reasons.append(f"🚫 GRAY DOMINANCE TINGGI: B&W mungkin ({gray_ratio:.3f})")
                elif gray_ratio > 0.8:
                    fake_score += 80
                    reasons.append(f"🚫 GRAY DOMINANCE: B&W suspect ({gray_ratio:.3f})")
            
            # PROTEKSI WAJAH ASLI
            base_threshold = 50  # Threshold dasar
            
            if hsv is not None:
                saturation = hsv[:,:,1]
                avg_sat = np.mean(saturation)
                sat_std = np.std(saturation)
                
                # Proteksi untuk gambar berwarna yang baik
                if avg_sat > 40 and sat_std > 20:
                    fake_score = max(0, fake_score - 200)
                    base_threshold = 150
                    reasons.append(f"🛡️ PROTEKSI ULTIMATE: Excellent color detected")
                elif avg_sat > 30 and sat_std > 15:
                    fake_score = max(0, fake_score - 150)
                    base_threshold = 120
                    reasons.append(f"🛡️ PROTEKSI STRONG: Very good color detected")
                elif avg_sat > 25 and sat_std > 12:
                    fake_score = max(0, fake_score - 120)
                    base_threshold = 100
                    reasons.append(f"🛡️ PROTEKSI ENHANCED: Good color detected")
                elif avg_sat > 20 and sat_std > 8:
                    fake_score = max(0, fake_score - 80)
                    base_threshold = 80
                    reasons.append(f"🛡️ PROTEKSI MODERATE: Moderate color detected")
                
                # Emergency protection untuk gambar yang jelas berwarna
                if avg_sat > 50 and sat_std > 25:
                    print(f"🛡️ EMERGENCY PROTECTION: Clearly a color image!")
                    return False, 0.0, "✅ WAJAH ASLI - Color image protected"
            
            # FINAL DECISION
            print(f"🔍 ANTI-SPOOFING DEBUG:")
            print(f"   📊 Final Score: {fake_score:.1f}")
            print(f"   🎯 Threshold: {base_threshold}")
            print(f"   📋 Total Indicators: {len(reasons)}")
            
            # Ultra aggressive untuk score tinggi
            if fake_score >= 300:
                is_fake = True
                confidence = 1.0
                main_reason = f"🚫 B&W CONFIRMED: Score extremely high ({fake_score:.0f})"
            elif fake_score >= 200:
                is_fake = True
                confidence = 0.95
                main_reason = f"🚫 B&W VERY LIKELY: Score very high ({fake_score:.0f})"
            elif fake_score >= 150:
                is_fake = True
                confidence = 0.90
                main_reason = f"🚫 B&W LIKELY: Score high ({fake_score:.0f})"
            elif fake_score >= 100:
                is_fake = True
                confidence = 0.85
                main_reason = f"🚫 B&W POSSIBLE: Score moderate-high ({fake_score:.0f})"
            else:
                is_fake = fake_score >= base_threshold
                confidence = min(fake_score / 100.0, 1.0)
                if is_fake:
                    main_reason = f"🚫 FAKE DETECTED: Score {fake_score:.0f} >= {base_threshold}"
                else:
                    main_reason = f"✅ REAL FACE: Score {fake_score:.0f} < {base_threshold}"
            
            if is_fake:
                print(f"🚨 FAKE DETECTED: {main_reason}")
                if reasons:
                    print(f"   📋 Top reasons: {'; '.join(reasons[:3])}")
            else:
                print(f"✅ REAL FACE: {main_reason}")
                if reasons:
                    print(f"   📋 Minor indicators: {'; '.join(reasons[:2])}")

            return is_fake, confidence, main_reason

        except Exception as e:
            print(f"❌ Anti-spoofing error: {e}")
            return False, 0.0, f"Error: {str(e)}"

if __name__ == "__main__":
    anti_spoof = AntiSpoofing()
    print("✅ Enhanced Anti-spoofing module ready!")
    print("💡 Optimized for B&W photo detection")
