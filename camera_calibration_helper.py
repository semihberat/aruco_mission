#!/usr/bin/env python3
"""
Kamera Kalibrasyonu Yardımcısı
Daha hassas 3D pozisyon tespiti için kameranızı kalibre edin
"""

import cv2
import numpy as np
import glob
import pickle

class CameraCalibrationHelper:
    def __init__(self):
        """Kamera kalibrasyonu yardımcısı"""
        print("📐 Kamera Kalibrasyonu Yardımcısı")
        
        # Satranç tahtası ayarları
        self.chessboard_size = (9, 6)  # İç köşe sayısı
        self.square_size = 0.025  # Her karenin boyutu (2.5cm)
        
        # 3D nokta koordinatları (gerçek dünya)
        self.prepare_object_points()
        
    def prepare_object_points(self):
        """3D nesne noktalarını hazırla"""
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size
    
    def calibrate_from_images(self, image_folder="calibration_images/*.jpg"):
        """Fotoğraflardan kamera kalibrasyonu yap"""
        print(f"\n🔍 Kalibrasyon fotoğrafları aranıyor: {image_folder}")
        
        # Kalibrasyon için nokta listeleri
        objpoints = []  # 3D noktalar
        imgpoints = []  # 2D noktalar
        
        # Fotoğrafları yükle
        images = glob.glob(image_folder)
        
        if not images:
            print("❌ Kalibrasyon fotoğrafı bulunamadı!")
            print("📸 Önce satranç tahtası fotoğrafları çekin")
            return None, None
        
        print(f"📸 {len(images)} fotoğraf bulundu")
        
        valid_images = 0
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Satranç tahtası köşelerini bul
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            if ret:
                valid_images += 1
                print(f"   ✅ {fname} - Köşeler bulundu")
                
                objpoints.append(self.objp)
                
                # Alt-piksel hassasiyeti
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners2)
            else:
                print(f"   ❌ {fname} - Köşeler bulunamadı")
        
        if valid_images < 10:
            print(f"⚠️  Sadece {valid_images} geçerli fotoğraf! En az 10 gerekli")
            return None, None
        
        print(f"\n🔧 {valid_images} fotoğraf ile kalibrasyon yapılıyor...")
        
        # Kamera kalibrasyonu
        img_shape = gray.shape[::-1]
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )
        
        if ret:
            print("✅ Kalibrasyon başarılı!")
            print(f"📐 Kamera Matrisi:\n{camera_matrix}")
            print(f"🔍 Distorsiyon Katsayıları:\n{dist_coeffs}")
            
            # Kalibrasyonu kaydet
            calibration_data = {
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs,
                'rvecs': rvecs,
                'tvecs': tvecs,
                'image_shape': img_shape
            }
            
            with open('camera_calibration.pkl', 'wb') as f:
                pickle.dump(calibration_data, f)
            
            print("💾 Kalibrasyon kaydedildi: camera_calibration.pkl")
            return camera_matrix, dist_coeffs
        else:
            print("❌ Kalibrasyon başarısız!")
            return None, None
    
    def create_calibration_instructions(self):
        """Kalibrasyon talimatları"""
        print("\n📋 Kamera Kalibrasyonu Talimatları:")
        print("=" * 50)
        print("1️⃣ Satranç tahtası yazdırın (9x6 köşe)")
        print("2️⃣ Düz bir yüzeye yapıştırın")
        print("3️⃣ 15-20 farklı açıdan fotoğraf çekin:")
        print("   - Farklı mesafeler")
        print("   - Farklı açılar") 
        print("   - Görüntünün farklı bölgeleri")
        print("4️⃣ Fotoğrafları 'calibration_images/' klasörüne koyun")
        print("5️⃣ calibrate_from_images() fonksiyonunu çalıştırın")


def main():
    calibrator = CameraCalibrationHelper()
    
    print("🎯 Seçenekler:")
    print("1. Kalibrasyon talimatlarını göster")
    print("2. Mevcut fotoğraflardan kalibrasyon yap")
    
    choice = input("\nSeçiminiz (1-2): ")
    
    if choice == "1":
        calibrator.create_calibration_instructions()
    elif choice == "2":
        camera_matrix, dist_coeffs = calibrator.calibrate_from_images()
        if camera_matrix is not None:
            print("\n✅ Kalibrasyon tamamlandı!")
            print("📝 Bu değerleri realtime_camera_viewer.py dosyasında kullanın")
    else:
        print("❌ Geçersiz seçim!")


if __name__ == "__main__":
    main()
