#!/usr/bin/env python3
"""
Libcamera + OpenCV + Gerçek Zamanlı Görüntü Sistemi
ArUco DICT_4X4_50 - 3D Pozisyon Tespiti
"""

import cv2
import numpy as np
import time
import subprocess
import threading
import queue
import os
import signal
import pickle

class RealtimeCameraViewer:
    def __init__(self, target_marker_id=25):
        """Gerçek zamanlı kamera görüntü sistemi - DICT_4X4_50"""
        print("🍓 Gerçek Zamanlı Kamera + ArUco DICT_4X4_50 Sistemi")
        
        # Hedef marker ID'si
        self.target_marker_id = target_marker_id
        if self.target_marker_id >= 50:
            print(f"⚠️  DICT_4X4_50 için maksimum ID: 49")
            self.target_marker_id = 25
        
        print(f"🎯 Hedef Marker ID: {self.target_marker_id}")
        print("📋 Sadece bu marker tanınacak, diğerleri görmezden gelinecek")
        
        # ArUco setup - DICT_4X4_50
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector_params = cv2.aruco.DetectorParameters()
        
        print("📋 ArUco Sözlük: DICT_4X4_50 (4x4 bit, 50 marker)")
        print("📋 Marker ID Aralığı: 0-49")
        
        # Daha hassas tespit için parametreler
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 23
        self.detector_params.adaptiveThreshWinSizeStep = 10
        self.detector_params.minMarkerPerimeterRate = 0.03
        self.detector_params.maxMarkerPerimeterRate = 4.0
        
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        
        # ArUco marker boyutu (metre cinsinden - gerçek boyutu)
        self.marker_size = 0.05  # 5cm marker boyutu
        
        # 3D Pose estimation için kamera kalibrasyonu
        self.setup_camera_calibration()
        
        # Stream variables
        self.process = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=5)
        
        print("✅ Sistem hazır!")
    
    def setup_camera_calibration(self):
        """Kamera kalibrasyonu parametrelerini ayarla"""
        # Önce kaydedilmiş kalibrasyonu yüklemeyi dene
        if os.path.exists('camera_calibration.pkl'):
            print("🔄 Kaydedilmiş kalibrasyon yükleniyor...")
            try:
                with open('camera_calibration.pkl', 'rb') as f:
                    calibration_data = pickle.load(f)
                
                self.camera_matrix = calibration_data['camera_matrix']
                self.dist_coeffs = calibration_data['dist_coeffs']
                
                print("✅ Kişisel kalibrasyon yüklendi!")
                print(f"📐 Kamera Matrisi:")
                print(f"   fx={self.camera_matrix[0,0]:.1f}, fy={self.camera_matrix[1,1]:.1f}")
                print(f"   cx={self.camera_matrix[0,2]:.1f}, cy={self.camera_matrix[1,2]:.1f}")
                print(f"🔍 Distorsiyon: {len(self.dist_coeffs)} katsayı")
                print(f"📏 Marker Boyutu: {self.marker_size*100}cm")
                return
                
            except Exception as e:
                print(f"⚠️  Kalibrasyon yüklenemedi: {e}")
                print("📐 Varsayılan değerler kullanılacak...")
        
        # Varsayılan kalibrasyou parametreleri (kalibrasyon dosyası yoksa)
        print("📐 Varsayılan kamera kalibrasyonu kullanılıyor...")
        
        # Kamera matrisi (intrinsic parameters)
        # fx, fy: focal length (pixel cinsinden)
        # cx, cy: principal point (görüntü merkezi)
        self.camera_matrix = np.array([
            [500.0, 0.0, 320.0],    # fx=500, cx=320 (640/2)
            [0.0, 500.0, 240.0],    # fy=500, cy=240 (480/2)
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Distorsiyon katsayıları (lens bozulması)
        self.dist_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0], dtype=np.float32)
        
        print("📐 Varsayılan kalibrasyon ayarlandı:")
        print(f"   Focal Length: fx=500, fy=500")
        print(f"   Principal Point: cx=320, cy=240")
        print(f"   Marker Boyutu: {self.marker_size*100}cm")
        print("   ⚠️  Daha hassas sonuç için kameranızı kalibre edin!")
        print("   💡 auto_camera_calibration.py çalıştırarak kişisel kalibrasyon yapın")
    
    def calibrate_camera_interactive(self):
        """Interaktif kamera kalibrasyonu (opsiyonel)"""
        print("\n🔧 Gelişmiş Kamera Kalibrasyonu")
        print("Bu fonksiyon satranç tahtası ile kamera kalibrasyonu yapar")
        print("Şu an basit varsayılan değerler kullanılıyor")
        
        # Buraya daha gelişmiş kalibrasyon kodu eklenebilir
        pass
    
    def update_marker_size(self, size_in_meters):
        """Marker boyutunu güncelle (metre cinsinden)"""
        self.marker_size = size_in_meters
        print(f"📏 Marker boyutu güncellendi: {size_in_meters*100}cm")
    
    def estimate_pose(self, corners):
        """ArUco marker'ın 3D pozisyonunu ve oryantasyonunu hesapla"""
        if corners is None or len(corners) == 0:
            return None, None
        
        # Pose estimation
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, self.camera_matrix, self.dist_coeffs
        )
        
        return rvecs, tvecs
    
    def rotation_vector_to_euler(self, rvec):
        """Rotation vector'ı Euler açılarına çevir (derece cinsinden)"""
        # Rotation matrix'e çevir
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Euler açılarını hesapla (X-Y-Z sırası)
        sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = 0
        
        # Radyandan dereceye çevir
        return np.degrees([x, y, z])
    
    def draw_3d_axis(self, frame, rvec, tvec, camera_matrix, dist_coeffs, length=0.03):
        """3D eksenleri çiz (X=Kırmızı, Y=Yeşil, Z=Mavi)"""
        # 3D nokta tanımla (eksenlerin uç noktaları)
        axis_points = np.array([
            [0, 0, 0],           # Orijin
            [length, 0, 0],      # X ekseni (Kırmızı)
            [0, length, 0],      # Y ekseni (Yeşil) 
            [0, 0, -length]      # Z ekseni (Mavi) - negatif çünkü kamera koordinat sistemi
        ], dtype=np.float32)
        
        # 3D noktaları 2D'ye projekte et
        axis_2d, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
        axis_2d = axis_2d.astype(int)
        
        # Orijin noktası
        origin = tuple(axis_2d[0].ravel())
        x_point = tuple(axis_2d[1].ravel())
        y_point = tuple(axis_2d[2].ravel())
        z_point = tuple(axis_2d[3].ravel())
        
        # Eksenleri çiz
        cv2.line(frame, origin, x_point, (0, 0, 255), 3)    # X ekseni - Kırmızı
        cv2.line(frame, origin, y_point, (0, 255, 0), 3)    # Y ekseni - Yeşil
        cv2.line(frame, origin, z_point, (255, 0, 0), 3)    # Z ekseni - Mavi
        
        # Eksen etiketleri
        cv2.putText(frame, 'X', x_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, 'Y', y_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, 'Z', z_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return frame
    
    def create_marker(self, marker_id=None):
        """DICT_4X4_50 test marker oluştur"""
        # Hedef marker ID'sini kullan
        if marker_id is None:
            marker_id = self.target_marker_id
            
        print(f"\n🎯 DICT_4X4_50 Marker Oluşturuluyor (ID: {marker_id}):")
        
        # ID kontrolü (DICT_4X4_50 için 0-49 arası)
        if marker_id >= 50:
            print(f"⚠️  DICT_4X4_50 için maksimum ID: 49")
            marker_id = 25
            print(f"📋 ID {marker_id} kullanılacak")
        
        marker = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, 200)
        bordered = cv2.copyMakeBorder(marker, 50, 50, 50, 50, 
                                    cv2.BORDER_CONSTANT, value=255)
        
        filename = f'target_marker_id_{marker_id}.png'
        cv2.imwrite(filename, bordered)
        
        print(f"   ✅ Kaydedildi: {filename}")
        print(f"   🎯 Bu marker sistem tarafından tanınacak")
        return filename
    
    def start_camera_stream(self, width=640, height=480, fps=30):
        """Libcamera stream başlat"""
        print(f"\n📹 Kamera Stream Başlatılıyor:")
        print(f"   Çözünürlük: {width}x{height}")
        print(f"   FPS: {fps}")
        
        # Geçici pipe file
        pipe_path = "/tmp/camera_viewer_pipe"
        
        # Eski pipe'ı temizle
        if os.path.exists(pipe_path):
            os.remove(pipe_path)
        
        # FIFO oluştur
        os.mkfifo(pipe_path)
        
        # Libcamera komutu
        cmd = [
            'libcamera-vid',
            '--width', str(width),
            '--height', str(height),
            '--framerate', str(fps),
            '--timeout', '0',
            '--nopreview',
            '--codec', 'yuv420',
            '--output', pipe_path,
            '--flush'
        ]
        
        try:
            print("   🔄 Libcamera başlatılıyor...")
            self.process = subprocess.Popen(cmd, 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE,
                                          preexec_fn=os.setsid)
            
            time.sleep(2)
            
            if self.process.poll() is None:
                print("   ✅ Kamera çalışıyor!")
                return pipe_path
            else:
                print("   ❌ Kamera başlatılamadı!")
                return None
                
        except Exception as e:
            print(f"   ❌ Kamera hatası: {e}")
            return None
    
    def read_frames_thread(self, pipe_path, width, height):
        """Frame okuma thread'i"""
        print("📸 Frame okuma başladı")
        
        frame_size = width * height * 3 // 2  # YUV420
        
        try:
            with open(pipe_path, 'rb') as pipe:
                while self.running:
                    try:
                        data = pipe.read(frame_size)
                        
                        if len(data) != frame_size:
                            continue
                        
                        # YUV'yi BGR'ye çevir
                        yuv_array = np.frombuffer(data, dtype=np.uint8)
                        yuv_frame = yuv_array.reshape((height * 3 // 2, width))
                        bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
                        
                        # Queue'ya ekle (son frame'i tut)
                        if not self.frame_queue.full():
                            self.frame_queue.put(bgr_frame)
                        else:
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                            self.frame_queue.put(bgr_frame)
                    
                    except Exception as e:
                        if self.running:
                            print(f"   ⚠️  Frame hatası: {e}")
                        break
        
        except Exception as e:
            print(f"   ❌ Pipe hatası: {e}")
        
        finally:
            if os.path.exists(pipe_path):
                os.remove(pipe_path)
    
    def show_camera_with_detection(self):
        """Kamera görüntüsü ile birlikte ArUco tespiti"""
        print(f"\n📺 Gerçek Zamanlı Kamera Görüntüsü:")
        print("   📋 ESC tuşu ile çıkış")
        print("   📋 S tuşu ile screenshot")
        print("   📋 SPACE tuşu ile marker kaydet")
        
        # Stream başlat
        pipe_path = self.start_camera_stream()
        if not pipe_path:
            print("❌ Kamera başlatılamadı!")
            return
        
        # Frame okuma thread'ini başlat
        self.running = True
        read_thread = threading.Thread(
            target=self.read_frames_thread, 
            args=(pipe_path, 640, 480)
        )
        read_thread.daemon = True
        read_thread.start()
        
        # OpenCV penceresi oluştur
        cv2.namedWindow('ArUco Kamera', cv2.WINDOW_AUTOSIZE)
        
        # İstatistikler
        start_time = time.time()
        frame_count = 0
        detection_count = 0
        last_detection_time = 0
        
        print("\n🚀 Kamera görüntüsü başladı!")
        print("   🎯 Markeri kameraya gösterin!")
        
        try:
            while True:
                try:
                    # Frame al
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    frame_count += 1
                    current_time = time.time()
                    
                    # Frame kopyası (orijinali bozmamak için)
                    display_frame = frame.copy()
                    
                    # ArUco tespit
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    corners, ids, rejected = self.detector.detectMarkers(gray)
                    
                    # Sadece hedef marker'ı filtrele
                    if ids is not None:
                        # Hedef ID'yi ara
                        target_indices = []
                        for i, marker_id in enumerate(ids.flatten()):
                            if marker_id == self.target_marker_id:
                                target_indices.append(i)
                        
                        # Sadece hedef marker varsa işle
                        if target_indices:
                            # Sadece hedef marker'ın verilerini al
                            filtered_corners = [corners[i] for i in target_indices]
                            filtered_ids = np.array([[self.target_marker_id]])
                            
                            detection_count += 1
                            last_detection_time = current_time
                            
                            # Marker'ı çiz
                            cv2.aruco.drawDetectedMarkers(display_frame, filtered_corners, filtered_ids)
                            
                            # 3D Pose estimation
                            rvecs, tvecs = self.estimate_pose(filtered_corners)
                            
                            # Hedef marker için bilgi
                            for i, corner_set in enumerate(filtered_corners):
                                marker_id = self.target_marker_id
                                
                                # Merkez hesapla
                                center = corner_set[0].mean(axis=0).astype(int)
                                
                                # 3D pozisyon bilgileri varsa
                                if rvecs is not None and tvecs is not None:
                                    rvec = rvecs[i][0]
                                    tvec = tvecs[i][0]
                                    
                                    # 3D eksenleri çiz
                                    self.draw_3d_axis(display_frame, rvec, tvec, 
                                                    self.camera_matrix, self.dist_coeffs)
                                    
                                    # Pozisyon bilgileri (metre cinsinden)
                                    x, y, z = tvec
                                    
                                    # Oryantasyon bilgileri (Euler açıları - derece)
                                    euler_angles = self.rotation_vector_to_euler(rvec)
                                    roll, pitch, yaw = euler_angles
                                    
                                    # Pozisyon yazısı (cm cinsinden göster)
                                    pos_text = f"HEDEF ID:{marker_id} X:{x*100:.1f}cm Y:{y*100:.1f}cm Z:{z*100:.1f}cm"
                                    cv2.putText(display_frame, pos_text, 
                                              (center[0]-100, center[1]-40), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                    
                                    # Oryantasyon yazısı (derece cinsinden)
                                    rot_text = f"Roll:{roll:.1f}° Pitch:{pitch:.1f}° Yaw:{yaw:.1f}°"
                                    cv2.putText(display_frame, rot_text, 
                                              (center[0]-100, center[1]-25), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                                    
                                    # Konsola detaylı bilgi yazdır
                                    print(f"🎯 HEDEF Marker {marker_id}:")
                                    print(f"   📍 Pozisyon: X={x:.3f}m, Y={y:.3f}m, Z={z:.3f}m")
                                    print(f"   🔄 Rotasyon: Roll={roll:.1f}°, Pitch={pitch:.1f}°, Yaw={yaw:.1f}°")
                                    print(f"   📏 Mesafe: {np.linalg.norm(tvec):.3f}m")
                                
                                # Hedef marker vurgusu
                                cv2.putText(display_frame, f"🎯 HEDEF: {marker_id}", 
                                          (center[0]-40, center[1]-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)
                                
                                # Özel hedef marker çerçevesi
                                cv2.circle(display_frame, tuple(center), 8, (0, 255, 0), -1)
                                cv2.circle(display_frame, tuple(center), 15, (0, 255, 0), 3)
                                
                            # Hedef tespit mesajı
                            cv2.putText(display_frame, f"🎯 HEDEF BULUNDU: ID {self.target_marker_id}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                        else:
                            # Hedef marker yok, diğer marker'lar varsa görmezden gel
                            other_ids = [id for id in ids.flatten() if id != self.target_marker_id]
                            if other_ids:
                                cv2.putText(display_frame, f"❌ DİĞER MARKER'LAR: {other_ids} (GÖRMEZDİM GELİNDİ)", 
                                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        # Hiç marker tespit edilmedi
                        cv2.putText(display_frame, f"🔍 HEDEF MARKER ARAMASINDA... (ID: {self.target_marker_id})", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # FPS ve istatistikler
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    # Bilgi metinleri
                    info_text = [
                        f"FPS: {fps:.1f}",
                        f"Frame: {frame_count}",
                        f"Tespit: {detection_count}",
                        f"Süre: {elapsed:.1f}s"
                    ]
                    
                    # Bilgileri ekranda göster
                    for i, text in enumerate(info_text):
                        y_pos = display_frame.shape[0] - 100 + (i * 25)
                        cv2.putText(display_frame, text, (10, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(display_frame, text, (10, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    
                    # Son tespit zamanı
                    if detection_count > 0:
                        time_since_detection = current_time - last_detection_time
                        if time_since_detection < 2:  # 2 saniye içinde
                            cv2.putText(display_frame, "✅ HEDEF MARKER GÖRÜLÜYOR", 
                                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Hedef marker bilgisi (sürekli göster)
                    cv2.putText(display_frame, f"Hedef ID: {self.target_marker_id}", 
                              (10, display_frame.shape[0] - 130), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Hedef ID: {self.target_marker_id}", 
                              (10, display_frame.shape[0] - 130), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    
                    # Çerçeve göster
                    cv2.imshow('ArUco Kamera', display_frame)
                    
                    # Klavye kontrolü
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == 27:  # ESC
                        print("\n   ⏹️  ESC ile çıkış!")
                        break
                    elif key == ord('s') or key == ord('S'):  # Screenshot
                        screenshot_path = f'screenshot_{int(current_time)}.jpg'
                        cv2.imwrite(screenshot_path, display_frame)
                        print(f"   📸 Screenshot: {screenshot_path}")
                    elif key == ord(' '):  # SPACE - marker kaydet
                        if ids is not None:
                            marker_path = f'detected_marker_{int(current_time)}.jpg'
                            cv2.imwrite(marker_path, display_frame)
                            print(f"   💾 Marker kaydedildi: {marker_path}")
                
                except queue.Empty:
                    # Frame yoksa boş frame göster
                    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(empty_frame, "Kamera bekleniyor...", 
                              (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow('ArUco Kamera', empty_frame)
                    
                    key = cv2.waitKey(30) & 0xFF
                    if key == 27:
                        break
                    
                except KeyboardInterrupt:
                    print("\n   ⏹️  Program durduruldu!")
                    break
        
        except Exception as e:
            print(f"   ❌ Görüntü hatası: {e}")
        
        finally:
            # Temizlik
            cv2.destroyAllWindows()
            self.stop_stream()
            
            # Final stats
            total_time = time.time() - start_time
            print(f"\n📊 Final Sonuçlar:")
            print(f"   Süre: {total_time:.1f}s")
            print(f"   Frame: {frame_count}")
            print(f"   Tespit: {detection_count}")
            if total_time > 0:
                print(f"   FPS: {frame_count/total_time:.1f}")
            if frame_count > 0:
                print(f"   Tespit Oranı: {detection_count/frame_count:.2%}")
    
    def stop_stream(self):
        """Stream'i durdur"""
        print("\n⏹️  Kamera durduruluyor...")
        
        self.running = False
        
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=3)
                print("   ✅ Kamera durduruldu")
            except:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass
        
        # Queue temizle
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break


def main():
    print("🍓 Gerçek Zamanlı Kamera + ArUco DICT_4X4_50 3D Sistemi")
    print("=" * 50)
    
    # Hedef marker ID'sini kullanıcıdan al
    print("🎯 Hedef Marker ID Seçimi:")
    print("📋 DICT_4X4_50 için geçerli ID'ler: 0-49")
    
    try:
        target_id = int(input("Hedef Marker ID'sini girin (varsayılan: 25): ") or "25")
        if target_id < 0 or target_id >= 50:
            print(f"⚠️  Geçersiz ID! Varsayılan ID 25 kullanılacak")
            target_id = 25
    except ValueError:
        print(f"⚠️  Geçersiz girdi! Varsayılan ID 25 kullanılacak")
        target_id = 25
    
    system = RealtimeCameraViewer(target_marker_id=target_id)
    
    # Hedef marker oluştur
    print("\n1️⃣ Hedef Marker:")
    marker_file = system.create_marker()
    
    print("\n2️⃣ 3D Pozisyon Sistemi:")
    print("📐 Kamera kalibrasyonu: Aktif")
    print("📏 Marker boyutu: 5cm")
    print("🎯 X, Y, Z eksenleri görüntülenecek")
    print("📊 Pozisyon ve rotasyon bilgileri görünecek")
    print(f"🎯 Sadece ID {target_id} tanınacak, diğerleri görmezden gelinecek")
    
    print("\n3️⃣ Hazırlık:")
    print(f"📋 {marker_file} dosyasını yazdırın")
    print("📋 Marker boyutunu 5cm olarak ayarlayın")
    print("📋 Kamera penceresi açılacak")
    print("📋 SADECE hedef marker'ı kameraya gösterin")
    print("⏰ 3 saniye sonra başlıyor...")
    
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    try:
        # Kamera görüntüsü ile tespit
        system.show_camera_with_detection()
    except KeyboardInterrupt:
        print("\n⏹️  Program durduruldu!")
    finally:
        system.stop_stream()
    
    print(f"\n🎉 DICT_4X4_50 Hedef ID {target_id} sistemi tamamlandı!")


if __name__ == "__main__":
    main()
