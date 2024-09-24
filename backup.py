import os
import subprocess
import cv2
import easyocr
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter.filedialog  import askdirectory
from PIL import Image, ImageTk
import xml.etree.ElementTree as ET
import numpy as np

video_file = None
image_path = None
points = []
roi_points = []
fullscreen = False
roi_defined = False
output_dir_original = None
output_dir_transform = None
output_folder = None
screenshot_counter_original = 0
screenshot_counter_transform = 0

# Fungsi untuk berpindah halaman
def open_page(page):
    halaman_utama.pack_forget()
    tahap_1.pack_forget()
    tahap_2.pack_forget()
    tahap_3.pack_forget()
    tahap_4.pack_forget()
    tahap_5.pack_forget()
    page.pack(fill=tk.BOTH, expand=True)

# Fungsi untuk menampilkan pesan
def show_message(message):
    messagebox.showinfo("Informasi", message)

#Tahap 1 Pre-Processing
def process_images(input_folder, output_folder, new_size):
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i, filename in enumerate(os.listdir(input_folder)):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            img_path = os.path.join(input_folder, filename)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Cannot open file: {img_path}")
                    continue

                # Resize gambar
                img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
                
                # Mengubah Citra ke Grayscale
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                
                # Menyimpan Output Citra Hasil Preprocessing
                new_filename = f'image_{i+1:03d}.jpg'
                output_path = os.path.join(output_folder, new_filename)
                cv2.imwrite(output_path, img_gray)
                print(f'Processed and saved: {output_path}')
            except Exception as e:
                print(f"Error processing file {img_path}: {str(e)}")

def pre_processing():
    root = tk.Tk()  # Membuat objek root dari kelas Tk untuk aplikasi tkinter
    root.withdraw()  # Menyembunyikan jendela utama Tkinter

    input_folder = filedialog.askdirectory(title="Pilih Folder Input")
    if not input_folder:
        messagebox.showinfo("Info", "Tidak ada folder yang dipilih.")
        return
    
    output_folder = filedialog.askdirectory(title="Pilih Folder Output")
    if not output_folder:
        messagebox.showinfo("Info", "Tidak ada folder yang dipilih.")
        return

    try:
        # Input lebar dan tinggi citra baru untuk proses resize
        width = simpledialog.askinteger("Input", "Masukkan lebar baru:", minvalue=1)
        height = simpledialog.askinteger("Input", "Masukkan tinggi baru:", minvalue=1)
        if width is None or height is None:
            messagebox.showinfo("Info", "Ukuran tidak valid.")
            return
        new_size = (width, height)
    except Exception as e:
        messagebox.showinfo("Info", f"Input tidak valid: {str(e)}")
        return

    process_images(input_folder, output_folder, new_size)
    messagebox.showinfo("Info", "Proses resize dan konversi ke grayscale selesai.")

    root.destroy()

#Tahap 2 Training Haarcascade
# Anotasi Citra dengan Labelimg
# Fungsi untuk membuka LabelImg
def open_labelimg():
    labelimg_path = filedialog.askdirectory(title="Pilih Folder LabelImg")
    if labelimg_path:
        try:
            # Perintah untuk membuka LabelImg
            command = f'cd /d "{labelimg_path}" && python labelImg.py'
            subprocess.run(command, shell=True, check=True)
        except Exception as e:
            show_message(f"Error membuka LabelImg: {e}")

# Membuat file Info.txt
def buat_info_txt(xml_folder, output_txt_file):
    with open(output_txt_file, 'w') as f:
        for xml_file in os.listdir(xml_folder):
            if not xml_file.endswith(".xml"):
                continue
            xml_path = os.path.join(xml_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            filename = root.find('filename').text
            objects = root.findall('object')

            bboxes = []
            for obj in objects:
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                width = xmax - xmin
                height = ymax - ymin
                bboxes.append(f"{xmin} {ymin} {width} {height}")

            if bboxes:
                bbox_str = ' '.join(bboxes)
                f.write(f"mobil/{filename} {len(bboxes)} {bbox_str}\n")

    show_message(f"File '{output_txt_file}' berhasil dibuat!")

def info_main():
    root = tk.Tk()
    root.withdraw() 
    
    # Memilih folder yang berisi file XML
    xml_folder = filedialog.askdirectory(title="Pilih Folder yang Berisi File XML")
    if not xml_folder:
        print("Folder tidak dipilih. Program dihentikan.")
        return

    # Memilih folder tempat menyimpan file info.txt
    output_folder = filedialog.askdirectory(title="Pilih Folder Tempat Menyimpan File info.txt")
    if not output_folder:
        print("Folder tidak dipilih. Program dihentikan.")
        return

    # Nama file output
    output_txt_file = os.path.join(output_folder, "info.txt")

    # Konversi Pascal VOC XML ke Haarcascade TXT
    buat_info_txt(xml_folder, output_txt_file)

    print(f"Annotations have been converted to {output_txt_file}")

# Membuat List File Negatif
def create_negative_txt():
    # Fungsi untuk membuat file negatives.txt dari direktori yang dipilih
    neg_dir = filedialog.askdirectory(title="Select Negative Images Directory")
    if neg_dir:
        neg_list = os.path.join(neg_dir, 'bg.txt')
        try:
            with open(neg_list, 'w') as file:
                for filename in os.listdir(neg_dir):
                    if filename.endswith('.jpg') or filename.endswith('.png'):
                        file.write(f'{os.path.join(neg_dir, filename)}\n')
            messagebox.showinfo("Success", f"File info negatif berhasil dibuat dan disimpan di:\n{neg_list}")
        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat membuat file info negatif:\n{str(e)}")

#Membuat file vector
def buat_vector(info_file, vec_file, num, width, height):
    cmd = [
        'opencv_createsamples.exe',
        '-info', info_file,
        '-vec', vec_file,
        '-num', str(num),
        '-w', str(width),
        '-h', str(height)
    ]

    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        messagebox.showinfo("Sukses", "Berhasil Membuat File Vector")
        print("Output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Terjadi kesalahan saat membuat vector:\n{e.stderr}")
        print("Error output:")
        print(e.stderr)

def vector_main():
    # Inisialisasi Tkinter
    root = tk.Tk()
    root.withdraw()  # Sembunyikan jendela utama

    # Meminta pengguna untuk memilih file input
    info_file = filedialog.askopenfilename(title="Pilih File Input", filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")))

    if not info_file:
        messagebox.showerror("Error", "Tidak ada file input yang dipilih. Proses dibatalkan.")
        return

    # Meminta pengguna untuk memilih direktori output
    output_dir = filedialog.askdirectory(title="Pilih Direktori Output")

    if not output_dir:
        messagebox.showerror("Error", "Tidak ada direktori yang dipilih. Proses dibatalkan.")
        return

    vec_file = f'{output_dir}/positive.vec'
    num = 300
    width = 24
    height = 24

    buat_vector(info_file, vec_file, num, width, height)

# Mulai Pelatihan Haar Cascade
def train_cascade(data_dir, vec_file, bg_file, num_pos, num_neg, num_stages, val_buf_size, idx_buf_size, feature_type, width, height, min_hit_rate, max_false_alarm_rate, error_log_file):
    command = [
        'opencv_traincascade',
        '-data', data_dir,
        '-vec', vec_file,
        '-bg', bg_file,
        '-numPos', str(num_pos),
        '-numNeg', str(num_neg),
        '-numStages', str(num_stages),
        '-precalcValBufSize', str(val_buf_size),
        '-precalcIdxBufSize', str(idx_buf_size),
        '-featureType', feature_type,
        '-w', str(width),
        '-h', str(height),
        '-minHitRate', str(min_hit_rate),
        '-maxFalseAlarmRate', str(max_false_alarm_rate)
    ]

    with open(error_log_file, 'w') as err_file:
        subprocess.run(command, stderr=err_file, check=True)

def train_haarcascade():
    vector_file = filedialog.askopenfilename(title="Pilih file positives.vec", filetypes=[("VEC files", "*.vec")])
    if not vector_file:
        return
    
    negative_file = filedialog.askopenfilename(title="Pilih file negatives.txt", filetypes=[("TXT files", "*.txt")])
    if not negative_file:
        return

    classifier_output_dir = filedialog.askdirectory(title="Pilih direktori output classifier")
    if not classifier_output_dir:
        return
    
    width = 24
    height = 24
    num_stages = 10
    min_hit_rate = 0.99
    max_false_alarm_rate = 0.5
    num_pos = 250
    num_neg = 500
    feature_type = "HAAR"
    precalc_val_buf_size = 3072
    precalc_idx_buf_size = 3072
    error_log_file = "error_log.txt"
    
    print("Menjalankan pelatihan...")  # Logging perintah yang akan dijalankan
    
    try:
        train_cascade(
            data_dir=classifier_output_dir,
            vec_file=vector_file,
            bg_file=negative_file,
            num_pos=num_pos,
            num_neg=num_neg,
            num_stages=num_stages,
            val_buf_size=precalc_val_buf_size,
            idx_buf_size=precalc_idx_buf_size,
            feature_type=feature_type,
            width=width,
            height=height,
            min_hit_rate=min_hit_rate,
            max_false_alarm_rate=max_false_alarm_rate,
            error_log_file=error_log_file
        )
        messagebox.showinfo("Informasi", "Pelatihan classifier telah selesai.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        with open(error_log_file, 'r') as err_file:
            error_log = err_file.read()
        print(f"Stderr: {error_log}")
        messagebox.showerror("Error", f"Error: {e}\nStderr: {error_log}")

#Tahap 3 Memilih dan Memutar Video
# Mencari file video
def search_video():
    global video_file
    video_file = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")])
    if video_file:
        file_label3.config(text=f"File yang dipilih: {video_file}")
    else:
        video_file = None
        file_label3.config(text="File yang dipilih: ")

# Fungsi untuk memulai pemutaran video
def mulai_video():
    global video_file
    if video_file is None:
        messagebox.showwarning("Peringatan", "Silakan kembali ke tahap 1 untuk memilih video terlebih dahulu.")
        return
    else:
        putar_video()

# Fungsi untuk memutar video
def putar_video():
    global video_file
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        messagebox.showerror("Error", "Gagal membuka file video.")
        return
    
    target_width = 1920
    target_height = 1080
    
    window_width = 800
    window_height = 500
    
    window_name = 'Video Original'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)
    
    fullscreen = False
    
    while True:
        read, frame = cap.read()
        if read:
            # Mengubah ukuran frame video ke resolusi target
            frame_resized = cv2.resize(frame, (target_width, target_height))
            
            # Mendapatkan ukuran jendela
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                window_rect = cv2.getWindowImageRect(window_name)
                window_width, window_height = window_rect[2], window_rect[3]
            else:
                window_width, window_height = 640, 480
            
            # Membuat frame baru dengan padding hitam jika ukuran jendela lebih kecil
            top_padding = max(0, (window_height - target_height) // 2)
            bottom_padding = max(0, (window_height - target_height + 1) // 2)
            left_padding = max(0, (window_width - target_width) // 2)
            right_padding = max(0, (window_width - target_width + 1) // 2)
            
            padded_frame = cv2.copyMakeBorder(
                frame_resized, 
                top_padding, 
                bottom_padding, 
                left_padding, 
                right_padding, 
                cv2.BORDER_CONSTANT, 
                value=[0, 0, 0]
            )
            
            # Menampilkan frame yang diubah ukurannya dengan padding hitam
            cv2.imshow(window_name, padded_frame)
            
            # Mengatur delay untuk memastikan video diputar pada framerate yang diinginkan
            frame_rate = 60  # Framerate yang diinginkan 
            delay = int(1000 / frame_rate)  # Mengubah detik menjadi milidetik
            
            key = cv2.waitKey(delay) & 0xFF
            if key == 27:  # Tekan 'Esc' untuk keluar
                break
            elif key == ord('f'):  # Tekan 'f' untuk mengubah mode fullscreen
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Tahap 4 Pendeteksian Kendaraan
# Fungsi untuk deteksi kendaraan
def deteksi_kendaraan():
    global video_file, roi_points, roi_defined, fullscreen, output_dir_original, output_dir_transform

    if not video_file:
        messagebox.showerror("Error", "File video belum dipilih.")
        return

    if not output_dir_original or not output_dir_transform:
        messagebox.showerror("Error", "Direktori output belum dipilih.")
        return

    # Memuat classifier Haar Cascade untuk deteksi kendaraan
    car_cascade = cv2.CascadeClassifier('mobil.xml')
    motor_cascade = cv2.CascadeClassifier('motor.xml')

    # Membuka video
    cap = cv2.VideoCapture(video_file)

    # Memeriksa apakah video berhasil dibuka
    if not cap.isOpened():
        messagebox.showerror("Error", "Gagal membuka file video.")
        return

    # Resolusi target
    target_width = 1920
    target_height = 1080

    # Ukuran jendela awal
    window_width = 800
    window_height = 500

    # Ukuran jendela untuk tampilan "transform Video"
    transform_window_width = 800
    transform_window_height = 500

    # Membuat jendela
    cv2.namedWindow('Video Original', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Original', window_width, window_height)
    cv2.namedWindow('transform Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('transform Video', transform_window_width, transform_window_height)

    # Variabel untuk menyimpan gambar kendaraan terdeteksi
    screenshot_counter_original = 0
    screenshot_counter_transform = 0

    # Variabel untuk menghitung deteksi
    detection_counter = 0

    # Mendapatkan koordinat ROI
    roi_points = np.array(roi_points, dtype="float32")

    # Menentukan koordinat tujuan untuk transformasi perspektif
    (tl, tr, br, bl) = roi_points
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(roi_points, dst)
    M_inv = np.linalg.inv(M)

    # Memulai pemutaran video setelah ROI didefinisikan
    while True:
        read, frame = cap.read()
        if not read:
            print("Akhir video.")
            break

        # Mengubah ukuran frame ke resolusi target
        frame_resized = cv2.resize(frame, (target_width, target_height))

        # Menerapkan transformasi perspektif
        transform = cv2.warpPerspective(frame_resized, M, (maxWidth, maxHeight))

        # Deteksi kendaraan dan motor pada frame ROI
        if transform.size != 0:
            gray = cv2.cvtColor(transform, cv2.COLOR_BGR2GRAY)
            car = car_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=20, minSize=(380, 380))
            motors = motor_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=30, minSize=(180, 180))

            for (x, y, w, h) in car:
                # Menggambar kotak di sekitar kendaraan terdeteksi pada frame transform (ukuran asli)
                cv2.rectangle(transform, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Hanya menyimpan screenshot setiap 10 deteksi
                if detection_counter % 10 == 0:
                    # Menyimpan gambar kendaraan yang terdeteksi pada frame transform
                    vehicle_img_transform = transform[y:y + h, x:x + w]
                    screenshot_filename_transform = os.path.join(output_dir_transform, f"{screenshot_counter_transform}.png")
                    cv2.imwrite(screenshot_filename_transform, vehicle_img_transform, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    screenshot_counter_transform += 1

                    # Menghitung kembali koordinat kotak ke frame asli menggunakan M_inv
                    points = np.array([[(x, y)], [(x + w, y + h)]], dtype='float32')
                    points = points.reshape(-1, 1, 2)
                    original_points = cv2.perspectiveTransform(points, M_inv)
                    original_top_left = tuple(map(int, original_points[0][0]))
                    original_bottom_right = tuple(map(int, original_points[1][0]))

                    # Menggambar kotak pada frame asli
                    cv2.rectangle(frame_resized, original_top_left, original_bottom_right, (0, 255, 0), 2)

                    # Menyimpan gambar kendaraan yang terdeteksi pada frame asli
                    vehicle_img_original = frame_resized[original_top_left[1]:original_bottom_right[1], original_top_left[0]:original_bottom_right[0]]
                    screenshot_filename_original = os.path.join(output_dir_original, f"{screenshot_counter_original}.png")
                    cv2.imwrite(screenshot_filename_original, vehicle_img_original, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    screenshot_counter_original += 1

                detection_counter += 1

            for (x, y, w, h) in motors:
                # Menggambar kotak di sekitar motor terdeteksi pada frame transform (ukuran asli)
                cv2.rectangle(transform, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Hanya menyimpan screenshot setiap 10 deteksi
                if detection_counter % 10 == 0:
                    # Menyimpan gambar motor yang terdeteksi pada frame transform
                    motor_img_transform = transform[y:y + h, x:x + w]
                    screenshot_filename_transform = os.path.join(output_dir_transform, f"{screenshot_counter_transform}.png")
                    cv2.imwrite(screenshot_filename_transform, motor_img_transform, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    screenshot_counter_transform += 1

                    # Menghitung kembali koordinat kotak ke frame asli menggunakan M_inv
                    points = np.array([[(x, y)], [(x + w, y + h)]], dtype='float32')
                    points = points.reshape(-1, 1, 2)
                    original_points = cv2.perspectiveTransform(points, M_inv)
                    original_top_left = tuple(map(int, original_points[0][0]))
                    original_bottom_right = tuple(map(int, original_points[1][0]))

                    # Menggambar kotak pada frame asli
                    cv2.rectangle(frame_resized, original_top_left, original_bottom_right, (255, 0, 0), 2)

                    # Menyimpan gambar motor yang terdeteksi pada frame asli
                    motor_img_original = frame_resized[original_top_left[1]:original_bottom_right[1], original_top_left[0]:original_bottom_right[0]]
                    screenshot_filename_original = os.path.join(output_dir_original, f"{screenshot_counter_original}.png")
                    cv2.imwrite(screenshot_filename_original, motor_img_original, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    screenshot_counter_original += 1

                detection_counter += 1

        # Mengubah ukuran frame transform ke ukuran jendela yang diinginkan (termasuk kotak deteksi)
        transform_resized = cv2.resize(transform, (transform_window_width, transform_window_height))

        # Menggambar kotak ROI pada frame yang diubah ukurannya
        for i in range(4):
            cv2.line(frame_resized, tuple(map(int, roi_points[i])), tuple(map(int, roi_points[(i + 1) % 4])), (0, 0, 255), 2)

        # Menampilkan frame
        cv2.imshow('Video Original', frame_resized)

        # Menampilkan frame yang diubah perspektifnya dan diubah ukurannya
        cv2.imshow('transform Video', transform_resized)

        # Memeriksa penekanan tombol
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('f'):
            # Mengganti ke mode fullscreen
            if fullscreen:
                cv2.setWindowProperty('Video Original', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Video Original', window_width, window_height)
                cv2.setWindowProperty('transform Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            fullscreen = not fullscreen

    # Melepaskan objek video capture dan menutup semua jendela
    cap.release()
    cv2.destroyAllWindows()

def tentukan_roi():
    global video_file, roi_points, roi_defined, fullscreen

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        messagebox.showerror("Error", "Gagal membuka file video.")
        return False

    target_width = 1920
    target_height = 1080

    window_width = 800
    window_height = 500

    cv2.namedWindow('Video Original', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Original', window_width, window_height)

    cv2.setMouseCallback('Video Original', draw_roi)

    pause_video = True
    while not roi_defined:
        if pause_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Set frame ke awal
            read, frame = cap.read()
            frame_resized = cv2.resize(frame, (target_width, target_height))

            for point in roi_points:
                cv2.circle(frame_resized, point, 5, (255, 0, 0), -1)
            if len(roi_points) == 4:
                for i in range(4):
                    cv2.line(frame_resized, tuple(map(int, roi_points[i])), tuple(map(int, roi_points[(i + 1) % 4])), (255, 0, 0), 2)

            cv2.imshow('Video Original', frame_resized)
            key = cv2.waitKey(25)
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return False
            elif key == ord('f'):
                if fullscreen:
                    cv2.setWindowProperty('Video Original', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Video Original', window_width, window_height)
                else:
                    cv2.setWindowProperty('Video Original', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                fullscreen = not fullscreen
            elif key == ord('p'):  # Tombol 'p' untuk memutar video
                pause_video = False

    cap.release()
    cv2.destroyAllWindows()
    return True

# Fungsi untuk menentukan ROI
def draw_roi(event, x, y, flags, param):
    global roi_points, roi_defined

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(roi_points) < 4:
            roi_points.append((x, y))
            print(f"Point {len(roi_points)}: ({x}, {y})")
        if len(roi_points) == 4:
            roi_defined = True

# Fungsi untuk memilih direktori penyimpanan hasil output
def choose_output_directory():
    global output_dir_original, output_dir_transform

    output_dir_original = askdirectory(title="Pilih Direktori Penyimpanan Gambar Hasil Original")
    if not output_dir_original:
        messagebox.showwarning("Peringatan", "Direktori penyimpanan gambar hasil original belum dipilih.")
        return False

    output_dir_transform = askdirectory(title="Pilih Direktori Penyimpanan Gambar Hasil transform")
    if not output_dir_transform:
        messagebox.showwarning("Peringatan", "Direktori penyimpanan gambar hasil transform belum dipilih.")
        return False

    # Membuat direktori jika belum ada
    if not os.path.exists(output_dir_original):
        os.makedirs(output_dir_original)
    if not os.path.exists(output_dir_transform):
        os.makedirs(output_dir_transform)

    return True

# Memulai proses deteksi kendaraan
def mulai_deteksi_kendaraan():
    global video_file
    if video_file is None:
        messagebox.showwarning("Peringatan", "Silakan pilih video terlebih dahulu.")
        return
    else:
        if choose_output_directory() and tentukan_roi():
            deteksi_kendaraan()

#Tahap 5 Pendeteksian Plat
def select_file():
    global image_path
    image_path = filedialog.askopenfilename(title="Pilih Gambar", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if image_path:
        file_label.config(text=f"File terpilih: {image_path}")
    else:
        file_label.config(text="Tidak ada file yang dipilih.")
    return image_path

def display_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_text_to_file(texts, output_path):
    with open(output_path, 'w') as file:
        for text in texts:
            file.write(text + '\n')

def process_image():
    global output_folder
    if not image_path:
        messagebox.showwarning("Peringatan", "Harap pilih file gambar terlebih dahulu.")
        return
    
    output_folder = filedialog.askdirectory(title="Pilih Folder Output")
    if not output_folder:
        messagebox.showwarning("Peringatan", "Harap pilih folder output terlebih dahulu.")
        return

    # Inisialisasi pembaca EasyOCR dengan bahasa yang relevan
    reader = easyocr.Reader(['en', 'id'], model_storage_directory='./.easyocr_cache')

    # Membaca gambar menggunakan OpenCV
    image = cv2.imread(image_path)

    # Tampilkan gambar asli
    display_image("Gambar Asli", image)

    # Preprocessing: Konversi ke grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Tampilkan gambar grayscale
    display_image("Gambar Grayscale", gray_image)

    # # Preprocessing: Bilateral Filtering
    filtered_image = cv2.bilateralFilter(gray_image, d=8, sigmaColor=55, sigmaSpace=60)

    # # # Tampilkan gambar setelah bilateral filtering
    display_image("Gambar Setelah Bilateral Filtering", filtered_image)

    # # Melakukan deteksi teks pada gambar yang sudah dipreproses
    results = reader.readtext(filtered_image, detail=0)

    # Menampilkan hasil teks yang terdeteksi
    print("Teks yang terdeteksi:")
    for text in results:
        print(text)

    # Menggambar bounding box dan teks yang terdeteksi pada gambar asli dengan latar belakang untuk teks
    for (bbox, text, prob) in reader.readtext(filtered_image):
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Menggambar bounding box
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        # Menentukan posisi teks
        text_position = (top_left[0], top_left[1] - 10)

        # Menentukan ukuran teks
        font_scale = 0.8
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2

        # Menghitung ukuran teks
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Menentukan posisi latar belakang teks
        background_topleft = (text_position[0], text_position[1] - text_size[1])
        background_bottomright = (text_position[0] + text_size[0], text_position[1] + text_size[1] // 2)

        # Menggambar latar belakang teks
        cv2.rectangle(image, background_topleft, background_bottomright, (0, 255, 0), cv2.FILLED)

        # Menggambar teks di atas latar belakang
        cv2.putText(image, text, text_position, font, font_scale, (0, 0, 0), font_thickness)

    # Menentukan path output gambar
    output_image_path = os.path.join(output_folder, 'detected_image10.jpg')
    cv2.imwrite(output_image_path, image)

    # Menentukan path output file teks
    output_text_path = os.path.join(output_folder, 'plate10.txt')

    # Menyimpan teks yang terdeteksi ke dalam file teks
    save_text_to_file(results, output_text_path)

    # Tampilkan lokasi gambar dan teks yang disimpan
    print(f"Gambar dengan bounding box dan teks disimpan di: {output_image_path}")
    print(f"Teks yang terdeteksi disimpan di: {output_text_path}")

    # Tampilkan gambar akhir dengan bounding box dan teks
    display_image("Gambar dengan Teks Terdeteksi", image)

# Membuat window utama
app = tk.Tk()
app.title("DETEKSI PELANGGAR GARIS MARKA PADA TRAFFIC LIGHT")

# Mengatur tinggi dan lebar window
window_width = 800
window_height = 500

# Mendapatkan dimensi layar
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()

# Menghitung posisi tengah layar
center_x = int(screen_width / 2 - window_width / 2)
center_y = int(screen_height / 2 - window_height / 2)

# Mengatur posisi window
app.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

# Membuat frame utama
main_frame = tk.Frame(app)
main_frame.pack(fill=tk.BOTH, expand=True)

# Membuat headbar
headbar = tk.Frame(main_frame, height=40, bg="#41C9E2")
headbar.pack(side=tk.TOP, fill=tk.X)

# Membuat logo
try:
    logo_image = Image.open("logobl.png")
    logo_image = logo_image.resize((60, 60)) 
    logo_photo = ImageTk.PhotoImage(logo_image)
    # Menampilkan logo di headbar
    logo_label = tk.Label(headbar, image=logo_photo, bg="#41C9E2")
    logo_label.pack(side=tk.LEFT, padx=10)
except Exception as e:
    print(f"Error loading logo: {e}")
    logo_label = tk.Label(headbar, text="LOGO", bg="#41C9E2")
    logo_label.pack(side=tk.LEFT, padx=10)

# Membuat label di headbar
headbar_label = tk.Label(headbar, text="DETEKSI PELANGGAR GARIS MARKA PADA TRAFFIC LIGHT", fg="black", bg="#41C9E2", font=("Oswald", 15, "bold"))
headbar_label.pack(pady=10)

# Membuat frame untuk navbar dan konten
nav_content_frame = tk.Frame(main_frame)
nav_content_frame.pack(fill=tk.BOTH, expand=True)

# Membuat navbar
navbar_frame = tk.Frame(nav_content_frame, bg="#008DDA")
navbar_frame.pack(side=tk.TOP, fill=tk.X)

# Membuat tombol-tombol ke navbar
buttons = [
    ("Tahap 1", lambda: open_page(tahap_1)),
    ("Tahap 2", lambda: open_page(tahap_2)),
    ("Tahap 3", lambda: open_page(tahap_3)),
    ("Tahap 4", lambda: open_page(tahap_4)),
    ("Tahap 5", lambda: open_page(tahap_5)),
]

for (text, command) in buttons:
    button = tk.Button(navbar_frame, text=text, command=command, bg="#008DDA", fg="#FFFFFF", font=("Helvetica", 10, "bold"), padx=10, pady=5, bd=0, relief=tk.FLAT, activebackground="#03AED2")
    button.pack(side=tk.LEFT, padx=5, pady=5)

# Membuat frame untuk konten utama
content_frame = tk.Frame(nav_content_frame)
content_frame.pack(fill=tk.BOTH, expand=True)

# Membuat halaman-halaman pada content_frame
halaman_utama = tk.Frame(content_frame)
tahap_1 = tk.Frame(content_frame)
tahap_2 = tk.Frame(content_frame)
tahap_3 = tk.Frame(content_frame)
tahap_4 = tk.Frame(content_frame)
tahap_5 = tk.Frame(content_frame)

# Isi halaman utama
halaman_utama_paragraph = "MUHAMMAD HAFIZH HUSEIN - 2011500911 \n\n\n"
halaman_utama_paragraph += "Aplikasi ini dibuat untuk melakukan pendeteksian pelanggaran \n"
halaman_utama_paragraph += "garis marka pada traffic light"
halaman_utama_label = tk.Label(halaman_utama, text=halaman_utama_paragraph, font=("tahoma", 10, "bold"), padx=20, pady=10, justify='center')
halaman_utama_label.pack(expand=True, fill=tk.BOTH)

# Isi halaman tahap 1
tahap_1_paragraph = "Tahap 1: (Pre-Processing)\n\n\n"
tahap_1_paragraph += "Pada tahap pertama ini, merupakan proses resize dan merubah citra ke grayscale"
tahap_1_label = tk.Label(tahap_1, text=tahap_1_paragraph, font=("tahoma", 10, "bold"), padx=20, pady=10, justify='center')
tahap_1_label.pack(expand=True, fill=tk.BOTH)

tahap_1_button_frame = tk.Frame(tahap_1)
tahap_1_button_frame.pack(pady=10)
tahap_1_button = tk.Button(tahap_1_button_frame, text="Jalankan", command=pre_processing, bg="#008DDA", fg="#FFFFFF", font=("Helvetica", 10, "bold"), padx=10, pady=5)
tahap_1_button.grid(row=0, column=0, padx=5)

# Isi halaman tahap 2
tahap_2_paragraph = "Tahap 2: (Training Haarcascade)\n\n\n"
tahap_2_paragraph += "Tahap kedua merupakan proses training Haarcascade untuk melatih sistem\n"
tahap_2_paragraph += "dengan citra yang sudah diproses pada tahap sebelumnya"
tahap_2_label = tk.Label(tahap_2, text=tahap_2_paragraph, font=("tahoma", 10, "bold"), padx=20, pady=10, justify='center')
tahap_2_label.pack(expand=True, fill=tk.BOTH)

tahap_2_button_frame = tk.Frame(tahap_2)
tahap_2_button_frame.pack(pady=10)
tahap_2_button1 = tk.Button(tahap_2_button_frame, text="Anotasi Citra Positif", command=open_labelimg, bg="#008DDA", fg="#FFFFFF", font=("Helvetica", 10, "bold"), padx=10, pady=5)
tahap_2_button1.pack(side=tk.LEFT, padx=5)

tahap_2_button2 = tk.Button(tahap_2_button_frame, text="Buat List Citra Positif", command=info_main, bg="#008DDA", fg="#FFFFFF", font=("Helvetica", 10, "bold"), padx=10, pady=5)
tahap_2_button2.pack(side=tk.LEFT, padx=5)

tahap_2_button3 = tk.Button(tahap_2_button_frame, text="Buat List Citra Negatif", command=create_negative_txt, bg="#008DDA", fg="#FFFFFF", font=("Helvetica", 10, "bold"), padx=10, pady=5)
tahap_2_button3.pack(side=tk.LEFT, padx=5)

tahap_2_button4 = tk.Button(tahap_2_button_frame, text="Buat Vector Positif", command=vector_main, bg="#008DDA", fg="#FFFFFF", font=("Helvetica", 10, "bold"), padx=10, pady=5)
tahap_2_button4.pack(side=tk.LEFT, padx=5)

tahap_2_button5 = tk.Button(tahap_2_button_frame, text="Mulai Pelatihan", command=train_haarcascade, bg="#008DDA", fg="#FFFFFF", font=("Helvetica", 10, "bold"), padx=10, pady=5)
tahap_2_button5.pack(side=tk.LEFT, padx=5)

# Isi halaman tahap 3
tahap_3_paragraph = "Tahap 3: (Pilih Video)\n\n\n"
tahap_3_paragraph += "Tahap ini diharuskan memilih video untuk dilakukan pemrosesan pada tahap selanjutnya"
tahap_3_label = tk.Label(tahap_3, text=tahap_3_paragraph, font=("tahoma", 10, "bold"), padx=20, pady=10, justify='center')
tahap_3_label.pack(expand=True, fill=tk.BOTH)

file_label3 = tk.Label(tahap_3, font=("tahoma", 10), padx=20, pady=10, text="File yang dipilih: ", wraplength=666)
file_label3.pack()

tahap_3_button_frame = tk.Frame(tahap_3)
tahap_3_button_frame.pack(pady=10)
tahap_3_button1 = tk.Button(tahap_3_button_frame, text="Pilih Video", command=search_video, bg="#008DDA", fg="#FFFFFF", font=("Helvetica", 10, "bold"), padx=10, pady=5)
tahap_3_button1.pack(side=tk.LEFT, padx=5)

tahap_3_button2 = tk.Button(tahap_3_button_frame, text="Tampilkan Video", command=mulai_video, bg="#008DDA", fg="#FFFFFF", font=("Helvetica", 10, "bold"), padx=10, pady=5)
tahap_3_button2.pack(side=tk.LEFT, padx=5)

# Isi halaman tahap 4
tahap_4_paragraph = "Tahap 4: (Pendeteksian Pelanggar)\n\n\n"
tahap_4_paragraph += "Pada tahap ini dilakukan pendeteksian kendaraan yang melanggar lalu lintas"
tahap_4_label = tk.Label(tahap_4, text=tahap_4_paragraph, font=("tahoma", 10, "bold"), padx=20, pady=10, justify='center')
tahap_4_label.pack(expand=True, fill=tk.BOTH)

tahap_4_button_frame = tk.Frame(tahap_4)
tahap_4_button_frame.pack(pady=10)
tahap_4_button = tk.Button(tahap_4_button_frame, text="Jalankan", command=mulai_deteksi_kendaraan, bg="#008DDA", fg="#FFFFFF", font=("Helvetica", 10, "bold"), padx=10, pady=5)
tahap_4_button.grid(row=0, column=0, padx=5)

# Isi halaman tahap 5
tahap_5_paragraph = "Tahap 5: (Pendeteksian Plat Nomor)\n\n\n"
tahap_5_paragraph += "Pada tahap terakhir ini merupakan tahap pendeteksian plat nomor pelanggar\n"
tahap_5_paragraph += "yang telah didapat dari tahapan sebelumnya"
tahap_5_label = tk.Label(tahap_5, text=tahap_5_paragraph, font=("tahoma", 10, "bold"), padx=20, pady=10, justify='center')
tahap_5_label.pack(expand=True, fill=tk.BOTH)

file_label = tk.Label(tahap_5, text="Tidak ada file yang dipilih.")
file_label.pack()

tahap_5_button_frame = tk.Frame(tahap_5)
tahap_5_button_frame.pack(pady=10)
tahap_5_button1 = tk.Button(tahap_5_button_frame, text="Pilih Foto", command=select_file, bg="#008DDA", fg="#FFFFFF", font=("Helvetica", 10, "bold"), padx=10, pady=5)
tahap_5_button1.pack(side=tk.LEFT, padx=5)

tahap_5_button2 = tk.Button(tahap_5_button_frame, text="Jalankan", command=process_image, bg="#008DDA", fg="#FFFFFF", font=("Helvetica", 10, "bold"), padx=10, pady=5)
tahap_5_button2.pack(side=tk.LEFT, padx=5)

# Menampilkan halaman utama saat aplikasi dijalankan
open_page(halaman_utama)

# Membuat footer
footer = tk.Frame(main_frame, height=40, bg="#41C9E2")
footer.pack(side=tk.BOTTOM, fill=tk.X)

# Membuat label di footer
footer_label = tk.Label(footer, text="@copyright Muhammad Hafizh Husein - 2011500911", fg="black", bg="#41C9E2", font=("Tahoma", 10))
footer_label.pack(fill='both')

app.mainloop()