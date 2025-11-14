Steps để chạy annaconda
- conda create --name image-processing python=<version>: Tạo môi trường annaconda mới (3.10)
- conda activate image-processing (hoặc env mà mình cần chạy) - kích hoạt môi trường <env> của annaconda
- conda deactivate image-processing : Tắt môi trường annaconda
- conda env remove --name image-processing : Xoá môi trường <env> annaconda
- conda env list : Liệt kê các môi trường

Requirement
# Verify 
# Kiểm tra version TensorFlow/PyTorch hiện tại
pip show tensorflow torch

# Install
## 1: Cai dat NumPy DAU TIEN
pip install numpy==1.24.3

## 2: Cai dat TensorFlow on dinh
pip install tensorflow==2.13.0

## 3: Cai dat PyTorch voi CUDA
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121

## 4: Cai dat OpenCV
pip install opencv-python==4.8.1.78

## 5: Cai dat cac thu vien co ban
pip install pandas==2.1.4 matplotlib==3.8.2 scikit-learn==1.3.2 Pillow==10.1.0

## 6: Cai dat DeepFace CUOI CUNG
pip install deepface==0.0.79

# 4. Utilities
pip install pandas numpy matplotlib scikit-learn Pillow

# Downgrade NumPy về version tương thích (nếu lôĩ)
pip install "numpy<2" --force-reinstall
 
