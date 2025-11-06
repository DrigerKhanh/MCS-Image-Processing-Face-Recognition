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

#Install
# 1. TensorFlow với GPU support (đã tích hợp sẵn CUDA)
pip install tensorflow==2.15.0

# 2. PyTorch với CUDA support  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. DeepFace và computer vision
pip install deepface==0.0.79
pip install opencv-python==4.8.1.78
pip install retina-face==0.0.15
pip install mtcnn==0.1.1

# 4. Utilities
pip install pandas numpy matplotlib scikit-learn Pillow

# Downgrade NumPy về version tương thích (nếu lôĩ)
pip install "numpy<2" --force-reinstall
 
