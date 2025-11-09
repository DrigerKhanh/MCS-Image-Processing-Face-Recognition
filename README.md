### Steps để chạy annaconda
- conda create --name image-processing python=3.9
- conda activate image-processing (hoặc env mà mình cần chạy) - kích hoạt môi trường <env> của anaconda
- conda deactivate image-processing : Tắt môi trường anaconda
- conda env remove --name image-processing --all : Xoá môi trường <env> anaconda
- conda env list : Liệt kê các môi trường

### Installation
pip install numpy==1.24.3
conda install cudatoolkit=11.2 cudnn=8.1.0 -c conda-forge

#### Cài đặt TensorFlow với GPU support
pip install tensorflow-gpu==2.10.0

#### Cài đặt PyTorch với CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

#### Cài đặt các thư viện khác
pip install deepface==0.0.79
pip install opencv-python
pip install pandas
pip install numpy==1.24.3