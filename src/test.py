import tensorflow as tf
import sys
import subprocess
import os

print("=== PHÂN TÍCH HỆ THỐNG GPU ===")
print(f"Python: {sys.version}")
print(f"TensorFlow: {tf.__version__}")

# Kiểm tra GPU trong TensorFlow
gpus = tf.config.list_physical_devices('GPU')
print(f"\n1. TensorFlow GPU devices: {gpus}")

# Kiểm tra CUDA version
try:
    cuda_version = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
    print(f"\n2. CUDA Version:\n{cuda_version}")
except:
    print("\n2. CUDA Version: Not found or not in PATH")

# Kiểm tra NVIDIA driver
try:
    nvidia_smi = subprocess.check_output(["nvidia-smi"]).decode('utf-8')
    print(f"\n3. NVIDIA Driver - GPU detected")
    # Extract GPU info
    if "RTX 3060" in nvidia_smi:
        print("   ✅ RTX 3060 detected")
    if "RTX 4070" in nvidia_smi:
        print("   ✅ RTX 4070 Ti detected")
except:
    print("\n3. NVIDIA Driver: nvidia-smi not found")

# Kiểm tra environment variables
print(f"\n4. Environment Variables:")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"   PATH contains CUDA: {'CUDA' in os.environ.get('PATH', '')}")

# Kiểm tra TensorFlow GPU capability
print(f"\n5. TensorFlow GPU Test:")
try:
    # Test if TensorFlow can access GPU
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"   ✅ TensorFlow can create operations on GPU")
        print(f"   ✅ Operation device: {c.device}")
except Exception as e:
    print(f"   ❌ TensorFlow GPU error: {e}")

# Kiểm tra memory growth
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"   ✅ GPU memory growth enabled")
    except Exception as e:
        print(f"   ❌ GPU memory growth error: {e}")