import tensorflow as tf
import torch
import sys
import time


def check_gpu_support():
    print("🔍 Kiểm tra GPU support...")
    print(f"Python: {sys.version}")
    print()

    # Kiểm tra TensorFlow
    print("=== TENSORFLOW ===")
    print(f"Version: {tf.__version__}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Devices: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")

        # Test TensorFlow performance
        print("\n🧪 TensorFlow GPU Test:")
        with tf.device('/GPU:0'):
            # Large matrix multiplication
            size = 3000
            a = tf.random.normal([size, size])
            b = tf.random.normal([size, size])

            start = time.time()
            c = tf.matmul(a, b)
            tf_time = time.time() - start

            print(f"   Matrix {size}x{size}: {tf_time:.3f}s")
            print(f"   TensorFlow GPU: ✅ HOẠT ĐỘNG")
    else:
        print("❌ TensorFlow: Không tìm thấy GPU")

    print("\n=== PYTORCH ===")
    print(f"Version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"✅ CUDA: {torch.version.cuda}")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Test PyTorch performance
        print("\n🧪 PyTorch GPU Test:")
        device = torch.device('cuda')
        size = 3000

        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        torch_time = time.time() - start

        print(f"   Matrix {size}x{size}: {torch_time:.3f}s")
        print(f"   PyTorch GPU: ✅ HOẠT ĐỘNG")
    else:
        print("❌ PyTorch: Không tìm thấy CUDA")

    print("\n=== DEEPFACE ===")
    try:
        from deepface import DeepFace
        print(f"Version: {DeepFace.__version__}")
        print("✅ DeepFace: Đã cài đặt")
    except ImportError:
        print("❌ DeepFace: Chưa cài đặt")

    print("\n=== KẾT LUẬN ===")
    if gpus and torch.cuda.is_available():
        print("🎉 GPU HOẠT ĐỘNG HOÀN TOÀN!")
        print("🚀 Bạn có thể chạy face recognition code ngay bây giờ!")
    else:
        print("❌ Có vấn đề với GPU setup")


if __name__ == "__main__":
    check_gpu_support()