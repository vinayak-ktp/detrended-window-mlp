import subprocess
import sys


def run(cmd):
    print(f">>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def has_nvidia_gpu():
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def install_torch(gpu: bool):
    if gpu:
        print("GPU detected — installing PyTorch with CUDA 12.1 support")
        run([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
    else:
        print("No GPU detected — installing CPU-only PyTorch...")
        run([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])


def install_other_requirements():
    packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "seaborn",
        "ipython",
        "jupyter"
    ]
    print("\nInstalling other dependencies...")
    run([sys.executable, "-m", "pip", "install"] + packages)


if __name__ == "__main__":
    gpu = has_nvidia_gpu()
    print(f"GPU available: {gpu}\n")

    install_torch(gpu)
    install_other_requirements()

    print("\nAll dependencies installed successfully.")
