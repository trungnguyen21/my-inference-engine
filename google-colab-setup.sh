from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/my-inference-engine/models
!cp -n /content/drive/MyDrive/models/qwen-2.5.gguf /content/my-inference-engine/models/