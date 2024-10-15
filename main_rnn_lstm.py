import kagglehub

# Download latest version
path = kagglehub.dataset_download("nccvector/electromyography-emg-dataset")

print("Path to dataset files:", path)
