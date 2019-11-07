from tensorflow.python.client import device_lib

def check_tensorflow():
    available_devices = device_lib.list_local_devices()

    print("Detected GPUs:")

    for device in available_devices:
        if device.device_type == 'XLA_GPU' or device.device_type == 'GPU':
            print("Name: ", device.name)
            print("Memory: ", device.memory_limit)
