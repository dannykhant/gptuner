import psutil
import os

_HARDWARE_OVERRIDE = None

def set_hardware_override(cpu_cores=None, memory_gb=None, disk_gb=None, disk_type=None):
    """Set global hardware overrides for target database container/node."""
    global _HARDWARE_OVERRIDE
    _HARDWARE_OVERRIDE = {
        "cpu_cores": cpu_cores,
        "memory_gb": memory_gb,
        "disk_gb": disk_gb,
        "disk_type": disk_type,
    }

def get_hardware_info():
    if _HARDWARE_OVERRIDE and _HARDWARE_OVERRIDE.get("memory_gb") is not None:
        cpu = _HARDWARE_OVERRIDE.get("cpu_cores") or 2
        ram = int(_HARDWARE_OVERRIDE.get("memory_gb"))
        disk = int(_HARDWARE_OVERRIDE.get("disk_gb") or 10)
        return cpu, ram, disk

    try:
        available_cpu_cores = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory()
        total_memory = memory.total / (1024 * 1024 * 1024)
        root_disk = psutil.disk_usage('/')
        total_disk_space = root_disk.total / (1024 * 1024 * 1024)
        return available_cpu_cores, int(total_memory), int(total_disk_space)
    except Exception:
        return 2, 2, 10

def get_disk_type(devices=["sda", "nvme0n1", "vda", "hda"]):
    if _HARDWARE_OVERRIDE and _HARDWARE_OVERRIDE.get("disk_type") is not None:
        return _HARDWARE_OVERRIDE.get("disk_type")

    for device in devices:
        rotational_path = f'/sys/block/{device}/queue/rotational'
        if os.path.exists(rotational_path):
            with open(rotational_path, 'r') as file:
                rotational_value = file.read().strip()
                if rotational_value == '0':
                    return 'SSD'
                elif rotational_value == '1':
                    return 'HDD'

    return 'SSD'
