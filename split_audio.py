import pynvml
import wave
import whisper
import os
import time

ONE_GB = 1073741824
TWO_GB = 2147483648
FIVE_GB = 5368709120
TEN_GB = 10737418240

# RGB shortcut for fg text color
def rgb(r, g, b): return f"\u001b[38;2;{r};{g};{b}m"
def newFolder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    print(f"Created folder '{path}'")
    return path

# Create a new (.wav) audio file using two time stamps (in seconds, start & end)
def clipAudio(input_file, output_file, start, end):
    # Open the input .wav file
    with wave.open(input_file, 'rb') as wav_file:
        # Get the parameters of the input audio file
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        # Calculate the start and end frame positions based on the timestamps
        start_frame = int(start * frame_rate)
        end_frame = int(end * frame_rate)
        # Clamp end_frame to last frame value in .wav
        end_frame = min(end_frame, num_frames)
        # Move the file pointer to the start_frame position
        wav_file.setpos(start_frame)
        # Read audio data from start_frame to end_frame
        audio_data = wav_file.readframes(end_frame - start_frame)

    # Write the clipped audio data to a new .wav file
    with wave.open(output_file, 'wb') as new_wav_file:
        new_wav_file.setnchannels(channels)
        new_wav_file.setsampwidth(sample_width)
        new_wav_file.setframerate(frame_rate)
        new_wav_file.writeframes(audio_data)

# Get all of the avalible device (cpu and gpu) with vram and other stats
def getDeviceInfo():
    pynvml.nvmlInit()
    devices_info = []
    device_count = pynvml.nvmlDeviceGetCount() # Only accounts for Nvidia gpus

    # This will only be use if no gpus are avalible (index is included for formality, I don't see a way to use it)
    cpu_info = {
        "index": None,
        "name": "cpu",
        "type": "cpu",
        "vram_free_bytes": 0,
        "vram_free_gb": 0,
        "vram_total_bytes": 0,
        "vram_total_gb": 0
    }
    devices_info.append(cpu_info)

    # Get all nvidia gpus
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle) # Human-readable name (branding)
        vram_free_bytes = pynvml.nvmlDeviceGetMemoryInfo(handle).free
        vram_free_gb = round(vram_free_bytes / 1024**3, 2) # Convert from bytes to gigabytes, **3 = ^3
        vram_total_bytes = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        vram_total_gb = round(vram_total_bytes / 1024**3, 2) # Convert from bytes to gigabytes, **3 = ^3

        device_info = {
            "index": i,
            "name": name,
            "type": "cuda",
            "vram_free_bytes": vram_free_bytes,
            "vram_free_gb": vram_free_gb,
            "vram_total_bytes": vram_total_bytes,
            "vram_total_gb": vram_total_gb
        }
        devices_info.append(device_info)

    pynvml.nvmlShutdown()
    return devices_info

# Get device with the most avalible VRAM memory (larger quantity required for larger models)
def getBestDevice(devices):
    best_device = devices[0]
    largest_mem_free = 0
    for device in devices:
        if device.get("vram_free_bytes") > largest_mem_free:
            best_device = device
    return best_device

# Get readable device names (branding) (mostly a shorthand)
def getDeviceNames(devices):
    device_names = []
    for device in devices:
        device_names.append(device.get("name"))
    return device_names

# Determine which Whisper model to use based off of VRAM avalibility
def getBestModel(device_vram_free):
    if device_vram_free >= TEN_GB:
        return "large"
    elif device_vram_free >= FIVE_GB:
        return "medium"
    elif device_vram_free >= TWO_GB:
        return "small"
    elif device_vram_free >= ONE_GB:
        return "base"
    elif device_vram_free < ONE_GB:
        return "tiny"

def main():
    # Enable terminal color
    os.system('color')

    # Change color to green
    print(rgb(0, 255, 0))

    # Create output folder (for split audio), input folder (for orignial audio), and logs folder (for transcription logs)
    input_path = newFolder(os.getcwd()+"\\input")
    output_path = newFolder(os.getcwd()+"\output")

    # Search audio folder for usable audio file
    audio_names = []
    for file_name in os.listdir(input_path):
        file_ext = file_name[file_name.rindex('.'):]
        if (file_ext == ".mp3" or file_ext == ".wav"):
            audio_names.append(file_name)
    print(f"Audio files detected: {audio_names}")

    if len(audio_names) == 0:
        print(f"There are no audio files in '{input_path}'. Only .wav and .mp3 files are allowed.")
        print("Exiting in 3 seconds...")
        time.sleep(3)
        return

    # Get VRAM avalibility on GPU (check for all avalible gpus and use one with most avalible VRAM)
    devices = getDeviceInfo()
    device = getBestDevice(devices)
    print(f"Avalible devices: {getDeviceNames(devices)}")
    use_type = device.get("type")
    if use_type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device.get('index'))
    print(f"Using device '{device.get('name')}' ({device.get('type')}:{device.get('index')}) with {device.get('vram_free_gb')}/{device.get('vram_total_gb')} GB VRAM")

    # Model to-be-used for whisper https://github.com/openai/whisper/blob/main/README.md ("Avalible Models and languages" section)
    # Note that the smaller models may come at a sacrifice to accuracy, described more in the link above
    use_model = getBestModel(device.get("vram_free_bytes"))
    print(f"Using Whisper model '{use_model}'")

    for audio in audio_names:
        # Create new folder in output_path named after the audio file
        audio_output_path = newFolder(os.path.join(output_path, audio[:-4]))

        # Transcribe with original whisper model
        print(f"Loading Whisper model - You may need to download the model on first use")
        model = whisper.load_model(use_model, device=use_type)
        print(f"Beginning audio transcription - This might take a long time (dependant on file size, model, and system hardware)")

        # Change color to default
        print(rgb(192, 192, 192))

        result = model.transcribe(os.path.join(input_path, audio))

        # Get all different sentances and strip unnecessary data (for ease of use a and log readability)
        stripped_segments = []
        for segment in result["segments"]:
            stripped_segments.append({"id"   : segment.get("id"), 
                                      "start": round(segment.get("start"), 2), 
                                      "end"  : round(segment.get("end"), 2), 
                                      "text" : segment.get("text")})
        
        # Change color to green
        print(rgb(0, 255, 0))

        # Clip all audio files using stripped_segment "time_start" and "time_end" values
        for segment in stripped_segments:
            clipAudio(os.path.join(input_path, audio), 
                      os.path.join(audio_output_path, str(segment.get("id")))+".wav",
                      segment.get("start"),
                      segment.get("end"))
            print('{{"id": {id}, "start": {start}, "end": {end}, "text": "{text}"}}'.format(**segment))
    

        # Move original audio files from input_path to output_path
        os.replace(os.path.join(input_path, audio), os.path.join(audio_output_path, audio))
        print(f"'{audio}' and new .wav files moved to {audio_output_path}")
    
    print("Successfully split audio!")
    print("Exiting in 3 seconds...")
    time.sleep(3)
    
if __name__ == "__main__":
    main()