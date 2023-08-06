import os
import time
# Enable terminal color
os.system('color')
# RGB shortcut for fg text color
def rgb(r, g, b): return f"\u001b[38;2;{r};{g};{b}m"
def newFolder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    print(f"Created folder '{path}'")
    return path

print(rgb(0, 255, 0))
print("======================================= REQUIREMENTS REMINDER ========================================")
print("If you are coming across errors, check if the below requirements are met (more info on README):")
print("- You must have Python (3.10.x or over) installed and added to PATH Environment Variables")
print("- You must have FFmpeg installed and added to PATH Environment Variables")
print("If PyTorch cannot find a gpu, you may need to install [https://developer.nvidia.com/cuda-downloads]")
print("If you are still having problems, open up an issue in my Github repo. I'm still learning y'know? =)")
print("======================================================================================================")
print(rgb(192, 192, 192))

input_path = newFolder(os.getcwd()+"\\input")
output_path = newFolder(os.getcwd()+"\output")

os.system("cmd /c py -m pip install whisper")
os.system("cmd /c py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print(rgb(0, 255, 0))
print("Successfully installed! Now use start.bat for audio splitting.")
print("Exiting in 3 seconds...")
print(rgb(192, 192, 192))
time.sleep(3)