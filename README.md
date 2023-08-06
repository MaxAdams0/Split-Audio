# Split-Audio
A quick Python script to detect speach using Whisper, parse it by sentance, and then output each sentance into a new audio file. This is intended for use in AI Model Training, but use it in whatever project you like! Please make sure to abide by GPLv3 License rules.
## Requirements
The only requirements are related to installs on your computer, any required python libraries for the script will be installed automatically using setup.py.
- [Python](https://www.python.org/downloads/release/python-31011/) (only 3.10.11 was tested, others may work)
- [FFmpeg](https://ffmpeg.org/download.html)

You may need to add either of these to your PATH Environmental Variables (Windows)
## Use
1. Run `setup.py`, and wait until it is done
2. Put your audio file(s) into the input folder
3. Run `start.bat` and wait

## Bugs
- PyTorch cannot correctly find gpu index if run through python file directly, unknown why

## Future Updates
These are in no specific order and may be completed at different times... or never
- Add ***diarization*** to split voices into different audio input (composite into one or just use as separate files)
- Add logging feature, so troubleshooting should be easier
- Add compatability to detect AMD & Intel GPUs (currently only NVIDIA bc it is using the ***pynvml*** library to find GPUs and get info)
- Display specific cpu name instead of "*cpu*" replacement name