Video Censor - local video mosaic tool (Windows, NVIDIA GPU)
============================================================

Start:  double-click VideoCensor.exe   (keep VideoCensorWorker.exe and the _internal folder next to it; do not move the exe alone)

Everything runs on this PC. No video, frame, click or log is sent anywhere. Internet is only needed once, to download the model file below.

1. Model file (not included, ~185 MB)
   Download sam2.1_hiera_small.pt from the official SAM 2 download links (https://github.com/facebookresearch/sam2 ,
   see models\README.md) and put it in the  models  folder next to VideoCensor.exe.
   Check the model's license on that page before use. NudeNet's small detector model is already inside the application.

2. FFmpeg (not included)
   Put ffmpeg.exe and ffprobe.exe in  ffmpeg\bin  next to VideoCensor.exe, OR install FFmpeg and add it to the PATH.
   (FFmpeg builds have their own licenses (LGPL/GPL); they are not redistributed here.)

3. Needs: an NVIDIA GPU with a current driver. A separate CUDA installation is NOT needed (the CUDA runtime is inside _internal).
   GPU memory use is under 1 GB for the small model. The worker refuses to run on the CPU unless you pass --device cpu (very slow).

4. Your data stays in these folders next to the exe (never inside _internal):
   models   model files          logs   run logs, review logs, job registry (jobs.sqlite)
   temp     intermediate chunks  output default place for results (you can choose another)
   input    suggested place for your videos     config   remembered settings
   For long videos put temp (or choose another work folder) on an SSD, not a hard disk: the final merge is much faster there.

5. How to work: File > Open video, left-click the thing to cover (right-click = not this), Process > Start processing.
   After a run use the Review tab to check the flagged frames, click again where tracking went wrong, Process > Re-process.
   No tool can promise that nothing was missed: always look at the result, starting with the flagged frames.

6. Uninstall: delete the folder. Nothing is written elsewhere (no registry entries, no installer).

Third-party components and their licenses: THIRD_PARTY_NOTICES.md
Full documentation: README.md
