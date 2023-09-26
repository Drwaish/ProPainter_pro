# Pro Inpainting Tool
Using Pro-Painter and Segment-and-Track Anything to remove objects from video.

# Setup repo
## Parent Repo
```bash
git clone https://github.com/Drwaish/ProPainter_pro.git
cd ProPainter_pro
```
## Dependent Repos
### Segment and Track Anything
```bash
git clone https://github.com/z-x-yang/Segment-and-Track-Anything.git
cd Segment-and-Track-Anything/
```
Now download Models and install dependencies
Run following command in Segment and Track Anything

```bash
mkdir ./ckpt
```
Now run bash script to download model.
```bash
bash script/download_ckpt.sh
```
Installing dependencies
```bash
bash script/install.sh
```
Install gradio for interactive interface.
```bash
pip install gradio 
```
Change directory to Parent
```bash
cd ..
```
### Pro-Painter
```bash
git clone https://github.com/sczhou/ProPainter.git
cd ProPainter
```
Download weights in weights folder
```bash
cd weights
wget https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth
wget https://github.com/sczhou/ProPainter/releases/download/v0.1.0/i3d_rgb_imagenet.pt
wget https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth
wget https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth
```
Change directory to Pro-Painter
```bash
cd ..
```
Install mmcv-full 
```bash
pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```
Install following libraries
```bash
pip install einops
pip install av 
```
Change directory to Parent
```bash
cd ..
```
Now run helper function.
```bash
python helper.py
```

Everything is setup now time see magic of AI
```bash
cd Segment-and-Track-Anything
python app.py
```
Enter prompt and enjoy output.