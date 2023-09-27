""" Replace files in dependent folders"""
import os

# replace app.py with Segment-and-Track-Anything/app.py
os.replace("app.py", "Segment-and-Track-Anything/app.py")

# replace seg_track_anything.py with Segment-and-Track-Anything/seg_track_anything.py
os.replace("seg_track_anything.py","Segment-and-Track-Anything/seg_track_anything.py")

#replace inference_propainter.py with ProPainter/inference_propainter.py 
os.replace("inference_propainter.py", "Segment-and-Track-Anything/ProPainter/inference_propainter.py")
