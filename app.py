from PIL.ImageOps import colorize, scale
import gradio as gr
import importlib
import sys
import os
import pdb
from matplotlib.pyplot import step

from model_args import segtracker_args,sam_args,aot_args
from SegTracker import SegTracker
from tool.transfer_tools import draw_outline, draw_points
import cv2
from PIL import Image
from skimage.morphology.binary import binary_dilation
import argparse
import torch
import time, math
from seg_track_anything import aot_model2ckpt, tracking_objects_in_video, draw_mask
import gc
import numpy as np
import json
from tool.transfer_tools import mask2bbox
import time
import gc

def tracking_objects(Seg_Tracker, input_video, frame_num=0):
    fps = 8
    print("Start tracking !")
    # pdb.set_trace()
    # output_video, output_mask=tracking_objects_in_video(Seg_Tracker, input_video, input_img_seq, fps)
    # pdb.set_trace()
    return tracking_objects_in_video(Seg_Tracker, input_video,fps, frame_num)

def gd_detect(Seg_Tracker, origin_frame, grounding_caption):
    box_threshold = 0.5
    text_threshold = 0.5
    aot_model = "r50_deaotl"
    long_term_mem = 9999
    max_len_long_term = 9999
    sam_gap = 9999
    max_obj_num = 255
    points_per_side = 16
    if Seg_Tracker is None:
        Seg_Tracker, _ , _, _ = init_SegTracker(origin_frame)
    print("Detect")
    predicted_mask, annotated_frame= Seg_Tracker.detect_and_seg(origin_frame, grounding_caption, box_threshold, text_threshold)
    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)
    masked_frame = draw_mask(annotated_frame, predicted_mask)
    return Seg_Tracker, masked_frame, origin_frame

def pro_painter(mask_image_path, video_image_path):
    print("In proPainter")
    print(os.system(f"!python ProPainter/inference_propainter.py --video {mask_image_path} --mask {video_image_path}"))
    return True
def get_meta_from_video(Seg_Tracker, input_video, grounding_caption):
    if input_video is None:
        return None, None, None, ""
    print("get meta information of input video")
    cap = cv2.VideoCapture(input_video)
    _, first_frame = cap.read()
    cap.release()
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    Seg_Tracker, masked_frame, origin_frame = gd_detect(Seg_Tracker, first_frame, grounding_caption)
    mask_img, video_img  = tracking_objects(Seg_Tracker, input_video, frame_num=0)
    time.sleep(2)
    video_name = os.path.basename(input_video).split('.')[0]
    print("Video Name:" , video_name)
    mask_image_path = f"/tracking_results/{video_name}/{video_name}_masked_frames"
    video_image_path = f"/tracking_results/{video_name}/{video_name}_masks"

    print({"video_name": video_name,
           "mask_image_path" : mask_image_path,
           "video_image_path" : video_image_path,
           "video_img" : video_img,
           "mask_img" : mask_img
           })
    # output_video = pro_painter(mask_image_path, video_image_path)
    # if output_video:
        # result_path = f"Segment-and-Track-Anything/ProPainter/results/{video_name}/inpaint_out.mp4"
        # return Seg_Tracker, result_path
        # !python inference_propainter.py --video inputs/object_removal/bmx-trees --mask inputs/object_removal/bmx-trees_mask --fp16
    import subprocess

# no Python Exception is thrown!
    var = subprocess.call(f"python ProPainter/inference_propainter.py --video {video_img} --mask {mask_img} --height 320 --width 576 --subvideo_length 30 --fp16" , shell = True)
    # print(os.subprocess(f"python ProPainter/inference_propainter.py --video {mask_image_path} --mask {video_image_path}"))
    print("var", var)
    if var == 0:
        tokenize = video_img.split('/')
        print("tokenize", tokenize[-1])
        result_path = f"results/{tokenize[-1]}/inpaint_out.mp4"
        return Seg_Tracker, result_path
    else:
        return Seg_Tracker, input_video

def SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask):
    with torch.cuda.amp.autocast():
        # Reset the first frame's mask
        frame_idx = 0
        Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)
        Seg_Tracker.first_frame_mask = predicted_mask
    return Seg_Tracker
def init_SegTracker(origin_frame):
    aot_model = "r50_deaotl"
    long_term_mem = 9999
    max_len_long_term = 9999
    sam_gap = 9999
    max_obj_num = 255
    points_per_side = 16
    if origin_frame is None:
        return None, origin_frame, [[], []], ""
    # reset aot args
    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]
    aot_args["long_term_mem_gap"] = long_term_mem
    aot_args["max_len_long_term"] = max_len_long_term
    # reset sam args
    segtracker_args["sam_gap"] = sam_gap
    segtracker_args["max_obj_num"] = max_obj_num
    sam_args["generator_args"]["points_per_side"] = points_per_side
    
    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()

    return Seg_Tracker, origin_frame, [[], []], ""

def init_SegTracker_Stroke(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame):
    if origin_frame is None:
        return None, origin_frame, [[], []], origin_frame
    # reset aot args
    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]
    aot_args["long_term_mem_gap"] = long_term_mem
    aot_args["max_len_long_term"] = max_len_long_term
    # reset sam args
    segtracker_args["sam_gap"] = sam_gap
    segtracker_args["max_obj_num"] = max_obj_num
    sam_args["generator_args"]["points_per_side"] = points_per_side
    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()
    return Seg_Tracker, origin_frame, [[], []], origin_frame



def seg_track_app():
    ##########################################################
    ######################  Front-end ########################
    ##########################################################
    app = gr.Blocks()
    with app:
        gr.Markdown(
            '''
            <div style="text-align:center;">
                <span style="font-size:3em; font-weight:bold;">Remove Object From Video</span>
            </div>
            '''
        )

        click_stack = gr.State([[],[]]) # Storage clicks status
        origin_frame = gr.State(None)
        Seg_Tracker = gr.State(None)
        current_frame_num = gr.State(None)
        refine_idx = gr.State(None)
        frame_num = gr.State(None)
        aot_model = gr.State(None)
        sam_gap = gr.State(None)
        points_per_side = gr.State(None)
        max_obj_num = gr.State(None)
        with gr.Row():
            input_video = gr.Video(label='Input video')
            input_first_frame = gr.Video(label='Masked Object Frame')
        with gr.Column():
            grounding_caption = gr.Textbox(label="Detection Prompt")
            detect_button = gr.Button(value="Detect")
                    
    ##########################################################
    ######################  back-end #########################
    ##########################################################
        # listen to the input_video to get the first frame of video
        # input_video.change(
        #     fn=get_meta_from_video,
        #     inputs=[
        #         input_video
        #     ],
        #     outputs=[
        #         input_first_frame, origin_frame
        #     ]
        # )        
        #-------------- Input compont -------------
        # Use grounding-dino to detect object
        detect_button.click(
            fn=get_meta_from_video, 
            inputs=[
                Seg_Tracker, input_video, grounding_caption
                ], 
            outputs=[
                Seg_Tracker, input_first_frame
                ]
                )
        with gr.Tab(label='Video example'):
            gr.Examples(
                examples=[
                    os.path.join(os.path.dirname(__file__), "assets", "blackswan.mp4"),
                    os.path.join(os.path.dirname(__file__), "assets", "cars.mp4"),
                    os.path.join(os.path.dirname(__file__), "assets", "cell.mp4"),
                    ],
                inputs=[input_video],
            )
    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=True)
if __name__ == "__main__":
    seg_track_app()
