import re
from glob import glob
from moviepy.editor import *
import cv2
import math

iemocap_full_release_path = "/Users/julie/Desktop/cs231n project/IEMOCAP_full_release/"

def divide_videos_to_clips(iemocap_full_release_path):
        info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)
        for x in range(5):
            sess_name = "Session" + str(x+1)
            path_video = iemocap_full_release_path + sess_name + "/dialog/avi/DivX/"
            path_label= iemocap_full_release_path + sess_name + "/dialog/EmoEvaluation/"
            video_clip_path =iemocap_full_release_path + sess_name + "/dialog/avi/sentences_video_audio/"
            if not os.path.exists(video_clip_path):
               os.makedirs(video_clip_path)
            videos=glob(path_video+'*.avi')

            for video_name in videos:
                video_name=video_name.split("/")[-1]
                video_name_folder=video_clip_path+video_name.split(".")[0]+'/'
                if not os.path.exists(video_name_folder):
                   os.makedirs(video_name_folder)
                with open(path_label+video_name.split(".")[0]+'.txt') as f:
                   content = f.read()
                info_lines = re.findall(info_line, content)
                for line in info_lines[1:]:  # the first line is a header
                    print(path_label+video_name.split(".")[0]+'.txt')
                    start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
                    start_time, end_time = start_end_time[1:-1].split('-')
                    start_time, end_time = float(start_time), float(end_time)
                    video = VideoFileClip(
                        path_video+video_name)
                    if end_time>video.duration:
                        end_time=video.duration
                    print("wav_file_name {},start time {},end time {}".format(wav_file_name, start_time, end_time))

                    video=video.subclip(
                        start_time, end_time)
                    video.write_videofile(
                        video_name_folder+wav_file_name+".mp4",
                        codec='libx264',
                        audio_codec='aac',
                        temp_audiofile='temp-audio.m4a',
                        remove_temp=True)

def extract_video_frames(iemocap_full_release_path,num_images):
    for x in range(5):
        sess_name = "Session" + str(x + 1)
        video_clip_path = iemocap_full_release_path + sess_name + "/dialog/avi/sentences_video_audio/"
        images_path=iemocap_full_release_path + sess_name + "/dialog/avi/images_clip/"
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        folders=os.listdir(video_clip_path)
        for folder in folders:
            image_path_each=images_path+folder+"/"
            if not os.path.exists(image_path_each):
                os.makedirs(image_path_each)
            folder=video_clip_path+folder+"/"
            video_clips=os.listdir(folder)
            for video_clip in video_clips:
                image_path_clip=image_path_each+video_clip+"/"
                if not os.path.exists(image_path_clip):
                    os.makedirs(image_path_clip)

                cap = cv2.VideoCapture(folder+video_clip)
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(length)
                interval=length//num_images
                frameRate = cap.get(5)  # frame rate
                print(frameRate)
                x = 1
                while (cap.isOpened()):
                    frameId = cap.get(1)  # current frame number
                    ret, frame = cap.read()
                    if (ret != True):
                        break
                    if length%num_images==0:
                        length-=1
                    if (frameId<=(length-length%num_images)) and (frameId % math.floor(interval) == 0):
                    #if (frameId % math.floor(interval) == 0):
                        filename = image_path_clip +video_clip+ str(int(x)) + ".jpg";
                        x += 1
                        cv2.imwrite(filename, frame)

                cap.release()
                print("Done!")
'''divide videos to clips function'''
#divide_videos_to_clips(iemocap_full_release_path)
'''choose number of images to be 29 because frame_rate=29.97, choose 29 makes the number of images extracted from the video to be 30'''
#extract_video_frames(iemocap_full_release_path,29)









