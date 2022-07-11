import os
import argparse
import cv2
import json
from tqdm import tqdm
from pytube import YouTube

parser = argparse.ArgumentParser(description='Extracting frames and audio')
parser.add_argument(
        '-data_dir',
        dest='data_dir',
        default="../",
        type=str,
        help='Directory containing PACS data'
    )

args = parser.parse_args()

youtube_info = json.load(open(os.path.join(args.data_dir, "json", "youtube_info.json")))
obj_data = json.load(open(os.path.join(args.data_dir, "json", "objects.json")))
os.makedirs(os.path.join(args.data_dir, "videos"), exist_ok=True)
os.makedirs(os.path.join(args.data_dir, "midframes"), exist_ok=True)
os.makedirs(os.path.join(args.data_dir, "frame_with_box"), exist_ok=True)

for video_id in tqdm(youtube_info):
    try:
        url = "https://www.youtube.com/watch?v=" + video_id
        video = YouTube(url)
        video.streams.get_by_resolution("720p").download(os.path.join(args.data_dir, "videos"), video_id+".mp4")
    except:
        print(f"Failed to download {video_id} from YouTube")
        continue

    try:
        vid_clips = list(youtube_info[video_id].keys())
        midframes = [youtube_info[video_id][vid_clip][2] for vid_clip in vid_clips]
        range_mins = [youtube_info[video_id][vid_clip][0] for vid_clip in vid_clips]
        range_maxs = [youtube_info[video_id][vid_clip][1] for vid_clip in vid_clips]

        cap = cv2.VideoCapture(os.path.join(args.data_dir, "videos", video_id+".mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        success, img = cap.read()
        count = 0
        while success:
            if count in midframes:
                obj_name = vid_clips[midframes.index(count)]
                img_name = os.path.join(args.data_dir, "midframes", f"{obj_name}.png")
                cv2.imwrite(img_name, img)
                
            count += 1
            success, img = cap.read()

        video_path = os.path.join(args.data_dir, "videos", video_id+".mp4")
        
        for name, rmin, rmax in zip(vid_clips, range_mins, range_maxs):
            start_time = rmin/fps
            end_time = rmax/fps

            os.system(f"ffmpeg -i {video_path} -ss {start_time:.2f} -to {end_time:.2f} -r 30 -c:v libx264 -crf 17 -c:a copy {os.path.join(args.data_dir, 'videos', name+'.mp4')} >/dev/null 2>&1")
    except:
        print(f"Failed to process {video_id} and its clips")
    os.remove(video_path)