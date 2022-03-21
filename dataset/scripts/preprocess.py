import os
import argparse
import cv2
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Extracting frames and audio')
parser.add_argument(
        '-data_dir',
        dest='data_dir',
        default="../",
        type=str,
        help='Directory containing PACS data'
    )

args = parser.parse_args()

def get_box(box, xmax = 1280, ymax = 720, blowup = 1.25, ar=1):

    x0, y0, x1, y1 = box
    w = x1-x0
    h = y1-y0

    max_ar = ar
    min_ar = 1/ar

    ar = w/h
    if ar > max_ar:
        needed_height = int(w/max_ar + 0.5)
        avg = (y1+y0)/2
        y1 = int(avg + needed_height/2 + 0.5)
        y0 = int(avg - needed_height/2 - 0.5)
    elif ar < min_ar:
        needed_width = int(h*min_ar + 0.5)
        avg = (x1+x0)/2
        x1 = int(avg + needed_width/2 + 0.5)
        x0 = int(avg - needed_width/2 - 0.5)

    w = x1-x0
    h = y1-y0

    x0 -= (blowup-1)/2*w
    x1 += (blowup-1)/2*w
    y0 -= (blowup-1)/2*h
    y1 += (blowup-1)/2*h

    if x1 >= xmax:
        x0 -= (x1-xmax)
        x1 = xmax
    if y1 >= ymax:
        y0 -= (y1-ymax)
        y1 = ymax
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x0 < 0:
        x1 -= x0
        x0 = 0

    x0 = max(0, x0)
    x1 = min(x1, xmax)
    y0 = max(0, y0)
    y1 = min(y1, ymax)
    
    return [int(i) for i in [x0, y0, x1, y1]]

youtube_info = json.load(open(os.path.join(args.data_dir, "json", "youtube_info.json")))
obj_data = json.load(open(os.path.join(args.data_dir, "json", "objects.json")))

os.makedirs(os.path.join(args.data_dir, "audio16000"), exist_ok=True)
os.makedirs(os.path.join(args.data_dir, "audio21992"), exist_ok=True)
os.makedirs(os.path.join(args.data_dir, "audio44100"), exist_ok=True)

os.makedirs(os.path.join(args.data_dir, "square_crop"), exist_ok=True)
os.makedirs(os.path.join(args.data_dir, "center_with_box"), exist_ok=True)
os.makedirs(os.path.join(args.data_dir, "frame_with_box"), exist_ok=True)

for vid in tqdm(os.listdir(os.path.join(args.data_dir, "videos"))):
    
    if not vid.endswith(".mp4"):
        continue

    vid_dir = os.path.join(args.data_dir, "videos", vid)

    # First extract the audio
    for freq in [16000, 21992, 44100]:
        output_dir = os.path.join(args.data_dir, f"audio{freq}", f"{vid[:-4]}.wav")
        os.system(f"ffmpeg -y -i {vid_dir} -vn -ar {freq} -ac 1 -codec:a pcm_s16le {output_dir} >/dev/null 2>&1")

    try:
        assert(os.path.exists(os.path.join(args.data_dir, "midframes", vid[:-4] + ".png")))
    except:
        print(f"{vid} does not have a midframe")
        continue

    midframe = cv2.imread(os.path.join(args.data_dir, "midframes", vid[:-4] + ".png"))
    box = obj_data[vid[:-4]]["bounding_box"]

    new_img = cv2.rectangle(midframe, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(args.data_dir, "frame_with_box", f"{vid[:-4]}.png"), new_img)

    new_box = get_box(box)
    coords = [box[0] - new_box[0], box[1] - new_box[1], box[2] - new_box[0], box[3] - new_box[1]]

    new_img = midframe[new_box[1]:new_box[3], new_box[0]:new_box[2]]
    cv2.imwrite(os.path.join(args.data_dir, "square_crop", vid[:-4] + ".png"), new_img)

    new_img = cv2.rectangle(new_img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(args.data_dir, "center_with_box", vid[:-4] + ".png"), new_img)
    
    # print("Extracting video frames")
    cap = cv2.VideoCapture(vid_dir)
    success, img = cap.read()
    count = 1
    os.makedirs(os.path.join(args.data_dir, "frames486", vid[:-4]), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "frames252", vid[:-4]), exist_ok=True)
    while success:
        big_img = cv2.resize(img, (864, 486))
        cv2.imwrite(os.path.join(args.data_dir, "frames486", vid[:-4], f"{count:06}.png"), big_img)
        small_img = cv2.resize(img, (448, 252))
        cv2.imwrite(os.path.join(args.data_dir, "frames252", vid[:-4], f"{count:06}.png"), small_img)
        count += 1
        success, img = cap.read()
    