import os
from tqdm import tqdm
import cv2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("input_path",type=str)
parser.add_argument("output_path",type=str)
parser.add_argument("num_of_frames",type=int)

args = parser.parse_args()

def save_img_from_videos(PATH,DEST,split,fps_thresh=30,num_of_frames=10):
  split_path = os.path.join(PATH,split)
  dest_split_path = os.path.join(DEST,'processed_images',split)
  training_videos_folders = ["0","1"]
  skipped_vids = []
  for folder in training_videos_folders:
      vid_path = os.path.join(split_path,folder)
      videos_path = [i for i in os.listdir(vid_path) if "mp4" in i]
      vid_num = 0
      print("Reading Videos from " , vid_path)
      print("No of videos present " , len(videos_path))

      for video_path in tqdm(videos_path):
          vids = os.path.join(vid_path,video_path)
          vid = video_path.split("/")[-1].split(".")[0]     
          vid_num+=1
          cap = cv2.VideoCapture(vids)
          # print(vids)
          # print('Captured',cap)
          counter = 0
          TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

          if TOTAL_FRAMES < fps_thresh:
            print('Skipping',TOTAL_FRAMES,vid)
            skipped_vids.append(vid)

          else:
            if not os.path.exists(os.path.join(dest_split_path,folder)):
              #print(os.path.join(dest_split_path,folder,vid))
              os.makedirs(os.path.join(dest_split_path,folder))
              
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                filename = (os.path.join(dest_split_path,folder,vid+'_image_'+str(counter))+".jpg")
                # print(filename)
                cv2.imwrite(filename, frame)
                counter+=1
                if counter >num_of_frames:
                    cap.release()
                    continue
  print("processing done...")

if __name__ == "__main__":
    save_img_from_videos(args.input_path,args.output_path,"train",30,args.num_of_frames)
    save_img_from_videos(args.input_path,args.output_path,"test",30)
