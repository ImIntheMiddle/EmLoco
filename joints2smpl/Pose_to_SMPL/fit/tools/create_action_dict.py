import sys
import os
import argparse
import json
import tqdm

SPLIT = {'train': ['bytes-cafe-2019-02-07_0', 'gates-basement-elevators-2019-01-17_1', 'hewlett-packard-intersection-2019-01-24_0', 'huang-lane-2019-02-12_0', 'jordan-hall-2019-04-22_0', 'packard-poster-session-2019-03-20_2', 'stlc-111-2019-04-19_0', 'svl-meeting-gates-2-2019-04-08_0', 'svl-meeting-gates-2-2019-04-08_1', 'tressider-2019-03-16_1'], 'val': ['gates-ai-lab-2019-02-08_0'], 'test': ['packard-poster-session-2019-03-20_1', 'tressider-2019-03-16_0']}

ACTION = ["walking", "standing", "running", "going upstairs", "going downstairs"]
# ACTION = ["running", "going upstairs", "going downstairs"]

def main(opt):
    # construct the action directory
    action_dir = opt.action_dir
    action_dict = {}
    action_list = []
    bar_split = tqdm.tqdm(SPLIT.keys(), leave=True)
    # bar_split = tqdm.tqdm(["val"], leave=True)
    for split in bar_split:
      processed_samples = 0
      added_samples = 0
      bar_split.set_description(f"Processing {split}: Processed {processed_samples} samples")
      action_dict[split] = {}
      bar_scene = tqdm.tqdm(SPLIT[split], leave=False)
      for scene in bar_scene:
          bar_scene.set_description(f"Processing {scene}...")
          scene_path = os.path.join(action_dir, f"{scene}.json")
          scene_data = json.load(open(scene_path))
          action_dict[split][scene] = {}
          bar_frame = tqdm.tqdm(scene_data["labels"].keys(), leave=False)
          for frame in bar_frame:
              # import pdb; pdb.set_trace()
              frame_id = int(frame.split(".jpg")[0])
              action_dict[split][scene][frame_id] = {}
              for pedestrian in scene_data["labels"][frame]:
                  pedestrian_id = int(pedestrian["label_id"].split(":")[-1])
                  for action in pedestrian["action_label"].keys():
                      if action not in action_list:
                          action_list.append(action)
                      if (action in ACTION) and (pedestrian["action_label"][action] in [1, 2, 3]): # confidence level is 1 or 2
                          action_dict[split][scene][frame_id][pedestrian_id] = action
                          added_samples += 1
                  processed_samples += 1
              bar_split.set_description(f"Processing {split}: Processed {processed_samples} samples")
      print(f"\n Added {added_samples} samples out of {processed_samples} samples in {split} split.")
    print(f"Action list: {action_list}")
    # save the action dictionary
    with open("action_dict.json", "w") as f:
        json.dump(action_dict, f)
        print("Action dictionary saved to action_dict.json!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_dir", type=str, default='data/jrdb_2022/train_dataset_with_activity/labels/labels_2d_stitched', help="Directory of the JRDB-Act dataset.")
    opt = parser.parse_args()
    main(opt)