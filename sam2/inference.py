import time
import os
import numpy as np
from pathlib import Path
import torch
import cv2
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
from PIL import Image
import argparse


device = torch.device("cuda")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

model_checkoint = "../checkpoints/checkpoint_150.pt"
model_config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
vid_predictor = build_sam2_video_predictor(model_config, model_checkoint, device=device)
sam2_model = build_sam2(model_config, model_checkoint, device="cuda")
img_predictor = SAM2ImagePredictor(sam2_model)

def extract_frames(video_path, frames_path):
    # Create output directory based on video filename
    os.makedirs(frames_path, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no frames left

        frame_count += 1
        frame_filename = os.path.join(frames_path, f"{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)

    cap.release()
    print(f"Extracted {frame_count} frames to '{frames_path}'.")


def best_image_to_video_validation(video_dir, replay_vidname, background_vidname):

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # setup images
    inference_state = vid_predictor.init_state(video_path=video_dir)

    # take a look the first video frame
    num_frames = len(frame_names)
    weights = [0.5 + 0 * abs(i - (num_frames / 2)) / (num_frames / 2) for i in range(num_frames)]
    frame_diffs = []

    for i in range(1, len(frame_names)):
        prev_frame = np.array(Image.open(os.path.join(video_dir, frame_names[i-1]))).astype(np.float32)
        curr_frame = np.array(Image.open(os.path.join(video_dir, frame_names[i]))).astype(np.float32)
        diff = np.sum(np.abs(curr_frame - prev_frame) > 50) # Count pixels with differences greater than threshold
        weighted_diff = diff * weights[i]
        frame_diffs.append(weighted_diff)
        # Find the frame with the maximum weighted difference
    ann_frame_idx = np.argmax(frame_diffs) + 1

    # Promptless mask generation on first frame

    img_predictor.set_image(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))) # apply SAM image encoder to the image

    # prompt encoding

    #mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
    sparse_embeddings, dense_embeddings = img_predictor.model.sam_prompt_encoder(points=None,boxes=None,masks=None,)

    # mask decoder
    batched_mode = False  #unnorm_coords.shape[0] > 1 # multi object prediction
    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in img_predictor._features["high_res_feats"]]
    low_res_masks, prd_scores, _, _ = img_predictor.model.sam_mask_decoder(image_embeddings=img_predictor._features["image_embed"][-1].unsqueeze(0),image_pe=img_predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features,)
    prd_masks = img_predictor._transforms.postprocess_masks(low_res_masks, img_predictor._orig_hw[-1])# Upscale the masks to the original im
    prd_mask = torch.sigmoid(prd_masks[:, 0])

    # Clean the mask:
    cleaned_mask = prd_mask.detach().cpu().squeeze()

    clean_squash = cleaned_mask / cleaned_mask.max()
    cleaned_mask = torch.where(clean_squash >= 0.5, 1, torch.tensor(0.0))

    # Add new mask for first frame(0 - first frame, 1 - one object)
    frame_idx, obj_ids, video_res_masks = vid_predictor.add_new_mask(inference_state, ann_frame_idx, 1, cleaned_mask)

    # Composite video
    composite_frames = {}
    background_frames = {}

    video_segments = {}  # video_segments contains the per-frame segmentation results
    
    for out_frame_idx, out_obj_ids, out_mask_logits in vid_predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        assert len(video_segments[out_frame_idx].items()) == 1

        out_obj_id, out_mask = list(video_segments[out_frame_idx].items())[0]
        source_path = os.path.join(video_dir, frame_names[out_frame_idx])
        source_image = Image.open(source_path).convert('RGB')
        white_background = Image.new('RGB', source_image.size, (255, 255, 255, 255))

        # Apply the mask to the source image
        masked_image = Image.composite(source_image, white_background, Image.fromarray(out_mask.squeeze()))
        background_image = Image.composite(source_image, white_background, Image.fromarray(~out_mask.squeeze()))
        composite_frames[out_frame_idx] = np.array(masked_image.convert("RGB"))
        background_frames[out_frame_idx] = np.array(background_image.convert("RGB"))


    for out_frame_idx, out_obj_ids, out_mask_logits in vid_predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=False):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        assert len(video_segments[out_frame_idx].items()) == 1

        out_obj_id, out_mask = list(video_segments[out_frame_idx].items())[0]
        source_path = os.path.join(video_dir, frame_names[out_frame_idx])
        source_image = Image.open(source_path).convert('RGB')
        white_background = Image.new('RGB', source_image.size, (255, 255, 255, 255))

        # Apply the mask to the source image
        masked_image = Image.composite(source_image, white_background, Image.fromarray(out_mask.squeeze()))
        background_image = Image.composite(source_image, white_background, Image.fromarray(~out_mask.squeeze()))
        composite_frames[out_frame_idx] = np.array(masked_image.convert("RGB"))
        background_frames[out_frame_idx] = np.array(background_image.convert("RGB"))

    composite_frames = sorted(composite_frames.items(), key=lambda x: x[0])
    background_frames = sorted(background_frames.items(), key=lambda x: x[0])

    print("LEN BACKGROUND FRAMES", len(background_frames))
    size = composite_frames[0][1].shape[1], composite_frames[0][1].shape[0]
    fps = 10
    composite_out = cv2.VideoWriter(replay_vidname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[0], size[1]), True)
    background_out = cv2.VideoWriter(background_vidname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[0], size[1]), True)


    for i in range(len(composite_frames)):
        data = composite_frames[i][1].astype('uint8')
        rgb_image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        composite_out.write(rgb_image)

        data = background_frames[i][1].astype('uint8')
        rgb_image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        background_out.write(rgb_image)


    composite_out.release()
    background_out.release()

    return {
        "video_segments" : video_segments,
        "composites" : composite_frames,
    }

def convert_image(input_path, output_path):
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image {input_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a binary mask where white pixels become black and non-white become white
    _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(output_path, binary)
def process_directory(input_dir, output_dir):
    """Process all images in the input directory and save converted images to output directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]

    print(f"Found {len(image_files)} images to process")
    
    for i, image_file in enumerate(sorted(image_files)):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, f"{Path(image_file).stem}_converted.jpg")
        
        convert_image(input_path, output_path)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")



def run_inference(start_episode, end_episode, directory):


    for episode in range(start_episode, end_episode):
        episode_path = os.path.join(directory, str(episode))
        os.makedirs(episode_path, exist_ok=True)

        print(f"Processing episode {episode} on device {device}")
        replay_video_path = os.path.join(episode_path, f"trajectory_replay.mp4")
        replay_frames_path = os.path.join(episode_path, f"trajectory_replay_frames")
        background_path = os.path.join(episode_path, f"trajectory_background.mp4")
        mask_path = os.path.join(episode_path, f"mask_frames")
        best_image_to_video_validation(os.path.join((directory), f"{episode}/frames"),replay_video_path,background_path)
        extract_frames(video_path=replay_video_path, frames_path=replay_frames_path)
        process_directory(replay_frames_path, mask_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on episodes')
    parser.add_argument('--start', type=int, required=True, help='Starting episode number')
    parser.add_argument('--end', type=int, required=True, help='Ending episode number')
    parser.add_argument('--directory', type=str, required=True, help='Input directory')

    
    args = parser.parse_args()
    
    #print(f"Running inference on dataset {args.dataset} for episodes {args.start} to {args.end}")
    run_inference(args.start, args.end, args.directory)
