import argparse
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

#TODO: USE PROCESSING STEP FROM ROVI-AUG INSTEAD
def process_step_toto(step):
    arm_joints = step['observation']['state']
    gripper_joints = tf.cond(
        step['action']['open_gripper'],
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.02, 0.02], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints, gripper_joints], axis=0), image

def process_step_nyu(step):
    arm_joints = step['observation']['state']
    binary_gripper = step['action'][13]
    bg = tf.cond(
        binary_gripper > 0,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints[:7], bg], axis = 0), image

def process_step_berkeley_ur5(step):
    # Need to output [j0 -> j5, 0, 0, 0, 0, 0, 0, 0, 0]
    arm_joints = step['observation']['robot_state'][:14] # First 6 are joints
    is_gripper_closed = step['observation']['robot_state'][-2]
    is_gripper_closed = tf.cast(is_gripper_closed > 0.5, tf.bool)  # Assuming threshold 0.5 for True/False
    is_gripper_closed = tf.cond(is_gripper_closed, 
                                lambda: tf.constant([1.0, 0.025, 0.80, -0.80, 1.0, 0.0252768, 0.80, -0.80], dtype=tf.float32), 
                                lambda: tf.constant([0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32))
    # zeros = tf.stack([is_gripper_closed, 0, 0, 0, is_gripper_closed, 0, 0, 0])
    joints = tf.concat([arm_joints[:6], is_gripper_closed], axis = 0)

    # zeros = tf.constant([0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
    # is_gripper_closed = step['observation']['robot_state'][-2]
    # bg = tf.cond(
    #     is_gripper_closed < 0,
    #     lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
    #     lambda: tf.constant([0.025, 0.025], dtype=tf.float32),
    # )
    image = step['observation']['image'] # Extract the image from the dataset
    return joints, image

def process_step_ucsd_kitchen(step):
    arm_joints = step['observation']['state'][:7] # First seven joints
    gripper_open = step["action"][6]
    is_gripper_closed = tf.cast(gripper_open < 0.5, tf.bool)
    grip_control = tf.cond(
        is_gripper_closed,
        lambda: tf.constant([0.85, 0.844, 0.853, 0.85, 0.844, 0.853], dtype=tf.float32),
        lambda: tf.constant([0.185,0.185,0.188,0.185,0.185, 0.188], dtype=tf.float32)
    )

    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints, grip_control], axis = 0), image

def process_step_utokyo_xarm_pick_place(step):
    arm_joints = step['observation']['joint_state'][:7] # First seven joints
    is_gripper_closed = tf.cast(step["action"][-1] > 0.5, tf.bool)
    grip_control = tf.cond(
        is_gripper_closed,
        lambda: tf.constant([0.85, 0.844, 0.853, 0.85, 0.844, 0.853], dtype=tf.float32),
        lambda: tf.constant([0, 0, 0, 0, 0, 0], dtype=tf.float32)
    )

    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints, grip_control], axis = 0), image

def process_step_ucsd_pick_and_place(step):
    arm_joints = step['observation']['state'] # First seven joints
    is_gripper_closed = tf.cast(step["action"][-1] < 0, tf.bool)
    grip_control = tf.cond(
        is_gripper_closed,
        lambda: tf.constant([1.0,  0.75,  0.9,  1.0,  0.3, 1], dtype=tf.float32),
        lambda: tf.constant([0, 0, 0, 0, 0, 0], dtype=tf.float32)
    )

    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints, grip_control], axis = 0), image


def process_step_asu_table_top(step):
    arm_joints = step['observation']['state'][:6] # First six joints
    is_gripper_closed = tf.cast(step['observation']['state'][-1] > 0.2, tf.bool)
    is_gripper_closed = tf.cond(is_gripper_closed, 
                                lambda: tf.constant([0.498, 0.00155, 0.5, -0.482, 0.498,0.00154, 0.5,-0.482], dtype=tf.float32), 
                                lambda: tf.constant([0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32))

    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints, is_gripper_closed], axis = 0), image

def process_step_kaist_nonprehensile(step):
    arm_joints = step['observation']['state'][:14:2] # First six joints
    is_gripper_open = tf.cast(False, tf.bool) # Gripper does not open
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints, gripper_joints], axis = 0), image

def process_step_cmu_play_fusion(step):
    arm_joints = step['observation']['state'] # franka 
    gripper_dist = arm_joints[-1] 
    is_gripper_open = tf.cast(gripper_dist > 0.04, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints[:7], gripper_joints], axis = 0), image

def process_step_austin_buds(step):
    arm_joints = step['observation']['state'][:7] # franka 
    gripper_pos = step["observation"]["state"][7]
    is_gripper_open = tf.cast(gripper_pos > 0.04, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints[:7], gripper_joints], axis = 0), image

def process_step_austin_sailor(step):
    arm_joints = step['observation']['state_joint'] # franka 
    gripper_pos = step["observation"]["state"][-1]
    is_gripper_open = tf.cast(gripper_pos > 0.04, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints[:7], gripper_joints], axis = 0), image, gripper_pos

def process_step_austin_mutex(step):
    arm_joints = step['observation']['state'][:7] # franka 
    gripper_pos = step["observation"]["state"][7:8]
    is_gripper_open = tf.cast(gripper_pos > 0.05, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints[:7], gripper_joints], axis = 0), image

def process_step_austin_sirius(step):
    arm_joints = step['observation']['state'][:7] # franka 
    gripper_pos = step["observation"]["state"][7:8]
    is_gripper_open = tf.cast(gripper_pos > 0.05, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    image = step['observation']['image'] # Extract the image from the dataset
    return tf.concat([arm_joints[:7], gripper_joints], axis = 0), image

def process_step_viola(step):
    arm_joints = step['observation']['joint_states']  # franka
    gripper_width = step['observation']['gripper_states'][0]  
    is_gripper_open = tf.cast(gripper_width > 0.04, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    full_joint_state = tf.concat([arm_joints, gripper_joints], axis=0)
    image = step['observation']['agentview_rgb']
    #agentview_rgb: a fixed camera showing the workspace (like a third-person view)
    #eye_in_hand_rgb: a camera mounted on the robot arm (first-person)
    return full_joint_state, image

def process_step_taco_play(step):
    robot_obs = step['observation']['robot_obs']  # franka
    joint_positions = robot_obs[7:14]  
    gripper_width = robot_obs[2]       
    is_gripper_open = tf.cast(gripper_width > 0.04, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    full_joint_state = tf.concat([joint_positions, gripper_joints], axis=0)
    image = step['observation']['rgb_static'] #swap rgb_static with rgb_gripper if want eye-in-hand camera
    return full_joint_state, image

def process_step_iamlab_cmu_pickup_insert(step):
    state = step['observation']['state']  
    arm_joints = state[:7]                
    gripper_width = state[-1]             
    is_gripper_open = tf.cast(gripper_width > 0.04, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    full_joint_state = tf.concat([arm_joints, gripper_joints], axis=0)
    image = step['observation']['image']  
    return full_joint_state, image

def process_step_bridge(step):
    arm_joints = step['observation']['state']
    image = step['observation']['image'] # Extract the image from the dataset

    return tf.concat([arm_joints, arm_joints[-1:]], axis = 0), image

def process_step_stanford_hydra(step):
    arm_joints = step['observation']['state'][10:17]
    image = step['observation']['image'] # Extract the image from the dataset

    return tf.concat([arm_joints, arm_joints[-1:]], axis = 0), image

def process_step_stanford_hydra(step):
    arm_joints = step['observation']['state']
    image = step['observation']['image'] # Extract the image from the dataset
    gripper_pos = step["observation"]["state"][-1]
    is_gripper_open = tf.cast(gripper_pos > 0.5, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    return tf.concat([arm_joints[:7], gripper_joints], axis = 0), image

def process_step_stanford_hydra(step):
    arm_joints = step['observation']['joint_pos']
    image = step['observation']['image_side_1'] # Extract the image from the dataset
    gripper_pos = step["observation"]["state_gripper_pose"]
    is_gripper_open = tf.cast(gripper_pos > 0.5, tf.bool)
    gripper_joints = tf.cond(
        is_gripper_open,
        lambda: tf.constant([0.05, 0.05], dtype=tf.float32),
        lambda: tf.constant([0.0, 0.0], dtype=tf.float32),
    )
    return tf.concat([arm_joints[:7], gripper_joints], axis = 0), image

dataset_to_processing_function = {
    'toto': process_step_toto,
    'nyu_franka_play_dataset_converted_externally_to_rlds': process_step_nyu,
    'berkeley_autolab_ur5': process_step_berkeley_ur5,
    'ucsd_kitchen_dataset_converted_externally_to_rlds': process_step_ucsd_kitchen,
    'utokyo_xarm_pick_and_place_converted_externally_to_rlds': process_step_utokyo_xarm_pick_place,
    'kaist_nonprehensile_converted_externally_to_rlds': process_step_kaist_nonprehensile,
    'asu_table_top_converted_externally_to_rlds': process_step_asu_table_top,
    'austin_buds_dataset_converted_externally_to_rlds': process_step_austin_buds,
    'utaustin_mutex': process_step_austin_mutex,
    'austin_sailor_dataset_converted_externally_to_rlds': process_step_austin_sailor,
    'bridge': process_step_bridge,
    #'fractal20220817_data': process_step_fractal,
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds': process_step_iamlab_cmu_pickup_insert,
    'viola': process_step_viola,
    'taco_play': process_step_taco_play,
}


def load_dataset(dataset_name, start_episode, end_episode, directory):
    def extract(episode):
        return episode['steps'].map(lambda step: dataset_to_processing_function[dataset_name](step))
    
    builder = tfds.builder_from_directory(builder_dir=f'gs://gresearch/robotics/{dataset_name}/0.1.0/')
    read_config = tfds.ReadConfig(
    interleave_cycle_length=1,     # read shards one by one
    shuffle_seed=None,             # no file shuffling
    )
    ds = builder.as_dataset(
        split=f"train[{start_episode}:{end_episode+1}]",
        shuffle_files=False,
        read_config=read_config,
    )

    processed_data = ds.map(extract, num_parallel_calls=tf.data.AUTOTUNE)

    robot_states, images = [], []
    for episode in processed_data:
        episode_states, episode_images = [], []
        for state, image in episode:
            episode_states.append(state)
            episode_images.append(image)
        robot_states.append(episode_states)
        images.append(episode_images)

    results_folder = os.path.join(directory, dataset_name)
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    

    # Iterate over each episode
    for episode_idx, frames in enumerate(images, start=start_episode):
        # Create a subdirectory for the episode
        episode_dir = os.path.join(results_folder, f"{episode_idx}")
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)
        frames_folder = os.path.join(episode_dir, "frames")
        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder)

        for frame_idx, frame in enumerate(frames):
            filename = os.path.join(frames_folder, f"{frame_idx}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB))

    print("All frames saved successfully!")


def main():
    parser = argparse.ArgumentParser(description='Load dataset from Google Research Robotics dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to load')
    parser.add_argument('--start', type=int, required=True, help='Starting episode number')
    parser.add_argument('--end', type=int, required=True, help='Ending episode number')
    parser.add_argument('--save_dir', type=str, required=True, help='Base directory for saving the dataset')
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    print(f"Loading dataset {args.dataset} episodes {args.start} to {args.end}")
    for i in range (args.start, args.end+1):
        load_dataset(args.dataset, i, i+1, args.save_dir)

if __name__ == "__main__":
    main()