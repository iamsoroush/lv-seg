
import numpy as np
import os
import cv2
import tensorflow as tf
from featurizer import EchoNetFeaturizer


def load_model(model_path):
    """
    loads the model
    Args:
        model_path (str): model path
    Returns:
        model (tf.SavedModel): model
    """
    # loading the model
    model = tf.saved_model.load(model_path)
    return model

def load_seg_maps(file_name):
    """
    loads the npz file
    Args:
        file_name (str): npz files directory
    Returns:
        seg_maps (list): segmentation maps list
    """
    frame_files = os.listdir(file_name)
    frame_files.sort(key=lambda file_name: int(file_name.split('.')[0].split('_')[-1]))
    seg_maps = []
    for each_file in frame_files:
        if each_file.endswith('.npz'):
            npz_file = np.load(os.path.join(file_name, each_file))
            seg_maps.append((npz_file['seg_map']*255).astype(np.uint8))

    return seg_maps


if __name__ == '__main__':
    # loading the data
    print(os.path.abspath(os.curdir))
    main_video_dir = '../Videos/'
    target_video_dir = '../'

    # loading the model
    model_path = 'D:/AIMedic/FinalProject_echocardiogram/echoC_Codes/Experiments/exportmodel/'
    model = load_model(model_path)

    featurizer = EchoNetFeaturizer(csv_file_path='../FileList.csv',
                                    model = model,
                                    video_dir=main_video_dir,
                                    target_add=target_video_dir,
                                    handling_file_existance='ignore')

    featurizer.featurize()

    '''    
    # load_data = load_seg_maps(target_path)
    # print(len(load_data))

    # for i, video_frames in enumerate(processed_videos_list[0]):
    #     if not np.array_equal(load_data[i], video_frames*255):
    #         print('load_data[i].shape', load_data[i].shape)
    #         print('type(load_data[i])', type(load_data[i]))
    #         print('video_frames.shape', video_frames.shape)
    #         print('type(video_frames)', type(video_frames))
    #         print(load_data[i].shape)
    #         print(type(load_data[i]))
    #         print(load_data[i].dtype)
    #         print(load_data[i].max())
    #         print('NOT EQUAL'+ str(i))
    #     else:
    #         print(load_data[i].shape)
    #         print(type(load_data[i]))
    #         print(load_data[i].dtype)
    #         print(load_data[i].max())
    '''