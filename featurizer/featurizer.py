import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import time
from tqdm import tqdm
import shutil
import warnings


class EchoNetFeaturizer:
    """
    EchoNet featurizer class
    """

    def __init__(self, csv_file_path, model, video_dir='../', target_add='.', 
                do_preprocess=True, do_postprocess=True, handling_file_existance='overwrite'):
        """
        initializes the class
        Args:
            csv_file_path (str): csv file path
            model : segmentation model
            target_add (str): target directory address
            target_shape (tuple): target shape
        """

        self.video_dir = video_dir
        self.csv_file_path = csv_file_path
        self.model = model
        self.target_add = target_add
        self.do_preprocess = do_preprocess
        self.do_postprocess = do_postprocess
        self.handling_file_existance = handling_file_existance

    def get_main_dataframe(self):
        """
        returns the main dataframe
        """
        return pd.read_csv(self.csv_file_path)

    def featurize(self):
        """
        featurizes the videos
        """

        # direcotry to save the featurized videos
        assert os.path.exists(self.target_add), 'target directory does not exist'

        # getting the main dataframe
        df = self.get_main_dataframe()
        video_file_names = np.array(df.FileName)
        
        # loading/creating dataframes for the processed videos
        if not os.path.isfile(os.path.join(self.target_add, 'featurizer_FileList.csv')):
            target_dataframe = df
            target_dataframe['Features'] = ''
        else:
            target_dataframe = pd.read_csv(os.path.join(self.target_add, 'featurizer_FileList.csv'))

        # processing the loaded videos
        processed_videos_list = []
        start = time.time()
        for video_filename in tqdm(video_file_names[:2000], leave=True):

            # getting the video frames
            video_path = os.path.join(self.video_dir, video_filename + '.avi')
            video_frames, video_info = self.load_echo_video(video_path, to_gray=True)

            # adding the segmentation maps dir info to the dataframe
            video_features_dir = os.path.join(self.target_add, 'featurized_videos')
            target_path = os.path.join(video_features_dir, video_filename)
            target_dataframe.loc[target_dataframe['FileName']==video_filename , 'Features'] = target_path

            # check if the folder exists and not empty
            if os.path.isdir(target_path) and os.listdir(target_path) != []:
                warnings.warn(f'{video_filename}: The folder already exists and not empty')

                if self.handling_file_existance == 'manual':
                    answer = input(f'ignore/overwrite the {video_filename} (i/o)? ')
                    self.handling_file_existance = 'ignore' if answer=='i' else 'overwrite'
                
                if self.handling_file_existance == 'ignore':
                    continue
                elif self.handling_file_existance == 'overwrite':
                    shutil.rmtree(target_path)
                    os.mkdir(target_path)
                else:
                    raise ValueError(f'{self.handling_file_existance}: is not a valid value for handling_file_existance')
            else:
                os.makedirs(target_path, exist_ok=True)

            processed_video_frames = []
            for i, frame in enumerate(tqdm(video_frames, desc=video_filename, leave=False)):
                # getting the segmentation map
                prediction = self.process(frame)

                # saving the processed frame
                prediction = prediction.astype(bool)
                self.save_seg_map(prediction, 'frame_' + str(i), target_path)

                processed_video_frames.append(prediction)
            
            # updating the dataframe
            target_dataframe.to_csv(os.path.join(self.target_add, 'featurizer_FileList.csv'), index=False)

    def process(self, image, to_gray=False):
        """
        processes the video
        Args:
            image (np.arrays): image as frame of a video to process
            to_gray (bool): makes the frames gray-scale if needed
        Returns:
            seg_map (np.array): segmentation map array
        """

        # preprocessing the frame (if needed)
        if self.do_preprocess:
            image = self.preprocess_img(image)
        
        # getting the prediction
        prediction = self.model(image)

        # postprocessing the prediction
        if self.do_postprocess:
            prediction = self.postprocess_img(prediction, target_shape=self.original_image_shape)

        return prediction
    
    # (optional)
    def preprocess_img(self, image):
        """
        preprocesses the image for the unet baseline(for now)
        Args:
            img (np.array): image array
        Returns:
            img (np.array): preprocessed image array
        """
        self.original_image_shape = image.shape

        target_size = (128, 128)
        preprocessed_image = np.array(tf.image.resize(image[:,:,tf.newaxis],
                                                    target_size,
                                                    antialias=False,
                                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
    
        preprocessed_image=tf.convert_to_tensor(preprocessed_image, dtype=tf.float32)
        preprocessed_image=tf.expand_dims(preprocessed_image,axis=0)
        preprocessed_image = preprocessed_image / 255.0

        return preprocessed_image

    # (optional)
    @staticmethod
    def postprocess_img(image, target_shape=(112, 112)):
        """
        postprocesses the image
        Args:
            img (np.array): image array
        Returns:
            img (np.array): postprocessed image array
        """
        # converting image to numpy array
        postprocessed_image = image[0,:,:,:].numpy()

        # thresholding the segmentation map
        postprocessed_image = (postprocessed_image > 0.5).astype(int)

        # for the images with the type int, have to multiply it with 255
        if 'int' in str(postprocessed_image.dtype) and np.amax(postprocessed_image) <= 1:
            postprocessed_image = (postprocessed_image[:, :, 0] * 255).astype('uint8')
        else:
            postprocessed_image = postprocessed_image[:, :, 0].astype('uint8')

        # resizing the image to the original size
        postprocessed_image = cv2.resize(postprocessed_image, target_shape)

        return postprocessed_image
  
    def load_echo_video(self, video_dir, to_gray=False):
        """
        loads the video from the directory given
        Args:
            video_dir (str): video address directory
            to_gray (bool): makes the frames gray-scale if needed
        Returns:
            video_frames (np.array): video frames array
            video_info (list): list of video information ( frame count, fps, duration )
        """

        # a list to be used for storing the frames
        video_frames = []

        # capturing the video with OpenCV lib.
        vidcap = cv2.VideoCapture(video_dir)

        frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps

        # counting the frames
        count = 0
        while vidcap.isOpened():
            success, frame = vidcap.read()
            if success:
                if to_gray:
                    # converting the frames from rgb to gray-scale ( just for compression )
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # frame_label = self.process(frame)
                video_frames.append(frame)
                # video_labels.append(frame_label)
            else:
                break
            count += 1

        # releasing the video capture object
        vidcap.release()

        # validating the frame count
        if count != frame_count:
            frame_count = count

        video_info = [frame_count, fps, duration]
        return np.array(video_frames), video_info
    
    @staticmethod
    def save_seg_map(seg_map, file_name, target_add='.'):
        """
        saves the segmentation map in the target directory
        Args:
            seg_map (np.array): segmentation map array
            file_name (str): file name for the new segmentation map
            target_add (str): target directory address
            
        Returns:
            saved_seg_map (str): saved segmentation map directory
        """

        # saving the seg_map
        seg_map_add = os.path.join(target_add, file_name)
        np.savez_compressed(seg_map_add, seg_map = seg_map)
        return seg_map_add