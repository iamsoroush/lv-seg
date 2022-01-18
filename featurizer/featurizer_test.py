import numpy as np
import cv2
import os
import sys
import pytest
from featurizer import *
from featurizer_run_script import load_model, load_seg_maps

# testing class for the featurizer with pytest framework
class TestClass:        

    @pytest.fixture
    def setup_dirs(self):
        main_dir = './tests/'

        # csv file path
        csv_file_path = os.path.join(main_dir, 'small_test_filelist.csv')

        # creating a test video
        video_dir = os.path.join(main_dir, 'test_echo_vidoes/small_test_df/')
        
        # creating a test model
        model_path = '../../../../echoC_Codes/Experiments/exportmodel/'

        # creating a test output directory
        output_dir = './tests/featurizer_tests/'
        os.makedirs(name=output_dir, exist_ok=True)

        dirs = {
            'csv_file_path': csv_file_path,
            'video_dir': video_dir,
            'model_path': model_path,
            'output_dir': output_dir
        }
        return dirs

    @pytest.fixture
    def model(self, setup_dirs):
        # loading the model (tf.SavedModel)
        return load_model(setup_dirs['model_path'])

    def test(self, setup_dirs):
        print(setup_dirs)

    @pytest.mark.parametrize("handling_file_existance", [
        ('ignore'),
    ])
    def test_featurize(self, setup_dirs, model, handling_file_existance):

        featurizer = EchoNetFeaturizer(csv_file_path=setup_dirs['csv_file_path'], 
                                model=model, 
                                video_dir=setup_dirs['video_dir'],
                                target_add=setup_dirs['output_dir'],
                                handling_file_existance= handling_file_existance)

        featurizer.featurize()

        df = pd.read_csv(setup_dirs['csv_file_path'])
        target_df = pd.read_csv(setup_dirs['output_dir'] + 'featurizer_FileList.csv')

        # dirs
        featurized_videos_dir = setup_dirs['output_dir']+'featurized_videos/'

        print(target_df)
        print(set(df['FileName']))

        # checking if the featurizer created the correct number and name of files
        assert set(os.listdir(featurized_videos_dir)) == set(df.FileName)

        #checking if the featurizer created the correct number of frames
        for each_file in os.listdir(featurized_videos_dir):
            number_of_frames = df.loc[df['FileName'] == each_file, 'NumberOfFrames'].values[0]
            assert len(os.listdir(os.path.join(featurized_videos_dir, each_file))) == number_of_frames

            loaded_features = load_seg_maps(os.path.join(featurized_videos_dir, each_file))
            assert len(loaded_features) == number_of_frames
            
            for each_frame in loaded_features:
                assert each_frame.shape == (112, 112)

                assert each_frame.dtype == np.uint8

                values = set(np.unique(each_frame))
                assert len(values) == 2 or len(values) == 1

                assert values.issubset({0, 255})
            

