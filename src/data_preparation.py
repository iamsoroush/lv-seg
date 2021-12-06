import os
import cv2
import numpy as np
import pandas as pd
import skimage.draw
import skimage.color
import pydicom as dicom
from pydicom.datadict import add_dict_entry


class ExtractDicom:
    """

    This class contains implementations of some tools for converting data with DICOM format to AVI

    """

    def __init__(self, src_base, dst_base, features):
        """

        Initialize the required variables

        Args:
            src_base: Path of the folder which contains DICOM files
            dst_base: Path of the folder that converted AVI videos will be saved in it
            features: List of features stored in DICOM file
        """

        self.src_base_addr = src_base
        self.dst_base_addr = dst_base
        self.df_volume = pd.read_csv(os.path.join(src_base, 'VolumeTracings.csv'))
        # self.features = ['FileName',
        #                  'EF',
        #                  'ESV',
        #                  'EDV',
        #                  'FrameHeight',
        #                  'FrameWidth',
        #                  'FPS',
        #                  'NumberOfFrames',
        #                  'Split']
        self.features = features
        if not os.path.isdir(self.dst_base_addr):
            os.makedirs(self.dst_base_addr)

    def register_data_features(self):

        """

        In this method, the address (tag) assigned for each feature of data frame

        """

        i = 0
        # self.echo_data_info = {}
        for k in self.features:
            print(k)
            v = 0x10021001 + i
            char_hex = hex(v)
            # self.echo_data_info[k] = [int('0x' + char_hex[2:6], 16), int('0x' + char_hex[6:], 16)]
            if k in ['FileName', 'Split']:
                add_dict_entry(v, "CS", k, k)
            else:
                if k == 'NumberOfFrames':
                    k = 'NOF'
                add_dict_entry(v, "DS", k, k)
            i += 1

    def convert_dicom_to_avi(self, src_addr, dst_file_name, csv_addr):

        """

        Conversion of data file with DICOM format to AVI could be performed by this method

        Args:
            src_addr: Path of the DICOM file
            dst_file_name: Name of the file that wanted to be converted
            csv_addr: Path of the csv file that the feature of the DICOM file will be saved in that

        Returns:

        """

        defaults = ['Samples per Pixel', 'Photometric Interpretation', 'Rows',
                    'Columns', 'Bits Allocated', 'Bits Stored', 'Pixel Representation',
                    'Pixel Data', 'NOF']

        ds = dicom.dcmread(src_addr)
        data = {}
        file_name = ds.FileName
        ed_frame, es_frame = None, None
        if file_name + '.avi' in self.df_volume['FileName'].unique():
            ed_frame, es_frame = self.df_volume.loc[self.df_volume['FileName'] == file_name + '.avi', ['Frame']][
                'Frame'].unique()
        data['ED_Frame'] = ed_frame
        data['ES_Frame'] = es_frame

        subset = ds.Split.lower()
        data['image_address_ed'] = f'{self.dst_base_addr}/{subset}/seg_image/{file_name}_image_ed.jpg'
        data['image_address_es'] = f'{self.dst_base_addr}/{subset}/seg_image/{file_name}_image_es.jpg'
        data['label_address_ed'] = f'{self.dst_base_addr}/{subset}/seg_image/{file_name}_label_ed.jpg'
        data['label_address_es'] = f'{self.dst_base_addr}/{subset}/seg_image/{file_name}_label_es.jpg'

        if not os.path.isfile(csv_addr):
            for k in ds.keys():
                key = ds[k].name
                if key not in defaults:
                    data[key] = [ds[k].value]

            df = pd.DataFrame(data)
            df.to_csv(csv_addr, index=False)

        else:
            df = pd.read_csv(csv_addr)
            if ds.FileName not in list(df.FileName):
                for k in ds.keys():
                    key = ds[k].name
                    if key not in defaults:
                        data[key] = ds[k].value

                df = df.append(data, ignore_index=True)

                df.to_csv(csv_addr, index=False)

        pixel_array_numpy = ds.pixel_array

        width, height = pixel_array_numpy.shape[1:]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        sub_folder = f'{ds.Split.lower()}/Videos'
        sub_folder_path = os.path.join(sub_folder, dst_file_name)
        if not os.path.isdir(os.path.join(self.dst_base_addr, sub_folder)):
            os.makedirs(os.path.join(self.dst_base_addr, sub_folder))

        dst_path = os.path.join(self.dst_base_addr, sub_folder_path)
        video = cv2.VideoWriter(dst_path, fourcc, 1, (width, height))

        for p in pixel_array_numpy:
            frame = cv2.cvtColor(p.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            video.write(frame)

        cv2.destroyAllWindows()
        video.release()

    def convert_all_to_avi(self):

        """

        All DICOM videos existed in the specific address could be converted to the AVI file by using this method

        Returns:

        """

        self.register_data_features()
        src = os.path.join(self.src_base_addr, 'Videos')
        files = os.listdir(src)
        csv_addr = os.path.join(self.dst_base_addr, 'FileList.csv')
        for f in files:
            src_path = os.path.join(src, f)

            # dst_addr = os.path.join(dst_video_path, f.replace('dcm', 'avi'))
            dst_file_name = f.replace('dcm', 'avi')
            self.convert_dicom_to_avi(src_path, dst_file_name, csv_addr)

        df = pd.read_csv(csv_addr)
        df[df['Split'] == 'TRAIN'].to_csv(f'{self.dst_base_addr}/train_features.csv')
        df[df['Split'] == 'TEST'].to_csv(f'{self.dst_base_addr}/test_features.csv')
        df[df['Split'] == 'VAL'].to_csv(f'{self.dst_base_addr}/val_features.csv')

    def generate_img_label(self):
        for subset in ['train', 'test', 'val']:
            df = pd.read_csv(f'{self.dst_base_addr}/{subset}_features.csv')
            impath = f'{self.dst_base_addr}/{subset}/seg_image'
            if not os.path.isdir(impath):
                os.makedirs(impath)

            for data, ed_frame, es_frame in zip(df['FileName'], df['ED_Frame'], df['ES_Frame']):
                vidpath = f'{self.dst_base_addr}/{subset}/videos/{data}.avi'
                vidcap = cv2.VideoCapture(vidpath)
                success, image = vidcap.read()
                count = 1
                while success:
                    if ed_frame is None:
                        continue
                    elif count == int(ed_frame):
                        cv2.imwrite(os.path.join(impath, f"{data}_image_ed.jpg"), image)
                    if es_frame is None:
                        continue
                    elif count == int(es_frame):
                        cv2.imwrite(os.path.join(impath, f"{data}_image_es.jpg"), image)
                    success, image = vidcap.read()
                    count += 1

    @staticmethod
    def make_masks(contours_points):

        x1, y1, x2, y2 = contours_points[:, 0], contours_points[:, 1], contours_points[:, 2], contours_points[:, 3]

        x = np.concatenate((x1[1:], np.flip(x2[1:])))
        y = np.concatenate((y1[1:], np.flip(y2[1:])))

        r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (112, 112))
        mask = np.zeros((112, 112), np.float32)
        mask[r, c] = 1

        return mask

    def construct_all_mask(self):
        tracing_df = self.df_volume
        patient_unique = tracing_df['FileName'].unique()
        val_data_list = os.listdir(f'{self.dst_base_addr}/val/videos')
        test_data_list = os.listdir(f'{self.dst_base_addr}/test/videos')
        train_data_list = os.listdir(f'{self.dst_base_addr}/train/videos')
        for patient in patient_unique:
            patient_labels = tracing_df[tracing_df['FileName'] == patient]
            labels_frame = patient_labels['Frame'].unique()
            ed_label = patient_labels[patient_labels['Frame'] == labels_frame[0]]
            ed_label = np.array(ed_label.loc[:, ['X1', 'Y1', 'X2', 'Y2']])
            es_label = patient_labels[patient_labels['Frame'] == labels_frame[1]]
            es_label = np.array(es_label.loc[:, ['X1', 'Y1', 'X2', 'Y2']])
            label_path = None
            if patient in train_data_list:
                label_path = f'{self.dst_base_addr}/train/seg_image'

            if patient in test_data_list:
                label_path = f'{self.dst_base_addr}/test/seg_image'

            if patient in val_data_list:
                label_path = f'{self.dst_base_addr}/val/seg_image'

            if label_path is not None:
                ed_label_frame = self.make_masks(ed_label)
                skimage.io.imsave(os.path.join(label_path, f"{patient[:-4]}_label_ed.jpg"), ed_label_frame)
                es_label_frame = self.make_masks(es_label)
                skimage.io.imsave(os.path.join(label_path, f"{patient[:-4]}_label_es.jpg"), es_label_frame)

    def calculate_ratio(self):
        for subset in ['test', 'train', 'val']:
            df = pd.read_csv(f'{self.dst_base_addr}/{subset}_features.csv')
            df['ed_frame_class_ratio'] = None
            df['es_frame_class_ratio'] = None
            for i in range(len(df)):
                if df.at[i, 'ED_Frame'] is None:
                    break
                else:
                    for ed_es in ['ed', 'es']:
                        data = df.at[i, 'FileName']
                        label = cv2.imread(
                            f'{self.dst_base_addr}/{subset}/seg_image/{data}_label_{ed_es}.jpg')
                        label = skimage.color.rgb2gray(label) > 0.5
                        label = label.astype(int)
                        n_of_non_zero = cv2.countNonZero(label)
                        df.at[i, f'{ed_es}_frame_class_ratio'] = n_of_non_zero / (label.shape[0] * label.shape[1])
            df.to_csv(f'{self.dst_base_addr}/{subset}_features.csv')