import os
import cv2
import shutil
import numpy as np
import pandas as pd
import pydicom._storage_sopclass_uids
from pydicom.datadict import add_dict_entry


class DICOMHandler:

    """

    This class contains implementations of some tools for converting data with AVI format to DICOM

    """

    def __init__(self, src_base, dst_base):

        """

        Initialize the required variables

        Args:
            src_base: Path of the folder which contains AVI videos
            dst_base: Path of the folder that converted DICOM videos will be saved in it

        Attributes:
            echo_data_info: Dictionary contain the address (tag) of the features in the dicom file
            df_echo: Data frame of the csv file
        """

        csv_file_addr = os.path.join(src_base, 'FileList.csv')
        csv_volume_src = os.path.join(src_base, 'VolumeTracings.csv')
        csv_volume_dst = os.path.join(dst_base, 'VolumeTracings.csv')
        if not os.path.isdir(dst_base):
            os.makedirs(dst_base)
        shutil.copy(src=csv_volume_src, dst=csv_volume_dst)

        self.src_base_addr = src_base
        self.dst_base_addr = dst_base
        self.echo_data_info = {}
        self.df_echo = pd.read_csv(csv_file_addr)

    def register_data_features(self):

        """

        In this method, the address (tag) assigned for each feature of data frame

        """

        i = 0
        self.echo_data_info = {}
        for k in self.df_echo.keys():
            print(k)
            v = 0x10021001 + i
            char_hex = hex(v)
            self.echo_data_info[k] = [int('0x' + char_hex[2:6], 16), int('0x' + char_hex[6:], 16)]
            if k in ['FileName', 'Split']:
                add_dict_entry(v, "CS", k, k)
            else:
                if k == 'NumberOfFrames':
                    k = 'NOF'
                add_dict_entry(v, "DS", k, k)
            i += 1

    def convert_avi_to_dicom(self, img, dst_addr, attributes):

        """

        Conversion of data file with AVI format to DICOM could be performed by this method

        Args:
            img: 3D numpy array contains frame of the AVI video
            dst_addr: Address of the place that the converted DICOM file will be saved
            attributes: Features of the data sample which could be retrieved from csv file

        Returns:

        """

        # metadata
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        # dataset
        ds = pydicom.Dataset()
        ds.file_meta = file_meta

        ds.Rows = img.shape[1]
        ds.Columns = img.shape[2]
        ds.NumberOfFrames = img.shape[0]
        # ds.add_new([0x0028, 0x0008], "IS", int(image.shape[0]))
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsStored = 8

        ds.BitsAllocated = 8
        ds.PixelRepresentation = 1

        for k, v in attributes.items():
            if k in ['FileName', 'Split']:
                ds.add_new(self.echo_data_info[k], 'CS', v)
            else:
                ds.add_new(self.echo_data_info[k], 'DS', v)

        ds.PixelData = img.tobytes()

        # save
        ds.save_as(dst_addr, write_like_original=False)

    @staticmethod
    def read_avi_file(addr):

        """

        In this method, AVI videos will be read and the list of the frames will be returned

        Args:
            addr: Path of the AVI file

        Returns:
            List of frame represented in the AVI video

        """

        cap = cv2.VideoCapture(addr)
        all_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            else:
                break
        cap.release()
        return all_frames

    def convert_all_to_dicom(self):

        """

        All AVI videos existed in the specific address could be converted to the DICOM file by using this method

        Returns:

        """

        self.register_data_features()
        for k, v in self.df_echo.iterrows():

            # src_addr = os.path.join(self.src_base_addr, f'{v.Split.lower()}/videos/{v.FileName}.avi')

            src_addr = os.path.join(self.src_base_addr, f'Videos/{v.FileName}.avi')
            dst_video_path = os.path.join(self.dst_base_addr, 'Videos')
            # dst_folder = os.path.join(DST_BASE_ADDR, v.Split.lower())
            if not os.path.isdir(dst_video_path):
                os.makedirs(dst_video_path)

            dst_addr = os.path.join(dst_video_path, f'{v.FileName}.dcm')
            image = np.array(self.read_avi_file(src_addr))
            self.convert_avi_to_dicom(image, dst_addr, v)
