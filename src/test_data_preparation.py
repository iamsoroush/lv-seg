import os
import sys
import pytest
import pandas as pd
from .data_preparation import ExtractDicom


class TestExtractDicom:

    @pytest.fixture
    def addr(self):
        src = ''
        dst = ''
        return src, dst

    @pytest.fixture
    def extractor_instance(self, addr):
        features = ['FileName',
                    'EF',
                    'ESV',
                    'EDV',
                    'FrameHeight',
                    'FrameWidth',
                    'FPS',
                    'NumberOfFrames',
                    'Split']
        src, dst = addr
        return ExtractDicom(src, dst, features)

    @pytest.fixture
    def excel_columns(self):
        features = [
            ['FileName',
             'EF',
             'ESV',
             'EDV',
             'FrameHeight',
             'FrameWidth',
             'FPS',
             'NumberOfFrames',
             'Split',
             'ED_Frame',
             'ES_Frame',
             'image_address_ed',
             'image_address_es',
             'label_address_ed',
             'label_address_es',
             'ed_frame_class_ratio',
             'es_frame_class_ratio'
             ]
        ]
        return features

    @pytest.fixture
    def subset(self):
        return ['val', 'test', 'train']

    @pytest.fixture
    def image_label_addr(self):
        addr = ['image_address_ed',
                'image_address_es',
                'label_address_ed',
                'label_address_es']
        return addr

    def test_prepare_data(self, extractor_instance):
        extractor_instance.prepare_data()

    def test_existence_csv(self, addr, subset):
        all_excel = []
        src, dst = addr
        for s in subset:
            csv_addr = f'{dst}/{s}_features.csv'
            exist = os.path.isfile(csv_addr)
            all_excel.append(exist)
        assert all(all_excel)

    def test_csv_columns(self, addr, subset, excel_columns):
        src, dst = addr
        for s in subset:
            csv_addr = f'{dst}/{s}_features.csv'
            exist = os.path.isfile(csv_addr)
            if exist:
                df = pd.read_csv(csv_addr)
                assert all([c in df.columns for c in excel_columns])

    def test_existence_image_label_train(self, addr, image_label_addr):
        src, dst = addr
        csv_addr = f'{dst}/train_features.csv'
        df = pd.read_csv(csv_addr)
        check_addr = []
        for a in addr:
            for i in df.head(int(len(df) * 0.05))[a]:
                check_addr.append(os.path.isfile(i))
            assert all(check_addr)

    def test_existence_image_label_test(self, addr, image_label_addr):
        src, dst = addr
        csv_addr = f'{dst}/test_features.csv'
        df = pd.read_csv(csv_addr)
        check_addr = []
        for a in addr:
            for i in df.head(int(len(df) * 0.05))[a]:
                check_addr.append(os.path.isfile(i))
            assert all(check_addr)

    def test_existence_image_label_val(self, addr, image_label_addr):
        src, dst = addr
        csv_addr = f'{dst}/val_features.csv'
        df = pd.read_csv(csv_addr)
        check_addr = []
        for a in addr:
            for i in df.head(int(len(df) * 0.05))[a]:
                check_addr.append(os.path.isfile(i))
            assert all(check_addr)

