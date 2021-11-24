import pathlib


def test_requirements():
    packages = ['tensorflow>=2.7.0',
                'pandas',
                'scikit-learn',
                'scikit-image',
                'scipy',
                'matplotlib',
                'mlflow',
                'abstractions',
                'SimpleITK',
                'pyyaml',
                'albumentations',
                'tqdm',
                'pytest']

    project_dir = pathlib.Path(__file__).parent.parent
    print(project_dir)

    with open(project_dir.joinpath('requirements.txt'), 'r') as f:
        reqs = f.read().split()
        assert all(item in reqs for item in packages)
