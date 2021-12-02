import pathlib


def test_requirements():
    """Tests for main requirements."""

    packages = [
        "tensorflow>=2.7.0",
        "pandas",
        "scikit-learn",
        "scikit-image",
        "scipy",
        "matplotlib",
        "mlflow",
        "abstractions-aimedic",
        "SimpleITK",
        "pyyaml",
        "albumentations",
        "tqdm",
        "pytest",
        "pytest-dependency"
    ]

    project_dir = pathlib.Path(__file__).parent.parent
    print(project_dir)

    with open(project_dir.joinpath("requirements.txt"), "r") as f:
        reqs = f.read().split()
        assert all(item in reqs for item in packages)
