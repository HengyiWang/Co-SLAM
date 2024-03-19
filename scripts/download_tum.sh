mkdir -p data/TUM
cd data/TUM
if [[ ! -d rgbd_dataset_freiburg1_desk ]]; then
    wget -c https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
    tar -xvzf rgbd_dataset_freiburg1_desk.tgz
fi

if [[ ! -d rgbd_dataset_freiburg2_xyz ]]; then
    wget -c https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz
    tar -xvzf rgbd_dataset_freiburg2_xyz.tgz
fi

if [[ ! -d rgbd_dataset_freiburg3_long_office_household ]]; then
    wget -c https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz
    tar -xvzf rgbd_dataset_freiburg3_long_office_household.tgz
fi