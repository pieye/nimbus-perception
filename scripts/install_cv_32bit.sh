#!/usr/bin/env sh
sudo apt update
sudo apt upgrade
sudo apt install cmake gfortran
sudo apt install libjpeg-dev libtiff-dev libgif-dev
sudo apt install libavcodec-dev libavformat-dev libswscale-dev
sudo apt install libgtk2.0-dev libcanberra-gtk*
sudo apt install libxvidcore-dev libx264-dev libgtk-3-dev
sudo apt install libtbb2 libtbb-dev libdc1394-22-dev libv4l-dev
sudo apt install libopenblas-dev libatlas-base-dev libblas-dev
sudo apt install libjasper-dev liblapack-dev libhdf5-dev
sudo apt install gcc-arm* protobuf-compiler

cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip
unzip opencv.zip
unzip opencv_contrib.zip

mv opencv-4.2.0 opencv
mv opencv_contrib-4.2.0 opencv_contrib

cd ~/opencv/
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D ENABLE_NEON=ON \
      -D ENABLE_VFPV3=ON \
      -D WITH_OPENMP=ON \
      -D BUILD_TIFF=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_TBB=ON \
      -D BUILD_TBB=ON \
      -D BUILD_TESTS=OFF \
      -D WITH_EIGEN=OFF \
      -D WITH_V4L=ON \
      -D WITH_LIBV4L=ON \
      -D WITH_VTK=OFF \
      -D WITH_QT=OFF \
      -D OPENCV_EXTRA_EXE_LINKER_FLAGS=-latomic \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D INSTALL_C_EXAMPLES=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D BUILD_NEW_PYTHON_SUPPORT=ON \
      -D BUILD_opencv_python3=TRUE \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D BUILD_EXAMPLES=OFF ..

make -j4

sudo make install
sudo ldconfig
sudo apt update

cd ~
rm opencv.zip
rm opencv_contrib.zip
reboot