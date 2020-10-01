cd ~

wget -O tensorflow.zip https://github.com/tensorflow/tensorflow/archive/v2.3.0.zip

unzip tensorflow.zip
mv tensorflow-2.3.0 tensorflow
cd tensorflow

./tensorflow/lite/tools/make/download_dependencies.sh
./tensorflow/lite/tools/make/rpi_armv7l.sh


cd ~/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers
mkdir build
cd build
cmake ..
make -j4
sudo make install
sudo ldconfig