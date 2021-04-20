cmake_minimum_required(VERSION 2.8)

project( NeuralNetwork )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( NeuralNetwork Main.cpp NeuralNetwork.cpp)

target_link_libraries( NeuralNetwork ${OpenCV_LIBS} )