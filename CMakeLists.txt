cmake_minimum_required(VERSION 2.6)

project (caffe2_cpp_tutorial)
set(CMAKE_BUILD_TYPE "Debug")

find_package(Protobuf REQUIRED)  #REQUIRED表示如果没有找到，cmake会停止处理，并报告一个错误.**不管使用哪一种模式，只要找到*.cmake，*.cmake里面都会定义下面这些变量：<NAME>_FOUND；<NAME>_INCLUDE_DIRS or <NAME>_INCLUDES；<NAME>_LIBRARIES or <NAME>_LIBRARIES or <NAME>_LIBS；<NAME>_DEFINITIONS
# find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/modules/")
# find_package(Caffe2 REQUIRED)

find_library(CAFFE2_LIB caffe2)
find_library(GLOG_LIB glog)
find_library(GFLAGS_LIB gflags)
find_library(GTEST_LIB gtest)#No rule to make target '/usr/local/lib/libgtest.a', needed by 'blob'.  Stop.
find_library(CLOG_LIB clog)
find_library(GFLAGS_LIB gflags)
find_library(ONNX_LIB onnx) # Somehow necessary to prevent "undefined reference to `ncclReduceScatter'"
find_library(C10_LIB c10)
find_library(THREAD_LIB pthread)

# include_directories(modules)#由于包含的头文件只有一个，所以使用绝对路径即可
include_directories(${PROTOBUF_INCLUDE_DIRS})#由于以下几个include的结果都是一样的，所以使用一个即可
# include_directories(${Opencv_INCLUDE_DIRS})
# include_directories(${Caffe2_INCLUDE_DIR})
# include_directories(${GLOG_INCLUDE_DIRS})

#生成 .pb.cc and .pb.h

set(PB_SOURCE_DIR "${PROJECT_SOURCE_DIR}/modules/ModelHelpers")

  execute_process(
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --proto_path=${PB_SOURCE_DIR} --cpp_out=${PB_SOURCE_DIR} "${PB_SOURCE_DIR}/external.proto"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  
    execute_process(
        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}  --proto_path=${PB_SOURCE_DIR} --cpp_out=${PB_SOURCE_DIR} "${PB_SOURCE_DIR}/model.proto"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      

file(GLOB Third_LIB_SOURCES "${PROJECT_SOURCE_DIR}/modules/cvplot/*.cc"  "${PROJECT_SOURCE_DIR}/modules/ModelHelpers/*.cc"  ) 
add_library(caffe2_plot ${Third_LIB_SOURCES})

set(ALL_LIBRARIES)
list(APPEND ALL_LIBRARIES ${CAFFE2_LIB} ${PROTOBUF_LIBRARY} ${GLOG_LIB} ${CLOG_LIB} ${GFLAGS_LIB}  ${C10_LIB})#${ONNX_LIB}
list(APPEND ALL_LIBRARIES ${caffe2_plot})#使用自己涉及的cvplot函数




# message(STATUS ${Caffe2_INCLUDE_DIR})
# message(STATUS ${PROTOBUF_INCLUDE_DIRS})
# message(STATUS ${CAFFE2_LIB})
# message(STATUS ${PROTOBUF_LIBRARY})
# message(STATUS ${GTEST_LIB})
# message(STATUS ${GLOG_LIB})
# message(STATUS ${CLOG_LIB})
# message(STATUS ${GFLAGS_LIB})
# # message(STATUS ${ONNX_LIB})
# message(STATUS ${C10_LIB})
# message(STATUS ${THREAD_LIB})
# message(STATUS ${OpenCV_LIBS})

# execute_process(
# #按指定的先后顺序运行一个或多个命令，每个进程的输出通过管道连接作为下一个进程的输入。所有的进程使用单个的标准错误输出管道。
#   COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}  --proto_path=${PB_SOURCE_DIR} --cpp_out=${PB_SOURCE_DIR} "${PB_SOURCE_DIR}/model.proto"#有的需要--，但三有的不需要为什么？--proto_path=../caffe2
#   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})# 指定的目录将被设置为子进程的当前工作目录。
# #简单来说，就是如果多个proto文件之间有互相依赖，生成某个proto文件时，需要import其他几个proto文件，这时候就要用-I或者 -IPATH, --proto_path=PATH来指定搜索目录，否则在当前目录进行搜索。
  
  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC ")#设置编译选项可以通过add_compile_options命令，也可if(CMAKE_COMPILER_IS_GNUCXX) set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}"), 如果是gcc编译器,则在编译选项中加入c++11支持


ADD_EXECUTABLE(blob blob.cpp )
target_link_libraries(blob   ${ALL_LIBRARIES} )

# # ADD_EXECUTABLE(caffe2_benchmark caffe2_benchmark.cc )
# # target_link_libraries(caffe2_benchmark   ${ALL_LIBRARIES} )


# ADD_EXECUTABLE(convert_caffe_image_db convert_caffe_image_db.cc )
# target_link_libraries(convert_caffe_image_db   ${ALL_LIBRARIES} )

# ADD_EXECUTABLE(convert_db convert_db.cc )
# target_link_libraries(convert_db   ${ALL_LIBRARIES} )

# ADD_EXECUTABLE(convert_image_to_tensor convert_image_to_tensor.cc )# undefined reference to symbol '_ZN2cv6imreadERKNS_6StringEi'
# target_link_libraries(convert_image_to_tensor  ${ALL_LIBRARIES}  ${OpenCV_LIBS} )

# ADD_EXECUTABLE(db_throughput db_throughput.cc )
# target_link_libraries(db_throughput  ${ALL_LIBRARIES} ${THREAD_LIB})#-lpthread)

# ADD_EXECUTABLE(make_cifar_db make_cifar_db.cc )
# target_link_libraries(make_cifar_db  ${ALL_LIBRARIES} )

# ADD_EXECUTABLE(make_image_db make_image_db.cc )
# target_link_libraries(make_image_db  ${ALL_LIBRARIES} ${OpenCV_LIBS}  -lpthread )

# ADD_EXECUTABLE(make_mnist_db make_mnist_db.cc )
# target_link_libraries(make_mnist_db   ${ALL_LIBRARIES} )

# ADD_EXECUTABLE(predictor_verifier predictor_verifier.cc )
# target_link_libraries(predictor_verifier   ${ALL_LIBRARIES} )

# ADD_EXECUTABLE(print_registered_core_operators print_registered_core_operators.cc )
# target_link_libraries(print_registered_core_operators  ${ALL_LIBRARIES} )

# ADD_EXECUTABLE(run_plan  run_plan.cc )
# target_link_libraries(run_plan   ${ALL_LIBRARIES} )

# ADD_EXECUTABLE(speed_benchmark  speed_benchmark.cc )
# target_link_libraries(speed_benchmark  ${ALL_LIBRARIES} )

# ADD_EXECUTABLE(split_db split_db.cc )
# target_link_libraries(split_db  ${ALL_LIBRARIES} )

# ADD_EXECUTABLE(tsv_2_proto tsv_2_proto.cc )
# target_link_libraries(tsv_2_proto  ${ALL_LIBRARIES} )

# ADD_EXECUTABLE(intro intro.cc )
# target_link_libraries(intro  ${ALL_LIBRARIES} )

# ADD_EXECUTABLE(blob-workspace blob-workspace.cc )
# target_link_libraries(blob-workspace  ${ALL_LIBRARIES} )

# ADD_EXECUTABLE(operaters-workspace operaters-workspace.cc )
# target_link_libraries(operaters-workspace  ${ALL_LIBRARIES} )

# ADD_EXECUTABLE(net net.cc )
# target_link_libraries(net  ${ALL_LIBRARIES} )

# ADD_EXECUTABLE(toy toy.cc )
# target_link_libraries(toy  ${ALL_LIBRARIES} )



ADD_EXECUTABLE(mnist mnist.cc )
target_link_libraries(mnist  ${ALL_LIBRARIES} )



