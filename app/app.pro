#-------------------------------------------------
#
# Project created by QtCreator 2018-03-05T16:43:25
#
#-------------------------------------------------

QT       += core gui opengl sql

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = app
TEMPLATE = app
CONFIG += c++14

DEFINES += USE_OPENCV #delete if you don't want use OpenCV
DEFINES += CPU_ONLY #delete if you want use GPU

SOURCES += main.cpp\
        mainwindow.cpp \
    startscreen.cpp \
    c_qt_opencv_viewer_gl.cpp \
    workscreen.cpp \
    tracking/util/bounding_box.cpp \
    tracking/util/help_functions.cpp \
    tracking/caffe_binding.cpp \
    tracking/cascade_cnn.cpp \
    emotion/emotiw.cpp \
    frame_features.cpp \
    qperson.cpp \
    recognition/person_classifier.cpp \
    signscreen.cpp

HEADERS  += mainwindow.h \
    startscreen.h \
    c_qt_opencv_viewer_gl.h \
    workscreen.h \
    tracking/util/bounding_box.h \
    tracking/util/help_functions.h \
    tracking/caffe_binding.h \
    tracking/thread_group.inc.h \
    tracking/cascade_cnn.h \
    emotion/emotiw.h \
    frame_features.hpp \
    SimpleJSON/json.hpp \
    qperson.h \
    recognition/person_classifier.h \
    signscreen.h

FORMS    += mainwindow.ui \
    startscreen.ui \
    workscreen.ui \
    signscreen.ui


INCLUDEPATH += /usr/local/include/opencv

LIBS += -L/usr/local/lib \
        -lopencv_core \
        -lopencv_imgproc \
        -lopencv_videoio \
        -lopencv_highgui \
        -lopencv_imgcodecs \

LIBS += \
       -lboost_system\
       -lboost_thread\
       -lboost_filesystem\
       -lglog

LIBS += -lodbc

INCLUDEPATH += $$PWD/caffe/include
INCLUDEPATH += $$PWD/caffe/src

LIBS += -L$$PWD/caffe/build/lib/ -lcaffe

