#-------------------------------------------------
#
# Project created by QtCreator 2018-03-05T16:43:25
#
#-------------------------------------------------

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = app
TEMPLATE = app
CONFIG += c++11


SOURCES += main.cpp\
        mainwindow.cpp \
    startscreen.cpp \
    c_qt_opencv_viewer_gl.cpp

HEADERS  += mainwindow.h \
    startscreen.h \
    c_qt_opencv_viewer_gl.h

FORMS    += mainwindow.ui \
    startscreen.ui


INCLUDEPATH += /usr/local/include/opencv

LIBS += -L/usr/local/lib \
        -lopencv_core \
        -lopencv_imgproc \
        -lopencv_videoio \
        -lopencv_highgui \
        -lopencv_imgcodecs \
