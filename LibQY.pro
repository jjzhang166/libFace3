TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

HEADERS += \
    HCNetSDK.h \
    LinuxPlayM4.h \
    QY_CameraOperator.hpp \
    QY_FaceOperator.hpp \
    QY_HCNetWrapper.hpp \
    QY_Log.hpp


LIBS += -L./lib -lboost_system -lcurl -lb64 \
             -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_flann -lopencv_objdetect -lopencv_video\
             -lhcnetsdk -lPlayCtrl -lAudioRender -lSuperRender\
             -lpthread
