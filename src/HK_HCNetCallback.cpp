#include "HCNetWrapper.hpp"


#include <opencv/cv.hpp>


#include <iostream>
#include <thread>


void CALLBACK DecCBFun(int nPort, char* pBuf, int nSize, FRAME_INFO* pFrameInfo, void* nReserved1, int nReserved2) {
    /*
    pFrameInfo->nType :
    T_AUDIO16:101
    T_RGB32:7
    T_UYVY:1
    T_YV12:3
    */
    try{
        long lFrameType = pFrameInfo->nType;
        if (lFrameType == T_YV12) {
            cv::Mat pImg(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC3);
            cv::Mat src(pFrameInfo->nHeight + pFrameInfo->nHeight / 2, pFrameInfo->nWidth, CV_8UC1, pBuf);
            cv::cvtColor(src, pImg, CV_YUV2BGR_YV12);
            resize(pImg, pImg, cv::Size(640, 360), 0, 0, cv::INTER_LINEAR);
            std::cout << "show one frame!!!" << std::endl;
            //imshow("showImage", pImg);
            //waitKey(1);
            std::thread t([](cv::Mat img, double scale, bool tryflip){}, pImg, 1.0, false);
            t.detach();
        }
    }
    catch (std::exception& e) {
        std::cout << "\nexception thrown!" << std::endl;
        std::cout << e.what() << std::endl;
    }
}

void CALLBACK fRealDataCallBack(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser) {

    int nPort = -1;
    try{
        switch (dwDataType) {
        case NET_DVR_SYSHEAD: //系统头

            if (!PlayM4_GetPort(&nPort)) {		//获取播放库未使用的通道号
                break;
            }
            //m_iPort = lPort; //第一次回调的是系统头，将获取的播放库port号赋值给全局port，下次回调数据时即使用此port号播放
            if (dwBufSize > 0) {
                std::cout << "NET_DVR_SYSHEAD-dwBufSize:" <<  dwBufSize << std::endl;
                if (!PlayM4_SetStreamOpenMode(nPort, STREAME_REALTIME)) {		 //设置实时流播放模式
                    break;
                }

                if (!PlayM4_OpenStream(nPort, pBuffer, dwBufSize, 10 * 1024 * 1024)) {		//打开流接口
                    break;
                }

                if (!PlayM4_Play(nPort, NULL)) {		//播放开始
                    break;
                }
                if (!PlayM4_SetDecCallBack(nPort, DecCBFun)) {
                    break;
                }
            }
            break;
        case NET_DVR_STREAMDATA:   //码流数据
            if (dwBufSize > 0 && nPort != -1) {
                std::cout << "NET_DVR_STREAMDATA-dwBufSize:"<< dwBufSize << std::endl;
                if (!PlayM4_InputData(nPort, pBuffer, dwBufSize)) {
                    std::cout << "error" << PlayM4_GetLastError(nPort) << std::endl;
                    break;
                }
            }
            break;
        default: //其他数据
            if (dwBufSize > 0 && nPort != -1) {
                std::cout << "NET_DVR_otherData" << std::endl;
                if (!PlayM4_InputData(nPort, pBuffer, dwBufSize)) {
                    break;
                }
            }
            break;
        }
    }
    catch (std::exception& e) {
        std::cout << "\nexception thrown!" << std::endl;
        std::cout << e.what() << std::endl;
    }
}
