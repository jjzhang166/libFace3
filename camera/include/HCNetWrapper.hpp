#ifndef HCNetWrapper_HPP
#define HCNetWrapper_HPP

#include "HCNetSDK.h"
#include "LinuxPlayM4.h"

#include <string>

class HCNetWrapper {
    typedef struct {
        std::string cameraIP;
        int cameraPort;
        std::string username;
        std::string passwd;
        std::string channel;
    }_InfoConfig;

private:
    HCNetWrapper(const HCNetWrapper&);
    HCNetWrapper& operator=(const HCNetWrapper&);

public:
    typedef void CALLBACK (*RealDataCallBack)(LONG lRealHandle,
                                              DWORD dwDataType, BYTE *pBuffer,
                                              DWORD dwBufSize, void* pUser);

    HCNetWrapper(const std::string& ip, int port,
                 const std::string& uname,const std::string& passwd,
                 const std::string channel)
        :_lUserID(-1), _cbRealData(NULL), _cbData(NULL)
    {
        //init device
        NET_DVR_Init();
        //set connect time and reconnect time
        NET_DVR_SetConnectTime(20000, 1);
        NET_DVR_SetReconnect(10000, true);

        _config.cameraIP = ip;
        _config.cameraPort = port;
        _config.username = uname;
        _config.passwd = passwd;
        _config.channel = channel;
    }

    ~HCNetWrapper() {
        //release resource of SDK
        NET_DVR_Cleanup();
    }

    inline int LoginV30() {
        //lUserID = NET_DVR_Login_V30("192.168.1.64", 8000, "admin", "qy38888813", &struDeviceInfo);
        _lUserID = NET_DVR_Login_V30((char*)_config.cameraIP.c_str(), _config.cameraPort,
                                     (char*)_config.username.c_str(), (char*)_config.passwd.c_str(),
                                     &_struDeviceInfo);
        return _lUserID;
    }

    inline void LoginOut() {
        NET_DVR_Logout(_lUserID);
    }

    inline DWORD GetLastError() const {
        return NET_DVR_GetLastError();
    }

    //需要SDK解码时句柄设为有效值，仅取流不解码时可设为空
    //预览通道号
    //0-主码流，1-子码流，2-码流3，3-码流4，以此类推
    //0- TCP方式，1- UDP方式，2- 多播方式，3- RTP方式，4-RTP/RTSP，5-RSTP/HTTP
    inline void SetPlayInfo(int hPlayWnd, int lChannel, int dwStreamType, int dwLinkMode) {
        _struPlayInfo.hPlayWnd = hPlayWnd;
        _struPlayInfo.lChannel = lChannel;
        _struPlayInfo.dwStreamType = dwStreamType;
        _struPlayInfo.dwLinkMode = dwLinkMode;
    }

    inline int RealPlayV40() {
        return NET_DVR_RealPlay_V40(_lUserID, &_struPlayInfo, _cbRealData , _cbData);
    }

    inline void StopRealPlay(LONG realPlay) {
        NET_DVR_StopRealPlay(realPlay);
    }

    inline void SetRealDataCallBack(RealDataCallBack cb, void* pData = NULL) {
        _cbRealData = cb;
        _cbData = pData;
    }

private:
    //register device
    LONG _lUserID;
    NET_DVR_DEVICEINFO_V30 _struDeviceInfo;
    NET_DVR_PREVIEWINFO _struPlayInfo;
    _InfoConfig _config;
    RealDataCallBack _cbRealData;
    void* _cbData;
};


#endif // HCNetWrapper_HPP
