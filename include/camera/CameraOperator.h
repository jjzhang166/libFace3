#ifndef _LIBFACE_CameraOperator_H
#define _LIBFACE_CameraOperator_H

#include <opencv2/core.hpp>

#include <string>
#include <functional>


namespace libface {

namespace camera {

/* type define */
typedef std::function<void(cv::Mat)> Handler;

/* prototype */
void HandleFrameFromRtspCamera(const std::string& rtspAddress, Handler handler);
void HandleFrameFromUSBCamera(Handler handler);
void HandleFrameFromImage(const std::string& imagePath, Handler handler);

}	// namespace camera

}	// namespace libface

#endif // _LIBFACE_CameraOperator_H
