#ifndef QY_FACEOPERATOR_H
#define QY_FACEOPERATOR_H

#include "QY_Log.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <mutex>
#include <vector>


class QY_FaceOperator{
public:
    typedef std::function<void(cv::Mat)> DetectResHandle;

    static bool loadCascadeClassifier(const std::string& fliePath);
    static void faceDetect(cv::Mat img, double scale, bool tryflip,
                           DetectResHandle resultHandle);
    static void ExtractFeature();
    static void faceRecognition();

private:
    static std::mutex m_ClassierMutex;
    static cv::CascadeClassifier m_CascadeClassier;
};

/* below is static variable define */
std::mutex QY_FaceOperator::m_ClassierMutex;
cv::CascadeClassifier QY_FaceOperator::m_CascadeClassier;

/* below is static function define */

bool QY_FaceOperator::loadCascadeClassifier(const std::string &fliePath){
    return m_CascadeClassier.load(fliePath);
}

void QY_FaceOperator::faceDetect(cv::Mat img, double scale, bool tryflip,
                                 DetectResHandle resultHandle){
    using namespace cv;
    using namespace std;
    try
    {
        if(img.empty()){
            LOG_INFO ("img is empty\n");
            return;
        }

        int i = 0;
        vector<Rect> faces, faces2;
        faces.clear();
        faces2.clear();

        const static Scalar colors[] = { CV_RGB(0, 0, 255),
            CV_RGB(0, 128, 255),
            CV_RGB(0, 255, 255),
            CV_RGB(0, 255, 0),
            CV_RGB(255, 128, 0),
            CV_RGB(255, 255, 0),
            CV_RGB(255, 0, 0),
            CV_RGB(255, 0, 255) };

        Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);
        cvtColor(img, gray, CV_BGR2GRAY);
        resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
        equalizeHist(smallImg, smallImg);

        if(!m_ClassierMutex.try_lock()){
            cout << "CascadeClassie is working" << endl;
            return;
        }

        m_CascadeClassier.detectMultiScale(smallImg, faces,
                                           1.1, 2, 0
                                           //|CV_HAAR_FIND_BIGGEST_OBJECT
                                           //|CV_HAAR_DO_ROUGH_SEARCH
                                           | CV_HAAR_SCALE_IMAGE,
                                           Size(30, 30));
        if (tryflip)
        {
            flip(smallImg, smallImg, 1);
            m_CascadeClassier.detectMultiScale(smallImg, faces2,
                                               1.1, 2, 0
                                               | CV_HAAR_SCALE_IMAGE
                                               ,
                                               Size(30, 30));
            cout << "faces2.size():" << faces2.size() << endl;
            for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++)
            {
                faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
            }
        }
        m_ClassierMutex.unlock();

        if (faces.size() > 0){
            LOG_INFO ("faces size: %d.................\n", faces.size());
            for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
            {
                Mat procImage(smallImg, Rect((r->x)*scale, r->y*scale, r->width*scale, r->height*scale));
                // handle detect result
                if(resultHandle){
                    resultHandle(procImage);
                }
                else{
                    LOG_INFO ("face detect not result handler....\n");
                    continue;
                }
            }

        }
        else{
            cout << "This image is not a face, ignore it" << endl;
        }

    }
    catch (std::exception& e)
    {
        std::cout << "\nexception thrown!" << std::endl;
        std::cout << e.what() << std::endl;
    }
}

#endif // QY_FACEOPERATOR_H
