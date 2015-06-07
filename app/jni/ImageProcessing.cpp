#include "com_cabatuan_smiledetection_MainActivity.h"
#include <android/log.h>
#include <android/bitmap.h>
#include <stdlib.h>

#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#define  LOG_TAG    "SmileDetection"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define  DEBUG 0

using namespace std;
using namespace cv;

/* Function Headers */
void detectSmile(Mat& srcGray, Mat& mbgra);


/*
 * Class:     com_cabatuan_smiledetection_MainActivity
 * Method:    predict
 * Signature: (Landroid/graphics/Bitmap;[B)V
 */
JNIEXPORT void JNICALL Java_com_cabatuan_smiledetection_MainActivity_predict
  (JNIEnv * pEnv, jobject pClass, jobject pTarget, jbyteArray pSource){

  AndroidBitmapInfo bitmapInfo;
   uint32_t* bitmapContent; // Links to Bitmap content

   if(AndroidBitmap_getInfo(pEnv, pTarget, &bitmapInfo) < 0) abort();
   if(bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) abort();
   if(AndroidBitmap_lockPixels(pEnv, pTarget, (void**)&bitmapContent) < 0) abort();

   /// Access source array data... OK
   jbyte* source = (jbyte*)pEnv->GetPrimitiveArrayCritical(pSource, 0);
   if (source == NULL) abort();

   /// cv::Mat for YUV420sp source and output BGRA 
    Mat srcGray(bitmapInfo.height, bitmapInfo.width, CV_8UC1, (unsigned char *)source);
    Mat mbgra(bitmapInfo.height, bitmapInfo.width, CV_8UC4, (unsigned char *)bitmapContent);

/***********************************************************************************************/
    /// Native Image Processing HERE... 
    if(DEBUG){
      LOGI("Starting native image processing...");
    }

    detectSmile(srcGray, mbgra);

    if(DEBUG){
      LOGI("Successfully finished native image processing...");
    }
   
/************************************************************************************************/ 
   
   /// Release Java byte buffer and unlock backing bitmap
   pEnv-> ReleasePrimitiveArrayCritical(pSource,source,0);
   if (AndroidBitmap_unlockPixels(pEnv, pTarget) < 0) abort();
}







void detectSmile(Mat& srcGray, Mat& mbgra){

       /* Classifier variables */
       CascadeClassifier face_cascade;
       CascadeClassifier smile_cascade;
       char face_cascade_path[100];
       char smile_cascade_path[100];  
       sprintf( face_cascade_path, "%s/%s", getenv("ASSETDIR"), "haarcascade_frontalface_alt.xml");
       sprintf( smile_cascade_path, "%s/%s", getenv("ASSETDIR"), "haarcascade_smile.xml");

        /* Load the cascades */
       if( !face_cascade.load(face_cascade_path) ){ 
           LOGE("Error loading face cascade"); 
           abort(); 
       };
       
       if( !smile_cascade.load(smile_cascade_path) ){ 
           LOGE("Error loading smile cascade"); 
           abort(); 
       };

        /* Resize input image for faster computation */
        double scale = 2;
        Mat smallImg( cvRound (srcGray.rows/scale), cvRound(srcGray.cols/scale), CV_8UC1 );
        resize( srcGray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR ); 
  
    
        /* Equalize Histogram */
        //equalizeHist(srcGray, srcGray);


        /* Detect faces  */    
        std::vector<Rect> faces;
        face_cascade.detectMultiScale( smallImg, faces,
        1.2, 2, 0
        |CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        //|CASCADE_SCALE_IMAGE
        ,
        Size(30, 30) );



        int i = 0;
        for( vector<Rect>::iterator r = faces.begin(); r != faces.end(); r++, i++ )
        {
            Mat smallImgROI;
            std::vector<Rect> nestedObjects;
            Point center;
            int radius;

            double aspect_ratio = (double)r->width/r->height;

             if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
             {
               center.x = cvRound((r->x + r->width*0.5)*scale);
               center.y = cvRound((r->y + r->height*0.5)*scale);
               radius = cvRound((r->width + r->height)*0.25*scale);
               circle( srcGray, center, radius, Scalar(255,0,0), 3, 8, 0 );
              }
              else
               rectangle( srcGray, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
               cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                       Scalar(255,0,0), 3, 8, 0);

             const int half_height=cvRound((float)r->height/2);
             r->y=r->y + half_height;
             r->height = half_height;
             smallImgROI = smallImg(*r);
           
             smile_cascade.detectMultiScale( smallImgROI, nestedObjects,
            1.2, 0, 0
            |CASCADE_FIND_BIGGEST_OBJECT
            //|CASCADE_DO_ROUGH_SEARCH
            //|CASCADE_DO_CANNY_PRUNING
            //|CASCADE_SCALE_IMAGE
            ,
            Size(30, 30) );

        // The number of detected neighbors depends on image size (and also illumination, etc.). The
        // following steps use a floating minimum and maximum of neighbors. 
        // Intensity thus estimated will be
        //accurate only after a first smile has been displayed by the user.
        const int smile_neighbors = (int)nestedObjects.size();
        static int max_neighbors=-1;
        static int min_neighbors=-1;
        if (min_neighbors == -1) min_neighbors = smile_neighbors;
        max_neighbors = MAX(max_neighbors, smile_neighbors);

        // Draw rectangle on the left side of the image reflecting smile intensity
        float intensityZeroOne = ((float)smile_neighbors - min_neighbors) / (max_neighbors - min_neighbors + 1);
        int rect_height = cvRound((float)srcGray.rows * intensityZeroOne);
        CvScalar col = CV_RGB((float)255 * intensityZeroOne , 0, 0);
        rectangle(srcGray, cvPoint(0, srcGray.rows), cvPoint(srcGray.cols/10, srcGray.rows - rect_height), col, -1);
    }

        /// Display to Android
        cvtColor(srcGray, mbgra, CV_GRAY2BGRA);
}

