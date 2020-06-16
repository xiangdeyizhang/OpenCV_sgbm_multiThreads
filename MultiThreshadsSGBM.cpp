#include <iostream>
#include <iomanip>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"

using namespace cv;
using namespace std;
#define THREAD_NUMS 4

#define WIDTH  320
#define HEIGHT 240
#define M 2
#define N 2
#define SUB_WIDTH  WIDTH/ M
#define SUB_HEIGHT HEIGHT/ N

#define width_roi 80


/*paramThread用于传递线程需要的参数值*/
struct paramThread
{
	int w;
	int h;
	uchar * data1;
	uchar * data2;
	uchar * depth_data;
};
 
/********************************************************
*	@brief       : 多线程处理函数
*	@param  args : 多线程传入的参数
*	@return      : void
********************************************************/
void * threadProcess(void* args) {
 
	pthread_t myid = pthread_self();
	paramThread *para = (paramThread *)args;
	int w = WIDTH/2+width_roi;
	int h = HEIGHT/2;
	cv::Mat image_left(h,w,CV_8UC1,(uchar *)para->data1);
	cv::Mat image_right(h,w,CV_8UC1,(uchar *)para->data2);
	cv::Mat disp8(h,w,CV_8UC1,(uchar *)para->depth_data);

	//imshow("image_left",image_left);
	//imshow("image_right",image_right);
	//waitKey(0);

	//设置匹配参数
    /********************SGBM*************************************/
    ///(640,480)
    //SGBM
	int mindisparity  = 0;
	int ndisparities  = 48;  
	int SADWindowSize = 9; 
	Ptr<StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);
	int P1 = 8 * 1 * SADWindowSize* SADWindowSize;
	int P2 = 32* 1 * SADWindowSize* SADWindowSize;
	sgbm->setP1(P1);
	sgbm->setP2(P2);
	sgbm->setPreFilterCap(15);
	sgbm->setUniquenessRatio(10);
	sgbm->setSpeckleRange(2);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setDisp12MaxDiff(1);

	//cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	 //sgbm
	cv::Mat disp;
    sgbm->compute(image_left, image_right, disp);
	disp.convertTo(disp8, CV_8U, 255/(48*16.));

	printf("thread id = %d, w=%d, h=%d\n", myid,disp8.cols,disp8.rows);
	//cv::imshow("depth", disp8); cv::waitKey(0);
	pthread_exit(NULL);
	return NULL;
}
 
/********************************************************
*	@brief       : 实现图像分割，
*	@param  num  :  分割个数
*	@param  type : 0：垂直分割(推荐)，1：水平分割（不推荐）
*	@return      : vector<cv::Mat>
*   PS：使用水平分割时（type=1），处理完后必须调用catImage进行拼接，
*   使用垂直分割时（type=0），可以不进行catImage，因为是对原图进行操作的
********************************************************/
vector<cv::Mat> splitImage(cv::Mat image, int num,int type) {
	vector<Mat> v;
	Mat image_cut, roi_img1,roi_img2,roi_img3,roi_img4;
	cv::Rect rect1,rect2,rect3,rect4;

	rect1 =Rect(0 * SUB_WIDTH, 				0 * SUB_HEIGHT, SUB_WIDTH+width_roi, SUB_HEIGHT);
	rect2 =Rect(1 * SUB_WIDTH-width_roi, 	0 * SUB_HEIGHT, SUB_WIDTH+width_roi, SUB_HEIGHT);
	rect3 =Rect(0 * SUB_WIDTH, 				1 * SUB_HEIGHT, SUB_WIDTH+width_roi, SUB_HEIGHT);
	rect4 =Rect(1 * SUB_WIDTH-width_roi, 	1 * SUB_HEIGHT, SUB_WIDTH+width_roi, SUB_HEIGHT);
	roi_img1 =Mat(image, rect1).clone();
	roi_img2 =Mat(image, rect2).clone();
	roi_img3 =Mat(image, rect3).clone();
	roi_img4 =Mat(image, rect4).clone();

	v.push_back(roi_img1);
	v.push_back(roi_img2);
	v.push_back(roi_img3);
	v.push_back(roi_img4);

#if 0
	for(int j = 0; j < 2; j ++)
	{
		for (int i = 0; i < 2; i++)
		{
			Rect rect(i * SUB_WIDTH, j * SUB_HEIGHT, SUB_WIDTH, SUB_HEIGHT);
			image_cut = Mat(image, rect);
			roi_img = image_cut.clone();
			//imshow("roi",roi_img);
			//waitKey(0);
			v.push_back(roi_img);
		}
	}
#endif
	cout<<"v"<<v.size()<<endl;
	return  v;
}
 
/********************************************************
*	@brief       : 实现图像拼接，
*	@param  v    :  
*	@param  type : 0：垂直拼接，1：水平拼接
*	@return      : Mat
********************************************************/
cv::Mat catImage(vector<cv::Mat> v, int type) {
	cv::Mat dest= v.at(0);
	for (size_t i = 1; i < v.size(); i++)
	{
		if (type == 0)//垂直拼接
		{
			cv::vconcat(dest, v.at(i), dest);
		}
		else if (type == 1)//水平拼接
		{
			cv::hconcat(dest, v.at(i), dest);
		}
	}
	return dest;
}
 
int main() {

	//SGBM
	int mindisparity  = 0;
	int ndisparities  = 48;  
	int SADWindowSize = 9; 
	Ptr<StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);
	int P1 = 8 * 1 * SADWindowSize* SADWindowSize;
	int P2 = 32* 1 * SADWindowSize* SADWindowSize;
	sgbm->setP1(P1);
	sgbm->setP2(P2);
	sgbm->setPreFilterCap(15);
	sgbm->setUniquenessRatio(10);
	sgbm->setSpeckleRange(2);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setDisp12MaxDiff(1);
	//sgbm->setMode(cv::StereoSGBM::MODE_HH);
	cv::Mat disp, disp8;
	while(1)
	{
		cv::Mat leftsrc  = cv::imread("aloeL.jpg",0);
		cv::Mat rightsrc = cv::imread("aloeR.jpg",0);

		cv::Mat image1;
		cv::Mat image2;

		cv::resize(leftsrc,image1,cv::Size(320,240),0,0,1);
		cv::resize(rightsrc,image2,cv::Size(320,240),0,0,1);

		cv::resize(leftsrc,leftsrc,cv::Size(320,240),0,0,1);
		cv::resize(rightsrc,rightsrc,cv::Size(320,240),0,0,1);

		cout<<image1.size()<<endl;
 

 	#if 1
	//单线程
	//设置匹配参数
    	/********************SGBM*************************************/
    	///(640,480)

        int64 match_time = getTickCount();
        //sgbm
    	sgbm->compute(leftsrc, rightsrc, disp);
	
        //bm
        //bm->compute(rectifyImageL, rectifyImageR, disp);
		disp.convertTo(disp, CV_32F, 1.0 / 16);                //除以16得到真实视差值
		Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);       //显示
		normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
        cout << "match_time_one_thread=======" << ((getTickCount() - match_time) / getTickFrequency())*1000 << "ms"<<endl;

	imshow("depth_src",disp8U);
	waitKey(1);

	#endif
 

 #if 1
	/*使用多线程图像处理*/
	int64 match_time2 = getTickCount();
	int type = 0;
	vector<cv::Mat> v  = splitImage(image1, THREAD_NUMS, type);
	vector<cv::Mat> v1 = splitImage(image1, THREAD_NUMS, type);
	vector<cv::Mat> v2 = splitImage(image2, THREAD_NUMS, type);

	//for(int i=0;i<4;i++)
	//{
	   //imshow("left",v1[0]);
	   //imshow("right",v2[0]);
	   //waitKey(0);
	//}

	paramThread args[THREAD_NUMS];
	pthread_t pt[THREAD_NUMS];	//创建THREAD_NUMS个子线程
	for (size_t i = 0; i < THREAD_NUMS; i++)
	{
		args[i].h 			= v1.at(i).rows;
		args[i].w 			= v1.at(i).cols;
		args[i].data1 		= v1.at(i).data; 
		args[i].data2 		= v2.at(i).data;
		args[i].depth_data	= v.at(i).data;

		pthread_create(&pt[i], NULL, &threadProcess, (void *)(&args[i]));
	}
	/*等待全部子线程处理完毕*/
	for (size_t i = 0; i < THREAD_NUMS; i++)
	{
		pthread_join(pt[i], NULL);
	}
	//图像合并
	int t = 0; 
	Mat MergeImage(Size(WIDTH, HEIGHT), 0);
	cv::Rect rect1,rect2,rect3,rect4;
	cv::Mat roiImage1,roiImage2,roiImage3,roiImage4;

	rect1 =Rect(0 * SUB_WIDTH, 		0 * SUB_HEIGHT, SUB_WIDTH, SUB_HEIGHT);
	rect2 =Rect(1 * SUB_WIDTH, 		0 * SUB_HEIGHT, SUB_WIDTH, SUB_HEIGHT);
	rect3 =Rect(0 * SUB_WIDTH, 		1 * SUB_HEIGHT, SUB_WIDTH, SUB_HEIGHT);
	rect4 =Rect(1 * SUB_WIDTH, 		1 * SUB_HEIGHT, SUB_WIDTH, SUB_HEIGHT);

	roiImage1 =v[0](cv::Rect(0,0,  v[0].cols-width_roi,SUB_HEIGHT));
	roiImage2 =v[1](cv::Rect(width_roi,0 ,v[1].cols-width_roi,SUB_HEIGHT));
	roiImage3 =v[2](cv::Rect(0,0,  v[2].cols-width_roi,SUB_HEIGHT));
	roiImage4 =v[3](cv::Rect(width_roi,0, v[3].cols-width_roi,SUB_HEIGHT));

	roiImage1.copyTo(MergeImage(rect1));
	roiImage2.copyTo(MergeImage(rect2));
	roiImage3.copyTo(MergeImage(rect3));
	roiImage4.copyTo(MergeImage(rect4));
	t++;

	#if 0
	for (int j = 0; j < N; j++)
	{
		for (int i = 0; i < M; i++)
		{
			Rect ROI(i * SUB_WIDTH, j * SUB_HEIGHT, SUB_WIDTH, SUB_HEIGHT);
			v[t].copyTo(MergeImage(ROI));
			t++;
		}
	}
	#endif
	
	cout << "match_time_4_threasd======" << ((getTickCount() - match_time2) / getTickFrequency())*1000 << "ms"<<endl;
 
	cv::imshow("depth_all", MergeImage); 
	cv::waitKey(1);
#endif
	}

	
	//cv::imshow("image2", image2); cv::waitKey(30);
 
 
	//cv::waitKey(0);
	return 0;
}
