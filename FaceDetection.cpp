/*
 * File: DetectFaces.cpp
 * Description: Used to detect faces from either photos or webcam and
 *              is manipulated to match a particular description. 
 *				This file is used to create the database. 
 */

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(cv::Mat frame);

/** Global variables*/
String path = "E:/OpenCV-2.4.13/opencv/sources/data/haarcascades/";
String face_cascade_name = path + "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

int image_counter = 01;

String window_name = "Capture - Face Detection";
int main()
{
	cv::VideoCapture camera;
	cv::Mat frame;
	

	// -- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) {
		cout << "--(!)Error loading face cascade" << endl;
		return -1;
	}
	
	///*
	// -- 2.Read the video stream
	camera.open(0);
	if (!camera.isOpened()) {
		cout << "--(!)Error opening video capture" << endl;
		return -1;
	}

	int key = 0;
	do {

		camera.read(frame);
		if (frame.empty()) {
			cout << "--(!)No captured frame -- Break!" << endl;
			break;
		}

		// -- 3.Apply the classifier to the frame
		detectAndDisplay(frame);

		key = waitKey(5);

	} while (key != 27);
	//*/

	/*
	// --2. Detect faces in photos
	frame = cv::imread("s3_01.jpg", CV_LOAD_IMAGE_COLOR);
	// -- 3.Apply the classifier to the frame
	file << "image02" << image_counter << ".jpg";
	image_counter++;
	detectAndDisplay(frame);
	cv::waitKey(0);
	//*/

	/*
	// --2. Detect faces in videos
	string videoFile = "HeartRate1.avi";
	VideoCapture capture(videoFile);


	if (!capture.isOpened())
		throw "Error when reading steam_avi";

	for (;;)
	{
		capture >> frame;
		if (frame.empty())
			break;
		
		detectAndDisplay(frame);
		//waitKey(1); // waits to display frame
	}
	*/

	waitKey(0);
	return 0;
}

/* @function detectAndDisplay */
void detectAndDisplay(cv::Mat frame) {
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// -- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	cv::Mat resizedFace;
	if (!faces.empty()) {
		for (size_t i = 0; i < faces.size(); i++) {
			
			cv::Rect face_i = faces[i];
			cv::Mat face = frame_gray(faces[i]);
			rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(255, 0, 0), 4, 8, 0);
			cv::resize(face, resizedFace, Size(92, 112), 1.0, 1.0, INTER_CUBIC);
			

		}
	}

	// --Show what you got
	stringstream file;
	file << "image03" << image_counter << ".jpg";
	image_counter++;
	cv::imwrite(file.str(), resizedFace);
	imshow(window_name, frame);
	

}