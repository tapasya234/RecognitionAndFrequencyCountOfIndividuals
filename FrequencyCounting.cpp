/*
* File: FrequencyCounting.cpp
* Description: Used to recognize faces from the webcam and count the frequency of each recognized frame. 
*			   
*/

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

/* Global Vaviables*/
const int imgHeight = 112;
const int imgWidth = 92;

// Find out fps of webcamera
const int fps = 20;

bool isFaceFound(bool history[]){
	for (int i = 0; i < fps; i++){
		if (history[i]) return true;
	}
	return false;
}

void initArray(bool* s, int n){
	for (int i = 0; i < n; i++){
		s[i] = false;
	}
}

void printArrays(int prediction, bool s1History[], bool s2History[], bool s3History[]){
	//print arrays
	cout << "Prediction: " << prediction << endl << "S1: \t";
	cout << isFaceFound(s1History) << endl;
	cout << "S2: \t" << isFaceFound(s2History) << endl;
	cout << "S3: \t" << isFaceFound(s3History) << endl;


	/*for (int i = 0; i <fps; i++)
		cout << s1History[i] << " ";
	cout << endl << "S2: \t";
	for (int i = 0; i < fps; i++)
		cout << s2History[i] << " ";
	cout << endl << "S3: \t";
	for (int i = 0; i < fps; i++)
		cout << s3History[i] << " ";*/
	cout << endl << "------------------------------" <<endl;

}

int main() {

	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
	model->load("trainingModel.yml");

	// -- 1. Load the cascades
	CascadeClassifier face_cascade;

	string face_cascade_name = "E:/OpenCV-2.4.13/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
	if (!face_cascade.load(face_cascade_name)) {
		cout << "--(!)Error loading face cascade" << endl;
		return -1;
	}

	// -- 2.Read the video stream
	// variableto access the video camera
	int deviceId = 0;
	cv::VideoCapture camera;
	cv::Mat frame;
	camera.open(deviceId);
	if (!camera.isOpened()) {
		cout << "--(!)Error opening video capture" << endl;
		return -1;
	}

	//array to detect entrance into frame after one second or longer
	bool s1History[fps];
	bool s2History[fps];
	bool s3History[fps];
	int frameCounter = 0;
	initArray(s1History, fps);
	initArray(s2History, fps);
	initArray(s3History, fps);

	/*memset(s1History, '-1', 30);
	memset(s2History, '-1', 30);
	memset(s3History, '-1', 30);

	printArrays(0, s1History, s2History, s3History);
	cout << endl << endl;
	*/
	bool found = true;
	bool notFound = false;
	int s1Freq = 0, s2Freq = 0, s3Freq = 0;

	for (;;){

		camera >> frame;
		Mat original = frame.clone();

		Mat frame_gray;
		cvtColor(original, frame_gray, CV_BGR2GRAY);

		// Detect all faces in the video frame
		vector<Rect_<int> > faces;
		face_cascade.detectMultiScale(frame_gray, faces);

		// Try to recognize familiar faces by processing each face
		int prediction = 0;
		s1History[frameCounter] = notFound;
		s2History[frameCounter] = notFound;
		s3History[frameCounter] = notFound;
		for (int i = 0; i < faces.size(); i++){

			// Extract the face from the frame and manipulate it
			Rect face_i = faces[i];
			cv::Mat face = frame_gray(face_i);
			cv::Mat resizedFace;
			cv::resize(face, resizedFace, Size(imgHeight, imgWidth), 1.0, 1.0, INTER_CUBIC);

			// Predict the faces
			prediction = model->predict(resizedFace);
			//cout << prediction << endl;


			s1History[frameCounter] = notFound;
			s2History[frameCounter] = notFound;
			s3History[frameCounter] = notFound;

			if (prediction == 1){
				rectangle(original, face_i, CV_RGB(0, 255, 0), 1);

				if (!isFaceFound(s1History))
					s1Freq++;
				s1History[frameCounter] = found;
			}
			else if (prediction == 2){
				rectangle(original, face_i, CV_RGB(0, 0, 255), 1);

				if (!isFaceFound(s2History))
					s2Freq++;
				s2History[frameCounter] = found;
			}
			else if (prediction == 3){
				rectangle(original, face_i, CV_RGB(0, 0, 255), 1);

				if (!isFaceFound(s3History))
					s3Freq++;
				s3History[frameCounter] = found;
			}			
		}

		frameCounter++;
		if (frameCounter == fps)
			frameCounter = 0;
		printArrays(prediction, s1History, s2History, s3History);
		

		// Show the result:
		imshow("face_recognizer", original);
		// And display it:
		if (waitKey(10) == 27)
			break;
	}

	cout << "Frequency of S1: " << s1Freq << endl;
	cout << "Frequency of S2: " << s2Freq << endl;
	cout << "Frequency of S3: " << s3Freq << endl;
	cv::waitKey(0);
	return 0;
}

