/*
* File: FaceRecognition.cpp
* Description: Used to detect faces from the webcam and try to recognize them
*              based on the training model being built. 
*			   This file is to create the training model for the project.
*/

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main() {
	
	// Get the path to CSV and store them the vectors initialized below
	string fn_csv = "Dataset/Labels.csv";
	vector<Mat> images;
	vector<int> labels;

	// Read the CSV file
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}


	// If not sufficient images are present
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}

	// Get height and width from the 1st image to resize
	int imgHeight = images[0].rows;
	int imgWidth = images[0].rows;
	
	// train the model based on the dataset
	cout << "Training the model" << endl;
	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
	model->train(images, labels);
	model->save("trainingModel.yml");
	//model->load("trainingModel.yml");

	// Use the video camera to test the model and 
	// calculate the accuracy of the model

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

	for (;;){

		camera >> frame;
		Mat original = frame.clone();

		Mat frame_gray;
		cvtColor(original, frame_gray, CV_BGR2GRAY);

		// Detect all faces in the video frame
		vector<Rect_<int> > faces;
		face_cascade.detectMultiScale(frame_gray, faces);

		// Try to recognize familiar faces by processing each face
		for (int i = 0; i < faces.size(); i++){

			// Extract the face from the frame and manipulate it
			Rect face_i = faces[i];
			cv::Mat face = frame_gray(face_i);
			cv::Mat resizedFace;
			cv::resize(face, resizedFace, Size(imgHeight, imgWidth), 1.0, 1.0, INTER_CUBIC);

			// Predict the faces
			int prediction = model->predict(resizedFace);
			cout << prediction << endl;

			if (prediction == 1)
				rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
			else if (prediction == 2)
				rectangle(original, face_i, CV_RGB(0, 0, 255), 1);
			else if (prediction == 3)
				rectangle(original, face_i, CV_RGB(0, 0, 255), 1);

		}

		// Show the result:
		imshow("face_recognizer", original);
		// And display it:
		if (waitKey(10) == 27)
			break;
	}

	cv::waitKey(0);
	return 0;
}