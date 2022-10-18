#ifndef HW05_H_
#define HW05_H_

#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/objdetect.hpp"
#include <tuple>
#include <iterator>
#include "util/DebugManager.h"
using namespace cv;
using namespace cv::dnn;
using namespace std;

class hw05{

public:
	/**
	 * Takes as input path to an images, and outputs the year of birth of the person in the photo.
	 *
	 * @param inputImagePath input image path.
	 * @param modelDataPath Path to any model file that is used.
	 * @returns Age of the person in YYYY format, i.e. Year of birth of person. In case of an error, a -ve value is returned.
	 */
	static int getAge(std::string inputImagePath, std::string modelDataPath);

	static void getAge(std::string inputImagePath, int trainingMethod, std::string modelDataPath);

	static void getAge1(std::vector < cv::String >& images, std::string modelDataPath);
	static void getAge2(std::vector < cv::String >& images, std::string modelDataPath);
	static void getAge3(std::vector < cv::String >& images, std::string modelDataPath);
	static void getAge4(std::vector < cv::String >& images, std::string modelDataPath);

};


#endif
