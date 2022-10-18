#include "hw05.h"

//====================================================================================
vector<int> ageList;
vector<Mat> getCroppedFace(Mat inputImage){

	string detector_path = "../src/haarcascade_frontalface_default.xml";
	Mat frame_gray, croppedImage;
	std::vector<cv::Rect> features; // features define my region of interest my ROI
	vector<Mat> return_vec;


    cvtColor( inputImage, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

	CascadeClassifier classifier(detector_path);
	classifier.detectMultiScale(frame_gray, features);
	for (auto&& feature : features) {
    	cv::rectangle(inputImage, feature, cv::Scalar(0, 255, 0), 2);
    }
	if(features.size()==1){
		Mat faceCrop(inputImage, features[0]); // crop face from input image with features of ROI, this is only when single face was found in the image.
		croppedImage = faceCrop;
	}
	else{
		croppedImage = inputImage;
	}

	cv::resize(croppedImage, croppedImage, cv::Size(224, 224)); // Resize the face to match the standard input to the model

	// cv::imshow("Detected face!", inputImage); // Print Image with rectangular area of intrest
	// cv::imshow("Extracted face!", croppedImage); // Display cropped face
	// waitKey(0);
	return_vec.push_back(inputImage);
	return_vec.push_back(croppedImage);

	return return_vec; // This is a vector of Mats and its contains framed inputImage and cropped face
}

vector<float> getAgeProbs(Mat faceROI){

	// faceROI is a cropped face's region of intrest

    string ageProto = "../src/age_deploy.prototxt";
    string ageModel = "../src/age_net.caffemodel";

	Scalar MODEL_MEAN_VALUES = Scalar(78.4263377603, 87.7689143744, 114.895847746);

	for(int i = 0; i<=100; i++){
		ageList.push_back(i);
	}

	// Load Network
    Net ageNet = readNetFromCaffe(ageProto, ageModel);

	ageNet.setPreferableBackend(DNN_TARGET_CPU);

	Mat blob;
	cv::dnn::blobFromImage(faceROI, blob, 1, cv::Size(224, 224), MODEL_MEAN_VALUES, false);

	ageNet.setInput(blob);

	vector<float> ageProbs = ageNet.forward();

	return ageProbs;
}

int getAges1(vector<float> ageProbs){

	int max_indice_age = std::distance(ageProbs.begin(), max_element(ageProbs.begin(), ageProbs.end()));
	int age = ageList[max_indice_age];
	cout << "Age: " << age << endl;
	int label = age; // label

	return age;
}

//=====================================================================================
int hw05::getAge(std::string inputImagePath, std::string modelDataPath) {
	DebugManager::writePrintfToLog("hw05::getAge",
			"Starting getAge");
	try {
		Mat inputImage = imread(inputImagePath.c_str());
		if (inputImage.empty()) {
			std::stringstream errMsg;
			errMsg << "Could not load input image '" << inputImagePath << "'";
			throw std::runtime_error(errMsg.str());
		}else
			DebugManager::writePrintfToLog("hw05::getAge",
				"Finished reading image: %s", inputImagePath.c_str());
		// int age = 1991;
		// return age;
		vector<Mat> crop = getCroppedFace(inputImage);

		// Unpack vector crop to get cropped face and framed input Image
		int it = 0;
		Mat face, framedInputImage;
		for (Mat x : crop){
			if (it ==0){
				framedInputImage = x;
			}else{
				face = x;
			}
			it++;
		}

		vector<float> ageProbs = getAgeProbs(face);

		int age = getAges1(ageProbs);
		FILE* stream = fopen(modelDataPath.c_str(), "w");
		fprintf(stream, "%s, %d\n", inputImagePath.c_str(), age);
		fflush(stream);
		fclose(stream);

		// cv::imshow("Detected face!!", framedInputImage); // Print Image with rectangular area of intrest
		// cv::imshow("Extracted face!!", face); // Display cropped face
		// waitKey(10);


	} catch (std::exception const &errThrown) {
		std::cout << "--error start-- \n\n" << errThrown.what()
				<< "\n\n--error end-- \n\n";
		DebugManager::writePrintfToLog("hw05::getAge", "Error thrown %s",
				errThrown.what());
		return -1;
	}
}

void hw05::getAge(std::string inputImagePath, int trainingMethod, std::string modelDataPath){
	DebugManager::writePrintfToLog("hw05::getAge",
			"Starting hw05::getAge method=%d", trainingMethod);

	std::vector < cv::String > images;
	cv::glob(inputImagePath, images); //This will get all image files from the folder

	switch (trainingMethod){
					case 1:
						return getAge1(images, modelDataPath);
					case 2:
						return getAge2(images, modelDataPath);
					case 3:
						return getAge3(images, modelDataPath);
					case 4:
						return getAge4(images, modelDataPath);
	}
}


void hw05::getAge1(std::vector < cv::String >& images, std::string modelDataPath){


	FILE* stream = fopen(modelDataPath.c_str(), "w"); //Opening output file.
	cv::Mat inputImage;  //Mat file to store each input image.
	int classificationResult = 0;  //Result for each image to be stored in this variable.
	char errorMessage[1024];

	cout<<"Solution Model 1 is instantiated successfully!"<<endl;
	// Looping over all images in the directory
	Mat face, framedInputImage;
	for (int i { 0 }; i < images.size(); i++) {
		inputImage = cv::imread(images[i]);
		vector<Mat> crop = getCroppedFace(inputImage);

		// Unpack vector crop to get cropped face and framed input Image
		int it = 0;
		for (Mat x : crop){
			if (it ==0){
				framedInputImage = x;
			}else{
				face = x;
			}
			it++;
		}

		vector<float> ageProbs = getAgeProbs(face);

		int max_indice_age = std::distance(ageProbs.begin(), max_element(ageProbs.begin(), ageProbs.end()));
		max_indice_age = max_indice_age<101 ? max_indice_age:100; 
		int age = ageList[max_indice_age];
		fprintf(stream, "%s, %d\n", images[i].c_str(), age);

	}
	fflush(stream);
	fclose(stream);

	cv::imshow("Detected face!!", framedInputImage); // Print Image with rectangular area of intrest
	cv::imshow("Extracted face!!", face); // Display cropped face
	waitKey(10);
}


void hw05::getAge2(std::vector < cv::String >& images, std::string modelDataPath){


	FILE* stream = fopen(modelDataPath.c_str(), "w"); //Opening output file.
	cv::Mat inputImage;  //Mat file to store each input image.
	int classificationResult = 0;  //Result for each image to be stored in this variable.
	char errorMessage[1024];

	cout<<"Solution Model 2 is instantiated successfully!"<<endl;
	// Looping over all images in the directory
	for (int i { 0 }; i < images.size(); i++) {
		inputImage = cv::imread(images[i]);
		vector<Mat> crop = getCroppedFace(inputImage);

		// Unpack vector crop to get cropped face and framed input Image
		int it = 0;
		Mat face, framedInputImage;
		for (Mat x : crop){
			if (it ==0){
				framedInputImage = x;
			}else{
				face = x;
			}
			it++;
		}

		vector<float> ageProbs = getAgeProbs(face);

		// FIND AGES BY LINEAR INTERPOLATION
		float age = 0.0, denom = 0;
		for(int it =0; it< 101; it++){
			age += ageProbs[it]*ageList[it];
			denom += ageProbs[it];
		}
		age /= denom;

		int new_age = (int) age;
		fprintf(stream, "%s, %d\n", images[i].c_str(), new_age);

	}
	fflush(stream);
	fclose(stream);
}

void hw05::getAge3(std::vector < cv::String >& images, std::string modelDataPath){


	FILE* stream = fopen(modelDataPath.c_str(), "w"); //Opening output file.
	cv::Mat inputImage;  //Mat file to store each input image.
	int classificationResult = 0;  //Result for each image to be stored in this variable.
	char errorMessage[1024];

	cout<<"Solution Model 3 is instantiated successfully!"<<endl;
	// Looping over all images in the directory
	for (int i { 0 }; i < images.size(); i++) {
		inputImage = cv::imread(images[i]);
		vector<Mat> crop = getCroppedFace(inputImage);

		// Unpack vector crop to get cropped face and framed input Image
		int it = 0;
		Mat face, framedInputImage;
		for (Mat x : crop){
			if (it ==0){
				framedInputImage = x;
			}else{
				face = x;
			}
			it++;
		}

		vector<float> ageProbs = getAgeProbs(face);

		// FIND THE AGE BY NEIGHBORHOOD INTERPOLATION

		int max_indice_age = std::distance(ageProbs.begin(), max_element(ageProbs.begin(), ageProbs.end()));
		max_indice_age = max_indice_age<101 ? max_indice_age:100; 

		float age = 0.0, denom = 0;
		for(int it = max_indice_age-5; it< max_indice_age+5; it++){
			age += ageProbs[it]*ageList[it];
			denom += ageProbs[it];
		}
		age /= denom;

		int new_age = (int) age;


		fprintf(stream, "%s, %d\n", images[i].c_str(), new_age);

	}
	fflush(stream);
	fclose(stream);
}

void hw05::getAge4(std::vector < cv::String >& images, std::string modelDataPath){


	FILE* stream = fopen(modelDataPath.c_str(), "w"); //Opening output file.
	cv::Mat inputImage;  //Mat file to store each input image.
	int classificationResult = 0;  //Result for each image to be stored in this variable.
	char errorMessage[1024];

	cout<<"Solution Model 4 is instantiated successfully!"<<endl;
	// Looping over all images in the directory
	for (int i { 0 }; i < images.size(); i++) {
		inputImage = cv::imread(images[i]);
		vector<Mat> crop = getCroppedFace(inputImage);

		// Unpack vector crop to get cropped face and framed input Image
		int it = 0;
		Mat face, framedInputImage;
		for (Mat x : crop){
			if (it ==0){
				framedInputImage = x;
			}else{
				face = x;
			}
			it++;
		}

		vector<float> ageProbs = getAgeProbs(face);

		// FIND THE AGE BY NEIGHBORHOOD-LOCAL-LINEAR INTERPOLATION
		 
		int max_indice_age = std::distance(ageProbs.begin(), max_element(ageProbs.begin(), ageProbs.end()));
		max_indice_age = max_indice_age<101 ? max_indice_age:100; 

		int max_idx = max_indice_age;
		float age = 0.0, denom = 0;
		if(max_idx<3){			
			for(int it = 0; it< 3; it++){
				age += ageProbs[it]*ageList[it];
				denom += ageProbs[it];
		}
		}else if(max_idx<7){
			for(int it = 3; it< 7; it++){
				age += ageProbs[it]*ageList[it];
				denom += ageProbs[it];
		}
		}else if(max_idx<14){
			for(int it = 7; it< 14; it++){
				age += ageProbs[it]*ageList[it];
				denom += ageProbs[it];
		}
		}else if(max_idx<20){
			for(int it = 14; it< 20; it++){
				age += ageProbs[it]*ageList[it];
				denom += ageProbs[it];
		}
		}else if(max_idx<27){
			for(int it = 20; it< 27; it++){
				age += ageProbs[it]*ageList[it];
				denom += ageProbs[it];
		}
		}else if(max_idx<37){
			for(int it = 27; it< 37; it++){
				age += ageProbs[it]*ageList[it];
				denom += ageProbs[it];
		}
		}else if(max_idx<46){
			for(int it = 37; it< 46; it++){
				age += ageProbs[it]*ageList[it];
				denom += ageProbs[it];
		}
		}else if(max_idx<59){
			for(int it = 46; it< 59; it++){
				age += ageProbs[it]*ageList[it];
				denom += ageProbs[it];
		}
		}else if(max_idx<75){
			for(int it = 59; it< 75; it++){
				age += ageProbs[it]*ageList[it];
				denom += ageProbs[it];
		}
		}else{
			for(int it = 75; it< 101; it++){
				age += ageProbs[it]*ageList[it];
				denom += ageProbs[it];
		}
		}

		age /= denom;

		int new_age = (int) age;

		fprintf(stream, "%s, %d\n", images[i].c_str(), new_age);

	}
	fflush(stream);
	fclose(stream);
}
