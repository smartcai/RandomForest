#include "randomforest.hpp"
#include "randomforest_base.hpp"
#include "WriteData.h"
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;
using namespace handlib;

int main()
{
	CRandomForest randomforest_hand_detector;
	CTrainParam tp;

	//Train:
	//randomforest_hand_detector.TrainForest(tp);
	
	randomforest_hand_detector.LoadForest();

	//Test and save into TXT file:
	int num_image = 8252;
	string img_dir = "E:\\datasets\\nyu_hand_dataset_v2\\dataset\\test\\";
	string txt_dir = "E:\\datasets\\nyu_hand_dataset_v2\\dataset\\mask\\test\\";

	for (int i = 1; i <= num_image; i++)
	{
		stringstream file_name;
		stringstream txt_name;
		file_name << img_dir << "depth_1_00" << (i < 10 ? "0000" :\
			(i < 100 ? "000" : (i < 1000 ? "00" : (i < 10000 ? "0" : "" )))) << i << ".png";
		Mat img = imread(file_name.str());
		if (img.empty())
		{
			cout << file_name.str() << "not existd!" << endl;
			continue;
		}
	
		//imshow("res", randomforest_hand_detector.Detect(img));
		//imwrite("result.png", randomforest_hand_detector.Detect(img));

		txt_name << txt_dir << "depth_1_00" << (i < 10 ? "0000" : \
			(i < 100 ? "000" : (i < 1000 ? "00" : (i < 10000 ? "0" : "")))) << i << ".txt";
		WriteData( txt_name.str(), randomforest_hand_detector.Detect(img));

		img.release();
		cout << "Data " << file_name.str() << " tested and saved." << endl;
	}
	
	waitKey();
	return 0;
}