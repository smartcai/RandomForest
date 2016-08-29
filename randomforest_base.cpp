#include "randomforest.hpp"
#include "randomforest_base.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

namespace handlib
{


	CNode::CNode()
	{
		prob = -1;
	}

	CNode::CNode(CSplitCandidate phi)
	{
		this->phi = phi;
		prob = -1;
	}

	CNode::CNode(float p)
	{
		left = -1;
		right = -1;
		prob = p;
	}

	bool CNode::isLeaf()
	{
		return prob > -EPS;
	}

	CSplitCandidate CSplitCandidate::RandSplitCandidate(int range_offset)
	{
		CSplitCandidate phi;
		phi.du = RandFloat(range_offset);
		phi.dv = RandFloat(range_offset);
		if (rand() % 2)
			phi.du = -phi.du;
		if (rand() % 2)
			phi.dv = -phi.dv;
		return phi;
	}

	void CTrainingData::SortDataByFeature(int l, int r)
	{
		sort(data.begin() + l, data.begin() + r + 1);
	}

	int CTrainingData::GetDepth(int u, int v, Mat &img)
	{
		if (u < 0 || u >= img.rows || v < 0 || v >= img.cols)
		{
			return BACKGROUND_DEPTH;
		}
		return img.at<Vec3b>(u, v)[0] + (img.at<Vec3b>(u, v)[1] << 8);
	}

	float CTrainingData::GetLabel(int u, int v, Mat &img)
	{
		if (u < 0 || u >= img.rows || v < 0 || v >= img.cols)
		{
			return 0;
		}
		return img.at<Vec3b>(u, v)[2] / 255.0;
	}

	int CTrainingData::GetDepth(CPixel &p)
	{
		return GetDepth(p.u, p.v, images[p.id]);
	}

	float CTrainingData::GetLabel(CPixel &p)
	{
		return GetLabel(p.u, p.v, images[p.id]);
	}

	CTrainingData::CTrainingData(std::string img_dir, int num_image, int num_pixel)
	{
		images.clear();
		data.clear();
		float num_foreground = 0;
		for (int i = 0; i < num_image; i++)
		{
			stringstream file_name;

			file_name << img_dir << "depth_000" << (i < 10 ? "000" : \
				(i < 100 ? "00" : (i < 1000 ? "0" : ""))) << i << ".png";
			Mat img = imread(file_name.str());
			if (img.empty())
			{
				cout << file_name.str() << "not existd!" << endl;
				continue;
			}
			images.push_back(img.clone());

			int j = 0;
			while (j < num_pixel)
			{
				int u = 0, v = 0;
				while (GetDepth(u, v, img) >= BACKGROUND_DEPTH)
				{
					u = rand() % img.rows;
					v = rand() % img.cols;
				}
				//            cerr << int(img.at<Vec3b>(u, v)[0]) << " " << int(img.at<Vec3b>(u, v)[1]) << \
				//              " " << int(img.at<Vec3b>(u, v)[2]) << endl;
				//            cerr << "u v " << u << " " << v << endl;
				if (GetLabel(u, v, img) < EPS && rand() % FOREGROUND_BACKGROUND_BALANCE)
				{
					continue;
				}
				j++;
				num_foreground += GetLabel(u, v, img);
				data.push_back(CPixel(u, v, i));
			}
			img.release();
			cout << "Data " << file_name.str() << " loaded." << endl;
		}

		cout << "Load data ok. Foreground ratio is " << num_foreground / data.size() << "." << endl;
	}

	void CTrainingData::shuffle()
	{
		for (int i = 0; i < data.size(); i++)
		{
			int j = rand() % (data.size() - i) + i;
			swap(data[i], data[j]);
		}
	}

	CTrainingData:: ~CTrainingData()
	{
		images.clear();
		data.clear();
	}

	CTrainParam::CTrainParam()
	{
		num_tree = 6;
		num_image = 6736;
		num_pixel = 1000;
		num_offset = 100;
		max_dep = 25;
		min_sample = 5;
		rate_bagging = 0.6;
		range_offset = 100000;
		//img_dir = "D:\\Datasets\\dataset_rdf\\dataset\\";
		img_dir = "E:\\datasets\\nyu_hand_dataset_v2\\dataset_rdf\\dataset\\";
		out_name = "FOREST.model";
	}


}