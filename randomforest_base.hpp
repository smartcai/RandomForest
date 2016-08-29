#ifndef RANDOMFOREST_BASE_H
#define RANDOMFOREST_BASE_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <stack>
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace handlib
{

	const int BACKGROUND_DEPTH = 1994;
	const float EPS = 1e-6;
	const int FOREGROUND_BACKGROUND_BALANCE = 10;
	const float DELTA = 0.05;
	const float Inf = 1e30;

	inline float RandFloat(float R)
	{
		float tmp = 1.0 * rand() / ((unsigned)RAND_MAX + 1);
		return tmp * R;
	}

	inline float RandFloatLog(float R)
	{
		float tmp = log(R + 1);
		return exp(RandFloat(tmp)) - 1;
	}

	class CPixel
	{
	public:
		int u, v, id, f;
		CPixel(){}
		CPixel(int u, int v, int id) : u(u), v(v), id(id){}
		bool operator < (CPixel &u) const
		{
			return f < u.f;
		}
	};

	class CTrainingData
	{
	public:
		std::vector<cv::Mat> images;
		std::vector<CPixel> data;

		CTrainingData(){}
		~CTrainingData();
		CTrainingData(std::string img_dir, int num_image, int num_pixel);
		void shuffle();
		void SortDataByFeature(int l, int r);
		int GetDepth(int u, int v, cv::Mat &img);
		float GetLabel(int u, int v, cv::Mat &img);
		int GetDepth(CPixel &p);
		float GetLabel(CPixel &p);
	};

	class CSplitCandidate
	{
	public:
		int du, dv, tau;
		CSplitCandidate(){}
		CSplitCandidate(int du, int dv, int tau) : du(du), dv(dv), tau(tau){}
		static CSplitCandidate RandSplitCandidate(int range_offset);
	};

	class CNode
	{
	public:
		int left, right;
		CSplitCandidate phi;
		float prob;
		CNode();
		CNode(float p);
		CNode(CSplitCandidate phi);
		bool isLeaf();
	};

	class CStackElement
	{
	public:
		int node, l, r, dep;
		CStackElement(){}
		CStackElement(int node, int l, int r, int dep) : node(node), l(l), r(r), dep(dep){}
	};

	class CTrainParam
	{
	public:
		int num_tree;
		int num_image;
		int num_pixel;
		int num_offset;
		int max_dep;
		int min_sample;
		float rate_bagging;

		int range_offset;

		std::string img_dir;
		std::string out_name;

		CTrainParam();
	};









}




#endif