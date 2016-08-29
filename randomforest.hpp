#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include "randomforest_base.hpp"

namespace handlib
{

	class CRandomForest
	{
	private:
		CTrainParam tp;
		CTrainingData td;
		std::vector<std::vector<CNode> > trees;

		void TrainTree(int tree_id);
		inline bool TestLeaf(int l, int r, int dep);
		inline int GetFeature(CPixel &p, int du, int dv);
		inline int GetFeature(cv::Mat &img, int u, int v, int du, int dv);
		float GetProb(int l, int r);
		CSplitCandidate FindBestPhi(int l, int r);
		int SortData(int l, int r, CSplitCandidate &phi);

		float Predict(cv::Mat &img, int u, int v);

		void SaveNode(int tree_id, int node, std::ofstream &fout);
		void SaveTree(int tree_id, std::ofstream &fout);
		void LoadNode(int tree_id, std::string node_type, std::ifstream &fin);
		void LoadTree(int tree_id, std::ifstream &fin);
	public:
		CRandomForest() {}
		~CRandomForest();

		void TrainForest(CTrainParam &train_param);
		void SaveForest(std::string file_name = "FOREST.model");
		void LoadForest(std::string file_name = "FOREST.model");
		cv::Mat Detect(cv::Mat &img);
	};

}


#endif