#include "randomforest.hpp"
#include "randomforest_base.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


namespace handlib
{

	bool CRandomForest::TestLeaf(int l, int r, int dep)
	{
		if (r - l + 1 <= tp.min_sample)
		{
			cout << "Leaf node at " << l << " " << r << " : " << "too few samples." << endl;
			return true;
		}

		if (dep > tp.max_dep)
		{
			cout << "Leaf node at " << l << " " << r << " : " << "max depth reached." << endl;
			return true;
		}

		float avg_label = 0;
		for (int i = l; i <= r; i++)
			avg_label += td.GetLabel(td.data[i]);
		avg_label /= (r - l + 1);
		if (avg_label < DELTA || avg_label > 1 - DELTA)
		{
			cout << "Leaf node at " << l << " " << r << " : " << "label equal." << endl;
			return true;
		}

		return false;
	}

	float CRandomForest::GetProb(int l, int r)
	{
		float avg_label = 0;
		for (int i = l; i <= r; i++)
			avg_label += td.GetLabel(td.data[i]);
		return avg_label / (r - l + 1);
	}

	int CRandomForest::GetFeature(Mat &img, int u, int v, int du, int dv)
	{
		int d = td.GetDepth(u, v, img);
		return td.GetDepth(u + du / d, v + dv / d, img) - d;
	}

	int CRandomForest::GetFeature(CPixel &p, int du, int dv)
	{
		int d = td.GetDepth(p);
		return td.GetDepth(CPixel(p.u + du / d, p.v + dv / d, p.id)) - d;
	}

	int CRandomForest::SortData(int l, int r, CSplitCandidate &phi)
	{
		for (int i = l; i <= r; i++)
			td.data[i].f = GetFeature(td.data[i], phi.du, phi.dv);
		td.SortDataByFeature(l, r);

		for (int i = l; i <= r; i++)
		if (td.data[i].f >= phi.tau)
			return i;
		return r;
	}

	CSplitCandidate CRandomForest::FindBestPhi(int l, int r)
	{
		float best_gain = -Inf;
		CSplitCandidate best_phi;

		for (int t = 0; t < tp.num_offset; t++)
		{
			CSplitCandidate phi = CSplitCandidate::RandSplitCandidate(tp.range_offset);

			//cerr << "split " << t << " " << phi.du << " " << phi.dv << endl;

			float sum_label = 0;
			for (int i = l; i <= r; i++)
			{
				td.data[i].f = GetFeature(td.data[i], phi.du, phi.dv);
				sum_label += td.GetLabel(td.data[i]);
			}

			td.SortDataByFeature(l, r);
			float gain = -Inf;
			int card_L = 0;
			int card_R = r - l + 1;
			float cur_label = 0;
			for (int i = l; i < r; i++)
			{
				cur_label += td.GetLabel(td.data[i]);
				card_L++;
				card_R--;

				float pl = cur_label / card_L;
				float pr = (sum_label - cur_label) / card_R;
				float Hl, Hr;

				if (pl < EPS || pl > 1 - EPS)
					Hl = 0;
				else
					Hl = pl * log2(pl) + (1 - pl) * log2(1 - pl);

				if (pr < EPS || pr > 1 - EPS)
					Hr = 0;
				else
					Hr = pr * log2(pr) + (1 - pr) * log2(1 - pr);

				if (card_L * Hl + card_R * Hr > gain)
				{
					gain = card_L * Hl + card_R * Hr;
					phi.tau = (td.data[i].f + td.data[i + 1].f + 1) / 2;
				}
			}

			if (gain > best_gain)
			{
				best_gain = gain;
				best_phi = phi;
			}
		}

		return best_phi;
	}




	void CRandomForest::TrainTree(int tree_id)
	{
		stack<CStackElement> stk;
		int mid, cnt = 0;
		cout << "Training tree " << tree_id << "." << endl;

		td.shuffle();
		trees[tree_id].clear();
		trees[tree_id].push_back(CNode());
		stk.push(CStackElement(0, 0, int(td.data.size() * tp.rate_bagging), 0));

		while (!stk.empty())
		{
			int node = stk.top().node;
			int l = stk.top().l;
			int r = stk.top().r;
			int dep = stk.top().dep;

			cerr << "Running Range " << l << " " << r << endl;

			stk.pop();

			trees[tree_id][node].phi = FindBestPhi(l, r);
			mid = SortData(l, r, trees[tree_id][node].phi);

			if (TestLeaf(l, mid - 1, dep + 1))
			{
				trees[tree_id].push_back(CNode(GetProb(l, mid - 1)));
				trees[tree_id][node].left = trees[tree_id].size() - 1;
			}
			else
			{
				trees[tree_id].push_back(CNode());
				trees[tree_id][node].left = trees[tree_id].size() - 1;
				stk.push(CStackElement(trees[tree_id][node].left, l, mid - 1, dep + 1));
			}

			if (TestLeaf(mid, r, dep + 1))
			{
				trees[tree_id].push_back(CNode(GetProb(mid, r)));
				trees[tree_id][node].right = trees[tree_id].size() - 1;
			}
			else
			{
				trees[tree_id].push_back(CNode());
				trees[tree_id][node].right = trees[tree_id].size() - 1;
				stk.push(CStackElement(trees[tree_id][node].right, mid, r, dep + 1));
			}
		}
	}


	void CRandomForest::TrainForest(CTrainParam &train_param)
	{
		tp = train_param;
		td = CTrainingData(tp.img_dir, tp.num_image, tp.num_pixel);

		trees.resize(tp.num_tree);
		for (int i = 0; i < tp.num_tree; i++)
		{
			TrainTree(i);
		}
		cout << "Train forest ok." << endl;
		td.~CTrainingData();
		SaveForest(tp.out_name);
	}




	void CRandomForest::SaveNode(int tree_id, int node, ofstream &fout)
	{
		if (trees[tree_id][node].isLeaf())
			fout << "L " << trees[tree_id][node].prob << endl;
		else
			fout << "S " << trees[tree_id][node].phi.du << " " << \
			trees[tree_id][node].phi.dv << " " << \
			trees[tree_id][node].phi.tau << endl;
	}

	void CRandomForest::SaveTree(int tree_id, ofstream &fout)
	{
		stack<int> stk;

		SaveNode(tree_id, 0, fout);
		stk.push(0);
		while (!stk.empty())
		{
			int node = stk.top();
			stk.pop();

			int left = trees[tree_id][node].left;
			if (!(trees[tree_id][left].isLeaf()))
				stk.push(left);
			SaveNode(tree_id, left, fout);

			int right = trees[tree_id][node].right;
			if (!(trees[tree_id][right].isLeaf()))
				stk.push(right);
			SaveNode(tree_id, right, fout);
		}
	}

	void CRandomForest::SaveForest(string file_name)
	{
		ofstream fout(file_name);
		fout << trees.size() << endl;
		for (int i = 0; i < trees.size(); i++)
			SaveTree(i, fout);
		fout.close();
		cout << "Saved forest to " << file_name << " ." << endl;
	}


	void CRandomForest::LoadNode(int tree_id, string node_type, ifstream &fin)
	{
		if (node_type[0] == 'L')
		{
			float prob;
			fin >> prob;
			trees[tree_id].push_back(CNode(prob));
		}
		else
		if (node_type[0] == 'S')
		{
			CSplitCandidate phi;
			fin >> phi.du >> phi.dv >> phi.tau;
			trees[tree_id].push_back(CNode(phi));
		}
		else
		{
			cout << "Error Input Format!" << endl;
			exit(-1);
		}
	}

	void CRandomForest::LoadTree(int tree_id, ifstream &fin)
	{
		stack<int> stk;
		string node_type;

		fin >> node_type;
		LoadNode(tree_id, node_type, fin);

		stk.push(0);
		while (!stk.empty())
		{
			int node = stk.top();
			stk.pop();

			fin >> node_type;
			LoadNode(tree_id, node_type, fin);
			trees[tree_id][node].left = trees[tree_id].size() - 1;
			if (node_type[0] == 'S')
				stk.push(trees[tree_id][node].left);

			fin >> node_type;
			LoadNode(tree_id, node_type, fin);
			trees[tree_id][node].right = trees[tree_id].size() - 1;
			if (node_type[0] == 'S')
				stk.push(trees[tree_id][node].right);
		}
	}

	void CRandomForest::LoadForest(string file_name)
	{
		ifstream fin(file_name);
		int num_tree;
		fin >> num_tree;
		trees.resize(num_tree);
		for (int i = 0; i < num_tree; i++)
		{
			LoadTree(i, fin);
		}
		fin.close();
		cout << "Loaded forest from " << file_name << " ." << endl;
	}



	float CRandomForest::Predict(Mat &img, int u, int v)
	{
		float prob = 0;
		for (int tree_id = 0; tree_id < trees.size(); tree_id++)
		{
			int node = 0;
			while (!(trees[tree_id][node].isLeaf()))
			{
				CSplitCandidate phi = trees[tree_id][node].phi;
				int feature = GetFeature(img, u, v, phi.du, phi.dv);
				node = feature < phi.tau ? trees[tree_id][node].left : \
					trees[tree_id][node].right;
			}
			prob += trees[tree_id][node].prob;
		}
		prob /= trees.size();
		//    cerr << "prob " << prob << endl;
		return prob;
	}

	Mat CRandomForest::Detect(Mat &img)
	{
		//Mat res =  img.clone();
		Mat res = Mat(img.rows, img.cols, CV_8UC3);

		for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		if (res.at<Vec3b>(i, j)[0] + (res.at<Vec3b>(i, j)[1] << 8) < BACKGROUND_DEPTH)
		{
			//res.at<Vec3b>(i, j)[0] = int(Predict(img, i, j) * 255);
			//res.at<Vec3b>(i, j)[1] = int(Predict(img, i, j) * 255);
			//res.at<Vec3b>(i, j)[2] = int(Predict(img, i, j) * 255);
			res.at<Vec3b>(i, j)[2] = int(Predict(img, i, j)*255) ;
			//cerr << (int)res.at<Vec3b>(i, j)[2] << endl;

		}
		return res;
	}


	CRandomForest:: ~CRandomForest()
	{
		trees.clear();
	}


}