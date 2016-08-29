#include <iostream>  
#include <fstream>  
#include <iterator>  
#include <vector>  
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/*----------------------------
* 功能 : 将 cv::Mat 数据写入到 .txt 文件
*----------------------------
* 函数 : WriteData
* 访问 : public
* 返回 : -1：打开文件失败；0：写入数据成功；1：矩阵为空
*
* 参数 : fileName    [in]    文件名
* 参数 : matData [in]    矩阵数据
*/
int WriteData(string fileName, cv::Mat& matData)
{
	int retVal = 0;

	// 打开文件  
	ofstream outFile(fileName.c_str(), ios_base::out);  //按新建或覆盖方式写入  
	if (!outFile.is_open())
	{
		cout << "打开文件失败" << endl;
		retVal = -1;
		return (retVal);
	}

	// 检查矩阵是否为空  
	if (matData.empty())
	{
		cout << "矩阵为空" << endl;
		retVal = 1;
		return (retVal);
	}

	// 写入数据  
	for (int r = 0; r < matData.rows; r++)
	{
		for (int c = 0; c < matData.cols; c++)
		{
			int data = matData.at<Vec3b>(r, c)[2];  //读取数据，at<type> - type 是矩阵元素的具体数据格式  
			outFile << data << "\t";   //每列数据用 tab 隔开  
		}
		outFile << "\n";
	}

	return (retVal);
}