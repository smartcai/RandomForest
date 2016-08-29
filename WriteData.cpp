#include <iostream>  
#include <fstream>  
#include <iterator>  
#include <vector>  
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/*----------------------------
* ���� : �� cv::Mat ����д�뵽 .txt �ļ�
*----------------------------
* ���� : WriteData
* ���� : public
* ���� : -1�����ļ�ʧ�ܣ�0��д�����ݳɹ���1������Ϊ��
*
* ���� : fileName    [in]    �ļ���
* ���� : matData [in]    ��������
*/
int WriteData(string fileName, cv::Mat& matData)
{
	int retVal = 0;

	// ���ļ�  
	ofstream outFile(fileName.c_str(), ios_base::out);  //���½��򸲸Ƿ�ʽд��  
	if (!outFile.is_open())
	{
		cout << "���ļ�ʧ��" << endl;
		retVal = -1;
		return (retVal);
	}

	// �������Ƿ�Ϊ��  
	if (matData.empty())
	{
		cout << "����Ϊ��" << endl;
		retVal = 1;
		return (retVal);
	}

	// д������  
	for (int r = 0; r < matData.rows; r++)
	{
		for (int c = 0; c < matData.cols; c++)
		{
			int data = matData.at<Vec3b>(r, c)[2];  //��ȡ���ݣ�at<type> - type �Ǿ���Ԫ�صľ������ݸ�ʽ  
			outFile << data << "\t";   //ÿ�������� tab ����  
		}
		outFile << "\n";
	}

	return (retVal);
}