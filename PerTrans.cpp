#include <vector>
#include <opencv2/opencv.hpp>
#include<algorithm>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

enum elem
	{M11,M12,M13,
	M21,M22,M23,
	M31,M32,M33};

struct point4Op
{
	Point2f calcpoint;
	Point2f realpoint;
	int xIdx;
	int yIdx;
};


bool cmp_x(point4Op x, point4Op y)
{
	return x.calcpoint.x<y.calcpoint.x;
}

bool cmp_y(point4Op x, point4Op y)
{
	return x.calcpoint.y < y.calcpoint.y;
}

float distance(Point2f& a,Point2f& b)
{
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

class codiConvertor
{
public:
	Mat mapMartix;
	float loss;
	vector<point4Op> points4Op;

	codiConvertor() { mapMartix = Mat(); loss = 0; }
	void optimizer(float LR);
	double calcLoss(point4Op point);
	vector<double> getDerivative(point4Op& point);
};

void codiConvertor::optimizer(float LR)
{
	//cout << "size is:" << points4Op.size() << endl;
	float loss = 0;
	for (auto point : points4Op)
	{
		cout << point.calcpoint << endl;
		cout << point.realpoint << endl;
		//LR = LR * -1; //what the fack is that???
		vector<double> derivative = getDerivative(point);
		mapMartix.at<double>(0, 0) += LR * derivative[M11];
		mapMartix.at<double>(0, 1) += LR * derivative[M12];
		//mapMartix.at<double>(0, 2) += LR * derivative[M13];
		mapMartix.at<double>(1, 0) += LR * derivative[M21];
		mapMartix.at<double>(1, 1) += LR * derivative[M22];
		//mapMartix.at<double>(1, 2) += LR * derivative[M23];
		mapMartix.at<double>(2, 0) += LR * derivative[M31];
		mapMartix.at<double>(2, 1) += LR * derivative[M32];
		//mapMartix.at<double>(2, 2) += LR * derivative[M33];
		loss += calcLoss(point);
		//if (calcLoss(point) < 1)
			//return;
	}
	cout << "----------------------------loss is:" << loss << endl;
 }

double codiConvertor::calcLoss(point4Op point)
{
	double diff_x = abs(point.calcpoint.x - point.realpoint.x);
	double diff_y = abs(point.calcpoint.y - point.realpoint.y);
	return sqrt(diff_x * diff_x + diff_y * diff_y);
}
vector<double> codiConvertor::getDerivative(point4Op& point)
{
	double x = point.calcpoint.x;
	double y = point.calcpoint.y;
	double u = point.realpoint.x;
	double v = point.realpoint.y;
	double x_diff= point.calcpoint.x-point.realpoint.x;
	double y_diff= point.calcpoint.y-point.realpoint.y;

	double a11 = mapMartix.at<double>(0, 0);
	double a12 = mapMartix.at<double>(0, 1);
	double a13 = mapMartix.at<double>(0, 2);
	double a21 = mapMartix.at<double>(1, 0);
	double a22 = mapMartix.at<double>(1, 1);
	double a23 = mapMartix.at<double>(1, 2);
	double a31 = mapMartix.at<double>(2, 0);
	double a32 = mapMartix.at<double>(2, 1);
	double a33 = mapMartix.at<double>(2, 2);
	//cout << mapMartix << endl;

	vector<double> result;
	double w = a13 * point.calcpoint.x + a23 * point.calcpoint.x + a33;


	result.push_back(x_diff * u / w); //M11
	result.push_back(y_diff * u / w); //M12
	result.push_back(x_diff * u * (a11 * u + a21 * v + a31) * log(w)); //M13
	result.push_back(x_diff * v / w); //M21
	result.push_back(y_diff * v / w); //M22
	result.push_back(y_diff * v * (a12 * u + a22 * v + a32) * log(w)); //M23
	result.push_back(x_diff / w); //M31
	result.push_back(y_diff / w); //M32
	result.push_back(y_diff * (a12 * u + a22 * v + a32) * log(w)); //M33
	//for (auto res : result)
		//cout << res << " ";
	//cout << endl;
	//system("pause");
	return result;
}

int main()
{
	codiConvertor* codiConverter =new codiConvertor();
	Mat img = imread("ganpicimg1.jpg", IMREAD_COLOR);
	Mat rectpos;
	inRange(img, Scalar(10, 10, 10), Scalar(255, 30, 30), rectpos);


	vector<vector<Point>>counters;
	findContours(rectpos, counters, RETR_EXTERNAL, CHAIN_APPROX_NONE);


	vector<point4Op> cornPoints;
	point4Op center;
	float r;
	for (auto& counter : counters)
	{
		minEnclosingCircle(counter, center.calcpoint, r);
		cornPoints.push_back(center);
	}

	sort(cornPoints.begin(), cornPoints.end(), cmp_x);
	int i = 0;
	for (auto& recgPoint : cornPoints)
	{
		recgPoint.xIdx = i;
		i++;
	}

	sort(cornPoints.begin(), cornPoints.end(), cmp_y);
	i = 0;
	for (auto& recgPoint : cornPoints)
	{
		if (i < 2)
		{
			if (recgPoint.xIdx == 0 || recgPoint.xIdx == 1)
				recgPoint.realpoint = Point(100, 100);
			else
				recgPoint.realpoint = Point(400, 100);
		}
		else
		{
			if (recgPoint.xIdx == 0 || recgPoint.xIdx == 1)
				recgPoint.realpoint = Point(100, 300);
			else
				recgPoint.realpoint = Point(400, 300);
		}
		i++;
	}

	vector<Point2f> srcPoints;
	vector<Point2f> dstPoints;
	for (auto recgPoint : cornPoints)
	{
		srcPoints.push_back(recgPoint.realpoint);
		dstPoints.push_back(recgPoint.calcpoint);
	}
	//cout << srcPoints << endl;
	//cout << dstPoints << endl;

	Mat trans = getPerspectiveTransform(srcPoints, dstPoints);
	trans = trans.inv();
	codiConverter->mapMartix = trans;
	//warpPerspective(img, img, trans, Size(640, 480), 1, BORDER_REPLICATE);

	//imshow(" ", img);
	//waitKey();

	//cout << trans;

	//----------------------------------optimize---------------------------------------------//
	Mat pos;
	inRange(img, Scalar(10, 10, 230), Scalar(30, 30, 255), pos);

	counters.clear();
	counters.shrink_to_fit();
	findContours(pos, counters, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	//drawContours(img, counters, -1, Scalar(255, 255, 255));
	//imshow(" ", img);
	//waitKey();

	vector<Point2f> recgPoints;
	vector<Point2f> calcPoints;
	vector<Point2f> realPoints{ Point2f(150, 200), Point2f(350, 200), Point2f(200, 300), Point2f(600, 50), Point2f(650, 150) ,
		                                         Point2f(450, 150) , Point2f(600, 350), Point2f(50, 450), Point2f(50, 50) };
	Point2f recgPoint;
	for (auto& counter : counters)
	{
		minEnclosingCircle(counter, recgPoint, r);
		recgPoints.push_back(recgPoint);
	}

	Mat res;
	while (1)
	{
		perspectiveTransform(recgPoints, calcPoints, codiConverter->mapMartix);
		warpPerspective(img, res, trans, Size(640, 480), 1, BORDER_REPLICATE);
		imshow(" ", res);
		waitKey(5);
		vector<point4Op> opPoints;
		point4Op curcar;
		float mindis;
		for (auto& calcPoint : calcPoints)
		{
			mindis = 10000000;
			for (auto& realPoint : realPoints)
			{
				if (distance(calcPoint, realPoint) < mindis)
				{
					curcar.realpoint = realPoint;
					mindis = distance(calcPoint, realPoint);
				}
			}
			curcar.calcpoint = calcPoint;
			opPoints.push_back(curcar);
		}
		codiConverter->points4Op = opPoints;
		codiConverter->optimizer(0.000001);
	}


}
