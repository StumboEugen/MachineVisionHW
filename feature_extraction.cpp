#include <iostream>
#include <fstream>
#include <unistd.h>
#include <dirent.h>
#include <vector>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>

using namespace std;
using namespace cv;

const string savePath = "pics_output/";

/**
 * A ransac method to calculate the desired affine transformation Matrix
 * @param pairs the paired fetrured points
 * @param iters iter_nums
 * @param error what error arrange we can accept as good pairs
 * @return the desired affine transformation matrix(from 1 to 2)
 */
Mat ransac(const vector<pair<Point2f, Point2f>> & pairs, int iters = 100000 , double error = 10.0){
    int good_model_num = static_cast<int>(pairs.size() * 0.7f);

    int index;

    double model_error = 255.0;
    double dis_error;

    Mat warp_mat;
    Mat best_warp_mat;

    vector<Point2f> pointgood1;
    vector<Point2f> pointgood2;

    Point2f srcTri[40];
    Point2f dstTri[40];

    for(int i = 0; i < iters; i++){
        pointgood1.clear();
        pointgood2.clear();
        for(int i = 0; i < 40; i++){
            index = static_cast<int>(rand() % pairs.size());
            srcTri[i] = pairs[index].first;
            dstTri[i] = pairs[index].second;
        }

        warp_mat = getAffineTransform(srcTri, dstTri);

        Eigen::Matrix<double ,2,3> warp_matrix;
        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 3; j++) {
                warp_matrix(i,j) = warp_mat.at<double>(i,j);
            }
        }

        Point2f x1, x2;
        for (const auto &pair : pairs) {
            x1 = pair.first;
            x2 = pair.second;

            Eigen::Matrix<double ,3,1> v1_3d;
            v1_3d << x1.x, x1.y, 1;
            Eigen::Matrix<double, 2, 1> out = warp_matrix.cast<double>() * v1_3d;
            dis_error = static_cast<double>(sqrt(pow(out(0) - x2.x, 2) + pow(out(1) - x2.y, 2)));
            if(dis_error < error){
                pointgood1.push_back(x1);
                pointgood2.push_back(x2);
            }
        }

        double max_dist_error = 0;
        if(pointgood1.size() >= good_model_num){
            for(int i = 0; i < pointgood1.size(); i++){
                x1 = pointgood1[i];
                x2 = pointgood2[i];

                Eigen::Matrix<double ,3,1> v1_3d;
                v1_3d << x1.x, x1.y, 1;
                Eigen::Matrix<double, 2, 1> out = warp_matrix.cast<double>() * v1_3d;
                dis_error = static_cast<double>(sqrt(pow(out(0) - x2.x, 2) + pow(out(1) - x2.y, 2)));
                max_dist_error = max_dist_error > dis_error ? max_dist_error:dis_error;

            }
            if(max_dist_error < error && max_dist_error < model_error){
                model_error = max_dist_error;
                best_warp_mat = warp_mat;
            }
        }
    }
    cout<< "model error: "<< model_error << "\n";
    cout << "affine matrix:\n" << warp_mat << "\n" << endl;
    return best_warp_mat;
}

/**
 * get the ORB feature point pairs of two desired images
 * @param fileName the name of the match pics
 * @return the feature pairs
 */
vector<pair<Point2f, Point2f>> getMatchedPoints(Mat &img_1, Mat &img_2, const string &fileName) {
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    /** 第一步:检测 Oriented FAST 角点位置*/
    detector->detect ( img_1, keypoints_1 );
    detector->detect ( img_2, keypoints_2 );
    /** 第二步:根据角点位置计算 BRIEF 描述子*/
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );
    /** 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离*/
    vector<DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches, noArray());
    /** 第四步:匹配点对筛选*/
    double min_dist=10000, max_dist=0;
    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    min_dist = min_element( matches.begin(), matches.end(),
            [](const DMatch& m1, const DMatch& m2)
            {return m1.distance<m2.distance;} )->distance;
    max_dist = max_element( matches.begin(), matches.end(),
            [](const DMatch& m1, const DMatch& m2)
            {return m1.distance<m2.distance;} )->distance;
    cout << "-- Max dist :" << max_dist << "\n";
    cout << "-- Min dist :" << min_dist << endl;
    /**当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小
     * 设置一个经验值30作为下限.*/
    std::vector< DMatch > good_matches;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
    }
    Mat img_goodmatch;
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch );
    imwrite(savePath + "match" + fileName, img_goodmatch);

    vector<pair<Point2f, Point2f>> matchedPairs;
    for (const auto & match : good_matches) {
        matchedPairs.emplace_back(keypoints_1[match.queryIdx].pt,
                                  keypoints_2[match.trainIdx].pt);
    }
    cout << "size of point: " << matchedPairs.size() << endl;
    return std::move(matchedPairs);
}

/**
 * affine transform 1 image
 * @param fromImg
 * @param affineMatrix
 * @return the transformed image
 */
Mat affineImage(const Mat &fromImg, const Mat & affineMatrix) {
    Mat warp_dst = Mat::zeros( fromImg.rows, fromImg.cols, fromImg.type() );
    warpAffine( fromImg, warp_dst, affineMatrix, warp_dst.size() );
    return std::move(warp_dst);
}

/**
 * merge 2 images
 * @param img_1
 * @param img_2
 * @param affineMatrix
 * @return the merged images
 */
Mat mergeImages(const Mat & img_1, const Mat & img_2) {
    float AlphaBeta = 0.5f;
    Mat merged = Mat::zeros( img_1.rows, img_1.cols, img_1.type() );
    addWeighted( img_1, AlphaBeta, img_2, AlphaBeta, 0.0, merged);
    return std::move(merged);
}

/**
 * match and merge 2 images
 * @param img_1
 * @param img_2
 * @param fileName
 */
void matchImages(Mat & img_1, Mat & img_2, const string &fileName) {
    auto pairs = getMatchedPoints(img_1, img_2, fileName);

    Mat affineMatrix = ransac(pairs);

    Mat warp_dst = affineImage(img_1, affineMatrix);

    auto merged = mergeImages(warp_dst, img_2);

    imwrite(savePath + "merge" + fileName, merged);
}

int main ( int argc, char** argv )
{
    map<int, vector<string>> pics;
    string inputPath;
    char currentPath[100];
    auto charPtr = getcwd(currentPath, 100);
    string path(currentPath);
    inputPath = path + "/../pics";
    DIR *dir;
    dir = opendir(inputPath.c_str());
    if (dir == nullptr) {
        cout << "counld not find /pics" << endl;
        exit(1);
    }
    dirent * dirPtr;
    while ((dirPtr = readdir(dir)) != nullptr) {
        if (dirPtr->d_type == 8) {
            pics[dirPtr->d_name[0]].emplace_back(dirPtr->d_name);
        }
    }
    char camA = '1';
    char camB = '2';
    if (argc == 3) {
        camA = argv[1][0];
        camB = argv[2][0];
        if (pics[camA].empty() || pics[camB].empty()) {
            cout << "usage: feature_extraction camNo1 camNo2" << endl;
            exit(1);
        }
    } else {
        cout << "you didn't assign any cameras, process on Cam1 and Cam2 on default" << endl;
    }
    sort(pics[camA].begin(), pics[camA].end());
    sort(pics[camB].begin(), pics[camB].end());
    for (int i = 0; i < pics[camA].size(); i++) {
        Mat input1 = imread ( "pics/" + pics[camA][i], CV_LOAD_IMAGE_COLOR );
        Mat input2 = imread ( "pics/" + pics[camB][i], CV_LOAD_IMAGE_COLOR );
        Mat img_1, img_2;
        resize(input1, img_1, img_1.size(), 2, 2, INTER_CUBIC);
        resize(input2, img_2, img_2.size(), 2, 2, INTER_CUBIC);
        matchImages(img_1, img_2, camA + pics[camB][i]);
    }

    vector<string> fourPics ({"15-27-25.jpg", "15-28-28.jpg", "15-31-55.jpg", "15-32-43.jpg", "15-33-31.jpg"});
    if (argc == 1) {
        cout << "try to merge 4 imgaes..." << endl;
        for (int i = 0; i < fourPics.size(); i++) {
            Mat input1 = imread( "pics/1_" + fourPics[i], CV_LOAD_IMAGE_COLOR );
            Mat input2 = imread( "pics/2_" + fourPics[i], CV_LOAD_IMAGE_COLOR );
            Mat input3 = imread( "pics/3_" + fourPics[i], CV_LOAD_IMAGE_COLOR );
            Mat input4 = imread( "pics/4_" + fourPics[i], CV_LOAD_IMAGE_COLOR );
            Mat img1, img2, img3, img4;
            resize(input1, img1, img1.size(), 2, 2, INTER_CUBIC);
            resize(input2, img2, img2.size(), 2, 2, INTER_CUBIC);
            resize(input3, img3, img3.size(), 2, 2, INTER_CUBIC);
            resize(input4, img4, img4.size(), 2, 2, INTER_CUBIC);
            Mat trans12 = ransac(getMatchedPoints(img1, img2, "4_12_" + fourPics[i]) );
            Mat trans32 = ransac(getMatchedPoints(img3, img2, "4_32_" + fourPics[i]) );
            Mat trans42 = ransac(getMatchedPoints(img4, img2, "4_42_" + fourPics[i]) );
            Mat affined1 = affineImage(img1, trans12);
            Mat affined3 = affineImage(img3, trans32);
            Mat affined4 = affineImage(img4, trans42);
            Mat merged12 = mergeImages(affined1, img2);
            Mat merged34 = mergeImages(affined3, affined4);
            Mat merged1234 = mergeImages(merged12, merged34);
            imwrite(savePath + fourPics[i], merged1234);
        }
    }
    cout << "finished, merge images are in floder: /build/pics_output" << endl;
    return 0;
}
