#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


class Pixel
{
private:
    int id;
    int clusterID;
    std::vector<double> lab;
    // cv::Vec3b Lab;

public:
    Pixel(){}
    Pixel(int id, cv::Vec3b Lab){
        this->id = id;
        this->clusterID = 0;
        // this->Lab = Lab;
        for (int i = 0; i < 3; i++){
            lab.push_back(Lab[i] * 1.0/ 255.0);
        }
    }

    int getId() { return id; }

    int getClusterID() { return clusterID; }
    
    double getVal(int pos) { return lab[pos]; }

    void setVal(int pos, double val){
        this->lab[pos] = val;
    }

    void setCluster(int clusterID) { this->clusterID = clusterID; }

    double calc_distance(Pixel p1, Pixel p2){
        return std::sqrt( std::pow(p1.getVal(0) - p2.getVal(0), 2) + std::pow(p1.getVal(1) - p2.getVal(1), 2) +std::pow(p1.getVal(2) - p2.getVal(2), 2) );
    }
};

class Cluster
{
private:
    int clusterID;
    Pixel centroid;
    std::vector<Pixel> pixels;

public:
    Cluster(int clusterID, Pixel centroid){
        this->clusterID = clusterID;
        this->centroid = centroid;
    }
    Pixel getCentroid(){
        return this->centroid;
    }

    int getClusterID(){
        return clusterID;
    }

    Pixel getPixel(int pos) { return pixels[pos]; }

    void addPixel(Pixel p){
        p.setCluster(clusterID);
        pixels.push_back(p);
    }

    bool removePixel(int pixelID){
        for (int i = 0; i < pixels.size(); i++){
            if (pixels[i].getId() == pixelID){
                pixels.erase(pixels.begin() + i);
                return true;
            }
        }

        return false;
    }

    int getSize() { return pixels.size(); }

    void clearCluster(){pixels.clear();}

    void moveCentroid(int pos, double val){
        this->centroid.setVal(pos, val);
    }
};


class Kmeans
{
private:
    int K, iters, total_points;
    std::vector<Cluster> clusters;

    void clearClusters();

    int getNearestClusterId(Pixel pixel);

public:
    Kmeans(int K, int iters, int total_points);

    std::vector<int> run(std::vector<Pixel> &all_pixels);
};

Kmeans::Kmeans(int K, int iters, int total_points){
    this->K = K;
    this->iters = iters;
    this->total_points = total_points;
}

void Kmeans::clearClusters(){
    for (Cluster cluster : this->clusters){
        cluster.clearCluster();
    }
}

int Kmeans::getNearestClusterId(Pixel pixel)
{
    // for each cluster form 0 to K-1 count Euclidean norm and choose the nearest cluster
    double min_dist = DBL_MAX;
    int nearestCLusterID = 0;
    for (int i = 0; i < K; i++){
        Pixel current_centroid = clusters[i].getCentroid();
        double dist = std::sqrt( std::pow(current_centroid.getVal(0) - pixel.getVal(0), 2) + std::pow(current_centroid.getVal(1) - pixel.getVal(1), 2) +std::pow(current_centroid.getVal(2) - pixel.getVal(2), 2) );

        if (dist < min_dist){
            min_dist = dist;
            nearestCLusterID = clusters[i].getClusterID();
        }
    }

    return nearestCLusterID;
}


std::vector<int> Kmeans::run(std::vector<Pixel> &all_pixels){
    std::vector<int> labels;

    std::vector<int> used_pointIds;
    for (int i = 1; i <= K; i++)
    {
        while (true)
        {
            
            int index = std::rand() % this->total_points;

            if (std::find(used_pointIds.begin(), used_pointIds.end(), index) ==
                used_pointIds.end())
            {
                used_pointIds.push_back(index);
                
                std::cout<<index << " " << all_pixels.size()<< " - log \n";
                all_pixels[index].setCluster(i);
                Cluster cluster(i, all_pixels[index]);
                clusters.push_back(cluster);
                break;
            }
        }
    }
    std::cout << "Clusters initialized = " << clusters.size() << std::endl
              << std::endl;

    std::cout << "Running K-Means Clustering.." << std::endl;
    int iter = 1;
    while (true)
    {
        std::cout << "Iter - " << iter << "/" << iters << std::endl;
        bool done = true;

// Add all points to their nearest cluster
#pragma omp parallel for reduction(&& : done) num_threads(16)
        for (int i = 0; i < total_points; i++)
        {
            int currentClusterId = all_pixels[i].getClusterID();
            int nearestClusterId = getNearestClusterId(all_pixels[i]);

            if (currentClusterId != nearestClusterId)
            {
                all_pixels[i].setCluster(nearestClusterId);
                done = false;
            }
        }

        // clear all existing clusters
        clearClusters();

        // reassign points to their new clusters
        for (int i = 0; i < total_points; i++)
        {
            // cluster index is ID-1
            clusters[all_pixels[i].getClusterID() - 1].addPixel(all_pixels[i]);
        }

        // Recalculating the center of each cluster
        for (int i = 0; i < K; i++)
        {
            int ClusterSize = clusters[i].getSize();

            for (int j = 0; j < 3; j++)
            {
                double sum = 0.0;
                if (ClusterSize > 0)
                {
#pragma omp parallel for reduction(+ : sum) num_threads(16)
                    for (int p = 0; p < ClusterSize; p++)
                    {
                        sum += clusters[i].getPixel(p).getVal(j);
                    }
                    clusters[i].moveCentroid(j,  sum * 1.0 / ClusterSize);
                }
            }
        }

        if (done || iter >= iters)
        {
            std::cout << "Clustering completed in iteration : " << iter << std::endl
                      << std::endl;
            break;
        }
        iter++;
    }

    for (int i = 0; i < total_points; i++)
    {
        labels.push_back(all_pixels[i].getClusterID());
    }

    return labels;
   
}

int main(){

    std::string ORIGINAL_DIR = "/home/den/CV_labs/Lab4/img/original/";
    std::string RES_DIR = "/home/den/CV_labs/Lab4/img/outputs/segmentation3/";
    cv::Mat img = cv::imread(ORIGINAL_DIR + "5.jpg");
    
    cv::Mat img_Lab;
    cv::cvtColor(img, img_Lab, cv::COLOR_BGR2Lab);
    std::vector<Pixel> imgData;
    int id_counter = 0;
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            Pixel pixel{id_counter, img_Lab.at<cv::Vec3b>(y, x)};
            id_counter++;
            imgData.push_back(pixel);
        }
    }
    std::cout << img_Lab.at<cv::Vec3b>(0, 0) << "LOG \n";
    Kmeans model{3, 100, img.cols * img.rows};
    std::vector<int> labels = model.run(imgData);
    
    int height = img.rows;
    int width = img.cols;
    
    cv::Mat outputImage(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < labels.size(); i++) {
        int spatialRow = i / img.cols;
        int spatialCol = i % img.cols;
        switch (labels[i] % 3) {
        case 0:
            outputImage.at<cv::Vec3b>(spatialRow, spatialCol)[0] = 255;
            outputImage.at<cv::Vec3b>(spatialRow, spatialCol)[1] = 0;
            outputImage.at<cv::Vec3b>(spatialRow, spatialCol)[2] = 0;
            break;
        case 1:
            outputImage.at<cv::Vec3b>(spatialRow, spatialCol)[0] = 0;
            outputImage.at<cv::Vec3b>(spatialRow, spatialCol)[1] = 255;
            outputImage.at<cv::Vec3b>(spatialRow, spatialCol)[2] = 0;
            break;
        case 2:
            outputImage.at<cv::Vec3b>(spatialRow, spatialCol)[0] = 0;
            outputImage.at<cv::Vec3b>(spatialRow, spatialCol)[1] = 0;
            outputImage.at<cv::Vec3b>(spatialRow, spatialCol)[2] = 255;
            break;
        default:
            outputImage.at<cv::Vec3b>(spatialRow, spatialCol)[0] = 255;
            outputImage.at<cv::Vec3b>(spatialRow, spatialCol)[1] = 255;
            outputImage.at<cv::Vec3b>(spatialRow, spatialCol)[2] = 255;
            break;
        }
    }

    cv::imwrite(RES_DIR + "clustered_img_5.jpeg", outputImage);

    return 0;
}
