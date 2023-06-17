#ifndef SPARSEINST_H
#define SPARSEINST_H
#include "common.h"

class SparseInst {

public:
    explicit SparseInst(const std::string& param_path, const std::string& bin_path);
    void run(const std::string& input_path, const std::string& output_path) noexcept;
    void inference(const cv::Mat& image, std::vector<MaskObject>& objects) noexcept;
    
private:
    int target_size = 608;
    float conf_thre = 0.45;
    float mask_thre = 0.45;
    ncnn::Net sparseinst;

};


#endif