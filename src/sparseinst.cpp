#include "sparseinst.h"


SparseInst::SparseInst(const std::string& param_path, const std::string& bin_path) {
    this->sparseinst.opt.use_vulkan_compute = true;
    this->sparseinst.load_param(param_path.c_str());
    this->sparseinst.load_model(bin_path.c_str());
}


void SparseInst::run(const std::string& input_path, const std::string& output_path) noexcept {
    std::vector<std::string> image_names = get_names(input_path);

    for (auto& image_name : image_names) {
        fprintf(stderr, "Inference image: %s \n", image_name.c_str());
        auto save_name = replace(image_name, input_path, output_path);
        cv::Mat image = cv::imread(image_name);
        std::vector<MaskObject> objects;

        this->inference(image, objects);
        visualize_mask(image, objects, save_name);
    }
}

void SparseInst::inference(const cv::Mat& image, std::vector<MaskObject>& objects) noexcept {
    int img_w = image.cols; 
    int img_h = image.rows;

    int w = img_w; 
    int h = img_h; 
    float scale = 1.f;

    if (w > h) {
        scale = (float) this->target_size / w;
        w = this->target_size;
        h = h * scale;
    } else {
        scale = (float) this->target_size / h;
        h = this->target_size;
        w = w  * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, img_w, img_h, w, h);

    int w_pad = this->target_size - w;
    int h_pad = this->target_size - h;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, 0, h_pad, 0, w_pad, ncnn::BORDER_CONSTANT, 114.f);

    ncnn::Extractor ex = this->sparseinst.create_extractor();

    ncnn::Mat masks; ncnn::Mat scores;

    ex.input("input_image", in_pad);
    ex.extract("masks", masks);
    ex.extract("scores", scores);

    int num_proposals = scores.h;
    int num_classes = scores.w;
	int mask_h = masks.h;
	int mask_w = masks.w;

    int ori_mask_h = (int) (mask_h / scale + 0.5); 
    int ori_mask_w = (int) (mask_w / scale + 0.5); 
    for (int y = 0; y < num_proposals; y++) {
        float* per_ins_scores = scores.row(y);
        ncnn::Mat mask_ncnn = masks.channel(y);
        int cls = std::max_element(per_ins_scores, per_ins_scores + num_classes) - per_ins_scores;
		float score = per_ins_scores[cls];
		if (score < this->conf_thre) continue;
		cv::Mat ori_mask = cv::Mat::zeros(mask_w, mask_h, CV_32FC1);
        memcpy((uchar*)ori_mask.data, mask_ncnn.data, mask_ncnn.w * mask_ncnn.h * sizeof(float));

        cv::Mat tmp_mask = cv::Mat::zeros(mask_w, mask_h, CV_8UC1);
        tmp_mask = ori_mask > this->mask_thre;
        cv::Mat pad_mask = cv::Mat::zeros(ori_mask_w, ori_mask_h, CV_8UC1);
        cv::resize(tmp_mask, pad_mask, pad_mask.size());
        objects.emplace_back(pad_mask, cls, score);
    }
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s [param_path] [bin_path] [input_path] [output_path]\n", argv[0]);
        return -1;
    }

    const char* param_path = argv[1];
    const char* bin_path = argv[2];
    const char* input_path = argv[3];
    const char* output_path = argv[4];

    SparseInst sparseinst(param_path, bin_path);

    sparseinst.run(input_path, output_path);
    return 0;
}