#include "common.h"
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

std::vector<std::string> get_names(const std::string& image_path) {
    std::vector<std::string> image_names;
    auto dir = opendir(image_path.c_str());

    if ((dir) != nullptr) {
        struct dirent *entry;
        entry = readdir(dir);
        while (entry) {
            auto temp = image_path + "/" + entry->d_name;
            if (strcmp(entry->d_name, "") == 0 || strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                entry = readdir(dir);
                continue;
            }
            image_names.emplace_back(temp);
            entry = readdir(dir);
        }
    }
    return image_names;
}

std::string replace(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

static bool isFile(const std::string& filename) {
    struct stat buffer;
    return S_ISREG(buffer.st_mode);
}
 
static bool isDirectory(const std::string& filefodler) {
    struct stat buffer;
    return S_ISDIR(buffer.st_mode);
}

void visualize_mask(cv::Mat& image, 
                    const std::vector<MaskObject>& objects, 
                    const std::string& save_path) noexcept {
    cv::Scalar txt_color = cv::Scalar(255, 255, 255);
    cv::Scalar rec_color = cv::Scalar(0, 0, 0);
    for (int i = 0; i < (int)objects.size(); i++) {
        const MaskObject& obj = objects[i];
        const cv::Scalar& box_color = Color::coco[obj.label];

        cv::Mat points;
        cv::findNonZero(obj.mask, points);
        cv::Rect rect = cv::boundingRect(points);

        // text
        int baseline = 0;
        std::string text = Category::coco[obj.label] + " | " + cv::format("%.3f", obj.prob);;
        cv::Size txt_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        cv::rectangle(image, cv::Rect(cv::Point(rect.x, rect.y), 
                      cv::Size(txt_size.width, txt_size.height + baseline)), 
                      rec_color, -1);
        cv::putText(image, text, cv::Point(rect.x, rect.y + txt_size.height + 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1);

        // mask
        cv::Mat color_mask(image.rows, image.cols, CV_8UC3, box_color);
        cv::Mat mask = cv::Mat::zeros(color_mask.size(), CV_8UC3);
        color_mask.copyTo(mask, obj.mask(cv::Rect(0, 0, image.cols, image.rows)));

        cv::addWeighted(image, 1.0, mask, 0.5, 0., image);

    }
    cv::imwrite(save_path, image);
}
