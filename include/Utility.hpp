#ifndef UTILITY_HPP
#define UTILITY_HPP


string getExtName(string filePath) {
    string filename = filePath;

    // Remove extension if present.
    const size_t period_idx = filename.rfind('.');
    if (std::string::npos != period_idx) {
        filename.erase(0, period_idx + 1);
    }
    return filename;
}

int IsImg(string extName) {
    if (!extName.compare("jpg") || !extName.compare("jpeg")
            || !extName.compare("bmp") || !extName.compare("tiff") || !extName.compare("JPG")
            || !extName.compare("JPEG") || !extName.compare("BMP") || !extName.compare("png")
            || !extName.compare("PNG")) {
        return 1;
    } else {
        return 0;
    }
}



#endif // UTILITY_HPP
