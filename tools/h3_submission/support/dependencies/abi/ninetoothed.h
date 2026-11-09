#ifndef NINETOOTHED_H
#define NINETOOTHED_H

#include <stdint.h>

#define NINETOOTHED_TRUE 1

#define NINETOOTHED_FALSE 0

#define NINETOOTHED_NONE 0

enum NineToothedDataType {
    NINETOOTHED_INT8,
    NINETOOTHED_INT16,
    NINETOOTHED_INT32,
    NINETOOTHED_INT64,
    NINETOOTHED_UINT8,
    NINETOOTHED_UINT16,
    NINETOOTHED_UINT32,
    NINETOOTHED_UINT64,
    NINETOOTHED_FLOAT16,
    NINETOOTHED_BFLOAT16,
    NINETOOTHED_FLOAT32,
    NINETOOTHED_FLOAT64
};

typedef struct {
    void *data;
    uint64_t *shape;
    int64_t *strides;
} NineToothedTensor;

typedef void *NineToothedStream;

typedef int NineToothedResult;

#ifdef __cplusplus
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace ninetoothed {

class AutoTuningCache {
public:
    AutoTuningCache(const std::string& path) : path{path} {}

    std::vector<int> lookup(const std::vector<int>& config) {
        load();

        auto iter = cache.find(config);

        if (iter != cache.end()) {
            return iter->second;
        }

        return default_meta;
    }

private:
    void load() {
        auto mtime = std::filesystem::last_write_time(path);

        if (mtime == last_mtime) {
            return;
        }

        last_mtime = mtime;

        cache.clear();

        std::ifstream file{path};
        std::string config_line;
        std::string meta_line;

        if (std::getline(file, config_line) && std::getline(file, meta_line)) {
            default_meta = parse(meta_line);
        }

        while (std::getline(file, config_line) && std::getline(file, meta_line)) {
            cache[parse(config_line)] = parse(meta_line);
        }
    }

    static std::vector<int> parse(const std::string& line) {
        std::vector<int> values;
        std::stringstream ss{line};
        std::string token;

        while (std::getline(ss, token, ',')) {
            auto trimmed = trim(token);

            if (!trimmed.empty()) {
                values.push_back(std::stoi(trimmed));
            }
        }

        return values;
    }

    static std::string trim(const std::string& s) {
        auto start = s.find_first_not_of(" \t\r\n");

        if (start == std::string::npos) {
            return "";
        }

        auto end = s.find_last_not_of(" \t\r\n");

        return s.substr(start, end - start + 1);
    }

    std::string path;

    std::map<std::vector<int>, std::vector<int>> cache;

    std::vector<int> default_meta;

    std::filesystem::file_time_type last_mtime;
};

}

#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace ninetoothed {

template<typename Key, typename Value>
class ThreadSafeUnorderedMap {
public:
    Value& operator[](const Key& key) {
        {
            std::shared_lock lock{mutex};

            auto iter = map.find(key);

            if (iter != map.end()) {
                return iter->second;
            }
        }

        std::unique_lock lock{mutex};

        auto iter = map.find(key);

        if (iter == map.end()) {
            iter = map.emplace(key, Value{}).first;
        }

        return iter->second;
    }

private:
    std::unordered_map<Key, Value> map;

    mutable std::shared_mutex mutex;
};

}
#endif

#endif // NINETOOTHED_H
