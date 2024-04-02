/*
 * Copyright 2021 4Paradigm
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/str_split.h"
#include "boost/algorithm/string.hpp"
#include "boost/algorithm/string/join.hpp"
#include "boost/algorithm/string/regex.hpp"
#include "codec/list_iterator_codec.h"
#include "codec/type_codec.h"
#include "udf/containers.h"
#include "udf/default_udf_library.h"
#include "udf/udf.h"
#include "udf/udf_registry.h"
#include "vm/jit_runtime.h"

namespace hybridse {
namespace udf {
namespace v1 {

template <typename T>
struct BinaryLabel {
    using Args = std::tuple<T>;
    bool operator()(T v) { return v; }
};

template <typename T>
struct MulticlassLabel {
    using Args = std::tuple<T>;
    int64_t operator()(T v) { return v; }
};

template <typename T>
struct RegressionLabel {
    using Args = std::tuple<T>;
    double operator()(T v) { return v; }
};

template <typename T>
struct Numeric {
    using Args = std::tuple<T>;
    T operator()(T v) { return v; }
};

template <typename T> struct Category;
template <typename T> struct Category<std::tuple<T>> {
    using Args = std::tuple<T>;
    using ParamType = typename DataTypeTrait<T>::CCallArgType;

    int64_t operator()(ParamType v) {
        return FarmFingerprint(CCallDataTypeTrait<ParamType>::to_bytes_ref(&v));
    }
};

template <typename T> struct Category<std::tuple<T, int32_t>> {
    using Args = std::tuple<T, int32_t>;
    using ParamType = typename DataTypeTrait<T>::CCallArgType;

    int64_t operator()(ParamType v, int32_t bucket_size) {
        uint64_t hash = FarmFingerprint(CCallDataTypeTrait<ParamType>::to_bytes_ref(&v));
        uint64_t mod = static_cast<int64_t>(bucket_size);
        return mod ? hash % mod : hash;
    }
};

template <typename T> struct Category<std::tuple<T, int64_t>> {
    using Args = std::tuple<T, int64_t>;
    using ParamType = typename DataTypeTrait<T>::CCallArgType;

    int64_t operator()(ParamType v, int64_t bucket_size) {
        uint64_t hash = FarmFingerprint(CCallDataTypeTrait<ParamType>::to_bytes_ref(&v));
        uint64_t mod = bucket_size;
        return mod ? hash % mod : hash;
    }
};

bool InstanceFormatGetInteger(codec::RowView& row_view, int idx,
    ::hybridse::type::Type column_type, std::string* ret) {
    switch (column_type) {
        case ::hybridse::type::kBool: {
            bool v = 0;
            if (row_view.GetBool(idx, &v) == 0) {
                *ret =  std::to_string(static_cast<int>(v));
                return true;
            }
        }
        case ::hybridse::type::kInt16: {
            int16_t v = 0;
            if (row_view.GetInt16(idx, &v) == 0) {
                *ret = std::to_string(v);
                return true;
            }
        }
        case ::hybridse::type::kInt32: {
            int32_t v = 0;
            if (row_view.GetInt32(idx, &v) == 0) {
                *ret = std::to_string(v);
                return true;
            }
        }
        case ::hybridse::type::kInt64: {
            int64_t v = 0;
            if (row_view.GetInt64(idx, &v) == 0) {
                *ret = std::to_string(v);
                return true;
            }
        }
        default: {
            return false;
        }
    }
    return false;
}

bool InstanceFormatGetFloat(codec::RowView& row_view, int idx,
    ::hybridse::type::Type column_type, std::string* ret) {
    switch (column_type) {
        case ::hybridse::type::kFloat: {
            float v = 0;
            if (row_view.GetFloat(idx, &v) == 0) {
                *ret = std::to_string(v);
                return true;
            }
        }
        case ::hybridse::type::kDouble: {
            double v = 0;
            if (row_view.GetDouble(idx, &v) == 0) {
                *ret = std::to_string(v);
                return true;
            }
        }
        default: {
            return false;
        }
    }
    return false;
}

bool GCFormatGetBinaryLabel(codec::RowView& row_view, int idx,
    ::hybridse::type::Type column_type, std::string* ret) {
    if (column_type == ::hybridse::type::kBool) {
        return InstanceFormatGetInteger(row_view, idx, column_type, ret);
    }
    return false;
}

bool GCFormatGetMulticlassLabel(codec::RowView& row_view, int idx,
    ::hybridse::type::Type column_type, std::string* ret) {
    return InstanceFormatGetInteger(row_view, idx, column_type, ret);
}

bool GCFormatGetRegressionLabel(codec::RowView& row_view, int idx,
    ::hybridse::type::Type column_type, std::string* ret) {
    switch (column_type) {
        case ::hybridse::type::kBool:
        case ::hybridse::type::kInt16:
        case ::hybridse::type::kInt32:
        case ::hybridse::type::kInt64: {
            return InstanceFormatGetInteger(row_view, idx, column_type, ret);
        }
        case ::hybridse::type::kFloat:
        case ::hybridse::type::kDouble: {
            return InstanceFormatGetFloat(row_view, idx, column_type, ret);
        }
        default: {
            return false;
        }
    }
    return false;
}

bool GCFormatGetNumeric(codec::RowView& row_view, int idx,
    ::hybridse::type::Type column_type, std::string* ret) {
    switch (column_type) {
        case ::hybridse::type::kBool:
        case ::hybridse::type::kInt16:
        case ::hybridse::type::kInt32:
        case ::hybridse::type::kInt64: {
            return InstanceFormatGetInteger(row_view, idx, column_type, ret);
        }
        case ::hybridse::type::kFloat:
        case ::hybridse::type::kDouble: {
            return InstanceFormatGetFloat(row_view, idx, column_type, ret);
        }
        default: {
            return false;
        }
    }
    return false;
}

bool GCFormatGetCategory(codec::RowView& row_view, int idx,
    ::hybridse::type::Type column_type, std::string* ret) {
    ret->clear();
    return InstanceFormatGetInteger(row_view, idx, column_type, ret);
}

codec::Row GCFormat(const codec::Row& row, const vm::InstanceFormatInfo& instance_format) {
    if (row.empty()) {
        return row;
    }
    // Init current run step runtime
    codec::Row instance_row;
    vm::JitRuntime::get()->InitRunStep();
    
    size_t slot_number = 1;
    std::string instance_label = "|";
    std::string instance_feature;
    const node::FeatureSignatureList& feature_signature_list = instance_format.feature_signature_list();
    codec::RowView row_view(*instance_format.format_input_schema(), row.buf(), row.size());
    for (size_t idx = 0; idx < feature_signature_list.size(); ++idx) {
        int i = static_cast<int>(idx);
        if (i >= 0 && i < row_view.GetSchema()->size()) {
            std::string data_string;
            ::hybridse::type::Type column_type = row_view.GetSchema()->Get(idx).type();
            switch (feature_signature_list[idx]) {
                case node::kFeatureSignatureBinaryLabel: {
                    if (GCFormatGetBinaryLabel(row_view, i, column_type, &data_string)) {
                        instance_label = data_string + "|";
                    }
                    break;
                }
                case node::kFeatureSignatureMulticlassLabel: {
                    if (GCFormatGetMulticlassLabel(row_view, i, column_type, &data_string)) {
                        instance_label = data_string + "|";
                    }
                    break;
                }
                case node::kFeatureSignatureRegressionLabel: {
                    if (GCFormatGetRegressionLabel(row_view, i, column_type, &data_string)) {
                        instance_label = data_string + "|";
                    }
                    break;
                }
                case node::kFeatureSignatureNumeric: {
                    if (GCFormatGetNumeric(row_view, i, column_type, &data_string)) {
                        instance_feature += " " + std::to_string(slot_number) + ":0:" + data_string;
                    }
                    ++slot_number;
                    break;
                }
                case node::kFeatureSignatureCategory: {
                    if (GCFormatGetCategory(row_view, i, column_type, &data_string)) {
                        instance_feature += " " + std::to_string(slot_number) + ":" + data_string;
                    }
                    ++slot_number;
                    break;
                }
                default: {
                    break;
                }
            }
        }
    }
    if (instance_format.format_fn_schema()->size() == 1) {
        codec::RowBuilder row_builder(*instance_format.format_fn_schema());
        std::string instance_string = instance_label + instance_feature;
        uint32_t total_length = row_builder.CalTotalLength(instance_string.length());
        int8_t* row_buffer = static_cast<int8_t*>(malloc(total_length));
        row_builder.SetBuffer(row_buffer, total_length);
        row_builder.AppendString(instance_string.c_str(), instance_string.length());
        instance_row = codec::Row(base::RefCountedSlice::CreateManaged(row_buffer, total_length));
    }
    vm::JitRuntime::get()->ReleaseRunStep();
    return instance_row;
}

}  // namespace v1

void DefaultUdfLibrary::InitFeatureSignature() {
    RegisterExternalTemplate<v1::BinaryLabel>("binary_label")
        .doc(R"(
        )")
        .args_in<bool, int16_t, int32_t, int64_t>();

    RegisterExternalTemplate<v1::MulticlassLabel>("multiclass_label")
        .doc(R"(
        )")
        .args_in<int16_t, int32_t, int64_t>();

    RegisterExternalTemplate<v1::RegressionLabel>("regression_label")
        .doc(R"(
        )")
        .args_in<bool, int16_t, int32_t, int64_t, float, double>();

    RegisterExternalTemplate<v1::Numeric>("numeric")
        .doc(R"(
        )")
        .args_in<bool, int16_t, int32_t, int64_t, float, double>();

    RegisterExternalTemplate<v1::Category>("category")
        .doc(R"(
        )")
        .args_in<std::tuple<bool>, std::tuple<bool, int32_t>, std::tuple<bool, int64_t>,
                 std::tuple<int16_t>, std::tuple<int16_t, int32_t>, std::tuple<int16_t, int64_t>, 
                 std::tuple<int32_t>, std::tuple<int32_t, int32_t>, std::tuple<int32_t, int64_t>,
                 std::tuple<int64_t>, std::tuple<int64_t, int32_t>, std::tuple<int64_t, int64_t>,
                 std::tuple<float>, std::tuple<float, int32_t>, std::tuple<float, int64_t>,
                 std::tuple<double>, std::tuple<double, int32_t>, std::tuple<double, int64_t>,
                 std::tuple<StringRef>, std::tuple<StringRef, int32_t>, std::tuple<StringRef, int64_t>,
                 std::tuple<Timestamp>, std::tuple<Timestamp, int32_t>, std::tuple<Timestamp, int64_t>,
                 std::tuple<Date>, std::tuple<Date, int32_t>, std::tuple<Date, int64_t> >();

    RegisterAlias("continuous", "numeric");
    RegisterAlias("discrete", "category");
    RegisterFeatureSignature("binary_label", node::kFeatureSignatureBinaryLabel);
    RegisterFeatureSignature("multiclass_label", node::kFeatureSignatureMulticlassLabel);
    RegisterFeatureSignature("regression_label", node::kFeatureSignatureRegressionLabel);
    RegisterFeatureSignature("numeric", node::kFeatureSignatureNumeric);
    RegisterFeatureSignature("continuous", node::kFeatureSignatureNumeric);
    RegisterFeatureSignature("discrete", node::kFeatureSignatureCategory);
    RegisterFeatureSignature("category", node::kFeatureSignatureCategory);
    RegisterInstanceFormat("GCFormat", v1::GCFormat);
}

}  // namespace udf
}  // namespace hybridse
