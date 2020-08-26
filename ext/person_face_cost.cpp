//
// Created by supermc on 2020/8/25.
//

#include "torch/extension.h"


at::Tensor person_face_cost(const at::Tensor &person_boxes,
                            const at::Tensor &face_boxes) {
//    if (person_boxes.numel() == 0) {
//        return at::empty({0}, person_boxes.options().dtype(at::kLong).device(at::kCPU));
//    }
//    if (face_boxes.numel() == 0) {
//        return at::empty({0}, face_boxes.options().dtype(at::kLong).device(at::kCPU));
//    }

    auto p_x1_t = person_boxes.select(1, 0).contiguous();
    auto p_y1_t = person_boxes.select(1, 1).contiguous();
    auto p_x2_t = person_boxes.select(1, 2).contiguous();
    auto p_y2_t = person_boxes.select(1, 3).contiguous();

    auto f_x1_t = face_boxes.select(1, 0).contiguous();
    auto f_y1_t = face_boxes.select(1, 1).contiguous();
    auto f_x2_t = face_boxes.select(1, 2).contiguous();
    auto f_y2_t = face_boxes.select(1, 3).contiguous();

    auto n_person = person_boxes.size(0);
    auto n_face = face_boxes.size(0);
    at::Tensor suppressed_t = at::zeros({n_person * n_face}, person_boxes.options().dtype(at::kFloat).device(at::kCPU));
    auto suppressed = suppressed_t.data<float_t>();
    auto p_x1 = p_x1_t.data<int32_t>();
    auto p_y1 = p_y1_t.data<int32_t>();
    auto p_x2 = p_x2_t.data<int32_t>();
    auto p_y2 = p_y2_t.data<int32_t>();

    auto f_x1 = f_x1_t.data<int32_t>();
    auto f_y1 = f_y1_t.data<int32_t>();
    auto f_x2 = f_x2_t.data<int32_t>();
    auto f_y2 = f_y2_t.data<int32_t>();


    for (int32_t j = 0; j < n_face; ++j) {
        auto j_f_x1 = f_x1[j];
        auto j_f_y1 = f_y1[j];
        auto j_f_x2 = f_x2[j];
        auto j_f_y2 = f_y2[j];
        auto j_f_area = float_t((j_f_x2 - j_f_x1) * (j_f_y2 - j_f_y1));
        for (int32_t i = 0; i < n_person; ++i) {
            auto x1 = std::max(p_x1[i], j_f_x1);
            auto y1 = std::max(p_y1[i], j_f_y1);
            auto x2 = std::min(p_x2[i], j_f_x2);
            auto y2 = std::min(p_y2[i], j_f_y2);

            auto w = std::max(static_cast<int32_t>(0), x2 - x1);
            auto h = std::max(static_cast<int32_t>(0), y2 - y1);

            auto inter = float_t(w * h);
            float_t cost = inter / j_f_area;
            suppressed[i * n_face + j] = cost;
        }
    }
    auto shape = std::vector<int64_t>{n_person, n_face};
    return suppressed_t.reshape(shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("person_face_cost", &person_face_cost, "person boxes and face boxes ious");
}


