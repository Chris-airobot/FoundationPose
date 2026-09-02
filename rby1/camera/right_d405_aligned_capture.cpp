// One-shot aligned RGB-D capture for the RBY1 right-hand Intel RealSense D405.
//
// Verified on the Nvidia U-PC on 2026-09-03 with D405 serial 230422272237.
// Uses the locally installed librealsense (/usr/local) and rs2::align to align
// depth into the color geometry. This path successfully produced a 640x480
// RGB frame, aligned 16-bit depth, K.txt, and depth_scale.txt.
//
// IMPORTANT: stop rs-right-iris.service before running so this process owns
// the right D405 exclusively.
//
// Build on Nvidia U-PC:
//   g++ -std=c++17 rby1/camera/right_d405_aligned_capture.cpp \
//     -I/usr/local/include -L/usr/local/lib \
//     -Wl,-rpath,/usr/local/lib -lrealsense2 \
//     -o /tmp/right_d405_aligned_capture
//
// Run:
//   sudo systemctl stop rs-right-iris.service
//   /tmp/right_d405_aligned_capture
//
// Output directory is /home/nvidia/fp_capture.

#include <librealsense2/rs.hpp>
#include <fstream>
#include <iostream>
#include <cstdint>

int main()
{
    const int W = 640;
    const int H = 480;

    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_device("230422272237");
    cfg.enable_stream(RS2_STREAM_COLOR, W, H, RS2_FORMAT_RGB8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, W, H, RS2_FORMAT_Z16, 30);

    auto profile = pipe.start(cfg);
    rs2::align align_to_color(RS2_STREAM_COLOR);

    rs2::frameset frames;
    for (int i = 0; i < 30; ++i)
        frames = pipe.wait_for_frames();

    auto aligned = align_to_color.process(frames);

    auto color = aligned.get_color_frame();
    auto depth = aligned.get_depth_frame();

    if (!color || !depth) {
        std::cerr << "ERROR: missing color/depth frame\n";
        return 1;
    }

    auto cp = color.get_profile().as<rs2::video_stream_profile>();
    auto K = cp.get_intrinsics();
    const float depth_scale = depth.get_units();

    {
        std::ofstream f("/home/nvidia/fp_capture/rgb.ppm", std::ios::binary);
        f << "P6\n" << W << " " << H << "\n255\n";

        const uint8_t *data =
            reinterpret_cast<const uint8_t *>(color.get_data());
        const int stride = color.get_stride_in_bytes();

        for (int y = 0; y < H; ++y)
            f.write(reinterpret_cast<const char *>(data + y * stride), W * 3);
    }

    {
        std::ofstream f(
            "/home/nvidia/fp_capture/depth_u16.raw", std::ios::binary);
        const uint8_t *data =
            reinterpret_cast<const uint8_t *>(depth.get_data());
        const int stride = depth.get_stride_in_bytes();

        for (int y = 0; y < H; ++y)
            f.write(
                reinterpret_cast<const char *>(data + y * stride),
                W * sizeof(uint16_t));
    }

    {
        std::ofstream f("/home/nvidia/fp_capture/K.txt");
        f << K.fx << " 0 " << K.ppx << "\n";
        f << "0 " << K.fy << " " << K.ppy << "\n";
        f << "0 0 1\n";
    }

    {
        std::ofstream f("/home/nvidia/fp_capture/depth_scale.txt");
        f << depth_scale << "\n";
    }

    std::cout << "===== FOUNDATIONPOSE CAPTURE READY =====\n";
    std::cout << "RGB:   640x480\n";
    std::cout << "Depth: 640x480 ALIGNED TO COLOR\n";
    std::cout << "K: fx=" << K.fx
              << " fy=" << K.fy
              << " cx=" << K.ppx
              << " cy=" << K.ppy << "\n";
    std::cout << "Depth scale: " << depth_scale << " m/unit\n";
    std::cout << "Saved: /home/nvidia/fp_capture\n";

    pipe.stop();
    return 0;
}
