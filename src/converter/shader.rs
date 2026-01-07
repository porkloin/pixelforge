//! Compute shader for color format conversion.
//!
//! This module contains the SPIR-V bytecode for the color conversion compute shader.
//! The shader converts RGB/BGR formats to various YUV formats using BT.601 coefficients.

/// Get the SPIR-V bytecode for the color conversion shader.
///
/// The shader expects:
/// - Push constants: width (u32), height (u32), input_format (u32), output_format (u32)
/// - Binding 0: Input buffer (RGB/BGR data)
/// - Binding 1: Output buffer (YUV data)
///
/// Workgroup size: 8x8x1.
pub fn get_spirv_code() -> Vec<u32> {
    // SPIR-V bytecode generated from the GLSL compute shader below.
    //
    // #version 450
    //
    // layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
    //
    // layout(push_constant) uniform PushConstants {
    //     uint width;
    //     uint height;
    //     uint input_format;   // 0=BGRx, 1=RGBx, 2=BGRA, 3=RGBA
    //     uint output_format;  // 0=NV12, 1=I420, 2=YUV444
    // } params;
    //
    // layout(std430, binding = 0) readonly buffer InputBuffer {
    //     uint input_data[];
    // };
    //
    // layout(std430, binding = 1) writeonly buffer OutputBuffer {
    //     uint output_data[];
    // };
    //
    // // BT.601 conversion coefficients
    // const float Y_R = 0.299;
    // const float Y_G = 0.587;
    // const float Y_B = 0.114;
    // const float U_R = -0.169;
    // const float U_G = -0.331;
    // const float U_B = 0.500;
    // const float V_R = 0.500;
    // const float V_G = -0.419;
    // const float V_B = -0.081;
    //
    // vec3 extract_rgb(uint pixel, uint format) {
    //     uint b0 = (pixel >> 0) & 0xFF;
    //     uint b1 = (pixel >> 8) & 0xFF;
    //     uint b2 = (pixel >> 16) & 0xFF;
    //
    //     if (format == 0 || format == 2) {
    //         // BGRx or BGRA
    //         return vec3(float(b2), float(b1), float(b0));
    //     } else {
    //         // RGBx or RGBA
    //         return vec3(float(b0), float(b1), float(b2));
    //     }
    // }
    //
    // vec3 rgb_to_yuv(vec3 rgb) {
    //     float y = Y_R * rgb.r + Y_G * rgb.g + Y_B * rgb.b;
    //     float u = 128.0 + U_R * rgb.r + U_G * rgb.g + U_B * rgb.b;
    //     float v = 128.0 + V_R * rgb.r + V_G * rgb.g + V_B * rgb.b;
    //     return vec3(clamp(y, 0.0, 255.0), clamp(u, 0.0, 255.0), clamp(v, 0.0, 255.0));
    // }
    //
    // void main() {
    //     uint x = gl_GlobalInvocationID.x;
    //     uint y = gl_GlobalInvocationID.y;
    //
    //     if (x >= params.width || y >= params.height) return;
    //
    //     uint pixel_idx = y * params.width + x;
    //     uint pixel = input_data[pixel_idx];
    //     vec3 rgb = extract_rgb(pixel, params.input_format);
    //     vec3 yuv = rgb_to_yuv(rgb);
    //
    //     uint pixel_count = params.width * params.height;
    //
    //     if (params.output_format == 2) {
    //         // YUV444: Full resolution Y, U, V planes
    //         output_data[pixel_idx] = uint(yuv.x);
    //         output_data[pixel_count + pixel_idx] = uint(yuv.y);
    //         output_data[2 * pixel_count + pixel_idx] = uint(yuv.z);
    //     } else {
    //         // YUV420: Write Y for every pixel
    //         // Write packed Y values (4 pixels per uint)
    //         uint y_byte_idx = pixel_idx;
    //         uint y_word_idx = y_byte_idx / 4;
    //         uint y_byte_offset = y_byte_idx % 4;
    //
    //         atomicOr(output_data[y_word_idx], uint(yuv.x) << (y_byte_offset * 8));
    //
    //         // Only process UV for top-left pixel of each 2x2 block
    //         if ((x % 2 == 0) && (y % 2 == 0)) {
    //             uint uv_x = x / 2;
    //             uint uv_y = y / 2;
    //             uint uv_width = params.width / 2;
    //             uint uv_idx = uv_y * uv_width + uv_x;
    //
    //             // Average UV from 2x2 block for better quality
    //             vec3 yuv00 = yuv;
    //             vec3 yuv10 = rgb_to_yuv(extract_rgb(input_data[pixel_idx + 1], params.input_format));
    //             vec3 yuv01 = rgb_to_yuv(extract_rgb(input_data[pixel_idx + params.width], params.input_format));
    //             vec3 yuv11 = rgb_to_yuv(extract_rgb(input_data[pixel_idx + params.width + 1], params.input_format));
    //
    //             float avg_u = (yuv00.y + yuv10.y + yuv01.y + yuv11.y) / 4.0;
    //             float avg_v = (yuv00.z + yuv10.z + yuv01.z + yuv11.z) / 4.0;
    //
    //             if (params.output_format == 0) {
    //                 // NV12: Interleaved UV after Y plane
    //                 uint uv_base = pixel_count;
    //                 uint uv_byte_idx = uv_idx * 2;
    //                 uint uv_word_idx = uv_byte_idx / 4;
    //                 uint uv_byte_offset = uv_byte_idx % 4;
    //
    //                 uint uv_packed = (uint(avg_v) << 8) | uint(avg_u);
    //                 atomicOr(output_data[uv_base/4 + uv_word_idx], uv_packed << (uv_byte_offset * 8));
    //             } else {
    //                 // I420: Separate U and V planes
    //                 uint u_base = pixel_count;
    //                 uint v_base = pixel_count + pixel_count / 4;
    //
    //                 uint u_byte_idx = u_base + uv_idx;
    //                 uint u_word_idx = u_byte_idx / 4;
    //                 uint u_byte_offset = u_byte_idx % 4;
    //
    //                 uint v_byte_idx = v_base + uv_idx;
    //                 uint v_word_idx = v_byte_idx / 4;
    //                 uint v_byte_offset = v_byte_idx % 4;
    //
    //                 atomicOr(output_data[u_word_idx], uint(avg_u) << (u_byte_offset * 8));
    //                 atomicOr(output_data[v_word_idx], uint(avg_v) << (v_byte_offset * 8));
    //             }
    //         }
    //     }
    // }

    // For now, return a placeholder - we'll compile the actual shader.
    // This needs to be replaced with the actual SPIR-V bytecode.
    compile_glsl_to_spirv()
}

/// Compile GLSL to SPIR-V at runtime using shaderc.
fn compile_glsl_to_spirv() -> Vec<u32> {
    const SHADER_SOURCE: &str = r#"
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
    uint input_format;   // 0=BGRx, 1=RGBx, 2=BGRA, 3=RGBA
    uint output_format;  // 0=NV12, 1=I420, 2=YUV444, 3=P010, 4=YUV444P10
} params;

layout(std430, binding = 0) readonly buffer InputBuffer {
    uint input_data[];
};

layout(std430, binding = 1) buffer OutputBuffer {
    uint output_data[];
};

// BT.601 conversion coefficients.
const float Y_R = 0.299;
const float Y_G = 0.587;
const float Y_B = 0.114;
const float U_R = -0.169;
const float U_G = -0.331;
const float U_B = 0.500;
const float V_R = 0.500;
const float V_G = -0.419;
const float V_B = -0.081;

vec3 extract_rgb(uint pixel, uint format) {
    uint b0 = (pixel >> 0) & 0xFFu;
    uint b1 = (pixel >> 8) & 0xFFu;
    uint b2 = (pixel >> 16) & 0xFFu;

    if (format == 0u || format == 2u) {
        // BGRx or BGRA: B=b0, G=b1, R=b2
        return vec3(float(b2), float(b1), float(b0));
    } else {
        // RGBx or RGBA: R=b0, G=b1, B=b2
        return vec3(float(b0), float(b1), float(b2));
    }
}

vec3 rgb_to_yuv(vec3 rgb) {
    float y = Y_R * rgb.r + Y_G * rgb.g + Y_B * rgb.b;
    float u = 128.0 + U_R * rgb.r + U_G * rgb.g + U_B * rgb.b;
    float v = 128.0 + V_R * rgb.r + V_G * rgb.g + V_B * rgb.b;
    return vec3(clamp(y, 0.0, 255.0), clamp(u, 0.0, 255.0), clamp(v, 0.0, 255.0));
}

// Convert 8-bit value to 10-bit in 16-bit word (value in upper 10 bits).
uint to_10bit(float val) {
    // Scale from 0-255 to 0-1023 (10-bit range), then shift left by 6.
    uint val10 = uint(val * 4.0);  // 0-255 -> 0-1020, close to 0-1023.
    return (val10 << 6u) & 0xFFC0u;  // Mask to ensure upper 10 bits only.
}

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;

    if (x >= params.width || y >= params.height) return;

    uint pixel_idx = y * params.width + x;
    uint pixel = input_data[pixel_idx];
    vec3 rgb = extract_rgb(pixel, params.input_format);
    vec3 yuv = rgb_to_yuv(rgb);

    uint pixel_count = params.width * params.height;

    if (params.output_format == 2u) {
        // YUV444 8-bit: Full resolution, byte-packed into uints.
        // Each pixel writes one byte to Y, U, and V planes.
        uint y_byte_idx = pixel_idx;
        uint y_word_idx = y_byte_idx / 4u;
        uint y_byte_offset = y_byte_idx % 4u;
        atomicOr(output_data[y_word_idx], uint(yuv.x) << (y_byte_offset * 8u));

        uint u_base = pixel_count;
        uint u_byte_idx = u_base + pixel_idx;
        uint u_word_idx = u_byte_idx / 4u;
        uint u_byte_offset = u_byte_idx % 4u;
        atomicOr(output_data[u_word_idx], uint(yuv.y) << (u_byte_offset * 8u));

        uint v_base = 2u * pixel_count;
        uint v_byte_idx = v_base + pixel_idx;
        uint v_word_idx = v_byte_idx / 4u;
        uint v_byte_offset = v_byte_idx % 4u;
        atomicOr(output_data[v_word_idx], uint(yuv.z) << (v_byte_offset * 8u));
    } else if (params.output_format == 4u) {
        // YUV444P10 (10-bit): 2-plane semi-planar format.
        // Y plane: 16-bit per sample, full resolution.
        // UV plane: 16-bit per component, interleaved, full resolution.
        uint y_word_idx = pixel_idx;  // One 16-bit value per pixel, packed 2 per uint.
        uint y_half_offset = pixel_idx % 2u;
        uint y_packed_idx = pixel_idx / 2u;
        atomicOr(output_data[y_packed_idx], to_10bit(yuv.x) << (y_half_offset * 16u));

        // UV plane starts after Y plane (pixel_count 16-bit values = pixel_count/2 uints).
        uint uv_base_words = pixel_count / 2u;
        // Each pixel has one UV pair (2x 16-bit = 32-bit = one uint).
        uint uv_word_idx = uv_base_words + pixel_idx;
        uint uv_packed = to_10bit(yuv.y) | (to_10bit(yuv.z) << 16u);
        output_data[uv_word_idx] = uv_packed;
    } else if (params.output_format == 3u) {
        // P010 (10-bit NV12): 2-plane semi-planar, 4:2:0 subsampling.
        // Y plane: 16-bit per sample.
        uint y_half_offset = pixel_idx % 2u;
        uint y_packed_idx = pixel_idx / 2u;
        atomicOr(output_data[y_packed_idx], to_10bit(yuv.x) << (y_half_offset * 16u));

        // Only process UV for top-left pixel of each 2x2 block.
        if ((x % 2u == 0u) && (y % 2u == 0u)) {
            uint uv_x = x / 2u;
            uint uv_y = y / 2u;
            uint uv_width = params.width / 2u;
            uint uv_idx = uv_y * uv_width + uv_x;

            // Sample all 4 pixels in the 2x2 block for proper UV averaging.
            vec3 yuv00 = yuv;
            uint idx10 = pixel_idx + 1u;
            uint idx01 = pixel_idx + params.width;
            uint idx11 = pixel_idx + params.width + 1u;

            vec3 yuv10 = (x + 1u < params.width) ?
                rgb_to_yuv(extract_rgb(input_data[idx10], params.input_format)) : yuv00;
            vec3 yuv01 = (y + 1u < params.height) ?
                rgb_to_yuv(extract_rgb(input_data[idx01], params.input_format)) : yuv00;
            vec3 yuv11 = (x + 1u < params.width && y + 1u < params.height) ?
                rgb_to_yuv(extract_rgb(input_data[idx11], params.input_format)) : yuv00;

            float avg_u = (yuv00.y + yuv10.y + yuv01.y + yuv11.y) / 4.0;
            float avg_v = (yuv00.z + yuv10.z + yuv01.z + yuv11.z) / 4.0;

            // UV plane starts after Y plane (pixel_count 16-bit values = pixel_count/2 uints).
            uint uv_base_words = pixel_count / 2u;
            // Each UV pair is one uint (U 16-bit, V 16-bit).
            uint uv_word_idx = uv_base_words + uv_idx;
            uint uv_packed = to_10bit(avg_u) | (to_10bit(avg_v) << 16u);
            output_data[uv_word_idx] = uv_packed;
        }
    } else {
        // YUV420 8-bit (NV12 or I420): Write Y for every pixel.
        uint y_byte_idx = pixel_idx;
        uint y_word_idx = y_byte_idx / 4u;
        uint y_byte_offset = y_byte_idx % 4u;
        atomicOr(output_data[y_word_idx], uint(yuv.x) << (y_byte_offset * 8u));

        // Only process UV for top-left pixel of each 2x2 block.
        if ((x % 2u == 0u) && (y % 2u == 0u)) {
            uint uv_x = x / 2u;
            uint uv_y = y / 2u;
            uint uv_width = params.width / 2u;
            uint uv_idx = uv_y * uv_width + uv_x;

            // Sample all 4 pixels in the 2x2 block for proper UV averaging.
            vec3 yuv00 = yuv;

            uint idx10 = pixel_idx + 1u;
            uint idx01 = pixel_idx + params.width;
            uint idx11 = pixel_idx + params.width + 1u;

            // Bounds check for right and bottom edges.
            vec3 yuv10 = (x + 1u < params.width) ?
                rgb_to_yuv(extract_rgb(input_data[idx10], params.input_format)) : yuv00;
            vec3 yuv01 = (y + 1u < params.height) ?
                rgb_to_yuv(extract_rgb(input_data[idx01], params.input_format)) : yuv00;
            vec3 yuv11 = (x + 1u < params.width && y + 1u < params.height) ?
                rgb_to_yuv(extract_rgb(input_data[idx11], params.input_format)) : yuv00;

            float avg_u = (yuv00.y + yuv10.y + yuv01.y + yuv11.y) / 4.0;
            float avg_v = (yuv00.z + yuv10.z + yuv01.z + yuv11.z) / 4.0;

            if (params.output_format == 0u) {
                // NV12: Interleaved UV after Y plane.
                uint uv_base_bytes = pixel_count;
                uint uv_byte_idx = uv_base_bytes + uv_idx * 2u;
                uint uv_word_idx = uv_byte_idx / 4u;
                uint uv_byte_offset = uv_byte_idx % 4u;

                // Pack U and V together.
                if (uv_byte_offset <= 2u) {
                    uint uv_packed = (uint(avg_v) << 8u) | uint(avg_u);
                    atomicOr(output_data[uv_word_idx], uv_packed << (uv_byte_offset * 8u));
                } else {
                    // Split across word boundary.
                    atomicOr(output_data[uv_word_idx], uint(avg_u) << 24u);
                    atomicOr(output_data[uv_word_idx + 1u], uint(avg_v));
                }
            } else {
                // I420: Separate U and V planes.
                uint uv_plane_size = pixel_count / 4u;

                uint u_base_bytes = pixel_count;
                uint u_byte_idx = u_base_bytes + uv_idx;
                uint u_word_idx = u_byte_idx / 4u;
                uint u_byte_offset = u_byte_idx % 4u;

                uint v_base_bytes = pixel_count + uv_plane_size;
                uint v_byte_idx = v_base_bytes + uv_idx;
                uint v_word_idx = v_byte_idx / 4u;
                uint v_byte_offset = v_byte_idx % 4u;

                atomicOr(output_data[u_word_idx], uint(avg_u) << (u_byte_offset * 8u));
                atomicOr(output_data[v_word_idx], uint(avg_v) << (v_byte_offset * 8u));
            }
        }
    }
}
"#;

    let compiler = shaderc::Compiler::new().expect("Failed to create shaderc compiler");
    let options = shaderc::CompileOptions::new().expect("Failed to create compile options");

    let artifact = compiler
        .compile_into_spirv(
            SHADER_SOURCE,
            shaderc::ShaderKind::Compute,
            "color_convert.comp",
            "main",
            Some(&options),
        )
        .expect("Failed to compile shader");

    artifact.as_binary().to_vec()
}
