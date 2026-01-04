//! GPU-accelerated color format conversion using Vulkan compute shaders.
//!
//! This module provides efficient color space conversion on the GPU,
//! converting from RGB/BGR formats to YUV for video encoding.
//!
//! The output remains on the GPU as a `vk::Image` to avoid unnecessary
//! CPU round-trips when used with a GPU-based video encoder.

mod pipeline;
mod shader;

use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;

/// Supported input pixel formats for color conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum InputFormat {
    /// BGRx (32-bit, blue first, x = unused).
    BGRx,
    /// RGBx (32-bit, red first, x = unused).
    RGBx,
    /// BGRA (32-bit, blue first, alpha last).
    BGRA,
    /// RGBA (32-bit, red first, alpha last).
    RGBA,
}

impl InputFormat {
    /// Bytes per pixel for this format.
    pub fn bytes_per_pixel(&self) -> usize {
        4 // All current formats are 32-bit
    }
}

/// Supported output YUV formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// NV12: Y plane followed by interleaved UV (4:2:0).
    NV12,
    /// I420: Y plane, U plane, V plane (4:2:0).
    I420,
    /// YUV444: Full resolution Y, U, V planes.
    YUV444,
}

impl OutputFormat {
    /// Calculate output size in bytes for given dimensions.
    pub fn output_size(&self, width: u32, height: u32) -> usize {
        let pixel_count = (width * height) as usize;
        match self {
            OutputFormat::NV12 | OutputFormat::I420 => pixel_count * 3 / 2,
            OutputFormat::YUV444 => pixel_count * 3,
        }
    }

    /// Get the Vulkan format for this output format.
    pub fn vulkan_format(&self) -> vk::Format {
        match self {
            OutputFormat::NV12 => vk::Format::G8_B8R8_2PLANE_420_UNORM,
            OutputFormat::I420 => vk::Format::G8_B8_R8_3PLANE_420_UNORM,
            OutputFormat::YUV444 => vk::Format::G8_B8_R8_3PLANE_444_UNORM,
        }
    }
}

/// Configuration for the color converter.
#[derive(Clone, Debug)]
pub struct ColorConverterConfig {
    /// Input frame width.
    pub width: u32,
    /// Input frame height.
    pub height: u32,
    /// Input pixel format.
    pub input_format: InputFormat,
    /// Output YUV format.
    pub output_format: OutputFormat,
}

/// GPU-based color format converter.
///
/// Uses Vulkan compute shaders to convert RGB/BGR formats to YUV.
/// with minimal latency and high throughput.
pub struct ColorConverter {
    context: VideoContext,
    config: ColorConverterConfig,

    // Compute pipeline resources.
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,

    // Input buffer (host-visible for upload)
    input_buffer: vk::Buffer,
    input_memory: vk::DeviceMemory,

    // Output buffer (compute shader writes here)
    output_buffer: vk::Buffer,
    output_memory: vk::DeviceMemory,

    // Command resources.
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl ColorConverter {
    /// Create a new color converter.
    pub fn new(context: VideoContext, config: ColorConverterConfig) -> Result<Self> {
        pipeline::create_converter(context, config)
    }

    /// Get the video context.
    ///
    /// Returns a reference to the VideoContext used by this converter.
    pub fn context(&self) -> &VideoContext {
        &self.context
    }

    /// Get the output buffer.
    ///
    /// Returns the Vulkan buffer containing raw YUV data.
    /// This is useful for direct GPU-to-GPU transfers.
    pub fn output_buffer(&self) -> vk::Buffer {
        self.output_buffer
    }

    /// Build buffer-to-image copy regions for multi-planar YUV formats.
    ///
    /// For multi-planar formats like NV12, I420, and YUV444, we need separate.
    /// copy regions for each plane with the appropriate aspect mask.
    fn build_buffer_to_image_copy_regions(&self) -> Vec<vk::BufferImageCopy> {
        match self.config.output_format {
            OutputFormat::NV12 => {
                // NV12: Y plane (PLANE_0) followed by interleaved UV plane (PLANE_1)
                let y_size = (self.config.width * self.config.height) as u64;
                vec![
                    // Y plane.
                    vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                    // UV plane (interleaved, half resolution)
                    vk::BufferImageCopy {
                        buffer_offset: y_size,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width / 2,
                            height: self.config.height / 2,
                            depth: 1,
                        },
                    },
                ]
            }
            OutputFormat::I420 => {
                // I420: Y plane, U plane, V plane (all separate)
                let y_size = (self.config.width * self.config.height) as u64;
                let uv_size = y_size / 4;
                vec![
                    // Y plane.
                    vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                    // U plane.
                    vk::BufferImageCopy {
                        buffer_offset: y_size,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width / 2,
                            height: self.config.height / 2,
                            depth: 1,
                        },
                    },
                    // V plane.
                    vk::BufferImageCopy {
                        buffer_offset: y_size + uv_size,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_2,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width / 2,
                            height: self.config.height / 2,
                            depth: 1,
                        },
                    },
                ]
            }
            OutputFormat::YUV444 => {
                // YUV444: Y, U, V planes all at full resolution.
                let plane_size = (self.config.width * self.config.height) as u64;
                vec![
                    // Y plane.
                    vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                    // U plane.
                    vk::BufferImageCopy {
                        buffer_offset: plane_size,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                    // V plane.
                    vk::BufferImageCopy {
                        buffer_offset: plane_size * 2,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_2,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                ]
            }
        }
    }

    /// Convert an input image directly to a target image (zero-copy encoder path).
    ///
    /// This is the most efficient path for encoding: it converts the source image.
    /// directly into the encoder's input image, eliminating an intermediate copy.
    ///
    /// The target image must:
    /// - Have the same dimensions as the converter's configuration
    /// - Be in a format compatible with NV12/YUV (G8_B8R8_2PLANE_420_UNORM)
    /// - Have TRANSFER_DST usage flag
    ///
    /// After this call, the target image will be in VIDEO_ENCODE_SRC_KHR layout,
    /// ready for encoding.
    ///
    /// # Arguments
    /// * `src_image` - Source RGB/BGR image (e.g., from DMA-BUF import)
    /// * `target_image` - Target image to write YUV data to (e.g., encoder's input_image)
    ///
    /// # Returns
    /// Returns `Ok(())` on success. The target_image is transitioned to VIDEO_ENCODE_SRC_KHR.
    pub fn convert(&mut self, src_image: vk::Image, target_image: vk::Image) -> Result<()> {
        let device = self.context.device();

        // Reset and record command buffer.
        unsafe {
            device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

            // --- Phase 1: Copy source image to input buffer ---

            // Transition source image to TRANSFER_SRC_OPTIMAL.
            let src_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL) // DMA-BUF imported images are typically GENERAL
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(src_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[src_barrier],
            );

            // Copy image to buffer.
            let region = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,   // Tightly packed
                buffer_image_height: 0, // Tightly packed
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width: self.config.width,
                    height: self.config.height,
                    depth: 1,
                },
            };

            device.cmd_copy_image_to_buffer(
                self.command_buffer,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.input_buffer,
                &[region],
            );

            // Memory barrier: ensure buffer write is complete before compute shader reads it.
            let input_buffer_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .buffer(self.input_buffer)
                .size(vk::WHOLE_SIZE);

            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[input_buffer_barrier],
                &[],
            );

            // --- Phase 2: Run compute shader ---

            // Clear output buffer to zero before compute shader runs.
            // This is CRITICAL because the shader uses atomicOr to write values,
            // which would accumulate if the buffer isn't cleared between frames.
            let output_size = self
                .config
                .output_format
                .output_size(self.config.width, self.config.height);
            device.cmd_fill_buffer(
                self.command_buffer,
                self.output_buffer,
                0,
                output_size as vk::DeviceSize,
                0,
            );

            // Barrier: fill buffer write -> shader read/write
            let fill_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .buffer(self.output_buffer)
                .size(vk::WHOLE_SIZE);

            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[fill_barrier],
                &[],
            );

            // Bind pipeline and descriptor set.
            device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );

            device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            // Push constants: width, height, input_format, output_format.
            let push_constants: [u32; 4] = [
                self.config.width,
                self.config.height,
                self.config.input_format as u32,
                self.config.output_format as u32,
            ];
            let push_constants_bytes: &[u8] = std::slice::from_raw_parts(
                push_constants.as_ptr() as *const u8,
                std::mem::size_of_val(&push_constants),
            );
            device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constants_bytes,
            );

            // Dispatch workgroups (8x8 workgroup size)
            let workgroup_x = self.config.width.div_ceil(8);
            let workgroup_y = self.config.height.div_ceil(8);
            device.cmd_dispatch(self.command_buffer, workgroup_x, workgroup_y, 1);

            // --- Phase 3: Copy output buffer to target image (encoder's input) ---

            // Memory barrier: buffer write -> buffer read
            let output_buffer_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .buffer(self.output_buffer)
                .size(vk::WHOLE_SIZE);

            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[output_buffer_barrier],
                &[],
            );

            // Transition target image (encoder's input) to TRANSFER_DST layout.
            let target_barrier_to_transfer = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(target_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[target_barrier_to_transfer],
            );

            // Copy buffer to target image - use per-plane copies for multi-planar formats.
            let copy_regions = self.build_buffer_to_image_copy_regions();

            device.cmd_copy_buffer_to_image(
                self.command_buffer,
                self.output_buffer,
                target_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy_regions,
            );

            // Transition target image to VIDEO_ENCODE_SRC_KHR layout for encoding.
            let target_barrier_to_encode = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
                .image(target_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            // Also transition source image back to GENERAL for reuse.
            let src_barrier_back = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(src_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);

            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[target_barrier_to_encode, src_barrier_back],
            );

            device
                .end_command_buffer(self.command_buffer)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }

        // Submit and wait.
        unsafe {
            device
                .reset_fences(&[self.fence])
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

            let command_buffers = [self.command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

            device
                .queue_submit(self.context.transfer_queue(), &[submit_info], self.fence)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

            device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }

        Ok(())
    }
}

impl Drop for ColorConverter {
    fn drop(&mut self) {
        unsafe {
            let device = self.context.device();

            // Destroy buffers and their memory.
            device.destroy_buffer(self.input_buffer, None);
            device.free_memory(self.input_memory, None);
            device.destroy_buffer(self.output_buffer, None);
            device.free_memory(self.output_memory, None);

            // Destroy pipeline resources.
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            // Destroy command resources.
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================
    // InputFormat tests.
    // ========================

    #[test]
    fn test_input_format_bytes_per_pixel() {
        assert_eq!(InputFormat::BGRx.bytes_per_pixel(), 4);
        assert_eq!(InputFormat::RGBx.bytes_per_pixel(), 4);
        assert_eq!(InputFormat::BGRA.bytes_per_pixel(), 4);
        assert_eq!(InputFormat::RGBA.bytes_per_pixel(), 4);
    }

    #[test]
    fn test_input_format_enum_values() {
        // Verify enum values match shader expectations.
        assert_eq!(InputFormat::BGRx as u32, 0);
        assert_eq!(InputFormat::RGBx as u32, 1);
        assert_eq!(InputFormat::BGRA as u32, 2);
        assert_eq!(InputFormat::RGBA as u32, 3);
    }

    // ========================
    // OutputFormat tests.
    // ========================

    #[test]
    fn test_output_format_size_nv12() {
        // NV12: Y plane + interleaved UV at half resolution = 1.5 * pixel_count.
        assert_eq!(OutputFormat::NV12.output_size(8, 8), 96); // 64 * 1.5 = 96
        assert_eq!(OutputFormat::NV12.output_size(16, 16), 384); // 256 * 1.5 = 384
        assert_eq!(
            OutputFormat::NV12.output_size(1920, 1080),
            1920 * 1080 * 3 / 2
        );
    }

    #[test]
    fn test_output_format_size_i420() {
        // I420: Y plane + U plane (quarter) + V plane (quarter) = 1.5 * pixel_count.
        assert_eq!(OutputFormat::I420.output_size(8, 8), 96);
        assert_eq!(OutputFormat::I420.output_size(16, 16), 384);
        assert_eq!(
            OutputFormat::I420.output_size(1920, 1080),
            1920 * 1080 * 3 / 2
        );
    }

    #[test]
    fn test_output_format_size_yuv444() {
        // YUV444: Full resolution Y + U + V = 3 * pixel_count.
        assert_eq!(OutputFormat::YUV444.output_size(8, 8), 192); // 64 * 3 = 192
        assert_eq!(OutputFormat::YUV444.output_size(16, 16), 768); // 256 * 3 = 768
        assert_eq!(
            OutputFormat::YUV444.output_size(1920, 1080),
            1920 * 1080 * 3
        );
    }

    #[test]
    fn test_output_format_size_standard_resolutions() {
        // Common video resolutions.
        let resolutions = [
            (320, 240),   // QVGA
            (640, 480),   // VGA
            (1280, 720),  // HD
            (1920, 1080), // Full HD
            (3840, 2160), // 4K
        ];

        for (width, height) in resolutions {
            let pixels = (width * height) as usize;

            // NV12 and I420 should be 1.5x pixels.
            assert_eq!(
                OutputFormat::NV12.output_size(width, height),
                pixels * 3 / 2
            );
            assert_eq!(
                OutputFormat::I420.output_size(width, height),
                pixels * 3 / 2
            );

            // YUV444 should be 3x pixels.
            assert_eq!(OutputFormat::YUV444.output_size(width, height), pixels * 3);
        }
    }

    #[test]
    fn test_output_format_enum_values() {
        // Verify enum values match shader expectations.
        assert_eq!(OutputFormat::NV12 as u32, 0);
        assert_eq!(OutputFormat::I420 as u32, 1);
        assert_eq!(OutputFormat::YUV444 as u32, 2);
    }

    // ========================
    // ColorConverterConfig tests.
    // ========================

    #[test]
    fn test_config_clone() {
        let config = ColorConverterConfig {
            width: 1920,
            height: 1080,
            input_format: InputFormat::BGRx,
            output_format: OutputFormat::NV12,
        };

        let cloned = config.clone();
        assert_eq!(cloned.width, 1920);
        assert_eq!(cloned.height, 1080);
        assert_eq!(cloned.input_format, InputFormat::BGRx);
        assert_eq!(cloned.output_format, OutputFormat::NV12);
    }

    #[test]
    fn test_config_debug() {
        let config = ColorConverterConfig {
            width: 640,
            height: 480,
            input_format: InputFormat::RGBA,
            output_format: OutputFormat::I420,
        };

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("640"));
        assert!(debug_str.contains("480"));
        assert!(debug_str.contains("RGBA"));
        assert!(debug_str.contains("I420"));
    }

    // ========================
    // ColorConverter Vulkan tests (require hardware)
    // ========================
    // These tests require Vulkan support and are gated behind a feature.
    // or will be skipped if Vulkan initialization fails.

    /// Helper to create a Vulkan context for testing.
    /// Returns None if Vulkan is not available.
    fn create_test_context() -> Option<VideoContext> {
        use crate::VideoContextBuilder;

        VideoContextBuilder::new()
            .app_name("ColorConverter Test")
            .enable_validation(false)
            .build()
            .ok()
    }

    #[test]
    fn test_converter_creation() {
        let Some(context) = create_test_context() else {
            eprintln!("Skipping test_converter_creation: Vulkan not available");
            return;
        };

        let config = ColorConverterConfig {
            width: 64,
            height: 64,
            input_format: InputFormat::BGRx,
            output_format: OutputFormat::NV12,
        };

        let result = ColorConverter::new(context, config);
        assert!(
            result.is_ok(),
            "Failed to create ColorConverter: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_converter_creation_various_formats() {
        let Some(context) = create_test_context() else {
            eprintln!("Skipping test: Vulkan not available");
            return;
        };

        let input_formats = [
            InputFormat::BGRx,
            InputFormat::RGBx,
            InputFormat::BGRA,
            InputFormat::RGBA,
        ];
        let output_formats = [OutputFormat::NV12, OutputFormat::I420, OutputFormat::YUV444];

        for input_format in &input_formats {
            for output_format in &output_formats {
                let config = ColorConverterConfig {
                    width: 32,
                    height: 32,
                    input_format: *input_format,
                    output_format: *output_format,
                };

                let result = ColorConverter::new(context.clone(), config);
                assert!(
                    result.is_ok(),
                    "Failed to create ColorConverter with {:?} -> {:?}: {:?}",
                    input_format,
                    output_format,
                    result.err()
                );
            }
        }
    }
}
