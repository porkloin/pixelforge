//! H.265/HEVC encoder implementation using Vulkan Video.
//!
//! This module implements H.265/HEVC video encoding using Vulkan Video extensions.

mod api;
mod encode;
mod init;

use ash::vk;
use tracing::debug;

use crate::encoder::dpb::DecodedPictureBuffer;
use crate::encoder::gop::GopStructure;
use crate::encoder::resources::{upload_image_to_input, UploadParams};
use crate::encoder::EncodeConfig;
use crate::error::Result;
use crate::vulkan::VideoContext;

/// Minimum bitstream buffer size.
const MIN_BITSTREAM_BUFFER_SIZE: usize = 2 * 1024 * 1024;

/// H.265 Coding Tree Block (CTB) size in pixels.
pub const CTB_SIZE: u32 = 32;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReferenceInfo {
    pub dpb_slot: u8,
    pub poc: i32,
}

/// H.265 encoder.
pub struct H265Encoder {
    context: VideoContext,
    config: EncodeConfig,
    dpb: DecodedPictureBuffer,
    gop: GopStructure,

    /// Aligned width (to CTB size).
    aligned_width: u32,
    /// Aligned height (to CTB size).
    aligned_height: u32,

    // Video session.
    video_queue_fn: ash::khr::video_queue::Device,
    video_encode_fn: ash::khr::video_encode_queue::Device,
    session: vk::VideoSessionKHR,
    session_params: vk::VideoSessionParametersKHR,
    session_memory: Vec<vk::DeviceMemory>,

    // Frame counters.
    input_frame_num: u64,
    encode_frame_num: u64,

    // Resources
    input_image: vk::Image,
    input_image_memory: vk::DeviceMemory,
    input_image_view: vk::ImageView,
    /// Current Vulkan image layout of `input_image` (tracked to avoid UB when transitioning).
    input_image_layout: vk::ImageLayout,
    /// DPB images.
    dpb_images: Vec<vk::Image>,
    dpb_image_memories: Vec<vk::DeviceMemory>,
    dpb_image_views: Vec<vk::ImageView>,
    /// Number of DPB slots allocated.
    dpb_slot_count: usize,
    bitstream_buffer: vk::Buffer,
    bitstream_buffer_memory: vk::DeviceMemory,
    /// Persistently mapped pointer to the bitstream buffer (avoids per-frame map/unmap).
    bitstream_buffer_ptr: *mut u8,

    // Command resources.
    command_pool: vk::CommandPool,
    upload_command_buffer: vk::CommandBuffer,
    upload_fence: vk::Fence,
    encode_command_buffer: vk::CommandBuffer,
    encode_fence: vk::Fence,
    query_pool: vk::QueryPool,

    // Parameter sets - cached header data (VPS/SPS/PPS)
    header_data: Option<Vec<u8>>,

    // Reference picture tracking.
    /// Whether we have a backward reference (for B-frames, L1).
    has_backward_reference: bool,
    /// POC of the L1 (backward) reference picture.
    backward_reference_poc: i32,
    /// DPB slot for L1 (backward) reference.
    backward_reference_dpb_slot: u8,
    /// Current DPB slot to use for setup (the reconstructed picture).
    current_dpb_slot: u8,
    /// Active L0 reference pictures (for P-frames).
    l0_references: Vec<ReferenceInfo>,
    /// Number of active reference frames.
    active_reference_count: u32,

    // DPB slot activation tracking.
    /// Tracks which DPB slots have been activated (used at least once).
    dpb_slot_active: Vec<bool>,
}

impl H265Encoder {
    /// Upload input frame from a GPU image.
    ///
    /// This copies from a source NV12 image directly to the encoder's input image,
    /// avoiding any CPU-side data copies. The source image must be in NV12 format
    /// with the same dimensions as the encoder configuration. The source image
    /// should be in GENERAL layout.
    fn upload_from_image(&mut self, src_image: vk::Image) -> Result<()> {
        if src_image == self.input_image {
            debug!("Source image is the encoder's input image, skipping upload copy");
            return Ok(());
        }

        let params = UploadParams {
            upload_command_buffer: self.upload_command_buffer,
            upload_fence: self.upload_fence,
            src_image,
            dst_image: self.input_image,
            width: self.config.dimensions.width,
            height: self.config.dimensions.height,
            pixel_format: self.config.pixel_format,
            input_image_layout: self.input_image_layout,
        };

        upload_image_to_input(&self.context, &params)?;

        // Update tracked layout.
        self.input_image_layout = vk::ImageLayout::VIDEO_ENCODE_SRC_KHR;

        Ok(())
    }
}

// SAFETY: The raw pointer bitstream_buffer_ptr is only used within the encoder's
// thread and is properly synchronized via Vulkan fences before access
unsafe impl Send for H265Encoder {}

impl Drop for H265Encoder {
    fn drop(&mut self) {
        unsafe {
            // Wait for device to be idle before destroying resources.
            let _ = self.context.device().device_wait_idle();

            // Destroy query pool.
            self.context
                .device()
                .destroy_query_pool(self.query_pool, None);

            // Destroy fences.
            self.context.device().destroy_fence(self.upload_fence, None);
            self.context.device().destroy_fence(self.encode_fence, None);

            // Destroy bitstream buffer.
            self.context
                .device()
                .unmap_memory(self.bitstream_buffer_memory);
            self.context
                .device()
                .destroy_buffer(self.bitstream_buffer, None);
            self.context
                .device()
                .free_memory(self.bitstream_buffer_memory, None);

            // Destroy input image.
            self.context
                .device()
                .destroy_image_view(self.input_image_view, None);
            self.context.device().destroy_image(self.input_image, None);
            self.context
                .device()
                .free_memory(self.input_image_memory, None);

            // Destroy DPB images.
            for i in 0..self.dpb_images.len() {
                self.context
                    .device()
                    .destroy_image_view(self.dpb_image_views[i], None);
                self.context
                    .device()
                    .destroy_image(self.dpb_images[i], None);
                self.context
                    .device()
                    .free_memory(self.dpb_image_memories[i], None);
            }

            // Free command buffers.
            self.context.device().free_command_buffers(
                self.command_pool,
                &[self.upload_command_buffer, self.encode_command_buffer],
            );

            // Destroy command pool.
            self.context
                .device()
                .destroy_command_pool(self.command_pool, None);

            // Destroy video session parameters.
            (self
                .video_queue_fn
                .fp()
                .destroy_video_session_parameters_khr)(
                self.context.device().handle(),
                self.session_params,
                std::ptr::null(),
            );

            // Destroy video session.
            (self.video_queue_fn.fp().destroy_video_session_khr)(
                self.context.device().handle(),
                self.session,
                std::ptr::null(),
            );

            // Free session memory.
            for mem in &self.session_memory {
                self.context.device().free_memory(*mem, None);
            }
        }
    }
}
