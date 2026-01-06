//! H.264 encoder implementation using Vulkan Video.
//!
//! This module implements H.264 video encoding using Vulkan Video extensions.

mod api;
mod encode;
mod init;

use ash::vk;
use tracing::debug;

use crate::encoder::resources::{upload_image_to_input, UploadParams};
use crate::error::Result;

use crate::encoder::dpb::DecodedPictureBuffer;
use crate::encoder::gop::GopStructure;
use crate::encoder::EncodeConfig;
use crate::vulkan::VideoContext;

/// Minimum bitstream buffer size.
const MIN_BITSTREAM_BUFFER_SIZE: usize = 2 * 1024 * 1024;

/// H.264 macroblock size in pixels.
pub const MB_SIZE: u32 = 16;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReferenceInfo {
    pub dpb_slot: u8,
    pub frame_num: u32,
    pub poc: i32,
}

/// H.264 encoder.
pub struct H264Encoder {
    context: VideoContext,
    config: EncodeConfig,
    dpb: DecodedPictureBuffer,
    gop: GopStructure,

    // Video session.
    video_queue_fn: ash::khr::video_queue::Device,
    video_encode_fn: ash::khr::video_encode_queue::Device,
    session: vk::VideoSessionKHR,
    session_params: vk::VideoSessionParametersKHR,
    session_memory: Vec<vk::DeviceMemory>,

    // Frame counters.
    input_frame_num: u64,
    encode_frame_num: u64,
    frame_num_syntax: u32,
    idr_pic_id: u32,

    // Resources
    input_image: vk::Image,
    input_image_memory: vk::DeviceMemory,
    input_image_view: vk::ImageView,
    /// Current Vulkan image layout of `input_image` (tracked to avoid UB when transitioning).
    input_image_layout: vk::ImageLayout,
    /// DPB images (up to MAX_DPB_SLOTS for B-frame and long-term reference support).
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

    // SPS/PPS written flag.
    sps_written: bool,

    // Reference picture tracking.
    /// Whether we have a backward reference (for B-frames, L1).
    has_backward_reference: bool,
    /// Frame number of the L1 (backward) reference picture (for B-frames).
    backward_reference_frame_num: u32,
    /// POC of the L1 (backward) reference picture.
    backward_reference_poc: i32,
    /// DPB slot for L1 (backward) reference.
    backward_reference_dpb_slot: u8,
    /// Current DPB slot to use for setup (the reconstructed picture).
    current_dpb_slot: u8,
    /// Active L0 reference pictures (for P-frames). Ordered from most recent to oldest.
    l0_references: Vec<ReferenceInfo>,
    /// Number of active reference frames (as configured/negotiated).
    active_reference_count: u32,
}

impl H264Encoder {
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
unsafe impl Send for H264Encoder {}

impl Drop for H264Encoder {
    fn drop(&mut self) {
        unsafe {
            let _ = self.context.device().device_wait_idle();
            self.context
                .device()
                .destroy_query_pool(self.query_pool, None);
            self.context.device().destroy_fence(self.upload_fence, None);
            self.context.device().destroy_fence(self.encode_fence, None);
            self.context
                .device()
                .destroy_command_pool(self.command_pool, None);
            self.context
                .device()
                .destroy_buffer(self.bitstream_buffer, None);
            // Unmap the persistently mapped bitstream buffer before freeing memory.
            self.context
                .device()
                .unmap_memory(self.bitstream_buffer_memory);
            self.context
                .device()
                .free_memory(self.bitstream_buffer_memory, None);

            for i in 0..self.dpb_slot_count {
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

            self.context
                .device()
                .destroy_image_view(self.input_image_view, None);
            self.context.device().destroy_image(self.input_image, None);
            self.context
                .device()
                .free_memory(self.input_image_memory, None);

            (self
                .video_queue_fn
                .fp()
                .destroy_video_session_parameters_khr)(
                self.context.device().handle(),
                self.session_params,
                std::ptr::null(),
            );
            (self.video_queue_fn.fp().destroy_video_session_khr)(
                self.context.device().handle(),
                self.session,
                std::ptr::null(),
            );

            for memory in &self.session_memory {
                self.context.device().free_memory(*memory, None);
            }
        }
    }
}
