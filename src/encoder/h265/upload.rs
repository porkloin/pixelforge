//! H.265/HEVC encoder frame upload functionality.

use super::H265Encoder;

use crate::encoder::resources::{upload_image_to_input, UploadParams};
use crate::error::Result;
use ash::vk;
use tracing::debug;

impl H265Encoder {
    /// Upload input frame from a GPU image.
    ///
    /// This copies from a source NV12 image directly to the encoder's input image,
    /// avoiding any CPU-side data copies. The source image must be in NV12 format
    /// with the same dimensions as the encoder configuration. The source image
    /// should be in GENERAL layout.
    pub(super) fn upload_from_image(&mut self, src_image: vk::Image) -> Result<()> {
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
