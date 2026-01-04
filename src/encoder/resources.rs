use crate::encoder::{BitDepth, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;
use std::ptr;

pub fn query_supported_video_formats(
    context: &VideoContext,
    profile_info: &vk::VideoProfileInfoKHR,
    image_usage: vk::ImageUsageFlags,
) -> Result<Vec<vk::Format>> {
    let video_queue_fn = ash::khr::video_queue::Instance::load(context.entry(), context.instance());

    // Vulkan expects a profile list in the pNext chain.
    let profiles = [*profile_info];
    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);

    let mut format_info = vk::PhysicalDeviceVideoFormatInfoKHR::default().image_usage(image_usage);
    format_info.p_next = (&mut profile_list as *mut vk::VideoProfileListInfoKHR).cast();

    let physical_device = context.physical_device();
    let mut count = 0u32;
    let result = unsafe {
        (video_queue_fn
            .fp()
            .get_physical_device_video_format_properties_khr)(
            physical_device,
            &format_info,
            &mut count,
            ptr::null_mut(),
        )
    };

    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::NoSuitableDevice(format!(
            "Failed to query video format properties for usage {:?}: {:?}",
            image_usage, result
        )));
    }

    if count == 0 {
        return Ok(Vec::new());
    }

    let mut props = vec![vk::VideoFormatPropertiesKHR::default(); count as usize];
    let result = unsafe {
        (video_queue_fn
            .fp()
            .get_physical_device_video_format_properties_khr)(
            physical_device,
            &format_info,
            &mut count,
            props.as_mut_ptr(),
        )
    };

    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::NoSuitableDevice(format!(
            "Failed to enumerate video format properties for usage {:?}: {:?}",
            image_usage, result
        )));
    }

    props.truncate(count as usize);
    Ok(props.into_iter().map(|p| p.format).collect())
}

/// Get the Vulkan format for a given pixel format and bit depth.
///
/// Supports YUV420 and YUV444 in 8-bit and 10-bit.
/// For YUV444, uses 2-plane (semi-planar) formats from VK_EXT_ycbcr_2plane_444_formats
/// which are supported by NVIDIA hardware for video encoding.
pub fn get_video_format(pixel_format: PixelFormat, bit_depth: BitDepth) -> vk::Format {
    match (pixel_format, bit_depth) {
        (PixelFormat::Yuv420, BitDepth::Eight) => vk::Format::G8_B8R8_2PLANE_420_UNORM,
        (PixelFormat::Yuv420, BitDepth::Ten) => {
            vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
        }
        // Use 2-plane semi-planar formats for YUV444 (supported by NVIDIA for video encoding).
        (PixelFormat::Yuv444, BitDepth::Eight) => vk::Format::G8_B8R8_2PLANE_444_UNORM,
        (PixelFormat::Yuv444, BitDepth::Ten) => {
            vk::Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16
        }
        // TODO: Add support for YUV422 formats.
        _ => unimplemented!(
            "Unsupported pixel format / bit depth combination: {:?} / {:?}",
            pixel_format,
            bit_depth
        ),
    }
}

/// Create a codec name array for Vulkan from a string.
///
/// This creates a null-terminated i8 array of 256 bytes for use with Vulkan
/// video extensions.
pub fn make_codec_name(codec_name: &[u8]) -> [i8; 256] {
    let mut name = [0i8; 256];
    for (i, &byte) in codec_name.iter().enumerate() {
        if i < 255 {
            name[i] = byte as i8;
        }
    }
    name
}

pub fn find_memory_type(
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..memory_props.memory_type_count).find(|&i| {
        (type_filter & (1 << i)) != 0
            && memory_props.memory_types[i as usize]
                .property_flags
                .contains(properties)
    })
}

pub fn create_bitstream_buffer(
    context: &VideoContext,
    size: usize,
    profile_info: &vk::VideoProfileInfoKHR,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let profiles = [*profile_info];
    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);

    let mut create_info = vk::BufferCreateInfo::default()
        .size(size as vk::DeviceSize)
        .usage(vk::BufferUsageFlags::VIDEO_ENCODE_DST_KHR)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    create_info.p_next = (&mut profile_list as *mut vk::VideoProfileListInfoKHR).cast();

    let buffer = unsafe { context.device().create_buffer(&create_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("buffer creation: {}", e)))?;

    let mem_requirements = unsafe { context.device().get_buffer_memory_requirements(buffer) };

    let memory_type_index = find_memory_type(
        context.memory_properties(),
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| {
        PixelForgeError::MemoryAllocation("No suitable memory type for buffer".to_string())
    })?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index);

    let memory = unsafe { context.device().allocate_memory(&alloc_info, None) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    unsafe { context.device().bind_buffer_memory(buffer, memory, 0) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    Ok((buffer, memory))
}
/// Create an image for video encoding (input or DPB).
///
/// This creates a VkImage suitable for use with a video encoder.
/// For DPB images, the usage is VIDEO_ENCODE_DPB_KHR.
/// For input images, the usage is VIDEO_ENCODE_SRC_KHR | TRANSFER_DST.
///
/// # Arguments
/// * `context` - The Vulkan video context
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `format` - The Vulkan format to use for the image
/// * `is_dpb` - If true, create a DPB image; if false, create an input image
/// * `profile_info` - Video profile info for the encoder session
pub fn create_image(
    context: &VideoContext,
    width: u32,
    height: u32,
    format: vk::Format,
    is_dpb: bool,
    profile_info: &vk::VideoProfileInfoKHR,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let usage = if is_dpb {
        vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR
    } else {
        vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR | vk::ImageUsageFlags::TRANSFER_DST
    };

    let profiles = [*profile_info];
    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);

    let mut create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    create_info.p_next = (&mut profile_list as *mut vk::VideoProfileListInfoKHR).cast();

    let image = unsafe { context.device().create_image(&create_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("image creation: {}", e)))?;

    let mem_requirements = unsafe { context.device().get_image_memory_requirements(image) };

    let memory_type_index = find_memory_type(
        context.memory_properties(),
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| {
        PixelForgeError::MemoryAllocation("No suitable memory type for image".to_string())
    })?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index);

    let memory = unsafe { context.device().allocate_memory(&alloc_info, None) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    unsafe { context.device().bind_image_memory(image, memory, 0) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    let view_create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        })
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let view = unsafe { context.device().create_image_view(&view_create_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("image view creation: {}", e)))?;

    Ok((image, memory, view))
}
/// Allocate and bind memory for a video session.
///
/// Returns the allocated device memory handles.
pub fn allocate_session_memory(
    context: &VideoContext,
    session: vk::VideoSessionKHR,
    video_queue_fn: &ash::khr::video_queue::Device,
) -> Result<Vec<vk::DeviceMemory>> {
    // Query memory requirements count.
    let mut memory_requirements_count = 0u32;
    let result = unsafe {
        (video_queue_fn
            .fp()
            .get_video_session_memory_requirements_khr)(
            context.device().handle(),
            session,
            &mut memory_requirements_count,
            ptr::null_mut(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::MemoryAllocation(format!("{:?}", result)));
    }

    // Query actual requirements.
    let mut memory_requirements =
        vec![vk::VideoSessionMemoryRequirementsKHR::default(); memory_requirements_count as usize];
    let result = unsafe {
        (video_queue_fn
            .fp()
            .get_video_session_memory_requirements_khr)(
            context.device().handle(),
            session,
            &mut memory_requirements_count,
            memory_requirements.as_mut_ptr(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::MemoryAllocation(format!("{:?}", result)));
    }

    // Allocate and bind memory for each requirement.
    let mut session_memory = Vec::new();
    let mut bind_infos = Vec::new();

    for req in &memory_requirements {
        let memory_type_index = find_memory_type(
            context.memory_properties(),
            req.memory_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|| {
            find_memory_type(
                context.memory_properties(),
                req.memory_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::empty(),
            )
        })
        .ok_or_else(|| {
            PixelForgeError::MemoryAllocation(format!(
                "No suitable memory type for video session (type_bits: 0x{:x})",
                req.memory_requirements.memory_type_bits
            ))
        })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.memory_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { context.device().allocate_memory(&alloc_info, None) }
            .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

        bind_infos.push(
            vk::BindVideoSessionMemoryInfoKHR::default()
                .memory_bind_index(req.memory_bind_index)
                .memory(memory)
                .memory_offset(0)
                .memory_size(req.memory_requirements.size),
        );

        session_memory.push(memory);
    }

    // Bind all memory to the session.
    let result = unsafe {
        (video_queue_fn.fp().bind_video_session_memory_khr)(
            context.device().handle(),
            session,
            bind_infos.len() as u32,
            bind_infos.as_ptr(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::MemoryAllocation(format!("{:?}", result)));
    }

    Ok(session_memory)
}

/// Command resources for encoding operations.
pub struct CommandResources {
    /// Command pool.
    pub command_pool: vk::CommandPool,
    /// Command buffer for upload operations.
    pub upload_command_buffer: vk::CommandBuffer,
    /// Fence for upload synchronization.
    pub upload_fence: vk::Fence,
    /// Command buffer for encode operations.
    pub encode_command_buffer: vk::CommandBuffer,
    /// Fence for encode synchronization.
    pub encode_fence: vk::Fence,
}

/// Create command resources for encoding.
pub fn create_command_resources(
    context: &VideoContext,
    queue_family_index: u32,
) -> Result<CommandResources> {
    // Create command pool.
    let pool_create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    let command_pool = unsafe {
        context
            .device()
            .create_command_pool(&pool_create_info, None)
    }
    .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    // Allocate command buffers.
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(2);

    let command_buffers = unsafe { context.device().allocate_command_buffers(&alloc_info) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
    let upload_command_buffer = command_buffers[0];
    let encode_command_buffer = command_buffers[1];

    // Create fences.
    let fence_create_info = vk::FenceCreateInfo::default();
    let upload_fence = unsafe { context.device().create_fence(&fence_create_info, None) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
    let encode_fence = unsafe { context.device().create_fence(&fence_create_info, None) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    Ok(CommandResources {
        command_pool,
        upload_command_buffer,
        upload_fence,
        encode_command_buffer,
        encode_fence,
    })
}

/// Create DPB images for video encoding.
///
/// Returns vectors of images, memories, and views.
pub fn create_dpb_images(
    context: &VideoContext,
    width: u32,
    height: u32,
    format: vk::Format,
    count: usize,
    profile_info: &vk::VideoProfileInfoKHR,
) -> Result<(Vec<vk::Image>, Vec<vk::DeviceMemory>, Vec<vk::ImageView>)> {
    let mut dpb_images = Vec::with_capacity(count);
    let mut dpb_image_memories = Vec::with_capacity(count);
    let mut dpb_image_views = Vec::with_capacity(count);

    for _ in 0..count {
        let (dpb_image, dpb_image_memory, dpb_image_view) =
            create_image(context, width, height, format, true, profile_info)?;
        dpb_images.push(dpb_image);
        dpb_image_memories.push(dpb_image_memory);
        dpb_image_views.push(dpb_image_view);
    }

    Ok((dpb_images, dpb_image_memories, dpb_image_views))
}

/// Map a bitstream buffer for persistent access.
pub fn map_bitstream_buffer(
    context: &VideoContext,
    memory: vk::DeviceMemory,
    size: usize,
) -> Result<*mut u8> {
    let ptr = unsafe {
        context.device().map_memory(
            memory,
            0,
            size as vk::DeviceSize,
            vk::MemoryMapFlags::empty(),
        )
    }
    .map_err(|e| {
        PixelForgeError::MemoryAllocation(format!("Failed to map bitstream buffer: {}", e))
    })? as *mut u8;

    Ok(ptr)
}

/// Parameters for uploading an image to the encoder's input image.
pub struct UploadParams {
    /// The command buffer to use for the upload.
    pub upload_command_buffer: vk::CommandBuffer,
    /// The fence to use for synchronization.
    pub upload_fence: vk::Fence,
    /// The source image to copy from.
    pub src_image: vk::Image,
    /// The destination image to copy to.
    pub dst_image: vk::Image,
    /// The width of the image.
    pub width: u32,
    /// The height of the image.
    pub height: u32,
    /// The pixel format of the image.
    pub pixel_format: PixelFormat,
    /// The current layout of the input image.
    pub input_image_layout: vk::ImageLayout,
}

/// Upload an image to the encoder's input image via GPU-to-GPU copy.
///
/// This function handles:
/// - Resetting and beginning the command buffer
/// - Transitioning source image from GENERAL to TRANSFER_SRC
/// - Transitioning destination image from `input_image_layout` to TRANSFER_DST
/// - Copying Y and UV planes (NV12 format)
/// - Transitioning destination image to VIDEO_ENCODE_SRC
/// - Transitioning source image back to GENERAL
/// - Submitting the command buffer and waiting for completion
///
/// Returns Ok(()) on success, or an error if any Vulkan operation fails.
pub fn upload_image_to_input(
    context: &crate::vulkan::VideoContext,
    params: &UploadParams,
) -> Result<()> {
    let device = context.device();

    unsafe {
        device.reset_command_buffer(
            params.upload_command_buffer,
            vk::CommandBufferResetFlags::empty(),
        )
    }
    .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(params.upload_command_buffer, &begin_info) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    // Transition source image from GENERAL to TRANSFER_SRC.
    let src_barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(params.src_image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

    // Transition destination image to TRANSFER_DST.
    let dst_barrier = vk::ImageMemoryBarrier::default()
        .old_layout(params.input_image_layout)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(params.dst_image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

    unsafe {
        device.cmd_pipeline_barrier(
            params.upload_command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[src_barrier, dst_barrier],
        );
    }

    // Copy image to image using per-plane copy regions (NV12 format).
    // Copy Y plane (plane 0).
    let y_copy_region = vk::ImageCopy {
        src_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::PLANE_0,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        dst_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::PLANE_0,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        extent: vk::Extent3D {
            width: params.width,
            height: params.height,
            depth: 1,
        },
    };

    // Copy UV plane (plane 1).
    let (uv_width, uv_height) = match params.pixel_format {
        PixelFormat::Yuv420 => (params.width / 2, params.height / 2),
        PixelFormat::Yuv444 => (params.width, params.height),
        _ => (params.width / 2, params.height / 2),
    };

    let uv_copy_region = vk::ImageCopy {
        src_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::PLANE_1,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        dst_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::PLANE_1,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        extent: vk::Extent3D {
            width: uv_width,
            height: uv_height,
            depth: 1,
        },
    };

    unsafe {
        device.cmd_copy_image(
            params.upload_command_buffer,
            params.src_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            params.dst_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[y_copy_region, uv_copy_region],
        );
    }

    // Transition destination image to VIDEO_ENCODE_SRC.
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(params.dst_image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::empty());

    // Also transition source image back to GENERAL for reuse.
    let src_barrier_back = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(params.src_image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);

    unsafe {
        device.cmd_pipeline_barrier(
            params.upload_command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier, src_barrier_back],
        );
    }

    unsafe { device.end_command_buffer(params.upload_command_buffer) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    let submit_info = vk::SubmitInfo::default()
        .command_buffers(std::slice::from_ref(&params.upload_command_buffer));

    let encode_queue = context.video_encode_queue().ok_or_else(|| {
        PixelForgeError::NoSuitableDevice("No video encode queue available".to_string())
    })?;

    unsafe { device.queue_submit(encode_queue, &[submit_info], params.upload_fence) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    unsafe { device.wait_for_fences(&[params.upload_fence], true, u64::MAX) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    unsafe { device.reset_fences(&[params.upload_fence]) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    Ok(())
}
