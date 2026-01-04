use super::{H265Encoder, CTB_SIZE, MIN_BITSTREAM_BUFFER_SIZE};

use crate::encoder::dpb::{DecodedPictureBuffer, DecodedPictureBufferTrait, DpbConfig};
use crate::encoder::gop::GopStructure;
use crate::encoder::resources::{
    allocate_session_memory, create_bitstream_buffer, create_command_resources, create_dpb_images,
    create_image, get_video_format, make_codec_name, map_bitstream_buffer,
};
use crate::encoder::{BitDepth, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;
use std::ptr;
use tracing::{debug, info};

impl H265Encoder {
    /// Create a new H.265/HEVC encoder.
    pub fn new(context: VideoContext, config: crate::encoder::EncodeConfig) -> Result<Self> {
        // B-frames are not yet supported.
        if config.b_frame_count > 0 {
            panic!(
                "B-frame encoding is not yet supported. Set b_frame_count=0 in encoder config. \
                 Got b_frame_count={}",
                config.b_frame_count
            );
        }

        let width = config.dimensions.width;
        let height = config.dimensions.height;

        // H.265 uses CTB (Coding Tree Block) sizes of 16, 32, or 64 pixels.
        // We use 32 as the default CTB size.
        let aligned_width = (width + CTB_SIZE - 1) & !(CTB_SIZE - 1);
        let aligned_height = (height + CTB_SIZE - 1) & !(CTB_SIZE - 1);

        info!(
            "Creating H.265 encoder: {}x{} (aligned: {}x{}, CTB size: {}), pixel_format={:?}",
            width, height, aligned_width, aligned_height, CTB_SIZE, config.pixel_format
        );

        // Load video queue extension functions.
        let video_queue_fn =
            ash::khr::video_queue::Device::load(context.instance(), context.device());
        let video_encode_fn =
            ash::khr::video_encode_queue::Device::load(context.instance(), context.device());

        // Get chroma subsampling from pixel format via `From` impl
        let chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR = config.pixel_format.into();

        // Get bit depth flags from config
        let bit_depth_flags: vk::VideoComponentBitDepthFlagsKHR = config.bit_depth.into();
        let video_format = get_video_format(config.pixel_format, config.bit_depth);

        // Select profile based on pixel format and bit depth:
        // - Main for YUV420 8-bit
        // - Main 10 for YUV420 10-bit
        // - Main 4:4:4 for YUV444 8-bit
        // - Main 4:4:4 10 for YUV444 10-bit
        let profile_idc = match (config.pixel_format, config.bit_depth) {
            (PixelFormat::Yuv420, BitDepth::Eight) => {
                ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN
            }
            (PixelFormat::Yuv420, BitDepth::Ten) => {
                ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10
            }
            (PixelFormat::Yuv444, BitDepth::Eight) => {
                ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS
            }
            (PixelFormat::Yuv444, BitDepth::Ten) => {
                ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS
            }
            _ => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "Unsupported pixel format / bit depth combination for H.265: {:?} / {:?}",
                    config.pixel_format, config.bit_depth
                )));
            }
        };

        // Create H.265 encode profile
        let mut h265_profile_info =
            vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(profile_idc);

        let mut profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(bit_depth_flags)
            .chroma_bit_depth(bit_depth_flags);
        profile_info.p_next =
            (&mut h265_profile_info as *mut vk::VideoEncodeH265ProfileInfoKHR).cast();

        // Create video session.
        let std_header_version = vk::ExtensionProperties {
            extension_name: make_codec_name(b"VK_STD_vulkan_video_codec_h265_encode"),
            spec_version: vk::make_api_version(0, 1, 0, 0),
        };

        // Calculate required DPB slots based on GOP structure
        let dpb_slot_count = if config.b_frame_count > 0 {
            let needed = 2 + config.b_frame_count as usize + 2;
            needed.min(crate::encoder::dpb::MAX_DPB_SLOTS)
        } else {
            2
        };
        debug!("Allocating {} DPB slots", dpb_slot_count);

        let encode_queue_family = context.video_encode_queue_family().ok_or_else(|| {
            PixelForgeError::NoSuitableDevice("No video encode queue family available".to_string())
        })?;

        let session_create_info = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(encode_queue_family)
            .flags(vk::VideoSessionCreateFlagsKHR::empty())
            .video_profile(&profile_info)
            .picture_format(video_format)
            .max_coded_extent(vk::Extent2D {
                width: aligned_width,
                height: aligned_height,
            })
            .reference_picture_format(video_format)
            .max_dpb_slots(dpb_slot_count as u32)
            .max_active_reference_pictures((dpb_slot_count - 1) as u32)
            .std_header_version(&std_header_version);

        let mut session = vk::VideoSessionKHR::null();
        let result = unsafe {
            (video_queue_fn.fp().create_video_session_khr)(
                context.device().handle(),
                &session_create_info,
                ptr::null(),
                &mut session,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(PixelForgeError::VideoSessionCreation(format!(
                "{:?}",
                result
            )));
        }

        // Query and allocate session memory.
        let session_memory = allocate_session_memory(&context, session, &video_queue_fn)?;

        // Create VPS, SPS and PPS
        // H.265 coding block sizes:
        // CTB size (cuSize) = 32x32 -> log2_ctb_size = 5 -> cuSize enum = 2
        // Min CB size (cuMinSize) = 16x16 -> log2_min_cb_size = 4 -> cuMinSize enum = 1
        let ctb_log2_size_y: u8 = 5; // 32x32 CTB
        let min_cb_log2_size_y: u8 = 4; // 16x16 min CB
        let log2_min_transform_block_size: u8 = 2; // 4x4 min TU
        let log2_max_transform_block_size: u8 = 5; // 32x32 max TU

        // Calculate SPS parameters.
        let pic_width_in_luma_samples = aligned_width;
        let pic_height_in_luma_samples = aligned_height;

        // Conformance window for cropping.
        // SubWidthC and SubHeightC depend on chroma format:
        // - YUV420: SubWidthC=2, SubHeightC=2
        // - YUV444: SubWidthC=1, SubHeightC=1
        let (sub_width_c, sub_height_c) = match config.pixel_format {
            PixelFormat::Yuv420 => (2u32, 2u32),
            PixelFormat::Yuv444 => (1u32, 1u32),
            _ => (2u32, 2u32), // Default to 4:2:0
        };
        let conf_win_right_offset = (aligned_width - width) / sub_width_c;
        let conf_win_bottom_offset = (aligned_height - height) / sub_height_c;
        let conformance_window_flag = conf_win_right_offset > 0 || conf_win_bottom_offset > 0;

        // Profile tier level - Main/Main10/Main 4:4:4 profile, level 5.1 (sufficient for 4K)
        let profile_tier_level = ash::vk::native::StdVideoH265ProfileTierLevel {
            flags: ash::vk::native::StdVideoH265ProfileTierLevelFlags {
                _bitfield_align_1: [],
                _bitfield_1: ash::vk::native::StdVideoH265ProfileTierLevelFlags::new_bitfield_1(
                    0, // general_tier_flag (Main tier)
                    1, // general_progressive_source_flag
                    0, // general_interlaced_source_flag
                    0, // general_non_packed_constraint_flag
                    1, // general_frame_only_constraint_flag
                ),
                __bindgen_padding_0: [0; 3],
            },
            general_profile_idc: profile_idc,
            general_level_idc: ash::vk::native::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1,
        };

        // Decoded Picture Buffer Manager
        let dec_pic_buf_mgr = ash::vk::native::StdVideoH265DecPicBufMgr {
            max_latency_increase_plus1: [0; 7],
            max_dec_pic_buffering_minus1: [(dpb_slot_count - 1) as u8, 0, 0, 0, 0, 0, 0],
            max_num_reorder_pics: [0; 7], // No B-frame reordering by default
        };

        // Short-term reference picture set (in SPS for RPS in SPS mode)
        // Set up a simple RPS with one reference picture
        let short_term_ref_pic_set = ash::vk::native::StdVideoH265ShortTermRefPicSet {
            flags: ash::vk::native::StdVideoH265ShortTermRefPicSetFlags {
                _bitfield_align_1: [],
                _bitfield_1: ash::vk::native::StdVideoH265ShortTermRefPicSetFlags::new_bitfield_1(
                    0, // inter_ref_pic_set_prediction_flag
                    0, // delta_rps_sign
                ),
                __bindgen_padding_0: [0; 3],
            },
            delta_idx_minus1: 0,
            use_delta_flag: 0,
            abs_delta_rps_minus1: 0,
            used_by_curr_pic_flag: 0,
            used_by_curr_pic_s0_flag: 1, // First negative reference is used
            used_by_curr_pic_s1_flag: 0,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
            num_negative_pics: 1, // One backward reference
            num_positive_pics: 0,
            delta_poc_s0_minus1: [0; 16],
            delta_poc_s1_minus1: [0; 16],
        };

        let long_term_ref_pics_sps = ash::vk::native::StdVideoH265LongTermRefPicsSps {
            used_by_curr_pic_lt_sps_flag: 0,
            lt_ref_pic_poc_lsb_sps: [0; 32],
        };

        // SPS flags
        let sps_flags = ash::vk::native::StdVideoH265SpsFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoH265SpsFlags::new_bitfield_1(
                1,                                           // sps_temporal_id_nesting_flag
                0,                                           // separate_colour_plane_flag
                if conformance_window_flag { 1 } else { 0 }, // conformance_window_flag
                1, // sps_sub_layer_ordering_info_present_flag
                0, // scaling_list_enabled_flag
                0, // sps_scaling_list_data_present_flag
                1, // amp_enabled_flag (asymmetric motion partitions)
                1, // sample_adaptive_offset_enabled_flag
                0, // pcm_enabled_flag
                0, // pcm_loop_filter_disabled_flag
                0, // long_term_ref_pics_present_flag
                0, // sps_temporal_mvp_enabled_flag
                0, // strong_intra_smoothing_enabled_flag
                0, // vui_parameters_present_flag
                0, // sps_extension_present_flag
                0, // sps_range_extension_flag
                0, // transform_skip_rotation_enabled_flag
                0, // transform_skip_context_enabled_flag
                0, // implicit_rdpcm_enabled_flag
                0, // explicit_rdpcm_enabled_flag
                0, // extended_precision_processing_flag
                0, // intra_smoothing_disabled_flag
                0, // high_precision_offsets_enabled_flag
                0, // persistent_rice_adaptation_enabled_flag
                0, // cabac_bypass_alignment_enabled_flag
                0, // sps_scc_extension_flag
                0, // sps_curr_pic_ref_enabled_flag
                0, // palette_mode_enabled_flag
                0, // sps_palette_predictor_initializers_present_flag
                0, // intra_boundary_filtering_disabled_flag
            ),
        };

        // Calculate bit depth minus 8 values for SPS (0 for 8-bit, 2 for 10-bit)
        let bit_depth_minus8: u8 = match config.bit_depth {
            BitDepth::Eight => 0,
            BitDepth::Ten => 2,
        };

        // Get chroma_format_idc based on pixel format.
        let chroma_format_idc = match config.pixel_format {
            PixelFormat::Yuv420 => {
                ash::vk::native::StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_420
            }
            PixelFormat::Yuv444 => {
                ash::vk::native::StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_444
            }
            _ => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "Unsupported pixel format for H.265: {:?}",
                    config.pixel_format
                )));
            }
        };

        let sps = ash::vk::native::StdVideoH265SequenceParameterSet {
            flags: sps_flags,
            chroma_format_idc,
            pic_width_in_luma_samples,
            pic_height_in_luma_samples,
            sps_video_parameter_set_id: 0,
            sps_max_sub_layers_minus1: 0,
            sps_seq_parameter_set_id: 0,
            bit_depth_luma_minus8: bit_depth_minus8,
            bit_depth_chroma_minus8: bit_depth_minus8,
            log2_max_pic_order_cnt_lsb_minus4: 4, // POC LSB range = 256
            log2_min_luma_coding_block_size_minus3: min_cb_log2_size_y - 3,
            log2_diff_max_min_luma_coding_block_size: ctb_log2_size_y - min_cb_log2_size_y,
            log2_min_luma_transform_block_size_minus2: log2_min_transform_block_size - 2,
            log2_diff_max_min_luma_transform_block_size: log2_max_transform_block_size
                - log2_min_transform_block_size,
            max_transform_hierarchy_depth_inter: (ctb_log2_size_y - log2_min_transform_block_size)
                .max(1),
            max_transform_hierarchy_depth_intra: 3,
            num_short_term_ref_pic_sets: 1,
            num_long_term_ref_pics_sps: 0,
            pcm_sample_bit_depth_luma_minus1: 7,
            pcm_sample_bit_depth_chroma_minus1: 7,
            log2_min_pcm_luma_coding_block_size_minus3: min_cb_log2_size_y - 3,
            log2_diff_max_min_pcm_luma_coding_block_size: ctb_log2_size_y - min_cb_log2_size_y,
            reserved1: 0,
            reserved2: 0,
            palette_max_size: 0,
            delta_palette_max_predictor_size: 0,
            motion_vector_resolution_control_idc: 0,
            sps_num_palette_predictor_initializers_minus1: 0,
            conf_win_left_offset: 0,
            conf_win_right_offset,
            conf_win_top_offset: 0,
            conf_win_bottom_offset,
            pProfileTierLevel: ptr::null(), // Will be set below
            pDecPicBufMgr: ptr::null(),
            pScalingLists: ptr::null(),
            pShortTermRefPicSet: ptr::null(),
            pLongTermRefPicsSps: ptr::null(),
            pSequenceParameterSetVui: ptr::null(),
            pPredictorPaletteEntries: ptr::null(),
        };

        // VPS flags
        let vps_flags = ash::vk::native::StdVideoH265VpsFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoH265VpsFlags::new_bitfield_1(
                1, // vps_temporal_id_nesting_flag
                1, // vps_sub_layer_ordering_info_present_flag
                0, // vps_timing_info_present_flag
                0, // vps_poc_proportional_to_timing_flag
            ),
            __bindgen_padding_0: [0; 3],
        };

        let vps = ash::vk::native::StdVideoH265VideoParameterSet {
            flags: vps_flags,
            vps_video_parameter_set_id: 0,
            vps_max_sub_layers_minus1: 0,
            reserved1: 0,
            reserved2: 0,
            vps_num_units_in_tick: 0,
            vps_time_scale: 0,
            vps_num_ticks_poc_diff_one_minus1: 0,
            reserved3: 0,
            pDecPicBufMgr: ptr::null(),
            pHrdParameters: ptr::null(),
            pProfileTierLevel: ptr::null(),
        };

        // PPS flags
        let pps_flags = ash::vk::native::StdVideoH265PpsFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoH265PpsFlags::new_bitfield_1(
                0, // dependent_slice_segments_enabled_flag
                0, // output_flag_present_flag
                0, // sign_data_hiding_enabled_flag
                1, // cabac_init_present_flag
                0, // constrained_intra_pred_flag
                1, // transform_skip_enabled_flag
                1, // cu_qp_delta_enabled_flag
                0, // pps_slice_chroma_qp_offsets_present_flag
                0, // weighted_pred_flag
                0, // weighted_bipred_flag
                0, // transquant_bypass_enabled_flag
                0, // tiles_enabled_flag
                0, // entropy_coding_sync_enabled_flag
                0, // uniform_spacing_flag
                0, // loop_filter_across_tiles_enabled_flag
                1, // pps_loop_filter_across_slices_enabled_flag
                1, // deblocking_filter_control_present_flag
                0, // deblocking_filter_override_enabled_flag
                0, // pps_deblocking_filter_disabled_flag
                0, // pps_scaling_list_data_present_flag
                0, // lists_modification_present_flag
                0, // slice_segment_header_extension_present_flag
                0, // pps_extension_present_flag
                0, // cross_component_prediction_enabled_flag
                0, // chroma_qp_offset_list_enabled_flag
                0, // pps_curr_pic_ref_enabled_flag
                0, // residual_adaptive_colour_transform_enabled_flag
                0, // pps_slice_act_qp_offsets_present_flag
                0, // pps_palette_predictor_initializers_present_flag
                0, // monochrome_palette_flag
                0, // pps_range_extension_flag
            ),
        };

        let pps = ash::vk::native::StdVideoH265PictureParameterSet {
            flags: pps_flags,
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            sps_video_parameter_set_id: 0,
            num_extra_slice_header_bits: 0,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            init_qp_minus26: 0,
            diff_cu_qp_delta_depth: 0,
            pps_cb_qp_offset: 0,
            pps_cr_qp_offset: 0,
            pps_beta_offset_div2: 0,
            pps_tc_offset_div2: 0,
            log2_parallel_merge_level_minus2: 0,
            log2_max_transform_skip_block_size_minus2: 0,
            diff_cu_chroma_qp_offset_depth: 0,
            chroma_qp_offset_list_len_minus1: 0,
            cb_qp_offset_list: [0; 6],
            cr_qp_offset_list: [0; 6],
            log2_sao_offset_scale_luma: 0,
            log2_sao_offset_scale_chroma: 0,
            pps_act_y_qp_offset_plus5: 0,
            pps_act_cb_qp_offset_plus5: 0,
            pps_act_cr_qp_offset_plus3: 0,
            pps_num_palette_predictor_initializers: 0,
            luma_bit_depth_entry_minus8: bit_depth_minus8,
            chroma_bit_depth_entry_minus8: bit_depth_minus8,
            num_tile_columns_minus1: 0,
            num_tile_rows_minus1: 0,
            reserved1: 0,
            reserved2: 0,
            column_width_minus1: [0; 19],
            row_height_minus1: [0; 21],
            reserved3: 0,
            pScalingLists: ptr::null(),
            pPredictorPaletteEntries: ptr::null(),
        };

        // Box the structures so they live long enough for session parameter creation
        let profile_tier_level_boxed = Box::new(profile_tier_level);
        let dec_pic_buf_mgr_boxed = Box::new(dec_pic_buf_mgr);
        let short_term_ref_pic_set_boxed = Box::new(short_term_ref_pic_set);
        let long_term_ref_pics_sps_boxed = Box::new(long_term_ref_pics_sps);

        // Create mutable copies with correct pointers
        let mut sps_with_ptrs = sps;
        sps_with_ptrs.pProfileTierLevel = profile_tier_level_boxed.as_ref();
        sps_with_ptrs.pDecPicBufMgr = dec_pic_buf_mgr_boxed.as_ref();
        sps_with_ptrs.pShortTermRefPicSet = short_term_ref_pic_set_boxed.as_ref();
        sps_with_ptrs.pLongTermRefPicsSps = long_term_ref_pics_sps_boxed.as_ref();

        let mut vps_with_ptrs = vps;
        vps_with_ptrs.pProfileTierLevel = profile_tier_level_boxed.as_ref();
        vps_with_ptrs.pDecPicBufMgr = dec_pic_buf_mgr_boxed.as_ref();

        let vps_array = [vps_with_ptrs];
        let sps_array = [sps_with_ptrs];
        let pps_array = [pps];

        let h265_add_info = vk::VideoEncodeH265SessionParametersAddInfoKHR::default()
            .std_vp_ss(&vps_array)
            .std_sp_ss(&sps_array)
            .std_pp_ss(&pps_array);

        let mut h265_params_create_info =
            vk::VideoEncodeH265SessionParametersCreateInfoKHR::default()
                .max_std_vps_count(1)
                .max_std_sps_count(1)
                .max_std_pps_count(1)
                .parameters_add_info(&h265_add_info);

        let mut params_create_info =
            vk::VideoSessionParametersCreateInfoKHR::default().video_session(session);
        params_create_info.p_next = (&mut h265_params_create_info
            as *mut vk::VideoEncodeH265SessionParametersCreateInfoKHR)
            .cast();

        let mut session_params = vk::VideoSessionParametersKHR::null();
        let result = unsafe {
            (video_queue_fn.fp().create_video_session_parameters_khr)(
                context.device().handle(),
                &params_create_info,
                ptr::null(),
                &mut session_params,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(PixelForgeError::SessionParametersCreation(format!(
                "{:?}",
                result
            )));
        }

        // Create profile info for images/buffers
        let mut h265_profile_for_resources =
            vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(profile_idc);
        let mut profile_for_resources = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(bit_depth_flags)
            .chroma_bit_depth(bit_depth_flags);
        profile_for_resources.p_next =
            (&mut h265_profile_for_resources as *mut vk::VideoEncodeH265ProfileInfoKHR).cast();

        // Create input image
        let (input_image, input_image_memory, input_image_view) = create_image(
            &context,
            aligned_width,
            aligned_height,
            video_format,
            false,
            &profile_for_resources,
        )?;

        // Create DPB images.
        let (dpb_images, dpb_image_memories, dpb_image_views) = create_dpb_images(
            &context,
            aligned_width,
            aligned_height,
            video_format,
            dpb_slot_count,
            &profile_for_resources,
        )?;

        // Create bitstream buffer.
        let (bitstream_buffer, bitstream_buffer_memory) =
            create_bitstream_buffer(&context, MIN_BITSTREAM_BUFFER_SIZE, &profile_for_resources)?;

        // Persistently map the bitstream buffer to avoid per-frame map/unmap overhead.
        let bitstream_buffer_ptr =
            map_bitstream_buffer(&context, bitstream_buffer_memory, MIN_BITSTREAM_BUFFER_SIZE)?;

        // Create command pool, buffers, and fences.
        let cmd_resources = create_command_resources(&context, encode_queue_family)?;
        let command_pool = cmd_resources.command_pool;
        let upload_command_buffer = cmd_resources.upload_command_buffer;
        let encode_command_buffer = cmd_resources.encode_command_buffer;
        let upload_fence = cmd_resources.upload_fence;
        let encode_fence = cmd_resources.encode_fence;

        // Create query pool
        let mut h265_profile_info_query =
            vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(profile_idc);

        let mut profile_info_query = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(bit_depth_flags)
            .chroma_bit_depth(bit_depth_flags);
        profile_info_query.p_next =
            (&mut h265_profile_info_query as *mut vk::VideoEncodeH265ProfileInfoKHR).cast();

        let mut encode_feedback_create = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::default()
            .encode_feedback_flags(
                vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET
                    | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN,
            );

        encode_feedback_create.p_next =
            (&mut profile_info_query as *mut vk::VideoProfileInfoKHR).cast();

        let mut query_pool_create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
            .query_count(1);
        query_pool_create_info.p_next = (&mut encode_feedback_create
            as *mut vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR)
            .cast();

        let query_pool = unsafe {
            context
                .device()
                .create_query_pool(&query_pool_create_info, None)
        }
        .map_err(|e| PixelForgeError::QueryPool(e.to_string()))?;

        // Create DPB and GOP structure
        let mut dpb = DecodedPictureBuffer::new();
        let dpb_config = DpbConfig {
            dpb_size: dpb_slot_count as u32,
            max_num_ref_frames: if config.b_frame_count > 0 { 2 } else { 1 },
            use_multiple_references: config.b_frame_count > 0,
            max_long_term_refs: 0,
            log2_max_frame_num_minus4: 0,         // Not used in H.265
            log2_max_pic_order_cnt_lsb_minus4: 4, // max_poc_lsb = 256
            num_temporal_layers: 1,
        };
        dpb.h265.sequence_start(dpb_config);

        let mut gop = if config.b_frame_count > 0 {
            GopStructure::new(config.gop_size, config.b_frame_count, config.gop_size)
        } else {
            GopStructure::new_ip_only(config.gop_size)
        };

        // Set GOP parameters to match SPS values
        // log2_max_pic_order_cnt_lsb_minus4 = 4, so max_poc_lsb = 2^8 = 256
        gop.set_max_frame_num(4); // Not used in H.265 but set for compatibility
        gop.set_max_poc_lsb(4);

        // Initialize DPB slot activation tracking
        let dpb_slot_active = vec![false; dpb_slot_count];

        info!("H.265 encoder created successfully");

        Ok(Self {
            context,
            config: config.clone(),
            dpb,
            gop,
            aligned_width,
            aligned_height,
            video_queue_fn,
            video_encode_fn,
            session,
            session_params,
            session_memory,
            input_frame_num: 0,
            encode_frame_num: 0,
            input_image,
            input_image_memory,
            input_image_view,
            input_image_layout: vk::ImageLayout::UNDEFINED,
            dpb_images,
            dpb_image_memories,
            dpb_image_views,
            dpb_slot_count,
            bitstream_buffer,
            bitstream_buffer_memory,
            bitstream_buffer_ptr,
            command_pool,
            upload_command_buffer,
            upload_fence,
            encode_command_buffer,
            encode_fence,
            query_pool,
            header_data: None,
            has_reference: false,
            reference_poc: 0,
            has_backward_reference: false,
            backward_reference_poc: 0,
            backward_reference_dpb_slot: 2,
            current_dpb_slot: 0,
            reference_dpb_slot: 1,
            reference_pic_type: 0,
            dpb_slot_active,
        })
    }
}
