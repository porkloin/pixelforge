use super::{H264Encoder, MIN_BITSTREAM_BUFFER_SIZE};

use crate::encoder::gop::{GopFrameType, GopPosition};
use crate::error::{PixelForgeError, Result};
use ash::vk;
use tracing::debug;

impl H264Encoder {
    pub(super) fn encode_frame_internal(
        &mut self,
        gop_position: &GopPosition,
        frame_num: u32,
        pic_order_cnt: i32,
        is_idr: bool,
    ) -> Result<Vec<u8>> {
        let is_b_frame = gop_position.frame_type == GopFrameType::B;
        let is_reference = gop_position.is_reference;

        debug!(
            "encode_frame_internal: frame_num={}, poc={}, is_idr={}, has_reference={}, \
             current_dpb_slot={}, reference_dpb_slot={}, reference_frame_num={}, reference_poc={}",
            frame_num,
            pic_order_cnt,
            is_idr,
            self.has_reference,
            self.current_dpb_slot,
            self.reference_dpb_slot,
            self.reference_frame_num,
            self.reference_poc
        );

        // Rate control setup.
        let (rc_mode, average_bitrate, max_bitrate, qp) = match self.config.rate_control_mode {
            crate::encoder::RateControlMode::Cqp | crate::encoder::RateControlMode::Disabled => (
                vk::VideoEncodeRateControlModeFlagsKHR::VBR,
                100_000_000, // 100 Mbps
                100_000_000,
                self.config.quality_level as i32,
            ),
            crate::encoder::RateControlMode::Cbr => (
                vk::VideoEncodeRateControlModeFlagsKHR::CBR,
                self.config.target_bitrate,
                self.config.target_bitrate,
                26, // Default QP for rate control to adjust from
            ),
            crate::encoder::RateControlMode::Vbr => (
                vk::VideoEncodeRateControlModeFlagsKHR::VBR,
                self.config.target_bitrate,
                self.config.max_bitrate,
                26, // Default QP for rate control to adjust from
            ),
        };

        // Reset command buffer before recording.
        unsafe {
            self.context.device().reset_command_buffer(
                self.encode_command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

        // Begin command buffer.
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.context
                .device()
                .begin_command_buffer(self.encode_command_buffer, &begin_info)
        }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

        // Reset query pool.
        unsafe {
            self.context.device().cmd_reset_query_pool(
                self.encode_command_buffer,
                self.query_pool,
                0,
                1,
            );
        }

        // Transition DPB image to video encode DPB layout if needed.
        let dpb_barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::VIDEO_ENCODE_DPB_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.dpb_images[self.current_dpb_slot as usize])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::empty());

        unsafe {
            self.context.device().cmd_pipeline_barrier(
                self.encode_command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE, // Use BOTTOM_OF_PIPE as VIDEO_ENCODE requires sync2
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[dpb_barrier],
            );
        }

        // Set up H.264 specific encode info.
        let slice_type = if is_idr {
            ash::vk::native::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_I
        } else if is_b_frame {
            ash::vk::native::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_B
        } else {
            ash::vk::native::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_P
        };

        let picture_type = if is_idr {
            ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR
        } else if is_b_frame {
            ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_B
        } else {
            ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P
        };

        // Build StdVideoEncodeH264SliceHeader.
        let slice_header_flags = ash::vk::native::StdVideoEncodeH264SliceHeaderFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeH264SliceHeaderFlags::new_bitfield_1(
                0, // direct_spatial_mv_pred_flag
                0, // num_ref_idx_active_override_flag
                0, // reserved
            ),
        };

        let slice_qp_delta = match self.config.rate_control_mode {
            crate::encoder::RateControlMode::Cqp | crate::encoder::RateControlMode::Disabled => {
                ((self.config.quality_level as i32) - 26) as i8
            }
            _ => 0,
        };

        let slice_header = ash::vk::native::StdVideoEncodeH264SliceHeader {
            flags: slice_header_flags,
            first_mb_in_slice: 0,
            slice_type,
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
            slice_qp_delta,
            reserved1: 0,
            cabac_init_idc:
                ash::vk::native::StdVideoH264CabacInitIdc_STD_VIDEO_H264_CABAC_INIT_IDC_0,
            disable_deblocking_filter_idc: ash::vk::native::StdVideoH264DisableDeblockingFilterIdc_STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_DISABLED,
            pWeightTable: std::ptr::null(),
        };

        // Build StdVideoEncodeH264PictureInfo.
        let picture_info_flags = ash::vk::native::StdVideoEncodeH264PictureInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeH264PictureInfoFlags::new_bitfield_1(
                if is_idr { 1 } else { 0 },       // IdrPicFlag
                if is_reference { 1 } else { 0 }, // is_reference
                0,                                // no_output_of_prior_pics_flag
                0,                                // long_term_reference_flag
                0,                                // adaptive_ref_pic_marking_mode_flag
                0,                                // reserved
            ),
        };

        // For P-frames, we need a reference list.
        // STD_VIDEO_H264_NO_REFERENCE_PICTURE = 0xFF.
        const NO_REFERENCE_PICTURE: u8 = 0xFF;
        let mut ref_list0: [u8; 32] = [NO_REFERENCE_PICTURE; 32];
        let mut ref_list1: [u8; 32] = [NO_REFERENCE_PICTURE; 32];

        // Reference list modification operations (not used)
        let _ref_pic_list_mod_flags = ash::vk::native::StdVideoEncodeH264RefPicMarkingEntry {
            memory_management_control_operation:
                ash::vk::native::StdVideoH264MemMgmtControlOp_STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_END,
            difference_of_pic_nums_minus1: 0,
            long_term_pic_num: 0,
            long_term_frame_idx: 0,
            max_long_term_frame_idx_plus1: 0,
        };

        let ref_lists_info_flags = ash::vk::native::StdVideoEncodeH264ReferenceListsInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeH264ReferenceListsInfoFlags::new_bitfield_1(
                0, // ref_pic_list_modification_flag_l0
                0, // ref_pic_list_modification_flag_l1
                0, // reserved
            ),
        };

        // Set up reference lists for P-frames and B-frames.
        // P-frames: L0 only (reference previous frame)
        // B-frames: L0 (forward/past) and L1 (backward/future)
        let (num_ref_l0, num_ref_l1) =
            if is_b_frame && self.has_reference && self.has_backward_reference {
                // B-frame: both L0 and L1 references.
                ref_list0[0] = self.reference_dpb_slot; // L0: forward reference (past)
                ref_list1[0] = self.backward_reference_dpb_slot; // L1: backward reference (future)
                (1, 1)
            } else if !is_idr && self.has_reference {
                // P-frame: only L0 reference.
                ref_list0[0] = self.reference_dpb_slot;
                (1, 0)
            } else {
                // IDR: no references.
                (0, 0)
            };

        let ref_lists_info = ash::vk::native::StdVideoEncodeH264ReferenceListsInfo {
            flags: ref_lists_info_flags,
            num_ref_idx_l0_active_minus1: if num_ref_l0 > 0 {
                (num_ref_l0 - 1) as u8
            } else {
                0
            },
            num_ref_idx_l1_active_minus1: if num_ref_l1 > 0 {
                (num_ref_l1 - 1) as u8
            } else {
                0
            },
            RefPicList0: ref_list0,
            RefPicList1: ref_list1,
            refList0ModOpCount: 0,
            refList1ModOpCount: 0,
            refPicMarkingOpCount: 0,
            reserved1: [0; 7],
            pRefList0ModOperations: std::ptr::null(),
            pRefList1ModOperations: std::ptr::null(),
            pRefPicMarkingOperations: std::ptr::null(),
        };

        let picture_info = ash::vk::native::StdVideoEncodeH264PictureInfo {
            flags: picture_info_flags,
            seq_parameter_set_id: 0,
            pic_parameter_set_id: 0,
            idr_pic_id: self.idr_pic_id as u16,
            primary_pic_type: picture_type,
            frame_num,
            PicOrderCnt: pic_order_cnt,
            temporal_id: 0,
            reserved1: [0; 3],
            pRefLists: if !is_idr && self.has_reference {
                &ref_lists_info
            } else {
                std::ptr::null()
            },
        };

        // Create slice NAL unit entry.
        // constant_qp should only be set when rate control is DISABLED
        let constant_qp = if rc_mode == vk::VideoEncodeRateControlModeFlagsKHR::DISABLED {
            self.config.quality_level as i32
        } else {
            0
        };
        let nalu_slice_entries = [vk::VideoEncodeH264NaluSliceInfoKHR::default()
            .constant_qp(constant_qp)
            .std_slice_header(&slice_header)];

        // Create H.264 picture info.
        let mut h264_picture_info = vk::VideoEncodeH264PictureInfoKHR::default()
            .nalu_slice_entries(&nalu_slice_entries)
            .std_picture_info(&picture_info);

        // Set up source picture resource.
        let src_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: self.config.dimensions.width,
                height: self.config.dimensions.height,
            })
            .base_array_layer(0)
            .image_view_binding(self.input_image_view);

        // Set up DPB slot for reconstructed picture (setup slot)
        let setup_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: self.config.dimensions.width,
                height: self.config.dimensions.height,
            })
            .base_array_layer(0)
            .image_view_binding(self.dpb_image_views[self.current_dpb_slot as usize]);

        // Set up reference picture resource (for P-frames, references previous frame)
        let reference_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: self.config.dimensions.width,
                height: self.config.dimensions.height,
            })
            .base_array_layer(0)
            .image_view_binding(self.dpb_image_views[self.reference_dpb_slot as usize]);

        // Create H.264 reference info for the setup slot (this frame being encoded)
        let std_reference_info_flags = ash::vk::native::StdVideoEncodeH264ReferenceInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeH264ReferenceInfoFlags::new_bitfield_1(
                0, // used_for_long_term_reference
                0, // reserved
            ),
        };

        let std_reference_info = ash::vk::native::StdVideoEncodeH264ReferenceInfo {
            flags: std_reference_info_flags,
            primary_pic_type: if is_idr {
                ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR
            } else {
                ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P
            },
            FrameNum: frame_num,
            PicOrderCnt: pic_order_cnt,
            long_term_pic_num: 0,
            long_term_frame_idx: 0,
            temporal_id: 0,
        };

        // Create H.264 DPB slot info for setup.
        let mut h264_dpb_slot_info =
            vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(&std_reference_info);

        let mut setup_reference_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(self.current_dpb_slot as i32)
            .picture_resource(&setup_picture_resource);
        setup_reference_slot.p_next =
            (&mut h264_dpb_slot_info as *mut vk::VideoEncodeH264DpbSlotInfoKHR).cast();

        // For P-frames and B-frames, we need reference info for L0 (forward reference)
        let ref_std_reference_info = ash::vk::native::StdVideoEncodeH264ReferenceInfo {
            flags: std_reference_info_flags,
            primary_pic_type:
                ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P,
            FrameNum: self.reference_frame_num,
            PicOrderCnt: self.reference_poc,
            long_term_pic_num: 0,
            long_term_frame_idx: 0,
            temporal_id: 0,
        };

        let mut ref_h264_dpb_slot_info = vk::VideoEncodeH264DpbSlotInfoKHR::default()
            .std_reference_info(&ref_std_reference_info);

        // For B-frames, we also need L1 (backward reference - the future frame already encoded)
        // Only create these if B-frames are enabled (b_frame_count > 0)
        let has_b_frames = self.config.b_frame_count > 0;

        // Create a dummy image view for when B-frames are disabled.
        // This avoids index out of bounds when backward_reference_dpb_slot >= dpb_image_views.len()
        let backward_ref_image_view = if has_b_frames {
            self.dpb_image_views[self.backward_reference_dpb_slot as usize]
        } else {
            // Use first DPB slot as placeholder (won't actually be used)
            self.dpb_image_views[0]
        };

        let backward_reference_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: self.config.dimensions.width,
                height: self.config.dimensions.height,
            })
            .base_array_layer(0)
            .image_view_binding(backward_ref_image_view);

        let backward_ref_std_reference_info = ash::vk::native::StdVideoEncodeH264ReferenceInfo {
            flags: std_reference_info_flags,
            primary_pic_type:
                ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P,
            FrameNum: self.backward_reference_frame_num,
            PicOrderCnt: self.backward_reference_poc,
            long_term_pic_num: 0,
            long_term_frame_idx: 0,
            temporal_id: 0,
        };

        let mut backward_ref_h264_dpb_slot_info = vk::VideoEncodeH264DpbSlotInfoKHR::default()
            .std_reference_info(&backward_ref_std_reference_info);

        // Build reference slots array similar to NVIDIA:
        // - referenceSlotsInfo[0] = setup slot (will have slotIndex=-1 for begin)
        // - referenceSlotsInfo[1] = L0 reference slot (for P-frames and B-frames)
        // - referenceSlotsInfo[2] = L1 reference slot (for B-frames only)
        //
        // For encodeInfo:
        //   - pReferenceSlots points to slot[1...] (skip setup)
        //   - referenceSlotCount = number of actual references
        //
        // For beginInfo:
        //   - pReferenceSlots points to slot[0...] (includes all)
        //   - referenceSlotCount = setup + references
        //   - slot[0].slotIndex = -1 (mark setup as inactive)

        // Create a separate DPB slot info for the begin slot.
        let mut h264_begin_dpb_slot_info =
            vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(&std_reference_info);

        // Setup slot for begin (will be modified to slotIndex=-1)
        let mut setup_slot_for_begin = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(self.current_dpb_slot as i32) // Will be set to -1 for begin
            .picture_resource(&setup_picture_resource);
        setup_slot_for_begin.p_next =
            (&mut h264_begin_dpb_slot_info as *mut vk::VideoEncodeH264DpbSlotInfoKHR).cast();

        // L0 Reference slot for P-frames and B-frames (forward reference)
        let mut ref_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(self.reference_dpb_slot as i32)
            .picture_resource(&reference_picture_resource);
        ref_slot.p_next =
            (&mut ref_h264_dpb_slot_info as *mut vk::VideoEncodeH264DpbSlotInfoKHR).cast();

        // L1 Reference slot for B-frames (backward reference - future frame)
        let mut backward_ref_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(self.backward_reference_dpb_slot as i32)
            .picture_resource(&backward_reference_picture_resource);
        backward_ref_slot.p_next =
            (&mut backward_ref_h264_dpb_slot_info as *mut vk::VideoEncodeH264DpbSlotInfoKHR).cast();

        // Create encode info based on frame type.
        // For B-frames we need both L0 and L1 references.
        let reference_slots_for_encode_p = [ref_slot];
        let reference_slots_for_encode_b = [ref_slot, backward_ref_slot];

        let mut encode_info = if is_b_frame && self.has_reference && self.has_backward_reference {
            // B-frame: include both L0 (forward) and L1 (backward) reference slots.
            vk::VideoEncodeInfoKHR::default()
                .dst_buffer(self.bitstream_buffer)
                .dst_buffer_offset(0)
                .dst_buffer_range(MIN_BITSTREAM_BUFFER_SIZE as vk::DeviceSize)
                .src_picture_resource(src_picture_resource)
                .setup_reference_slot(&setup_reference_slot)
                .reference_slots(&reference_slots_for_encode_b)
        } else if !is_idr && self.has_reference {
            // P-frame: include L0 reference slot only.
            vk::VideoEncodeInfoKHR::default()
                .dst_buffer(self.bitstream_buffer)
                .dst_buffer_offset(0)
                .dst_buffer_range(MIN_BITSTREAM_BUFFER_SIZE as vk::DeviceSize)
                .src_picture_resource(src_picture_resource)
                .setup_reference_slot(&setup_reference_slot)
                .reference_slots(&reference_slots_for_encode_p)
        } else {
            // IDR: no reference slots needed.
            vk::VideoEncodeInfoKHR::default()
                .dst_buffer(self.bitstream_buffer)
                .dst_buffer_offset(0)
                .dst_buffer_range(MIN_BITSTREAM_BUFFER_SIZE as vk::DeviceSize)
                .src_picture_resource(src_picture_resource)
                .setup_reference_slot(&setup_reference_slot)
        };
        encode_info.p_next =
            (&mut h264_picture_info as *mut vk::VideoEncodeH264PictureInfoKHR).cast();

        // For begin video coding, we need to include all picture resources that will be used.
        // The setup slot must be included with slotIndex=-1 to indicate it's not yet active.
        // Following NVIDIA's approach:
        // - referenceSlotsInfo[0] = setup slot (slotIndex=-1)
        // - referenceSlotsInfo[1...] = reference slots (actual indices)
        //
        // - For begin: pReferenceSlots points to slot[0...]
        setup_slot_for_begin = setup_slot_for_begin.slot_index(-1);

        let reference_slots_for_begin =
            if is_b_frame && self.has_reference && self.has_backward_reference {
                // B-frame: include setup slot (inactive), L0 reference, and L1 reference.
                vec![setup_slot_for_begin, ref_slot, backward_ref_slot]
            } else if !is_idr && self.has_reference {
                // P-frame: include setup slot (inactive) and L0 reference slot.
                vec![setup_slot_for_begin, ref_slot]
            } else {
                // IDR: just the setup slot (inactive)
                vec![setup_slot_for_begin]
            };

        let min_qp_val = if rc_mode == vk::VideoEncodeRateControlModeFlagsKHR::DISABLED
            || self.config.rate_control_mode == crate::encoder::RateControlMode::Cqp
            || self.config.rate_control_mode == crate::encoder::RateControlMode::Disabled
        {
            qp // Clamp to fixed QP for CQP/Disabled simulation
        } else {
            18 // Allow high quality for H.264
        };
        let max_qp_val = if rc_mode == vk::VideoEncodeRateControlModeFlagsKHR::DISABLED
            || self.config.rate_control_mode == crate::encoder::RateControlMode::Cqp
            || self.config.rate_control_mode == crate::encoder::RateControlMode::Disabled
        {
            qp // Clamp to fixed QP for CQP/Disabled simulation
        } else {
            42 // Allow lower quality when needed
        };

        let min_qp = vk::VideoEncodeH264QpKHR {
            qp_i: min_qp_val,
            qp_p: min_qp_val,
            qp_b: min_qp_val,
        };

        let max_qp = vk::VideoEncodeH264QpKHR {
            qp_i: max_qp_val,
            qp_p: max_qp_val,
            qp_b: max_qp_val,
        };

        let mut h264_rc_layer_info = vk::VideoEncodeH264RateControlLayerInfoKHR::default()
            .use_min_qp(true)
            .min_qp(min_qp)
            .use_max_qp(true)
            .max_qp(max_qp);

        let mut rc_layer_info = vk::VideoEncodeRateControlLayerInfoKHR::default()
            .average_bitrate(average_bitrate as u64)
            .max_bitrate(max_bitrate as u64)
            .frame_rate_numerator(self.config.frame_rate_numerator)
            .frame_rate_denominator(self.config.frame_rate_denominator);
        rc_layer_info.p_next =
            (&mut h264_rc_layer_info as *mut vk::VideoEncodeH264RateControlLayerInfoKHR).cast();

        let rc_layers = [rc_layer_info];

        let mut h264_rc_info = vk::VideoEncodeH264RateControlInfoKHR::default()
            .gop_frame_count(self.config.gop_size)
            .idr_period(self.config.gop_size)
            .consecutive_b_frame_count(self.config.b_frame_count);

        let mut rc_info = vk::VideoEncodeRateControlInfoKHR::default().rate_control_mode(rc_mode);

        if rc_mode != vk::VideoEncodeRateControlModeFlagsKHR::DISABLED {
            rc_info = rc_info
                .layers(&rc_layers)
                .virtual_buffer_size_in_ms(1000)
                .initial_virtual_buffer_size_in_ms(1000);
            rc_info.p_next = &mut h264_rc_info as *mut _ as *mut std::ffi::c_void;
        }

        // Begin video coding.
        // For the first frame, don't include rate control in begin_coding - set it via control command after RESET.
        let is_first_frame = self.encode_frame_num == 0;

        let begin_info = if is_first_frame {
            vk::VideoBeginCodingInfoKHR::default()
                .video_session(self.session)
                .video_session_parameters(self.session_params)
                .reference_slots(&reference_slots_for_begin)
        } else {
            let mut info = vk::VideoBeginCodingInfoKHR::default()
                .video_session(self.session)
                .video_session_parameters(self.session_params)
                .reference_slots(&reference_slots_for_begin);
            info.p_next = (&mut rc_info as *mut vk::VideoEncodeRateControlInfoKHR).cast();
            info
        };

        unsafe {
            (self.video_queue_fn.fp().cmd_begin_video_coding_khr)(
                self.encode_command_buffer,
                &begin_info,
            );
        }

        // Reset video coding state for the first frame, then set rate control.
        if is_first_frame {
            let reset_control_info = vk::VideoCodingControlInfoKHR::default()
                .flags(vk::VideoCodingControlFlagsKHR::RESET);
            unsafe {
                (self.video_queue_fn.fp().cmd_control_video_coding_khr)(
                    self.encode_command_buffer,
                    &reset_control_info,
                );
            }

            // After RESET, set the rate control mode.
            let mut rate_control = vk::VideoCodingControlInfoKHR::default()
                .flags(vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL);
            rate_control.p_next = (&mut rc_info as *mut vk::VideoEncodeRateControlInfoKHR).cast();
            unsafe {
                (self.video_queue_fn.fp().cmd_control_video_coding_khr)(
                    self.encode_command_buffer,
                    &rate_control,
                );
            }
        }

        // Begin query.
        unsafe {
            self.context.device().cmd_begin_query(
                self.encode_command_buffer,
                self.query_pool,
                0,
                vk::QueryControlFlags::empty(),
            );
        }

        // Encode
        unsafe {
            (self.video_encode_fn.fp().cmd_encode_video_khr)(
                self.encode_command_buffer,
                &encode_info,
            );
        }

        // End query.
        unsafe {
            self.context
                .device()
                .cmd_end_query(self.encode_command_buffer, self.query_pool, 0);
        }

        // End video coding.
        let end_info = vk::VideoEndCodingInfoKHR::default();
        unsafe {
            (self.video_queue_fn.fp().cmd_end_video_coding_khr)(
                self.encode_command_buffer,
                &end_info,
            );
        }

        // End command buffer.
        unsafe {
            self.context
                .device()
                .end_command_buffer(self.encode_command_buffer)
        }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

        // Submit
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&self.encode_command_buffer));

        let encode_queue = self.context.video_encode_queue().ok_or_else(|| {
            PixelForgeError::NoSuitableDevice("No video encode queue available".to_string())
        })?;

        unsafe {
            self.context
                .device()
                .queue_submit(encode_queue, &[submit_info], self.encode_fence)
        }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

        // Wait for encode to complete.
        unsafe {
            self.context
                .device()
                .wait_for_fences(&[self.encode_fence], true, u64::MAX)
        }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

        unsafe { self.context.device().reset_fences(&[self.encode_fence]) }
            .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

        // Read back query results to get actual encoded size.
        #[repr(C)]
        struct QueryResult {
            offset: u32,
            bytes_written: u32,
        }

        let mut query_results = [QueryResult {
            offset: 0,
            bytes_written: 0,
        }];

        unsafe {
            self.context.device().get_query_pool_results(
                self.query_pool,
                0, // first_query
                &mut query_results,
                vk::QueryResultFlags::WAIT,
            )
        }
        .map_err(|e| PixelForgeError::QueryPool(e.to_string()))?;

        let query_result = &query_results[0];

        debug!(
            "Encode complete: offset={}, bytes_written={}",
            query_result.offset, query_result.bytes_written
        );

        // Read back the bitstream data using the persistently mapped buffer pointer.
        // This avoids per-frame map/unmap overhead (the buffer is mapped once at init)
        // Note: The Vulkan encoder output already includes NAL start codes (Annex B format)
        let mut encoded_data = Vec::with_capacity(query_result.bytes_written as usize);

        unsafe {
            let src = std::slice::from_raw_parts(
                self.bitstream_buffer_ptr.add(query_result.offset as usize),
                query_result.bytes_written as usize,
            );
            encoded_data.extend_from_slice(src);
        }

        Ok(encoded_data)
    }
}
