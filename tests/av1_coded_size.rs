//! The frame size AV1 streams come out at, on whatever device runs the test.
//!
//! A device may code AV1 frames larger than the display size, to its
//! `codedPictureAlignment`. The Vulkan spec says how (its "AV1 resolution
//! override"): AV1 codes in 8x8 blocks and crops the rest implicitly, so a size
//! whose 8-aligned value meets the alignment is coded as it is, and any other
//! is coded at that value aligned up. For each size below, this encodes a few
//! frames, reads the frame size ffprobe finds in the stream, and checks it
//! against that rule for the device's alignment. It prints a table (run with
//! `--nocapture`) so results from different devices can be compared.
//!
//! Ignored by default: requires a Vulkan Video device that encodes AV1, and
//! ffprobe. Run with `cargo test --test av1_coded_size -- --ignored --nocapture`.

use std::fs::File;
use std::io::Write;

#[allow(dead_code)]
mod common;

use ash::vk::{self, TaggedStructure};
use pixelforge::{
    Codec, EncodeConfig, Encoder, Frame, RateControl, VideoContext, VideoContextBuilder, YuvLayout,
    YuvUploader,
};

/// Display sizes that streams are commonly asked for, and ones that land on
/// either side of common alignments.
const SIZES: [(u32, u32); 9] = [
    (1280, 720),
    (1366, 768),
    (1600, 900),
    (1920, 1080),
    (1920, 1082),
    (2560, 1080),
    (2560, 1440),
    (3440, 1440),
    (3840, 2160),
];

/// The coded size by the Vulkan spec's AV1 resolution override.
fn expected(value: u32, alignment: u32) -> u32 {
    let blocks = value.div_ceil(8) * 8;
    let aligned = blocks.div_ceil(alignment.max(1)) * alignment.max(1);
    if aligned == blocks { value } else { aligned }
}

/// The device's AV1 encode `codedPictureAlignment`, whether it has
/// `FRAME_SIZE_OVERRIDE`, and its largest coded extent, for 8-bit 4:2:0 Main.
fn capabilities(context: &VideoContext) -> (vk::Extent2D, bool, vk::Extent2D) {
    let video_queue = ash::khr::video_queue::Instance::load(context.entry(), context.instance());
    let mut av1_profile = vk::VideoEncodeAV1ProfileInfoKHR::default()
        .std_profile(ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN);
    let profile = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_AV1)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .push(&mut av1_profile);
    let mut av1 = vk::VideoEncodeAV1CapabilitiesKHR::default();
    let mut encode = vk::VideoEncodeCapabilitiesKHR::default();
    let mut caps = vk::VideoCapabilitiesKHR::default()
        .push(&mut av1)
        .push(&mut encode);
    // SAFETY: the profile and capability chains are valid for the call.
    let result = unsafe {
        (video_queue.fp().get_physical_device_video_capabilities_khr)(
            context.physical_device(),
            &profile,
            &mut caps,
        )
    };
    assert_eq!(result, vk::Result::SUCCESS, "AV1 encode capabilities");
    let max = caps.max_coded_extent;
    (
        av1.coded_picture_alignment,
        av1.flags
            .contains(vk::VideoEncodeAV1CapabilityFlagsKHR::FRAME_SIZE_OVERRIDE),
        max,
    )
}

/// The frame size ffprobe reads from the stream at `path`.
fn stream_size(path: &std::path::Path) -> Result<(u32, u32), Box<dyn std::error::Error>> {
    let output = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-f", "obu", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=width,height", "-of", "csv=p=0"])
        .arg(path)
        .output()?;
    assert!(
        output.status.success(),
        "ffprobe failed on {}",
        path.display()
    );
    let text = String::from_utf8(output.stdout)?;
    let (w, h) = text.trim().split_once(',').ok_or("no size from ffprobe")?;
    Ok((w.parse()?, h.parse()?))
}

#[test]
#[ignore = "requires a Vulkan Video device that encodes AV1, and ffprobe"]
fn av1_frames_are_coded_at_the_spec_size() -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();
    let context = VideoContextBuilder::new()
        .app_name("pixelforge-av1-coded-size")
        .require_encode(Codec::AV1)
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .build()?;
    let (alignment, frame_size_override, max) = capabilities(&context);
    let device = context
        .device_properties()
        .device_name_as_c_str()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    println!(
        "{device}: codedPictureAlignment {}x{}, FRAME_SIZE_OVERRIDE {}",
        alignment.width, alignment.height, frame_size_override
    );
    println!("| display | expected | stream | |\n|---|---|---|---|");

    let dir = common::scratch_dir("av1-coded-size");
    let mut wrong = Vec::new();
    for (width, height) in SIZES {
        if width > max.width || height > max.height {
            println!(
                "| {width}x{height} | | | larger than the device's {}x{} |",
                max.width, max.height
            );
            continue;
        }
        let config = EncodeConfig::new(Codec::AV1, width, height)
            .with_rate_control(RateControl::Cqp { qp: 30 })
            .with_frame_rate(60, 1);
        let mut encoder = Encoder::new(context.clone(), config)?;
        let mut input = YuvUploader::for_encoder(&encoder)?;
        let frame = vec![128u8; (width * height * 3 / 2) as usize];
        let mut bitstream = Vec::new();
        for _ in 0..3 {
            input.upload(YuvLayout::I420, &frame)?;
            let packet = pollster::block_on(encoder.submit(Frame::image(input.source()))?)?;
            bitstream.extend_from_slice(&packet.data);
        }
        encoder.flush()?;
        let path = dir.join(format!("{width}x{height}.obu"));
        File::create(&path)?.write_all(&bitstream)?;

        let want = (
            expected(width, alignment.width),
            expected(height, alignment.height),
        );
        let got = stream_size(&path)?;
        let verdict = if got == want { "ok" } else { "**wrong**" };
        println!(
            "| {width}x{height} | {}x{} | {}x{} | {verdict} |",
            want.0, want.1, got.0, got.1
        );
        if got != want {
            wrong.push(format!(
                "{width}x{height}: expected {}x{}, stream {}x{}",
                want.0, want.1, got.0, got.1
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "coded sizes off the spec: {}",
        wrong.join("; ")
    );
    Ok(())
}
