use std::ffi::CStr;
use std::os::raw::c_void;

use crate::config::config::LogLevel;
use crate::graphics::vulkan::transformation::Matrix4x4;
use crate::graphics::vulkan::vertex::Vertex;
use crate::utils::math::{VECTOR3_BACKWARD, VECTOR3_DOWN, VECTOR3_FORWARD, VECTOR3_LEFT, VECTOR3_RIGHT, VECTOR3_UP};
use anyhow::anyhow;
use cgmath::{vec3, vec4, Angle, Deg, Rad};
use log::{debug, error, trace, warn};
use thiserror::Error;
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::KhrSurfaceExtensionInstanceCommands;
use vulkanalia::Version;

pub(crate) const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

pub(crate) const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

pub(crate) const EXT_VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

pub(crate) const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[
    vk::KHR_SWAPCHAIN_EXTENSION.name,
    vk::KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION.name,
];

#[derive(Debug, Error)]
#[error("Suitability Error: {0}.")]
pub struct CompatibilityError(pub &'static str);

//Cube without normals with shared vertices
// pub static VERTICES: [Vertex; 8] = [
//     Vertex::new(Vector3::new(-1.0, -1.0, -1.0), Vector3::zero(),  Vector4::new(0.1, 0.1, 0.1, 1.0)), //0
//     Vertex::new(Vector3::new(1.0, -1.0, -1.0), Vector3::zero(), Vector4::new(1.0, 0.0, 0.0, 1.0)), //1
//     Vertex::new(Vector3::new(1.0, 1.0, -1.0), Vector3::zero(), Vector4::new(1.0, 1.0, 0.0, 1.0)), //2
//
//     Vertex::new(Vector3::new(-1.0, 1.0, -1.0), Vector3::zero(), Vector4::new(0.0, 1.0, 0.0, 1.0)), //3
//     Vertex::new(Vector3::new(-1.0, -1.0, 1.0), Vector3::zero(), Vector4::new(0.0, 0.0, 1.0, 1.0)), //4
//     Vertex::new(Vector3::new(1.0, -1.0, 1.0), Vector3::zero(), Vector4::new(1.0, 0.0, 1.0, 1.0)), //5
//
//     Vertex::new(Vector3::new(1.0, 1.0, 1.0), Vector3::zero(), Vector4::new(1.0, 1.0, 1.0, 1.0)),  //6
//     Vertex::new(Vector3::new(-1.0, 1.0, 1.0), Vector3::zero(), Vector4::new(0.0, 1.0, 1.0, 1.0)), //7
// ];
//
//
// pub static INDICES: &[u16] = &[
//     0, 1, 3, 3, 1, 2,
//     1, 5, 2, 2, 5, 6,
//     5, 4, 6, 6, 4, 7,
//     4, 0, 7, 7, 0, 3,
//     3, 2, 7, 7, 2, 6,
//     4, 5, 0, 0, 5, 1
// ];

//Cube with per vertex normals
pub static VERTICES: [Vertex; 24] = [
    // Front face
    Vertex::new(vec3(-1.0, -1.0, 1.0), VECTOR3_BACKWARD, vec4(0.5, 0.5, 0.5, 1.0)), // Bottom-left
    Vertex::new(vec3(1.0, -1.0, 1.0), VECTOR3_BACKWARD, vec4(0.5, 0.5, 0.5, 1.0)),  // Bottom-right
    Vertex::new(vec3(1.0, 1.0, 1.0), VECTOR3_BACKWARD, vec4(0.5, 0.5, 0.5, 1.0)),   // Top-right
    Vertex::new(vec3(-1.0, 1.0, 1.0), VECTOR3_BACKWARD, vec4(0.5, 0.5, 0.5, 1.0)),  // Top-left
    // Back face
    Vertex::new(vec3(-1.0, -1.0, -1.0), VECTOR3_FORWARD, vec4(0.5, 0.5, 0.5, 1.0)), // Bottom-left
    Vertex::new(vec3(1.0, -1.0, -1.0), VECTOR3_FORWARD, vec4(0.5, 0.5, 0.5, 1.0)),  // Bottom-right
    Vertex::new(vec3(1.0, 1.0, -1.0), VECTOR3_FORWARD, vec4(0.5, 0.5, 0.5, 1.0)),   // Top-right
    Vertex::new(vec3(-1.0, 1.0, -1.0), VECTOR3_FORWARD, vec4(0.5, 0.5, 0.5, 1.0)),  // Top-left
    // Left face
    Vertex::new(vec3(-1.0, 1.0, 1.0), VECTOR3_LEFT, vec4(0.5, 0.5, 0.5, 1.0)), // Top-left
    Vertex::new(vec3(-1.0, 1.0, -1.0), VECTOR3_LEFT, vec4(0.5, 0.5, 0.5, 1.0)), // Top-right
    Vertex::new(vec3(-1.0, -1.0, -1.0), VECTOR3_LEFT, vec4(0.5, 0.5, 0.5, 1.0)), // Bottom-right
    Vertex::new(vec3(-1.0, -1.0, 1.0), VECTOR3_LEFT, vec4(0.5, 0.5, 0.5, 1.0)), // Bottom-left
    // Right face
    Vertex::new(vec3(1.0, 1.0, 1.0), VECTOR3_RIGHT, vec4(0.5, 0.5, 0.5, 1.0)), // Top-left
    Vertex::new(vec3(1.0, 1.0, -1.0), VECTOR3_RIGHT, vec4(0.5, 0.5, 0.5, 1.0)), // Top-right
    Vertex::new(vec3(1.0, -1.0, -1.0), VECTOR3_RIGHT, vec4(0.5, 0.5, 0.5, 1.0)), // Bottom-right
    Vertex::new(vec3(1.0, -1.0, 1.0), VECTOR3_RIGHT, vec4(0.5, 0.5, 0.5, 1.0)), // Bottom-left
    // Top face
    Vertex::new(vec3(-1.0, 1.0, -1.0), VECTOR3_UP, vec4(0.5, 0.5, 0.5, 1.0)), // Top-left
    Vertex::new(vec3(1.0, 1.0, -1.0), VECTOR3_UP, vec4(0.5, 0.5, 0.5, 1.0)),  // Top-right
    Vertex::new(vec3(1.0, 1.0, 1.0), VECTOR3_UP, vec4(0.5, 0.5, 0.5, 1.0)),   // Bottom-right
    Vertex::new(vec3(-1.0, 1.0, 1.0), VECTOR3_UP, vec4(0.5, 0.5, 0.5, 1.0)),  // Bottom-left
    // Bottom face
    Vertex::new(vec3(-1.0, -1.0, -1.0), VECTOR3_DOWN, vec4(0.5, 0.5, 0.5, 1.0)), // Top-left
    Vertex::new(vec3(1.0, -1.0, -1.0), VECTOR3_DOWN, vec4(0.5, 0.5, 0.5, 1.0)),  // Top-right
    Vertex::new(vec3(1.0, -1.0, 1.0), VECTOR3_DOWN, vec4(0.5, 0.5, 0.5, 1.0)),   // Bottom-right
    Vertex::new(vec3(-1.0, -1.0, 1.0), VECTOR3_DOWN, vec4(0.5, 0.5, 0.5, 1.0)),  // Bottom-left
];

pub static INDICES: &[u16] = &[
    // Front face
    0, 1, 2, 2, 3, 0, // Back face
    4, 5, 6, 6, 7, 4, // Left face
    8, 9, 10, 10, 11, 8, // Right face
    12, 13, 14, 14, 15, 12, // Top face
    16, 17, 18, 18, 19, 16, // Bottom face
    20, 21, 22, 22, 23, 20,
];

//Triangle
// pub static VERTICES: [Vertex; 3] = [
//     Vertex::new(Vector3::new(-0.5, -0.5, 0.0), Vector4::new(0.0, 0.0, 1.0, 1.0)),
//     Vertex::new(Vector3::new(0.0, 0.5, 0.0), Vector4::new(0.0, 1.0, 0.0, 1.0)),
//     Vertex::new(Vector3::new(0.5, -0.5, 0.0), Vector4::new(1.0, 0.0, 0.0, 1.0)),
// ];
//
// pub static INDICES: &[u16] = &[0, 1, 2];

pub const PERSPECTIVE_CORRECTION: Matrix4x4 = Matrix4x4::new(
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0 / 2.0,
    0.0,
    0.0,
    0.0,
    1.0 / 2.0,
    1.0,
);

pub extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message).to_string_lossy() };

    debug!("Hello from callback");

    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => trace!("({:?}) {}", message_type, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => debug!("({:?}) {}", message_type, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => warn!("({:?}) {}", message_type, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => error!("({:?}) {}", message_type, message),
        _ => unreachable!(),
    }

    vk::FALSE
}

#[derive(Debug, Copy, Clone)]
pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub transfer: u32,
}

impl QueueFamilyIndices {
    const QUEUE_FLAGS: vk::QueueFlags = vk::QueueFlags::GRAPHICS;

    pub fn get(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> anyhow::Result<QueueFamilyIndices> {
        let properties = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        //ToDo: Extract present_queue based on instance.get_physical_device_surface_support, and not assume graphics is present
        let maybe_index = properties
            .iter()
            .enumerate()
            .position(|(i, p)| {
                p.queue_flags.contains(Self::QUEUE_FLAGS)
                    && unsafe { instance.get_physical_device_surface_support_khr(physical_device, i as u32, surface) }
                        .unwrap()
            })
            .map(|i| i as u32);

        if let Some(i) = maybe_index {
            Ok(QueueFamilyIndices {
                graphics: i,
                transfer: i,
            })
        } else {
            Err(anyhow!(CompatibilityError("Missing required queue families")))
        }
    }
}

pub fn perspective_matrix(fovy: f32, view_width: f32, view_height: f32, near: f32, far: f32) -> Matrix4x4 {
    let aspect = view_width / view_height;

    let half_fov = fovy * 0.5;
    let sin_fov = Rad::sin(Deg(half_fov).into());
    let cos_fov = Rad::cos(Deg(half_fov).into());

    let h = cos_fov / sin_fov;
    let w = h / aspect;
    let range = far / (far - near);

    Matrix4x4::new(
        w,
        0.0,
        0.0,
        0.0,
        0.0,
        h,
        0.0,
        0.0,
        0.0,
        0.0,
        range,
        1.0,
        0.0,
        0.0,
        -range * near,
        0.0,
    )
}

impl From<LogLevel> for vk::DebugUtilsMessageSeverityFlagsEXT {
    fn from(log_level: LogLevel) -> Self {
        match log_level {
            LogLevel::Verbose => vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
            LogLevel::Info => vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            LogLevel::Warning => vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            LogLevel::Error => vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        }
    }
}
