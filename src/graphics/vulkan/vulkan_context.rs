pub use vulkanalia::prelude::v1_2::*;
use winit::dpi::PhysicalSize;
use winit::window::Window;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Default, Hash)]
pub struct Size<T> {
    pub width: T,
    pub height: T,
}

impl From<PhysicalSize<u32>> for Size<u32> {
    fn from(size: PhysicalSize<u32>) -> Self {
        Self {
            width: size.width,
            height: size.height,
        }
    }
}

pub struct VulkanContext {
    window: Window,
    pub instance: Instance,
    pub device: Device,
    pub physical_device: vk::PhysicalDevice,
    pub surface: vk::SurfaceKHR,
    pub graphics_family: u32,
    pub present_family: u32,
}

impl VulkanContext {
    pub fn window_size(&self) -> Size<u32> {
        self.window.inner_size().into()
    }
}
