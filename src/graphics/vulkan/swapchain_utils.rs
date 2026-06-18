use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::KhrSurfaceExtensionInstanceCommands;

pub struct SwapchainCapabilities {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainCapabilities {
    pub fn query(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> anyhow::Result<SwapchainCapabilities> {
        let capabilities = unsafe { instance.get_physical_device_surface_capabilities_khr(physical_device, surface) }?;
        let formats = unsafe { instance.get_physical_device_surface_formats_khr(physical_device, surface) }?;
        let present_modes =
            unsafe { instance.get_physical_device_surface_present_modes_khr(physical_device, surface) }?;

        Ok(SwapchainCapabilities {
            capabilities,
            formats,
            present_modes,
        })
    }
}
