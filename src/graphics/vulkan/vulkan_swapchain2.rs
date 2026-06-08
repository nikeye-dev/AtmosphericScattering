use crate::graphics::vulkan::vulkan_context::*;
use anyhow::Result;
use vulkanalia::vk::{KhrSurfaceExtensionInstanceCommands, KhrSwapchainExtensionDeviceCommands};

struct SwapchainCapabilities {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

pub struct VulkanSwapchain {
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub surface_format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
}

impl VulkanSwapchain {
    pub fn new(context: &VulkanContext, old_swapchain: Option<VulkanSwapchain>) -> Result<Self> {
        let swapchain_capabilities = Self::query_capabilities(&context.instance, context.physical_device, context.surface)?;
        let surface_format = Self::choose_surface_format(&swapchain_capabilities.formats);
        let present_mode = Self::choose_present_mode(&swapchain_capabilities.present_modes);
        let extent = Self::choose_extent(context.window_size(), swapchain_capabilities.capabilities);
        let image_count = Self::image_count(swapchain_capabilities.capabilities);

        let swapchain_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(context.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(swapchain_capabilities.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(
                old_swapchain
                    .as_ref()
                    .map_or(vk::SwapchainKHR::null(), |s| s.swapchain),
            );

        let swapchain = unsafe { context.device.create_swapchain_khr(&swapchain_info, None)? };
        match old_swapchain {
            Some(old_swapchain) => {
                old_swapchain.destroy(&context.device);
            }
            None => {}
        }

        let images = unsafe { context.device.get_swapchain_images_khr(swapchain)? };
        let image_views = Self::create_image_views(&context.device, &images, surface_format.format)?;

        Ok(Self {
            swapchain,
            images,
            image_views,
            surface_format,
            extent,
        })
    }

    pub fn destroy(&self, device: &Device) {
        self.image_views
            .iter()
            .for_each(|v| unsafe { device.destroy_image_view(*v, None) });

        unsafe { device.destroy_swapchain_khr(self.swapchain, None) };
    }

    fn query_capabilities(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<SwapchainCapabilities> {
        let capabilities = unsafe { instance.get_physical_device_surface_capabilities_khr(physical_device, surface)? };
        let formats = unsafe { instance.get_physical_device_surface_formats_khr(physical_device, surface)? };
        let present_modes = unsafe { instance.get_physical_device_surface_present_modes_khr(physical_device, surface)? };

        Ok(SwapchainCapabilities {
            capabilities,
            formats,
            present_modes,
        })
    }

    fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        formats
            .iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .copied()
            .unwrap_or(formats[0])
    }

    fn choose_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        }
    }

    fn choose_extent(size: Size<u32>, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }

        vk::Extent2D::builder()
            .width(
                size.width
                    .clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
            )
            .height(
                size.height
                    .clamp(capabilities.min_image_extent.height, capabilities.max_image_extent.height),
            )
            .build()
    }

    fn image_count(capabilities: vk::SurfaceCapabilitiesKHR) -> u32 {
        (capabilities.min_image_count + 1).min(
            capabilities
                .max_image_count
                .max(capabilities.min_image_count + 1),
        )
    }

    fn create_image_views(device: &Device, images: &[vk::Image], format: vk::Format) -> Result<Vec<vk::ImageView>> {
        let components = vk::ComponentMapping::builder()
            .r(vk::ComponentSwizzle::IDENTITY)
            .g(vk::ComponentSwizzle::IDENTITY)
            .b(vk::ComponentSwizzle::IDENTITY)
            .a(vk::ComponentSwizzle::IDENTITY);

        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let image_views = images
            .iter()
            .map(|i| {
                let info = vk::ImageViewCreateInfo::builder()
                    .image(*i)
                    .view_type(vk::ImageViewType::_2D)
                    .format(format)
                    .components(components)
                    .subresource_range(subresource_range);

                unsafe { device.create_image_view(&info, None) }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(image_views)
    }
}
