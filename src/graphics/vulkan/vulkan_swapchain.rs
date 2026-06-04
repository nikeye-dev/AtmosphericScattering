use anyhow::{anyhow, Result};
use std::cmp::min;

use crate::graphics::vulkan::vulkan_rhi_data::VulkanRHIData;
use crate::graphics::vulkan::vulkan_utils::RHIDestroy;
use vulkanalia::vk::{
    ColorSpaceKHR, CompositeAlphaFlagsKHR, DeviceMemory, DeviceV1_0, Extent2D, Format, Handle, HasBuilder, Image, ImageAspectFlags,
    ImageSubresourceRange, ImageTiling, ImageUsageFlags, ImageView, ImageViewCreateInfo, InstanceV1_0, KhrSurfaceExtensionInstanceCommands,
    KhrSwapchainExtensionDeviceCommands, MemoryPropertyFlags, MemoryRequirements, PhysicalDevice, PresentModeKHR, SharingMode,
    SurfaceCapabilitiesKHR, SurfaceFormatKHR, SurfaceKHR, SwapchainCreateInfoKHR, SwapchainKHR,
};
use vulkanalia::{vk, Device, Instance};
use winit::window::Window;

#[derive(Debug, Default)]
pub(crate) struct SwapchainData {
    pub swapchain: SwapchainKHR,
    pub swapchain_images: Vec<Image>,
    pub swapchain_format: Format,
    pub swapchain_extent: Extent2D,
    pub swapchain_image_views: Vec<ImageView>,

    pub depth_image: Image,
    pub depth_image_memory: DeviceMemory,
    pub depth_image_view: ImageView,
}

impl RHIDestroy for SwapchainData {
    fn destroy(&mut self, rhi_data: &VulkanRHIData) {
        unsafe {
            rhi_data
                .logical_device
                .destroy_image_view(self.depth_image_view, None);
            rhi_data
                .logical_device
                .free_memory(self.depth_image_memory, None);
            rhi_data
                .logical_device
                .destroy_image(self.depth_image, None);

            self.swapchain_image_views
                .iter()
                .for_each(|v| rhi_data.logical_device.destroy_image_view(*v, None));

            rhi_data
                .logical_device
                .destroy_swapchain_khr(self.swapchain, None);
        }
    }
}

#[derive(Default)]
pub struct SwapchainDataBuilder {}

impl SwapchainDataBuilder {
    pub fn build(self, window: &Window, rhi_data: &VulkanRHIData) -> Result<SwapchainData> {
        let SwapchainSupport {
            capabilities,
            formats,
            present_modes,
        } = SwapchainSupport::get(&rhi_data.instance, rhi_data.physical_device, rhi_data.surface)?;
        let extent = Self::get_swapchain_extent(window, capabilities);

        let swapchain = self.create_swapchain(rhi_data, extent, &formats, &present_modes, capabilities)?;
        let swapchain_images = unsafe { rhi_data.logical_device.get_swapchain_images_khr(swapchain) }?;

        let surface_format = Self::get_swapchain_surface_format(&formats).format;
        let swapchain_image_views = Self::create_swapchain_image_views(&swapchain_images, &rhi_data.logical_device, surface_format)?;

        let (depth_image, depth_image_memory, depth_image_view) = self.create_depth_objects(rhi_data, extent)?;

        Ok(SwapchainData {
            swapchain,
            swapchain_format: surface_format,
            swapchain_extent: extent,
            swapchain_images,
            swapchain_image_views,
            depth_image,
            depth_image_memory,
            depth_image_view,
        })
    }

    fn create_swapchain(
        &self,
        rhi_data: &VulkanRHIData,
        extent: Extent2D,
        formats: &[SurfaceFormatKHR],
        present_modes: &[PresentModeKHR],
        capabilities: SurfaceCapabilitiesKHR,
    ) -> Result<SwapchainKHR> {
        let surface_format = Self::get_swapchain_surface_format(formats);
        let present_mode = Self::get_swapchain_present_mode(present_modes);

        let image_count = min(capabilities.min_image_count + 1, capabilities.max_image_count);

        //Sharing mode between graphics and presentation queue. We rely on them being the same one, so we use Exclusive
        let image_sharing_mode = SharingMode::EXCLUSIVE;

        let info = SwapchainCreateInfoKHR::builder()
            .surface(rhi_data.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(SwapchainKHR::null());

        let swapchain = unsafe { rhi_data.logical_device.create_swapchain_khr(&info, None) }?;
        Ok(swapchain)
    }

    fn create_swapchain_image_views(
        swapchain_images: &[Image],
        logical_device: &Device,
        swapchain_format: Format,
    ) -> Result<Vec<ImageView>> {
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

        let image_views = swapchain_images
            .iter()
            .map(|i| {
                let info = vk::ImageViewCreateInfo::builder()
                    .image(*i)
                    .view_type(vk::ImageViewType::_2D)
                    .format(swapchain_format)
                    .components(components)
                    .subresource_range(subresource_range);

                unsafe { logical_device.create_image_view(&info, None) }.unwrap()
            })
            .collect::<Vec<ImageView>>();

        Ok(image_views)
    }

    fn get_swapchain_surface_format(formats: &[SurfaceFormatKHR]) -> SurfaceFormatKHR {
        formats
            .iter()
            .find(|f| f.format == Format::B8G8R8A8_SRGB && f.color_space == ColorSpaceKHR::SRGB_NONLINEAR)
            .unwrap_or_else(|| &formats[0])
            .clone()
    }

    fn get_swapchain_present_mode(present_modes: &[PresentModeKHR]) -> PresentModeKHR {
        present_modes
            .iter()
            .find(|m| **m == PresentModeKHR::MAILBOX)
            .unwrap_or(&PresentModeKHR::FIFO)
            .clone()
    }

    fn get_swapchain_extent(window: &Window, capabilities: SurfaceCapabilitiesKHR) -> Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            Extent2D::builder()
                .width(
                    window
                        .inner_size()
                        .width
                        .clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
                )
                .height(
                    window
                        .inner_size()
                        .height
                        .clamp(capabilities.min_image_extent.height, capabilities.max_image_extent.height),
                )
                .build()
        }
    }

    fn create_depth_objects(&self, rhi_data: &VulkanRHIData, swapchain_extent: Extent2D) -> Result<(Image, DeviceMemory, ImageView)> {
        let format = Format::D32_SFLOAT;
        let (depth_image, depth_image_memory) = self.create_image(
            rhi_data,
            swapchain_extent.width,
            swapchain_extent.height,
            format,
            ImageTiling::OPTIMAL,
            ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let image_view = self.create_image_view(rhi_data, depth_image, format, ImageAspectFlags::DEPTH)?;

        Ok((depth_image, depth_image_memory, image_view))
    }

    fn create_image_view(&self, rhi_data: &VulkanRHIData, image: Image, format: Format, aspects: ImageAspectFlags) -> Result<ImageView> {
        let subresource_range = ImageSubresourceRange::builder()
            .aspect_mask(aspects)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let info = ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::_2D)
            .format(format)
            .subresource_range(subresource_range);

        let image_view = unsafe { rhi_data.logical_device.create_image_view(&info, None) }?;
        Ok(image_view)
    }

    fn create_image(
        &self,
        rhi_data: &VulkanRHIData,
        width: u32,
        height: u32,
        format: Format,
        tiling: ImageTiling,
        usage: ImageUsageFlags,
        properties: MemoryPropertyFlags,
    ) -> Result<(Image, DeviceMemory)> {
        // Image

        let info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::_2D)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::_1);

        let logical_device = &rhi_data.logical_device;
        let image = unsafe { logical_device.create_image(&info, None) }.unwrap();

        // Memory

        let requirements = unsafe { logical_device.get_image_memory_requirements(image) };

        let info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(self.get_memory_type_index(rhi_data, properties, requirements)?);

        let image_memory = unsafe { logical_device.allocate_memory(&info, None) }?;

        unsafe { logical_device.bind_image_memory(image, image_memory, 0) }?;

        Ok((image, image_memory))
    }

    //ToDo: Unify with pipeline
    fn get_memory_type_index(
        &self,
        rhi_data: &VulkanRHIData,
        properties: MemoryPropertyFlags,
        requirements: MemoryRequirements,
    ) -> Result<u32> {
        let instance = &rhi_data.instance;
        let physical_device = rhi_data.physical_device;

        let memory = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        (0..memory.memory_type_count)
            .find(|i| {
                let suitable = (requirements.memory_type_bits & (1u32 << i)) != 0;
                let memory_type = memory.memory_types[*i as usize];
                suitable && memory_type.property_flags.contains(properties)
            })
            .ok_or_else(|| anyhow!("Failed to find suitable memory type"))
    }
}

pub(crate) struct SwapchainSupport {
    pub capabilities: SurfaceCapabilitiesKHR,
    pub formats: Vec<SurfaceFormatKHR>,
    pub present_modes: Vec<PresentModeKHR>,
}

impl SwapchainSupport {
    pub fn get(instance: &Instance, physical_device: PhysicalDevice, surface: SurfaceKHR) -> Result<Self> {
        let capabilities = unsafe { instance.get_physical_device_surface_capabilities_khr(physical_device, surface) }?;
        let formats = unsafe { instance.get_physical_device_surface_formats_khr(physical_device, surface) }?;
        let present_modes = unsafe { instance.get_physical_device_surface_present_modes_khr(physical_device, surface) }?;

        Ok(Self {
            capabilities,
            formats,
            present_modes,
        })
    }
}
