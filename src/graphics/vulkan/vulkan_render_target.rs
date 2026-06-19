use crate::graphics::vulkan::vulkan_image::VulkanImage;
use crate::graphics::vulkan::vulkan_resources::VulkanResources;
use anyhow::Result;
use vulkanalia::prelude::v1_2::*;
use vulkanalia_vma::{Alloc, AllocationOptions};

pub struct VulkanRenderTarget {
    pub framebuffer: vk::Framebuffer,
    pub image_view: vk::ImageView,
    pub depth_image: VulkanImage,
    pub extent: vk::Extent2D,
}

impl VulkanRenderTarget {
    pub fn new(
        device: &Device,
        resources: &VulkanResources,
        render_pass: vk::RenderPass,
        image_view: vk::ImageView,
        depth_format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<Self> {
        let depth_image = Self::create_depth_image(device, resources, depth_format, extent)?;
        let framebuffer = Self::create_framebuffer(device, render_pass, image_view, depth_image.view, extent)?;
        Ok(Self {
            framebuffer,
            image_view,
            depth_image,
            extent,
        })
    }

    pub fn destroy(&mut self, device: &Device, resources: &VulkanResources) {
        unsafe { device.destroy_framebuffer(self.framebuffer, None) }
        self.depth_image.destroy(device, resources);
    }

    fn create_framebuffer(
        device: &Device,
        render_pass: vk::RenderPass,
        image_view: vk::ImageView,
        depth_image_view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Result<vk::Framebuffer> {
        let attachments = [image_view, depth_image_view];

        let framebuffer_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(1);

        let framebuffer = unsafe { device.create_framebuffer(&framebuffer_info, None) }?;
        Ok(framebuffer)
    }

    fn create_depth_image(
        device: &Device,
        resources: &VulkanResources,
        format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<VulkanImage> {
        let alloc_info = vk::ImageCreateInfo::builder()
            .format(format)
            .image_type(vk::ImageType::_2D)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::_1);

        let alloc_options = AllocationOptions::default();

        let (image, allocation) = unsafe {
            resources
                .allocator
                .create_image(alloc_info, &alloc_options)?
        };

        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::DEPTH)
            .level_count(1)
            .layer_count(1);

        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::_2D)
            .format(format)
            .subresource_range(subresource_range);

        let image_view = unsafe { device.create_image_view(&view_info, None) }?;

        Ok(VulkanImage {
            image,
            view: image_view,
            allocation: Some(allocation),
        })
    }
}
