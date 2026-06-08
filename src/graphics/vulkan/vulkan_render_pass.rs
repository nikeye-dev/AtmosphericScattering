use crate::graphics::vulkan::vulkan_context::VulkanContext;
use crate::graphics::vulkan::vulkan_resources::VulkanResources;
use crate::graphics::vulkan::vulkan_swapchain2::VulkanSwapchain;
use anyhow::{anyhow, Result};
use vulkanalia::prelude::v1_2::*;
use vulkanalia_vma::{Alloc, Allocation, AllocationOptions};

pub struct VulkanRenderPass {
    pub render_pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub depth_format: vk::Format,
    depth_allocation: Allocation,
}

impl VulkanRenderPass {
    pub fn new(context: &VulkanContext, resources: &VulkanResources, swapchain: &VulkanSwapchain) -> Result<Self> {
        let color_attachment = vk::AttachmentDescription::builder()
            .format(swapchain.surface_format.format)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE);

        let depth_format = Self::choose_depth_format(&context.instance, context.physical_device)?;
        let (depth_image, depth_image_view, depth_allocation) =
            Self::create_depth_image(context, resources, depth_format, swapchain.extent)?;

        let depth_attachment = vk::AttachmentDescription::builder()
            .format(depth_format)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref))
            .depth_stencil_attachment(&depth_ref);

        let dependency_start = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);

        let dependency_end = vk::SubpassDependency::builder()
            .src_subpass(0)
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::empty());

        let attachments = [color_attachment, depth_attachment];
        let subpasses = [subpass];
        let dependencies = [dependency_start, dependency_end];
        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let render_pass = unsafe { context.device.create_render_pass(&render_pass_info, None)? };
        let framebuffers = Self::create_framebuffers(&context.device, swapchain, render_pass, depth_image_view)?;

        Ok(Self {
            render_pass,
            framebuffers,
            depth_image,
            depth_image_view,
            depth_allocation,
            depth_format,
        })
    }

    pub fn destroy(self, device: &Device, resources: &VulkanResources) {
        self.framebuffers.iter().for_each(|&framebuffer| {
            unsafe { device.destroy_framebuffer(framebuffer, None) };
        });

        unsafe {
            device.destroy_image_view(self.depth_image_view, None);
            resources
                .allocator
                .destroy_image(self.depth_image, self.depth_allocation);

            device.destroy_render_pass(self.render_pass, None);
        }
    }

    fn choose_depth_format(instance: &Instance, physical_device: vk::PhysicalDevice) -> Result<vk::Format> {
        let formats = [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];

        formats
            .iter()
            .copied()
            .find(|&format| {
                let properties = unsafe { instance.get_physical_device_format_properties(physical_device, format) };

                properties
                    .optimal_tiling_features
                    .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            })
            .ok_or_else(|| anyhow!("No supported depth format found"))
    }

    fn create_depth_image(
        context: &VulkanContext,
        resources: &VulkanResources,
        format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<(vk::Image, vk::ImageView, Allocation)> {
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

        let image_view = unsafe { context.device.create_image_view(&view_info, None)? };

        Ok((image, image_view, allocation))
    }

    fn create_framebuffers(
        device: &Device,
        swapchain: &VulkanSwapchain,
        render_pass: vk::RenderPass,
        depth_image_view: vk::ImageView,
    ) -> Result<Vec<vk::Framebuffer>> {
        let framebuffers = swapchain
            .image_views
            .iter()
            .map(|&view| {
                let attachments = [view, depth_image_view];

                let framebuffer_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain.extent.width)
                    .height(swapchain.extent.height)
                    .layers(1);

                unsafe { device.create_framebuffer(&framebuffer_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(framebuffers)
    }
}
