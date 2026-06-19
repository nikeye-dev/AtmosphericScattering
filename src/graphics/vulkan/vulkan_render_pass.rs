use crate::graphics::vulkan::vulkan_context::VulkanContext;
use crate::graphics::vulkan::vulkan_render_target::VulkanRenderTarget;
use crate::graphics::vulkan::vulkan_resources::VulkanResources;
use crate::graphics::vulkan::vulkan_utils::choose_depth_format;
use anyhow::Result;
use vulkanalia::prelude::v1_2::*;

#[derive(Default)]
pub struct VulkanRenderPass {
    pub handle: vk::RenderPass,
    render_targets: Vec<VulkanRenderTarget>,
}

impl VulkanRenderPass {
    pub fn new(context: &VulkanContext, color_format: vk::Format, depth_format: vk::Format) -> Result<Self> {
        let color_attachment = vk::AttachmentDescription::builder()
            .format(color_format)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE);

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
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);

        let dependency_end = vk::SubpassDependency::builder()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
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

        let render_pass = unsafe { context.device.create_render_pass(&render_pass_info, None) }?;

        Ok(Self {
            handle: render_pass,
            render_targets: Vec::new(),
        })
    }

    pub fn destroy(&mut self, device: &Device, resources: &VulkanResources) {
        self.destroy_render_targets(device, resources);

        unsafe { device.destroy_render_pass(self.handle, None) };
    }

    pub fn destroy_render_targets(&mut self, device: &Device, resources: &VulkanResources) {
        self.render_targets.drain(..).for_each(|mut rt| {
            rt.destroy(device, resources);
        });
    }

    pub fn recreate_render_targets(
        &mut self,
        device: &Device,
        resources: &VulkanResources,
        image_views: &[vk::ImageView],
        depth_format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<()> {
        self.destroy_render_targets(device, resources);

        let count = image_views.len();
        for i in 0..count {
            let image_view = image_views[i];
            let render_target = VulkanRenderTarget::new(device, resources, self.handle, image_view, depth_format, extent)?;

            self.render_targets.push(render_target);
        }

        Ok(())
    }

    pub fn render_target(&self, image_index: usize) -> Option<&VulkanRenderTarget> {
        self.render_targets.get(image_index)
    }
}
