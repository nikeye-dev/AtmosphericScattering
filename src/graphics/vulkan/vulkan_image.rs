use crate::graphics::vulkan::vulkan_resources::VulkanResources;
use vulkanalia::prelude::v1_2::*;
use vulkanalia_vma::Allocation;

pub struct VulkanImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: Option<Allocation>,
}

impl VulkanImage {
    pub fn destroy(&mut self, device: &Device, resources: &VulkanResources) {
        unsafe { device.destroy_image_view(self.view, None) };

        if let Some(allocation) = self.allocation.take() {
            unsafe { resources.allocator.destroy_image(self.image, allocation) };
        }
    }
}
