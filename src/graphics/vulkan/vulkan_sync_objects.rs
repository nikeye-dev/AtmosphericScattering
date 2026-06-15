use anyhow::Result;
use vulkanalia::prelude::v1_2::*;
use vulkanalia::Device;

use crate::graphics::vulkan::vulkan_rhi_data::VulkanRHIData;
use crate::graphics::vulkan::vulkan_utils::RHIDestroy;

pub(crate) struct SyncObjects {
    pub(crate) image_available_semaphores: Vec<vk::Semaphore>,
    pub(crate) render_finished_semaphores: Vec<vk::Semaphore>,

    pub(crate) in_flight_fences: Vec<vk::Fence>,
    pub(crate) images_in_flight: Vec<vk::Fence>,
}

impl SyncObjects {
    pub(crate) fn create(device: &Device, image_count: usize, max_frames: usize) -> Result<Self> {
        let create_info = vk::SemaphoreCreateInfo::builder();
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_semaphores = Vec::<vk::Semaphore>::with_capacity(max_frames);
        let mut render_semaphores = Vec::<vk::Semaphore>::with_capacity(max_frames);
        let mut fences = Vec::<vk::Fence>::with_capacity(max_frames);

        for _ in 0..max_frames {
            unsafe {
                image_semaphores.push(device.create_semaphore(&create_info, None)?);
                render_semaphores.push(device.create_semaphore(&create_info, None)?);
                fences.push(device.create_fence(&fence_info, None)?);
            };
        }

        let image_fences = vec![vk::Fence::null(); image_count];
        Ok(Self {
            image_available_semaphores: image_semaphores,
            render_finished_semaphores: render_semaphores,
            in_flight_fences: fences,
            images_in_flight: image_fences,
        })
    }

    pub(crate) fn set_image_fence(&mut self, image_index: usize, fence: vk::Fence) {
        self.images_in_flight[image_index] = fence;
    }
}

impl RHIDestroy for SyncObjects {
    fn destroy(&mut self, rhi_data: &VulkanRHIData) {
        unsafe {
            self.images_in_flight.clear();

            self.in_flight_fences
                .iter()
                .for_each(|f| rhi_data.logical_device.destroy_fence(*f, None));

            self.render_finished_semaphores
                .iter()
                .for_each(|s| rhi_data.logical_device.destroy_semaphore(*s, None));
            self.image_available_semaphores
                .iter()
                .for_each(|s| rhi_data.logical_device.destroy_semaphore(*s, None));
        }
    }
}
