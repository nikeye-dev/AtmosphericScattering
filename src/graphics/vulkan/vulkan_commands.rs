use crate::graphics::vulkan::vulkan_utils::QueueFamilyIndices;
use anyhow::Result;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk;

//Command pools and command buffers allocation
pub struct VulkanCommands {
    graphics_pool: vk::CommandPool,
    transfer_pool: vk::CommandPool,
    buffers_per_frame: Vec<vk::CommandBuffer>,
}

impl VulkanCommands {
    pub fn new(device: &Device, family_indices: QueueFamilyIndices, frames_in_flight: usize) -> Result<Self> {
        let graphics_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(family_indices.graphics)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let graphics_pool = unsafe { device.create_command_pool(&graphics_pool_create_info, None) }?;

        let transfer_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(family_indices.transfer)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);

        let transfer_pool = unsafe { device.create_command_pool(&transfer_pool_create_info, None) }?;

        let frame_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(graphics_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(frames_in_flight as u32);

        let buffers_per_frame = unsafe { device.allocate_command_buffers(&frame_buffer_allocate_info) }?;

        Ok(Self {
            graphics_pool,
            transfer_pool,
            buffers_per_frame,
        })
    }

    pub fn begin_frame(&self, device: &Device, frame_index: usize) -> Result<vk::CommandBuffer> {
        let buffer = self.buffers_per_frame[frame_index];

        unsafe {
            device.reset_command_buffer(buffer, vk::CommandBufferResetFlags::empty())?;
        }

        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(buffer, &begin_info)?;
        }

        Ok(buffer)
    }

    pub fn end_frame(&self, device: &Device, command_buffer: vk::CommandBuffer) -> Result<()> {
        unsafe {
            device.end_command_buffer(command_buffer)?;
        }

        Ok(())
    }

    pub fn begin_transfer(&self, device: &Device) -> Result<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.transfer_pool)
            .command_buffer_count(1)
            .build();

        let buffer;
        unsafe {
            buffer = device.allocate_command_buffers(&allocate_info)?[0];
        }

        let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(buffer, &begin_info)?;
        }

        Ok(buffer)
    }

    pub fn submit_transfer(&self, device: &Device, queue: vk::Queue, command_buffer: vk::CommandBuffer) -> Result<()> {
        unsafe {
            device.end_command_buffer(command_buffer)?;
        }

        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&[command_buffer])
            .build();

        //ToDo: use fence
        unsafe {
            device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
            device.queue_wait_idle(queue)?;
        }

        self.free_buffer(device, self.transfer_pool, command_buffer);
        Ok(())
    }

    fn allocate_buffer(&self, device: &Device, command_pool: vk::CommandPool) -> Result<vk::CommandBuffer> {
        Ok(self.allocate_buffers(device, command_pool, 1)?[0])
    }

    fn allocate_buffers(&self, device: &Device, command_pool: vk::CommandPool, count: u32) -> Result<Vec<vk::CommandBuffer>> {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .command_buffer_count(count);

        let buffers;
        unsafe {
            buffers = device.allocate_command_buffers(&allocate_info)?;
        }

        Ok(buffers)
    }

    fn free_buffer(&self, device: &Device, command_pool: vk::CommandPool, command_buffer: vk::CommandBuffer) {
        self.free_buffers(device, command_pool, &[command_buffer])
    }

    fn free_buffers(&self, device: &Device, command_pool: vk::CommandPool, command_buffers: &[vk::CommandBuffer]) {
        unsafe {
            device.free_command_buffers(command_pool, command_buffers);
        }
    }
}
