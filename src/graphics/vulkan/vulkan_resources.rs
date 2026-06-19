use crate::graphics::vulkan::vulkan_commands::VulkanCommands;
use crate::graphics::vulkan::vulkan_context::VulkanContext;
use anyhow;
use anyhow::Result;
use std::ptr::{copy_nonoverlapping, NonNull};
use vulkanalia::prelude::v1_0::*;
use vulkanalia_vma::{
    Alloc, Allocation, AllocationCreateFlags, AllocationOptions, Allocator, AllocatorOptions, MemoryUsage,
};

//Buffer creation and copying - vertex, index, texture, uniform
pub struct VulkanResources {
    pub allocator: Allocator,
}

pub struct Buffer {
    pub handle: vk::Buffer,
    pub allocation: Allocation,
    pub size: vk::DeviceSize,
}

pub struct DynamicBuffer {
    pub buffer: Buffer,
    mem_ptr: NonNull<u8>,
}

impl From<DynamicBuffer> for Buffer {
    fn from(value: DynamicBuffer) -> Self {
        value.buffer
    }
}

impl VulkanResources {
    pub fn new(context: &VulkanContext) -> Result<Self> {
        let options = AllocatorOptions::new(&context.instance, &context.device, context.physical_device);

        let allocator;
        unsafe {
            allocator = Allocator::new(&options)?;
        };

        Ok(Self { allocator })
    }

    pub fn destroy(&mut self) {}

    fn staging_buffer(&self, size: vk::DeviceSize) -> Result<DynamicBuffer> {
        self.dynamic_buffer(size, vk::BufferUsageFlags::TRANSFER_SRC)
    }

    fn static_buffer(&self, size: vk::DeviceSize, usage: vk::BufferUsageFlags) -> Result<Buffer> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let alloc_options = AllocationOptions {
            usage: MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (handle, allocation) = unsafe { self.allocator.create_buffer(buffer_info, &alloc_options) }?;
        Ok(Buffer {
            handle,
            allocation,
            size,
        })
    }

    pub fn dynamic_buffer(&self, size: vk::DeviceSize, usage: vk::BufferUsageFlags) -> Result<DynamicBuffer> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let alloc_options = AllocationOptions {
            flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE | AllocationCreateFlags::MAPPED,
            usage: MemoryUsage::Auto,
            ..Default::default()
        };

        let (handle, allocation) = unsafe { self.allocator.create_buffer(buffer_info, &alloc_options) }?;
        let mem_ptr = self.allocator.get_allocation_info(allocation).pMappedData;

        Ok(DynamicBuffer {
            buffer: Buffer {
                handle,
                allocation,
                size,
            },
            mem_ptr: NonNull::new(mem_ptr.cast()).expect("Invalid buffer mem_ptr"),
        })
    }

    pub fn destroy_buffer(&self, buffer: Buffer) {
        unsafe {
            self.allocator
                .destroy_buffer(buffer.handle, buffer.allocation)
        }
    }

    pub fn destroy_dynamic_buffer(&self, dynamic_buffer: DynamicBuffer) {
        self.destroy_buffer(dynamic_buffer.into());
    }

    pub fn static_upload_buffer<T>(
        &self,
        device: &Device,
        commands: &VulkanCommands,
        data: &[T],
        queue: vk::Queue,
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer> {
        let size = (size_of::<T>() * data.len()) as vk::DeviceSize;

        let staging_buffer = self.staging_buffer(size)?;

        unsafe {
            copy_nonoverlapping(data.as_ptr(), staging_buffer.mem_ptr.as_ptr().cast(), data.len());
        }

        let target_buffer = self.static_buffer(size, usage)?;
        let command_buffer = commands.begin_transfer(device)?;

        unsafe {
            let copy_region = vk::BufferCopy::builder().size(size);
            device.cmd_copy_buffer(
                command_buffer,
                staging_buffer.buffer.handle,
                target_buffer.handle,
                &[copy_region],
            );
        }

        commands.submit_transfer(device, queue, command_buffer)?;
        self.destroy_dynamic_buffer(staging_buffer);

        Ok(target_buffer)
    }

    pub fn update_buffer<T>(&self, buffer: &DynamicBuffer, data: &T) {
        unsafe {
            copy_nonoverlapping(data, buffer.mem_ptr.as_ptr().cast(), 1);
        }
    }

    //Upload buffer - e.g. create the staging buffer from T[] data. Then create and copy to the target buffer - e.g. vertex
}
