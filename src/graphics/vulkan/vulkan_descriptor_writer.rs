use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::CopyDescriptorSet;

pub struct VulkanDescriptorWriter {
    pending_writes: Vec<vk::WriteDescriptorSet>,
    buffers: Vec<Vec<vk::DescriptorBufferInfo>>,
}

impl VulkanDescriptorWriter {
    pub fn new() -> Self {
        Self {
            pending_writes: Vec::new(),
            buffers: Vec::new(),
        }
    }

    pub fn write_buffer(
        mut self,
        set: vk::DescriptorSet,
        binding: u32,
        descriptor_type: vk::DescriptorType,
        buffer: vk::DescriptorBufferInfoBuilder,
    ) -> Self {
        self.buffers.push(vec![buffer.build()]);
        let buffer_infos = self.buffers.last().unwrap();

        let write_info = vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(binding)
            .dst_array_element(0)
            .buffer_info(buffer_infos)
            .descriptor_type(descriptor_type);

        self.pending_writes.push(write_info.build());
        self
    }

    pub fn commit(self, device: &Device) {
        unsafe {
            device.update_descriptor_sets(&self.pending_writes, &[] as &[CopyDescriptorSet]);
        }
    }
}
