use anyhow::{anyhow, Result};
use std::collections::HashMap;
use vulkanalia::prelude::v1_2::*;

struct VulkanDescriptors {
    pool: vk::DescriptorPool,

    frame_layout: vk::DescriptorSetLayout,
    pass_layouts: HashMap<u32, vk::DescriptorSetLayout>,
    material_layouts: HashMap<u32, vk::DescriptorSetLayout>,
}

impl VulkanDescriptors {
    pub fn frame_layout(&self) -> vk::DescriptorSetLayout {
        self.frame_layout
    }

    pub fn add_pass_layout(
        &mut self,
        device: &Device,
        pass_id: u32,
        layout: VulkanDescriptorSetLayoutBuilder,
    ) -> Result<vk::DescriptorSetLayout> {
        Self::add_layout(&mut self.pass_layouts, device, pass_id, layout)
    }

    pub fn pass_layout(&self, pass_id: u32) -> Option<vk::DescriptorSetLayout> {
        self.pass_layouts.get(&pass_id).copied()
    }

    pub fn add_material_layout(
        &mut self,
        device: &Device,
        material_id: u32,
        layout: VulkanDescriptorSetLayoutBuilder,
    ) -> Result<vk::DescriptorSetLayout> {
        Self::add_layout(&mut self.material_layouts, device, material_id, layout)
    }

    pub fn material_layout(&self, material_id: u32) -> Option<vk::DescriptorSetLayout> {
        self.material_layouts.get(&material_id).copied()
    }

    fn add_layout(
        collection: &mut HashMap<u32, vk::DescriptorSetLayout>,
        device: &Device,
        id: u32,
        layout: VulkanDescriptorSetLayoutBuilder,
    ) -> Result<vk::DescriptorSetLayout> {
        if (collection.contains_key(&id)) {
            return Err(anyhow!("Layout with id {id} already defined"));
        }

        let new_layout = layout.build(device)?;
        collection.insert(id, new_layout);
        Ok(new_layout)
    }

    fn create_descriptor_pool(device: &Device, max_sets: u32, sizes: &HashMap<vk::DescriptorType, u32>) -> Result<vk::DescriptorPool> {
        let pool_sizes = sizes
            .iter()
            .map(|(&type_, &count)| {
                vk::DescriptorPoolSize::builder()
                    .type_(type_)
                    .descriptor_count(count)
            })
            .collect::<Vec<_>>();

        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(max_sets);

        let pool = unsafe { device.create_descriptor_pool(&info, None)? };
        Ok(pool)
    }
}

pub struct VulkanDescriptorSetLayoutBuilder {
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl VulkanDescriptorSetLayoutBuilder {
    pub fn new() -> Self {
        Self { bindings: Vec::new() }
    }

    pub fn binding(
        mut self,
        binding_index: u32,
        descriptor_type: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
        count: u32,
    ) -> Self {
        let binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(binding_index)
            .descriptor_type(descriptor_type)
            .stage_flags(stage_flags)
            .descriptor_count(count)
            .build();

        self.bindings.push(binding);
        self
    }

    fn build(self, device: &Device) -> Result<vk::DescriptorSetLayout> {
        let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&self.bindings);
        let layout = unsafe { device.create_descriptor_set_layout(&create_info, None)? };
        Ok(layout)
    }
}
