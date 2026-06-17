use anyhow::{anyhow, Result};
use std::collections::HashMap;
use vulkanalia::prelude::v1_2::*;

pub struct VulkanDescriptors {
    pool: vk::DescriptorPool,

    frame_layouts: HashMap<u32, vk::DescriptorSetLayout>,
    pass_layouts: HashMap<u32, vk::DescriptorSetLayout>,
    material_layouts: HashMap<u32, vk::DescriptorSetLayout>,
}

impl VulkanDescriptors {
    pub fn new(device: &Device, frames_in_flight: u32, num_passes: u32, num_materials: u32) -> Result<Self> {
        let max_sets = frames_in_flight + num_passes + num_materials;
        let sizes = Self::calculate_pool_sizes(frames_in_flight, num_passes, num_materials);
        let pool = Self::create_descriptor_pool(device, max_sets, &sizes)?;

        Ok(Self {
            pool,
            frame_layouts: HashMap::new(),
            pass_layouts: HashMap::new(),
            material_layouts: HashMap::new(),
        })
    }

    pub fn destroy(self, device: &Device) {
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);

            self.frame_layouts.iter().for_each(|(_id, layout)| {
                device.destroy_descriptor_set_layout(*layout, None);
            });

            self.pass_layouts.iter().for_each(|(_id, layout)| {
                device.destroy_descriptor_set_layout(*layout, None);
            });

            self.material_layouts.iter().for_each(|(_id, layout)| {
                device.destroy_descriptor_set_layout(*layout, None);
            });
        }
    }

    pub fn add_frame_layout(
        &mut self,
        device: &Device,
        frame_id: u32,
        layout: VulkanDescriptorSetLayoutBuilder,
    ) -> Result<vk::DescriptorSetLayout> {
        Self::add_layout(&mut self.frame_layouts, device, frame_id, layout)
    }

    pub fn frame_layout(&self, frame_id: u32) -> Option<vk::DescriptorSetLayout> {
        self.pass_layouts.get(&frame_id).copied()
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

    pub fn allocate_set(&self, device: &Device, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet> {
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.pool)
            .set_layouts(std::slice::from_ref(&layout));

        let sets = unsafe { device.allocate_descriptor_sets(&info)? };
        Ok(sets[0])
    }

    pub fn allocate_sets(&self, device: &Device, layout: vk::DescriptorSetLayout, count: usize) -> Result<Vec<vk::DescriptorSet>> {
        let layouts = vec![layout; count];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);

        let sets = unsafe { device.allocate_descriptor_sets(&info)? };
        Ok(sets)
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

    fn calculate_pool_sizes(frames_in_flight: u32, num_passes: u32, num_materials: u32) -> HashMap<vk::DescriptorType, u32> {
        let mut sizes = HashMap::<vk::DescriptorType, u32>::new();

        //4 buffers per frame for now
        let mut ubo_count = frames_in_flight * 4 + num_materials;
        ubo_count = (ubo_count as f32 * 1.2).ceil() as u32;

        //2 samplers per pass and 4 samplers per material for now
        let mut sampler_count = num_passes * 2 + num_materials * 4;
        sampler_count = (sampler_count as f32 * 1.2).ceil() as u32;

        sizes.insert(vk::DescriptorType::UNIFORM_BUFFER, ubo_count);
        sizes.insert(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, sampler_count);

        sizes
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

    fn create_descriptor_sets(
        device: &Device,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        count: usize,
    ) -> Result<Vec<vk::DescriptorSet>> {
        let layouts = vec![layout; count];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .set_layouts(&layouts)
            .descriptor_pool(pool);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&info)? };
        Ok(descriptor_sets)
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

    pub fn build(self, device: &Device) -> Result<vk::DescriptorSetLayout> {
        let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&self.bindings);
        let layout = unsafe { device.create_descriptor_set_layout(&create_info, None)? };
        Ok(layout)
    }
}
