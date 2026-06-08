use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use vulkanalia::bytecode::Bytecode;
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::{BlendFactor, BlendOp, ColorComponentFlags, PipelineShaderStageCreateInfo};

#[derive(Debug, Hash, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Geometry,
    TesselationControl,
    TesselationEvaluation,
}

pub enum BlendMode {
    Opaque,
    Transparent,
    Additive,
    Premultiplied,
}

impl BlendMode {
    pub fn to_vulkan_attachment_state(&self) -> vk::PipelineColorBlendAttachmentStateBuilder {
        match self {
            BlendMode::Opaque => vk::PipelineColorBlendAttachmentState::builder()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::all()),
            BlendMode::Transparent => vk::PipelineColorBlendAttachmentState::builder()
                .blend_enable(true)
                .src_color_blend_factor(BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(BlendOp::ADD)
                .src_alpha_blend_factor(BlendFactor::ONE)
                .dst_alpha_blend_factor(BlendFactor::ZERO)
                .alpha_blend_op(BlendOp::ADD)
                .color_write_mask(ColorComponentFlags::all()),
            BlendMode::Additive => vk::PipelineColorBlendAttachmentState::builder()
                .blend_enable(true)
                .src_color_blend_factor(BlendFactor::ONE)
                .dst_color_blend_factor(BlendFactor::ONE)
                .color_blend_op(BlendOp::ADD)
                .color_write_mask(ColorComponentFlags::all()),
            BlendMode::Premultiplied => vk::PipelineColorBlendAttachmentState::builder()
                .blend_enable(true)
                .src_color_blend_factor(BlendFactor::ONE)
                .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(BlendOp::ADD)
                .color_write_mask(ColorComponentFlags::all()),
        }
    }
}

impl From<ShaderStage> for vk::ShaderStageFlags {
    fn from(value: ShaderStage) -> Self {
        match value {
            ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
            ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
            ShaderStage::Geometry => vk::ShaderStageFlags::GEOMETRY,
            ShaderStage::TesselationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
            ShaderStage::TesselationEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        }
    }
}

pub struct VulkanPipeline {}

pub struct VulkanPipelineBuilder {
    graphics_shaders: HashMap<ShaderStage, PathBuf>,
    compute_shader: Option<PathBuf>,

    vertex_bindings: Vec<vk::VertexInputBindingDescription>,
    vertex_attributes: Vec<vk::VertexInputAttributeDescription>,

    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    push_constant_ranges: Vec<vk::PushConstantRange>,

    cull_mode: vk::CullModeFlags,
    front_face: vk::FrontFace,
    polygon_mode: vk::PolygonMode,

    depth_test: bool,
    depth_write: bool,
    depth_compare: vk::CompareOp,

    blend_mode: BlendMode,
    sample_count: vk::SampleCountFlags,
}

impl VulkanPipelineBuilder {
    pub fn new() -> Self {
        Self {
            graphics_shaders: HashMap::new(),
            compute_shader: None,
            vertex_bindings: vec![],
            vertex_attributes: vec![],
            descriptor_set_layouts: vec![],
            push_constant_ranges: vec![],
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            polygon_mode: vk::PolygonMode::FILL,
            depth_test: true,
            depth_write: true,
            depth_compare: vk::CompareOp::LESS_OR_EQUAL,
            blend_mode: BlendMode::Opaque,
            sample_count: vk::SampleCountFlags::_1,
        }
    }

    pub fn build_graphics(self) -> Result<VulkanPipeline> {}

    pub fn graphics_shader(&mut self, stage: ShaderStage, shader_path: PathBuf) -> Result<()> {
        if (self.compute_shader.is_some()) {
            return Err(anyhow!("Graphics shader used in compute pipeline"));
        }

        self.graphics_shaders.insert(stage, shader_path);
        Ok(())
    }

    pub fn compute_shader(&mut self, shader_path: PathBuf) -> Result<()> {
        if (self.graphics_shaders.is_empty() == false) {
            return Err(anyhow!("Compute shader used in graphics pipeline"));
        }

        self.compute_shader = Some(shader_path);
        Ok(())
    }

    pub fn vertex_bindings(&mut self, bindings: Vec<vk::VertexInputBindingDescription>) {
        self.vertex_bindings = bindings;
    }

    pub fn vertex_attributes(&mut self, attributes: Vec<vk::VertexInputAttributeDescription>) {
        self.vertex_attributes = attributes;
    }

    pub fn descriptor_set_layouts(&mut self, layouts: Vec<vk::DescriptorSetLayout>) {
        self.descriptor_set_layouts = layouts;
    }

    pub fn push_constant_ranges(&mut self, ranges: Vec<vk::PushConstantRange>) {
        self.push_constant_ranges = ranges;
    }

    pub fn cull_mode(&mut self, cull_mode: vk::CullModeFlags) {
        self.cull_mode = cull_mode;
    }

    pub fn front_face(&mut self, front_face: vk::FrontFace) {
        self.front_face = front_face;
    }

    pub fn polygon_mode(&mut self, polygon_mode: vk::PolygonMode) {
        self.polygon_mode = polygon_mode;
    }

    pub fn depth_test(&mut self, depth_test: bool) {
        self.depth_test = depth_test;
    }

    pub fn depth_write(&mut self, depth_write: bool) {
        self.depth_write = depth_write;
    }

    pub fn depth_compare(&mut self, depth_compare: vk::CompareOp) {
        self.depth_compare = depth_compare;
    }

    pub fn blend_mode(&mut self, blend_mode: BlendMode) {
        self.blend_mode = blend_mode;
    }

    pub fn sample_count(&mut self, sample_count: vk::SampleCountFlags) {
        self.sample_count = sample_count;
    }

    fn create_shader_stage(&self, stage: ShaderStage, module: vk::ShaderModule) -> vk::PipelineShaderStageCreateInfoBuilder {
        PipelineShaderStageCreateInfo::builder()
            .stage(stage.into())
            .module(module)
            .name(b"main\0")
    }

    fn load_shader(&self, device: &Device, stage: ShaderStage) -> Result<vk::ShaderModule> {
        let shader_data = fs::read(&self.graphics_shaders[&stage])?;
        let shader_code = Bytecode::new(&shader_data)?;

        let shader_info = vk::ShaderModuleCreateInfo::builder()
            .code_size(shader_code.code_size())
            .code(shader_code.code());

        let module = unsafe { device.create_shader_module(&shader_info, None)? };
        Ok(module)
    }
}
