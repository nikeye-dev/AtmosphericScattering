use crate::graphics::vulkan::vulkan_render_pass::VulkanRenderPass;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use vulkanalia::bytecode::Bytecode;
use vulkanalia::prelude::v1_2::*;

#[derive(Debug, Hash, PartialEq, Eq, Copy, Clone)]
pub enum GraphicsShaderStage {
    Vertex,
    Fragment,
    Geometry,
    TesselationControl,
    TesselationEvaluation,
}

pub enum BlendMode {
    Opaque,
    Transparent,//Alpha Blend
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
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::all()),
            BlendMode::Additive => vk::PipelineColorBlendAttachmentState::builder()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE)
                .color_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::all()),
            BlendMode::Premultiplied => vk::PipelineColorBlendAttachmentState::builder()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::all()),
        }
    }
}

impl From<&GraphicsShaderStage> for vk::ShaderStageFlags {
    fn from(value: &GraphicsShaderStage) -> Self {
        match value {
            GraphicsShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
            GraphicsShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
            GraphicsShaderStage::Geometry => vk::ShaderStageFlags::GEOMETRY,
            GraphicsShaderStage::TesselationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
            GraphicsShaderStage::TesselationEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        }
    }
}

pub struct VulkanPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl VulkanPipeline {
    pub fn destroy(self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

pub struct VulkanGraphicsPipelineBuilder {
    shaders: HashMap<GraphicsShaderStage, PathBuf>,

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

impl VulkanGraphicsPipelineBuilder {
    pub fn new() -> Self {
        Self {
            shaders: HashMap::new(),
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

    pub fn build(self, device: &Device, render_pass: &VulkanRenderPass) -> Result<VulkanPipeline> {
        if !self.shaders.contains_key(&GraphicsShaderStage::Vertex) {
            return Err(anyhow!("Graphics pipeline requires vertex shader"));
        }

        let shader_modules = self
            .shaders
            .iter()
            .map(|(stage, path)| {
                let shader_module = self.load_shader(device, path)?;
                Ok((*stage, shader_module))
            })
            .collect::<Result<Vec<_>>>()?;

        let shader_stages = shader_modules
            .iter()
            .map(|(stage, module)| -> Result<vk::PipelineShaderStageCreateInfoBuilder> {
                let result = vk::PipelineShaderStageCreateInfo::builder()
                    .stage(stage.into())
                    .module(*module)
                    .name(b"main\0");

                Ok(result)
            })
            .collect::<Result<Vec<_>>>()?;

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&self.vertex_bindings)
            .vertex_attribute_descriptions(&self.vertex_attributes);

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(self.polygon_mode)
            .line_width(1.0)
            .cull_mode(self.cull_mode)
            .front_face(self.front_face)
            .depth_bias_enable(false);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(self.sample_count);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(self.depth_test)
            .depth_write_enable(self.depth_write)
            .depth_compare_op(self.depth_compare)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let color_blend_attachment = self.blend_mode.to_vulkan_attachment_state();
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&self.descriptor_set_layouts)
            .push_constant_ranges(&self.push_constant_ranges);

        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };
        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .dynamic_state(&dynamic_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .layout(layout)
            .render_pass(render_pass.render_pass)
            .subpass(0);

        let (pipelines, _) = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)? };

        for (_, module) in shader_modules {
            unsafe { device.destroy_shader_module(module, None) };
        }

        Ok(VulkanPipeline {
            pipeline: pipelines[0],
            layout,
        })
    }

    pub fn shader(mut self, stage: GraphicsShaderStage, shader_path: PathBuf) -> Self {
        self.shaders.insert(stage, shader_path);
        self
    }

    pub fn vertex_bindings(mut self, bindings: Vec<vk::VertexInputBindingDescription>) -> Self {
        self.vertex_bindings = bindings;
        self
    }

    pub fn vertex_attributes(mut self, attributes: Vec<vk::VertexInputAttributeDescription>) -> Self {
        self.vertex_attributes = attributes;
        self
    }

    pub fn descriptor_set_layouts(mut self, layouts: Vec<vk::DescriptorSetLayout>) -> Self {
        self.descriptor_set_layouts = layouts;
        self
    }

    pub fn push_constant_ranges(mut self, ranges: Vec<vk::PushConstantRange>) -> Self {
        self.push_constant_ranges = ranges;
        self
    }

    pub fn cull_mode(mut self, cull_mode: vk::CullModeFlags) -> Self {
        self.cull_mode = cull_mode;
        self
    }

    pub fn front_face(mut self, front_face: vk::FrontFace) -> Self {
        self.front_face = front_face;
        self
    }

    pub fn polygon_mode(mut self, polygon_mode: vk::PolygonMode) -> Self {
        self.polygon_mode = polygon_mode;
        self
    }

    pub fn depth_test(mut self, depth_test: bool) -> Self {
        self.depth_test = depth_test;
        self
    }

    pub fn depth_write(mut self, depth_write: bool) -> Self {
        self.depth_write = depth_write;
        self
    }

    pub fn depth_compare(mut self, depth_compare: vk::CompareOp) -> Self {
        self.depth_compare = depth_compare;
        self
    }

    pub fn blend_mode(mut self, blend_mode: BlendMode) -> Self {
        self.blend_mode = blend_mode;
        self
    }

    pub fn sample_count(mut self, sample_count: vk::SampleCountFlags) -> Self {
        self.sample_count = sample_count;
        self
    }

    fn load_shader(&self, device: &Device, path: &Path) -> Result<vk::ShaderModule> {
        let shader_data = fs::read(path)?;
        let shader_code = Bytecode::new(&shader_data)?;

        let shader_info = vk::ShaderModuleCreateInfo::builder()
            .code_size(shader_code.code_size())
            .code(shader_code.code());

        let module = unsafe { device.create_shader_module(&shader_info, None)? };
        Ok(module)
    }
}
