use std::cmp::max;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::intrinsics::copy_nonoverlapping;
use std::mem::size_of;

use anyhow::anyhow;
use anyhow::Result;
use vulkanalia::bytecode::Bytecode;
use vulkanalia::vk::{AccessFlags, AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, BlendFactor, BlendOp, Buffer, BufferCopy, BufferCreateInfo, BufferUsageFlags, ColorComponentFlags, CommandBuffer, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsageFlags, CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo, CompareOp, CopyDescriptorSet, CullModeFlags, DescriptorBufferInfo, DescriptorPool, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType, DeviceMemory, DeviceSize, DeviceV1_0, Fence, Format, Framebuffer, FramebufferCreateInfo, FrontFace, GraphicsPipelineCreateInfo, Handle, HasBuilder, ImageLayout, InstanceV1_0, LogicOp, MemoryAllocateInfo, MemoryMapFlags, MemoryPropertyFlags, MemoryRequirements, Offset2D, Pipeline, PipelineBindPoint, PipelineCache, PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo, PipelineDepthStencilStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PipelineLayout, PipelineLayoutCreateInfo, PipelineMultisampleStateCreateInfo, PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo, PipelineStageFlags, PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode, PrimitiveTopology, PushConstantRange, Rect2D, RenderPass, RenderPassCreateInfo, SampleCountFlags, ShaderModule, ShaderModuleCreateInfo, ShaderStageFlags, SharingMode, SubmitInfo, SubpassDependency, SubpassDescription, Viewport, WriteDescriptorSet, SUBPASS_EXTERNAL, WHOLE_SIZE};
use vulkanalia::Device;

use crate::graphics::vulkan::atmopsheric_scattering::{AtmosphereSampleData, ScatteringMedium};
use crate::graphics::vulkan::push_constants::PushConstants;
use crate::graphics::vulkan::transformation::Transformation;
use crate::graphics::vulkan::vertex::Vertex;
use crate::graphics::vulkan::view_state::ViewState;
use crate::graphics::vulkan::vulkan_rhi_data::VulkanRHIData;
use crate::graphics::vulkan::vulkan_swapchain::SwapchainData;
use crate::graphics::vulkan::vulkan_utils::{QueueFamilyIndices, RHIDestroy, INDICES, VERTICES};

#[derive(Debug, Default)]
pub struct PipelineData {
    pub(crate) pipeline_layout: PipelineLayout,
    pub(crate) render_pass: RenderPass,
    pub(crate) pipeline: Pipeline,
    pub(crate) framebuffers: Vec<Framebuffer>,

    pub(crate) global_command_pool: CommandPool,

    pub(crate) command_pools: Vec<CommandPool>,
    pub(crate) primary_command_buffers: Vec<CommandBuffer>,
    pub(crate) secondary_command_buffers: Vec<Vec<CommandBuffer>>,

    pub(crate) vertex_buffer: Buffer,
    pub(crate) vertex_buffer_memory: DeviceMemory,

    pub(crate) index_buffer: Buffer,
    pub(crate) index_buffer_memory: DeviceMemory,

    pub(crate) descriptor_set_layout: DescriptorSetLayout,

    /*  Fixed size:
         0 - transformation
         1 - viewState
         2 - atmospheric scattering medium
         3 - atmospheric scattering sample data
    */
    pub(crate) uniform_buffers: Vec<[Buffer; 4]>,
    pub(crate) uniform_buffers_memory: Vec<[DeviceMemory; 4]>,

    pub(crate) descriptor_pool: DescriptorPool,
    pub(crate) descriptor_sets: Vec<DescriptorSet>,
}

impl PipelineData {
    //ToDo: Move to vulkan_commands.rs
    pub fn get_or_allocate_secondary_buffer(&mut self, image_index: usize, buffer_index: usize, logical_device: &Device) -> CommandBuffer {
        self.secondary_command_buffers.resize_with(image_index + 1, Vec::new);
        let command_buffers = &mut self.secondary_command_buffers[image_index];

        let new_buffers_count = max(1, buffer_index) - command_buffers.len();
        if new_buffers_count > 0 {
            let allocate_info = CommandBufferAllocateInfo::builder()
                .command_pool(self.command_pools[image_index])
                .level(CommandBufferLevel::SECONDARY)
                .command_buffer_count(new_buffers_count as u32);

            let mut new_buffers = unsafe { logical_device.allocate_command_buffers(&allocate_info) }.unwrap();
            command_buffers.append(&mut new_buffers);
        }

        let command_buffer = command_buffers[buffer_index];
        command_buffer
    }
}

impl RHIDestroy for PipelineData {
   fn destroy(&mut self, rhi_data: &VulkanRHIData) {
       let logical_device = &rhi_data.logical_device;

       unsafe {
           self.primary_command_buffers.iter().enumerate().for_each(|(i, buffer)| {
               logical_device.free_command_buffers(self.command_pools[i], &[*buffer]);
           });

           self.command_pools.iter().for_each(|p| logical_device.destroy_command_pool(*p, None));
           logical_device.destroy_command_pool(self.global_command_pool, None);

           self.framebuffers
               .iter()
               .for_each(|fb| logical_device.destroy_framebuffer(*fb, None));

           logical_device.destroy_render_pass(self.render_pass, None);
           logical_device.destroy_pipeline_layout(self.pipeline_layout, None);
           logical_device.destroy_pipeline(self.pipeline, None);

           logical_device.destroy_buffer(self.index_buffer, None);
           logical_device.free_memory(self.index_buffer_memory, None);

           logical_device.destroy_buffer(self.vertex_buffer, None);
           logical_device.free_memory(self.vertex_buffer_memory, None);

           logical_device.destroy_descriptor_pool(self.descriptor_pool, None);

           logical_device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

           for i in 0..self.uniform_buffers.len() {
               for j in 0..self.uniform_buffers[i].len() {
                   logical_device.destroy_buffer(self.uniform_buffers[i][j], None);
                   logical_device.free_memory(self.uniform_buffers_memory[i][j], None);
               }
           }

           self.uniform_buffers.clear();
           self.uniform_buffers_memory.clear();
       }
   }
}

//ToDo: Support multiple pipeline creation
pub struct PipelineDataBuilder<'a> {
    value: PipelineData,
    rhi_data: &'a VulkanRHIData,
    swapchain_data: &'a SwapchainData,
    shaders: HashMap<ShaderStageFlags, &'a str>,
}

impl<'a> PipelineDataBuilder<'a> {
    pub fn new(rhi_data: &'a VulkanRHIData, swapchain_data: &'a SwapchainData) -> Self {
        Self {
            rhi_data,
            swapchain_data,
            value: PipelineData::default(),
            shaders: HashMap::new()
        }
    }

    pub fn shader(mut self, stage: ShaderStageFlags, shader_path: &'a str) -> Self {
        self.shaders.insert(stage, shader_path);
        self
    }

    pub fn build(mut self) -> Result<PipelineData> {
        self.create_descriptor_set_layout()?;
        self.create_pipeline()?;
        self.create_framebuffers()?;
        self.create_command_pools()?;

        self.create_vertex_buffer()?;
        self.create_index_buffer()?;
        self.create_uniform_buffers()?;
        self.create_descriptor_pool()?;
        self.create_descriptor_sets()?;
        self.create_command_buffers()?;

        Ok(self.value)
    }

    fn create_pipeline(&mut self) -> Result<()> {
        assert!(self.shaders.contains_key(&ShaderStageFlags::VERTEX));
        assert!(self.shaders.contains_key(&ShaderStageFlags::FRAGMENT));

        let logical_device = &self.rhi_data.logical_device;

        let vert = fs::read(self.shaders[&ShaderStageFlags::VERTEX])?;

        let vert_module = self.create_shader_module(&vert[..])?;
        let vert_stage = PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(b"main\0");

        //Create vertex input
        let binding_descriptions = &[Vertex::binding_description()];
        let attribute_descriptions = Vertex::attribute_descriptions();
        let vertex_input_state = PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions)
            ;

        let frag = fs::read(self.shaders[&ShaderStageFlags::FRAGMENT])?;

        let frag_module = self.create_shader_module(&frag[..])?;
        let frag_stage = PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(b"main\0");

        let input_assembly_state = PipelineInputAssemblyStateCreateInfo::builder()
            .topology(PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(self.swapchain_data.swapchain_extent.width as f32)
            .height(self.swapchain_data.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)
            ;

        let scissor = Rect2D::builder()
            .offset(Offset2D {x: 0, y: 0})
            .extent(self.swapchain_data.swapchain_extent)
            ;

        let viewports = &[viewport];
        let scissors = &[scissor];
        let viewport_state = PipelineViewportStateCreateInfo::builder()
            .viewports(viewports)
            .scissors(scissors)
            ;

        let rasterization_state = PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(CullModeFlags::empty())
            .front_face(FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            ;

        let multisample_state = PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(SampleCountFlags::_1)
            ;

        let depth_stencil_state = PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let color_blend_attachment = PipelineColorBlendAttachmentState::builder()
            .color_write_mask(ColorComponentFlags::all())
            .blend_enable(false)
            .src_color_blend_factor(BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(BlendFactor::ONE)
            .color_blend_op(BlendOp::ADD)
            .src_alpha_blend_factor(BlendFactor::ONE)
            .dst_alpha_blend_factor(BlendFactor::ZERO)
            .alpha_blend_op(BlendOp::ADD)
            ;

        let attachments = &[color_blend_attachment];
        let color_blend_state = PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(LogicOp::COPY)
            .attachments(attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0])
            ;

        let vert_push_constant_range = PushConstantRange::builder()
            .stage_flags(ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<PushConstants>() as u32)
        ;

        let layouts = &[self.value.descriptor_set_layout];
        let push_constant_ranges = &[vert_push_constant_range];
        let layout_info = PipelineLayoutCreateInfo::builder()
            .set_layouts(layouts)
            .push_constant_ranges(push_constant_ranges)
        ;

        self.value.pipeline_layout = unsafe { logical_device.create_pipeline_layout(&layout_info, None) }?;

        self.create_render_pass()?;

        let stages = &[vert_stage, frag_stage];
        let pipeline_info = GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .depth_stencil_state(&depth_stencil_state)
            .layout(self.value.pipeline_layout)
            .render_pass(self.value.render_pass)
            .subpass(0)
            .base_pipeline_handle(Pipeline::null())
            .base_pipeline_index(-1)
            ;

        self.value.pipeline = unsafe { logical_device.create_graphics_pipelines(PipelineCache::null(), &[pipeline_info], None) }?.0[0];

        unsafe {
            logical_device.destroy_shader_module(vert_module, None);
            logical_device.destroy_shader_module(frag_module, None);
        }

        Ok(())
    }

    fn create_shader_module(&self, bytecode: &[u8]) -> Result<ShaderModule> {
        let bytecode = Bytecode::new(bytecode)?;

        let shader_info = ShaderModuleCreateInfo::builder()
            .code_size(bytecode.code_size())
            .code(bytecode.code())
            ;

        Ok(unsafe { self.rhi_data.logical_device.create_shader_module(&shader_info, None) }?)
    }

    fn create_render_pass(&mut self) -> Result<()> {
        let swapchain_data = self.swapchain_data;
        let color_attachment = AttachmentDescription::builder()
            .format(swapchain_data.swapchain_format)
            .samples(SampleCountFlags::_1)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::PRESENT_SRC_KHR)
            ;

        let color_attachment_ref = AttachmentReference::builder()
            .attachment(0)
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            ;

        let depth_stencil_attachment = AttachmentDescription::builder()
            .format(Format::D32_SFLOAT)
            .samples(SampleCountFlags::_1)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let depth_stencil_attachment_ref = AttachmentReference::builder()
            .attachment(1)
            .layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_attachments = &[color_attachment_ref];
        let subpass = SubpassDescription::builder()
            .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
            .color_attachments(color_attachments)
            .depth_stencil_attachment(&depth_stencil_attachment_ref)
            ;

        let dependency = SubpassDependency::builder()
            .src_subpass(SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .src_access_mask(AccessFlags::empty())
            .dst_stage_mask(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .dst_access_mask(AccessFlags::COLOR_ATTACHMENT_WRITE | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            ;

        let attachments = &[color_attachment, depth_stencil_attachment];
        let subpasses = &[subpass];
        let dependencies = &[dependency];
        let render_pass_info = RenderPassCreateInfo::builder()
            .attachments(attachments)
            .subpasses(subpasses)
            .dependencies(dependencies)
            ;

        self.value.render_pass = unsafe { self.rhi_data.logical_device.create_render_pass(&render_pass_info, None) }?;

        Ok(())
    }

    fn create_framebuffers(&mut self) -> Result<()> {
        let swapchain_data = self.swapchain_data;

        let framebuffers = swapchain_data.swapchain_image_views
            .iter()
            .map(|iv| {
                let attachments = &[*iv, swapchain_data.depth_image_view];
                let create_info = FramebufferCreateInfo::builder()
                    .attachments(attachments)
                    .render_pass(self.value.render_pass)
                    .width(swapchain_data.swapchain_extent.width)
                    .height(swapchain_data.swapchain_extent.height)
                    .layers(1)
                    ;

                unsafe { self.rhi_data.logical_device.create_framebuffer(&create_info, None) }.unwrap()
            })
            .collect::<Vec<_>>();

        self.value.framebuffers = framebuffers;

        Ok(())
    }

    fn create_command_pool(&self) -> Result<CommandPool> {
        let indices = QueueFamilyIndices::get(&self.rhi_data.instance, self.rhi_data.physical_device, self.rhi_data.surface)?;

        let create_info = CommandPoolCreateInfo::builder()
            .queue_family_index(indices.graphics)
            .flags(CommandPoolCreateFlags::TRANSIENT)
            ;

        Ok(unsafe { self.rhi_data.logical_device.create_command_pool(&create_info, None) }?)
    }

    fn create_command_pools(&mut self) -> Result<()> {
        self.value.global_command_pool = self.create_command_pool()?;

        for _ in 0..self.swapchain_data.swapchain_images.len() {
            let command_pool = self.create_command_pool()?;
            self.value.command_pools.push(command_pool);
        }

        Ok(())
    }

    fn create_command_buffers(&mut self) -> Result<()> {
        let logical_device = &self.rhi_data.logical_device;

        for image_index in 0..self.swapchain_data.swapchain_images.len() {
            let allocate_info = CommandBufferAllocateInfo::builder()
                .level(CommandBufferLevel::PRIMARY)
                .command_pool(self.value.command_pools[image_index])
                .command_buffer_count(1)
                ;

            let command_buffers = unsafe { logical_device.allocate_command_buffers(&allocate_info) }?;
            self.value.primary_command_buffers.push(command_buffers[0]);
        }

        self.value.secondary_command_buffers = vec![vec![]; self.swapchain_data.swapchain_images.len()];

        Ok(())
    }

    //Vertex buffer
    fn create_vertex_buffer(&mut self) -> Result<()> {
        let logical_device = &self.rhi_data.logical_device;

        let size = (size_of::<Vertex>() * VERTICES.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = self.create_buffer(size, BufferUsageFlags::TRANSFER_SRC, MemoryPropertyFlags::HOST_COHERENT | MemoryPropertyFlags::HOST_VISIBLE)?;

        let app_memory = unsafe { logical_device.map_memory(staging_buffer_memory, 0, size, MemoryMapFlags::empty()) }?;

        unsafe {
            copy_nonoverlapping(VERTICES.as_ptr(), app_memory.cast(), VERTICES.len());
            logical_device.unmap_memory(staging_buffer_memory);
        }

        let (buffer, buffer_memory) = self.create_buffer(size, BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER, MemoryPropertyFlags::DEVICE_LOCAL)?;

        self.value.vertex_buffer = buffer;
        self.value.vertex_buffer_memory = buffer_memory;

        self.copy_buffer(staging_buffer, self.value.vertex_buffer, size)?;

        unsafe {
            logical_device.destroy_buffer(staging_buffer, None);
            logical_device.free_memory(staging_buffer_memory, None);
        }

        Ok(())
    }

    fn get_memory_type_index(&self, properties: MemoryPropertyFlags, requirements: MemoryRequirements) -> Result<u32> {
        let instance = &self.rhi_data.instance;
        let physical_device = self.rhi_data.physical_device;

        let memory = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        (0..memory.memory_type_count)
            .find(|i| {
                let suitable = (requirements.memory_type_bits & (1u32 << i)) != 0;
                let memory_type = memory.memory_types[*i as usize];
                suitable && memory_type.property_flags.contains(properties)
            })
            .ok_or_else(|| anyhow!("Failed to find suitable memory type"))
    }

    fn create_buffer(&self, size: DeviceSize, usage: BufferUsageFlags, properties: MemoryPropertyFlags) -> Result<(Buffer, DeviceMemory)> {
        let logical_device = &self.rhi_data.logical_device;

        let buffer_info = BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(SharingMode::EXCLUSIVE)
        ;

        let buffer = unsafe { logical_device.create_buffer(&buffer_info, None) }?;
        let requirements = unsafe { logical_device.get_buffer_memory_requirements(buffer) };

        let memory_info = MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(self.get_memory_type_index(properties, requirements)?);

        let buffer_memory = unsafe { logical_device.allocate_memory(&memory_info, None) }?;

        unsafe { logical_device.bind_buffer_memory(buffer, buffer_memory, 0) }?;

        Ok((buffer, buffer_memory))
    }

    fn copy_buffer(&self, src: Buffer, dst: Buffer, size: DeviceSize) -> Result<()> {
        let logical_device = &self.rhi_data.logical_device;

        let info = CommandBufferAllocateInfo::builder()
            .level(CommandBufferLevel::PRIMARY)
            .command_pool(self.value.global_command_pool)
            .command_buffer_count(1)
        ;

        let command_buffer = unsafe { logical_device.allocate_command_buffers(&info) }?[0];
        let begin_info = CommandBufferBeginInfo::builder()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { logical_device.begin_command_buffer(command_buffer, &begin_info) }?;

            let regions = BufferCopy::builder().size(size);
        unsafe { logical_device.cmd_copy_buffer(command_buffer, src, dst, &[regions]); }

        unsafe { logical_device.end_command_buffer(command_buffer) }?;

            let command_buffers = &[command_buffer];
            let info = SubmitInfo::builder()
                .command_buffers(command_buffers)
            ;

        unsafe { logical_device.queue_submit(self.rhi_data.graphics_queue, &[info], Fence::null()) }?;
        unsafe { logical_device.queue_wait_idle(self.rhi_data.graphics_queue)}?;

        unsafe { logical_device.free_command_buffers(self.value.global_command_pool, &[command_buffer]); }

        Ok(())
    }

    //Index buffer
    //ToDo: Unify with Vertex buffer creation
    fn create_index_buffer(&mut self) -> Result<()> {
        let logical_device = &self.rhi_data.logical_device;

        let size = (size_of::<u16>() * INDICES.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = self.create_buffer(size, BufferUsageFlags::TRANSFER_SRC, MemoryPropertyFlags::HOST_COHERENT | MemoryPropertyFlags::HOST_VISIBLE)?;

        let app_memory = unsafe { logical_device.map_memory(staging_buffer_memory, 0, size, MemoryMapFlags::empty()) }?;

        unsafe {
            copy_nonoverlapping(INDICES.as_ptr(), app_memory.cast(), INDICES.len());
            logical_device.unmap_memory(staging_buffer_memory);
        }

        let (buffer, buffer_memory) = self.create_buffer(size, BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::INDEX_BUFFER, MemoryPropertyFlags::DEVICE_LOCAL)?;

        self.value.index_buffer = buffer;
        self.value.index_buffer_memory = buffer_memory;

        self.copy_buffer(staging_buffer, self.value.index_buffer, size)?;

        unsafe {
            logical_device.destroy_buffer(staging_buffer, None);
            logical_device.free_memory(staging_buffer_memory, None);
        }

        Ok(())
    }

    //Uniform Buffers
    fn create_descriptor_set_layout(&mut self) -> Result<()> {
        let ubo0_binding = DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT);

        let ubo1_binding = DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT);

        let ubo2_binding = DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_type(DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(ShaderStageFlags::FRAGMENT);

        let ubo3_binding = DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_type(DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(ShaderStageFlags::FRAGMENT);

        let bindings = &[ubo0_binding, ubo1_binding, ubo2_binding, ubo3_binding];
        let info = DescriptorSetLayoutCreateInfo::builder()
            .bindings(bindings)
        ;

        self.value.descriptor_set_layout = unsafe { self.rhi_data.logical_device.create_descriptor_set_layout(&info, None) }?;
        Ok(())
    }

    fn destroy_uniform_buffers(&mut self) {
        unsafe {
            for i in 0..self.value.uniform_buffers.len() {
                for j in 0..self.value.uniform_buffers[i].len() {
                    self.rhi_data.logical_device.destroy_buffer(self.value.uniform_buffers[i][j], None);
                    self.rhi_data.logical_device.free_memory(self.value.uniform_buffers_memory[i][j], None);
                }
            }
        }

        self.value.uniform_buffers.clear();
        self.value.uniform_buffers_memory.clear();
    }

    fn create_uniform_buffers(&mut self) -> Result<()> {
        self.destroy_uniform_buffers();

        for _swapchain_image in &self.swapchain_data.swapchain_images {
            let b0_size = size_of::<Transformation>() as u64;
            let (b0, b0_memory) = self.create_buffer(b0_size, BufferUsageFlags::UNIFORM_BUFFER,
                                                     MemoryPropertyFlags::HOST_COHERENT | MemoryPropertyFlags::HOST_VISIBLE)?;

            let b1_size = size_of::<ViewState>() as u64;
            let (b1, b1_memory) = self.create_buffer(b1_size, BufferUsageFlags::UNIFORM_BUFFER,
                                                     MemoryPropertyFlags::HOST_COHERENT | MemoryPropertyFlags::HOST_VISIBLE)?;

            let b2_size = size_of::<ScatteringMedium>() as u64;
            let (b2, b2_memory) = self.create_buffer(b2_size, BufferUsageFlags::UNIFORM_BUFFER,
                                                     MemoryPropertyFlags::HOST_COHERENT | MemoryPropertyFlags::HOST_VISIBLE)?;

            let b3_size = size_of::<AtmosphereSampleData>() as u64;
            let (b3, b3_memory) = self.create_buffer(b3_size, BufferUsageFlags::UNIFORM_BUFFER,
                                                     MemoryPropertyFlags::HOST_COHERENT | MemoryPropertyFlags::HOST_VISIBLE)?;

            self.value.uniform_buffers.push([b0, b1, b2, b3]);
            self.value.uniform_buffers_memory.push([b0_memory, b1_memory, b2_memory, b3_memory]);
        }

        Ok(())
    }

    fn create_descriptor_pool(&mut self) -> Result<()> {
        let pool_size = DescriptorPoolSize::builder()
            .type_(DescriptorType::UNIFORM_BUFFER)
            .descriptor_count((self.swapchain_data.swapchain_images.len() * 4) as u32)
        ;

        let pool_sizes = &[pool_size];
        let info = DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(self.swapchain_data.swapchain_images.len() as u32);

        self.value.descriptor_pool = unsafe { self.rhi_data.logical_device.create_descriptor_pool(&info, None) }?;

        Ok(())
    }

    fn create_descriptor_sets(&mut self) -> Result<()> {
        let layouts = vec![self.value.descriptor_set_layout; self.swapchain_data.swapchain_images.len()];
        let info = DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.value.descriptor_pool)
            .set_layouts(&layouts)
        ;

        self.value.descriptor_sets = unsafe { self.rhi_data.logical_device.allocate_descriptor_sets(&info) }?;

        for i in 0..self.swapchain_data.swapchain_images.len() {
            let descriptor_set = self.value.descriptor_sets[i];
            let buffers = self.value.uniform_buffers[i];

            let buffer_infos = buffers.iter()
                .map(|b| {
                        DescriptorBufferInfo::builder()
                            .buffer(*b)
                            .offset(0)
                            .range(WHOLE_SIZE as u64)
                        })
                .collect::<Vec<_>>();

            let write_infos = buffers.iter()
                .enumerate()
                .map(|(i, _)|
                {
                    WriteDescriptorSet::builder()
                        .dst_set(descriptor_set)
                        .dst_binding(i as u32)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&buffer_infos[i..=i])
                })
                .collect::<Vec<_>>();

            unsafe { self.rhi_data.logical_device.update_descriptor_sets(&write_infos, &[] as &[CopyDescriptorSet]) }
        }

        Ok(())
    }
}