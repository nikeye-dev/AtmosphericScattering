use anyhow::{anyhow, Result};
use cgmath::{vec3, vec4, Deg, Euler, Quaternion, Rotation};
use std::mem::size_of;
use std::sync::{Arc, RwLock};
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::{DeviceV1_0, KhrSwapchainExtensionDeviceCommands, SubpassContents, WHOLE_SIZE};
use winit::window::Window;

use crate::config::config::GraphicsConfig;
use crate::graphics::rhi::Renderer;
use crate::graphics::vulkan::atmopsheric_scattering::{AtmosphereSampleData, ScatteringMedium};
use crate::graphics::vulkan::push_constants::PushConstants;
use crate::graphics::vulkan::transformation::{Matrix4x4, Transformation};
use crate::graphics::vulkan::vertex::Vertex;
use crate::graphics::vulkan::view_state::ViewState;
use crate::graphics::vulkan::vulkan_commands::VulkanCommands;
use crate::graphics::vulkan::vulkan_context::VulkanContext;
use crate::graphics::vulkan::vulkan_descriptor_writer::VulkanDescriptorWriter;
use crate::graphics::vulkan::vulkan_descriptors::{VulkanDescriptorSetLayoutBuilder, VulkanDescriptors};
use crate::graphics::vulkan::vulkan_pipeline::{
    BlendMode, GraphicsShaderStage, VulkanGraphicsPipelineBuilder, VulkanPipeline,
};
use crate::graphics::vulkan::vulkan_render_pass::VulkanRenderPass;
use crate::graphics::vulkan::vulkan_renderer::LayoutIds::FrameMain;
use crate::graphics::vulkan::vulkan_resources::{Buffer, DynamicBuffer, VulkanResources};
use crate::graphics::vulkan::vulkan_swapchain::VulkanSwapchain;
use crate::graphics::vulkan::vulkan_sync_objects::SyncObjects;
use crate::graphics::vulkan::vulkan_utils::{perspective_matrix, INDICES, PERSPECTIVE_CORRECTION, VERTICES};
use crate::utils::math::VECTOR3_FORWARD;
use crate::world::transform::OwnedTransform;
use crate::world::world::World;

#[repr(u32)]
#[derive(strum_macros::Display)]
enum LayoutIds {
    FrameMain = 1,
}

struct UniformBuffers {
    //ToDo: Merge and simplify
    transform_buffers: Vec<DynamicBuffer>,
    view_buffers: Vec<DynamicBuffer>,
    atmosphere_medium_buffers: Vec<DynamicBuffer>,
    atmosphere_buffers: Vec<DynamicBuffer>,
}

impl UniformBuffers {
    pub fn destroy(&mut self, resources: &VulkanResources) {
        self.transform_buffers.drain(..).for_each(|buffer| {
            resources.destroy_dynamic_buffer(buffer);
        });

        self.view_buffers.drain(..).for_each(|buffer| {
            resources.destroy_dynamic_buffer(buffer);
        });

        self.atmosphere_medium_buffers.drain(..).for_each(|buffer| {
            resources.destroy_dynamic_buffer(buffer);
        });

        self.atmosphere_buffers.drain(..).for_each(|buffer| {
            resources.destroy_dynamic_buffer(buffer);
        });
    }
}

struct TempBuffers {
    planet_vertex_buffer: Option<Buffer>,
    planet_index_buffer: Option<Buffer>,
}

impl TempBuffers {
    pub fn destroy(&mut self, resources: &VulkanResources) {
        if let Some(planet_vertex_buffer) = self.planet_vertex_buffer.take() {
            resources.destroy_buffer(planet_vertex_buffer);
        }

        if let Some(planet_index_buffer) = self.planet_index_buffer.take() {
            resources.destroy_buffer(planet_index_buffer);
        }
    }
}

pub struct VulkanRenderer {
    is_destroyed: bool,
    frame_index: usize,

    //Keep
    sync_objects: SyncObjects,

    //New
    context: VulkanContext,
    commands: VulkanCommands,
    resources: VulkanResources,
    swapchain: VulkanSwapchain,
    render_pass: VulkanRenderPass,
    pipeline: VulkanPipeline,
    descriptors: VulkanDescriptors,

    uniforms: UniformBuffers,
    per_frame_descriptor_sets: Vec<vk::DescriptorSet>,

    //ToDo: remove in favor of collecting objects for rendering
    world: Option<Arc<RwLock<World>>>,
    temp_buffers: TempBuffers,
}

impl Renderer for VulkanRenderer {
    fn initialize(&mut self, world: Arc<RwLock<World>>) -> Result<()> {
        self.world = Some(world);
        Ok(())
    }
    fn update(&mut self) {
        todo!()
    }

    fn render(&mut self, window: &Window) -> Result<()> {
        if self.swapchain.is_dirty() {
            self.recreate_swapchain(window)?;
        }

        //Wait for fences
        let fence = self.sync_objects.in_flight_fences[self.frame_index];
        unsafe {
            self.context
                .device
                .wait_for_fences(&[fence], true, u64::MAX)?;
        }

        //Acquire next image
        let image_result =
            self.swapchain
                .acquire_next_image(&self.context.device, &self.sync_objects, self.frame_index);

        let image_index = match image_result {
            Ok((image_index, _)) => image_index,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => {
                self.swapchain.mark_dirty();
                return Err(anyhow!(vk::ErrorCode::OUT_OF_DATE_KHR));
            }
            Err(e) => return Err(e.into()),
        };

        //Reset fences
        unsafe { self.context.device.reset_fences(&[fence]) }?;

        //Record buffers
        self.update_uniform_buffers()?;
        let command_buffers = self.update_command_buffers()?;

        //Submit to graphics queue
        let wait_semaphores = [self.sync_objects.image_available_semaphores[self.frame_index]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.sync_objects.render_finished_semaphores[self.frame_index]];

        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&command_buffers)
            .wait_semaphores(&wait_semaphores)
            .signal_semaphores(&signal_semaphores)
            .wait_dst_stage_mask(&wait_stages);

        unsafe {
            self.context
                .device
                .queue_submit(self.context.graphics_queue, &[submit_info], fence)?;
        }

        //Present
        let image_indices = [image_index];
        let swapchains = [self.swapchain.handle];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .image_indices(&image_indices)
            .swapchains(&swapchains);

        let present_result = unsafe {
            self.context
                .device
                .queue_present_khr(self.context.graphics_queue, &present_info)
        };

        match present_result {
            Ok(vk::SuccessCode::SUBOPTIMAL_KHR) => {
                self.swapchain.mark_dirty();
            }
            Ok(_) => {}
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => {
                self.swapchain.mark_dirty();
                return Err(anyhow!(vk::ErrorCode::OUT_OF_DATE_KHR));
            }
            Err(e) => return Err(anyhow!(e)),
        };

        Ok(())
    }

    fn destroy(&mut self) {
        self.is_destroyed = true;

        unsafe { self.context.device.device_wait_idle() }.unwrap();

        self.sync_objects.destroy(&self.context.device);
        self.pipeline.destroy(&self.context.device);
        self.descriptors.destroy(&self.context.device);
        self.render_pass
            .destroy(&self.context.device, &self.resources);
        self.swapchain.destroy(&self.context.device);
        self.commands.destroy(&self.context.device);

        self.uniforms.destroy(&self.resources);
        self.temp_buffers.destroy(&self.resources);

        self.resources.destroy();
        self.context.destroy();
    }

    fn get_width(&self) -> u32 {
        todo!()
    }

    fn get_height(&self) -> u32 {
        todo!()
    }
}

impl VulkanRenderer {
    pub fn new(window: &Window, config: GraphicsConfig) -> Result<Self> {
        let frames_in_flight = config.max_frames_in_flight;

        let loader = unsafe { LibloadingLoader::new(LIBRARY) }?;
        let entry = unsafe { Entry::new(loader) }.unwrap();
        let context = VulkanContext::new(window, window, &entry, &config)?;

        let commands = VulkanCommands::new(&context, frames_in_flight)?;
        let resources = VulkanResources::new(&context)?;
        let swapchain = VulkanSwapchain::new(&context, window.inner_size().into(), None)?;
        let render_pass = VulkanRenderPass::new(&context, &resources, &swapchain)?;

        let push_constant_ranges = vec![vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<PushConstants>() as u32)
            .build()];

        let uniforms = Self::create_uniform_buffers(&resources, frames_in_flight)?;
        let (descriptors, per_frame_descriptor_sets) =
            Self::create_descriptors(&context.device, frames_in_flight, &uniforms)?;

        let frame_descriptor_set_layout = descriptors
            .frame_layout(FrameMain as u32)
            .ok_or_else(|| anyhow!("Layout {FrameMain} not found"))?;

        let pipeline_descriptor_set_layouts = vec![frame_descriptor_set_layout];

        let pipeline = VulkanGraphicsPipelineBuilder::new()
            .shader(
                GraphicsShaderStage::Vertex,
                "./resources/shaders/compiled/basic_vert.spv".into(),
            )
            .shader(
                GraphicsShaderStage::Fragment,
                "./resources/shaders/compiled/atmosphere_frag.spv".into(),
            )
            .vertex_bindings(vec![Vertex::binding_description()])
            .vertex_attributes(Vertex::attribute_descriptions())
            .descriptor_set_layouts(pipeline_descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges)
            .blend_mode(BlendMode::Transparent)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::empty())
            .depth_test(true)
            .depth_write(true)
            .depth_compare(vk::CompareOp::LESS)
            .sample_count(vk::SampleCountFlags::_1)
            .build(&context.device, &render_pass)?;

        let sync_objects = SyncObjects::new(&context.device, swapchain.image_count(), frames_in_flight)?;

        let temp_buffers = TempBuffers {
            planet_vertex_buffer: Some(resources.static_upload_buffer(
                &context.device,
                &commands,
                &VERTICES,
                context.graphics_queue,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?),
            planet_index_buffer: Some(resources.static_upload_buffer(
                &context.device,
                &commands,
                &INDICES,
                context.graphics_queue,
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?),
        };

        Ok(Self {
            is_destroyed: false,
            frame_index: 0,
            world: None,
            sync_objects,
            context,
            commands,
            resources,
            swapchain,
            render_pass,
            pipeline,
            descriptors,
            uniforms,
            per_frame_descriptor_sets,
            temp_buffers,
        })
    }

    fn create_uniform_buffers(resources: &VulkanResources, frames_in_flight: usize) -> Result<UniformBuffers> {
        let transform_buffers = (0..frames_in_flight)
            .map(|_| {
                resources.dynamic_buffer(
                    size_of::<Transformation>() as vk::DeviceSize,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let view_buffers = (0..frames_in_flight)
            .map(|_| {
                resources.dynamic_buffer(
                    size_of::<ViewState>() as vk::DeviceSize,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let atmosphere_medium_buffers = (0..frames_in_flight)
            .map(|_| {
                resources.dynamic_buffer(
                    size_of::<ScatteringMedium>() as vk::DeviceSize,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let atmosphere_buffers = (0..frames_in_flight)
            .map(|_| {
                resources.dynamic_buffer(
                    size_of::<AtmosphereSampleData>() as vk::DeviceSize,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(UniformBuffers {
            transform_buffers,
            view_buffers,
            atmosphere_medium_buffers,
            atmosphere_buffers,
        })
    }

    fn create_descriptors(
        device: &Device,
        frames_in_flight: usize,
        uniform_buffers: &UniformBuffers,
    ) -> Result<(VulkanDescriptors, Vec<vk::DescriptorSet>)> {
        let descriptors = VulkanDescriptors::new(device, frames_in_flight as u32, 1, 0)?;
        let frame_layout = VulkanDescriptorSetLayoutBuilder::new()
            .binding(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .binding(
                1,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .binding(2, vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::FRAGMENT, 1)
            .binding(3, vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::FRAGMENT, 1)
            .build(device)?;

        //Create descriptors
        let per_frame_descriptor_sets = descriptors.allocate_sets(&device, frame_layout, frames_in_flight)?;

        for i in 0..frames_in_flight {
            let set = per_frame_descriptor_sets[i];
            VulkanDescriptorWriter::new()
                .write_buffer(
                    set,
                    0,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    vk::DescriptorBufferInfo::builder()
                        .buffer(uniform_buffers.transform_buffers[i].buffer.handle)
                        .offset(0)
                        .range(WHOLE_SIZE),
                )
                .write_buffer(
                    set,
                    0,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    vk::DescriptorBufferInfo::builder()
                        .buffer(uniform_buffers.view_buffers[i].buffer.handle)
                        .offset(1)
                        .range(WHOLE_SIZE),
                )
                .write_buffer(
                    set,
                    0,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    vk::DescriptorBufferInfo::builder()
                        .buffer(uniform_buffers.atmosphere_medium_buffers[i].buffer.handle)
                        .offset(2)
                        .range(WHOLE_SIZE),
                )
                .write_buffer(
                    set,
                    0,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    vk::DescriptorBufferInfo::builder()
                        .buffer(uniform_buffers.atmosphere_buffers[i].buffer.handle)
                        .offset(3)
                        .range(WHOLE_SIZE),
                )
                .commit(&device);
        }

        Ok((descriptors, per_frame_descriptor_sets))
    }

    fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        unsafe {
            self.context.device.device_wait_idle()?;
        }

        self.render_pass
            .destroy(&self.context.device, &self.resources);

        let old_swapchain = std::mem::take(&mut self.swapchain);
        self.swapchain = VulkanSwapchain::new(&self.context, window.inner_size().into(), Some(old_swapchain))?;

        self.render_pass = VulkanRenderPass::new(&self.context, &self.resources, &self.swapchain)?;
        Ok(())
    }

    //ToDo: Add transforms and move from here
    fn update_uniform_buffers(&self) -> Result<()> {
        let frame_index = self.frame_index;

        let world = self
            .world
            .as_ref()
            .ok_or_else(|| anyhow!("Invalid world reference"))?
            .read()
            .map_err(|_| anyhow!("Poisoned world reference"))?;
        
        let camera = world.active_camera();
        let view = camera.view_matrix();

        let camera_pos = camera.transform().location();
        let projection = PERSPECTIVE_CORRECTION
            * perspective_matrix(
                camera.view().fov,
                self.swapchain.extent.width as f32,
                self.swapchain.extent.height as f32,
                camera.view().near,
                camera.view().far,
            );

        let transformation = Transformation::new(view, projection);
        self.resources
            .update_buffer(&self.uniforms.transform_buffers[frame_index], &transformation);

        let light_rot = Quaternion::from(Euler {
            x: Deg(-65.0),
            y: Deg(25.0),
            z: Deg(0.0),
        });

        let light_dir = light_rot.rotate_vector(VECTOR3_FORWARD).extend(0.0);
        let light_illuminance_outer_space = vec4(1., 1., 1., 1.) * 100.0;

        let view_state = ViewState {
            world_camera_origin: camera_pos.extend(0.0),
            atmosphere_light_direction: light_dir,
            atmosphere_light_illuminance_outer_space: light_illuminance_outer_space,
        };

        self.resources
            .update_buffer(&self.uniforms.view_buffers[frame_index], &view_state);

        let unit_scale = 0.2;
        let scattering_ray = vec3(0.175287, 0.409607, 1.0);
        let medium = ScatteringMedium::new(0.2, scattering_ray);

        self.resources
            .update_buffer(&self.uniforms.atmosphere_medium_buffers[frame_index], &medium);

        let atmospheric_sample_data = AtmosphereSampleData {
            planet_pos: vec3(0.0, 0.0, 0.0).extend(0.0),
            planet_radius: 6.3710,
            atmosphere_thickness: 0.0600,
            sample_count: 100.0,
            sample_count_light: 15.0,
            unit_scale,
            light_dir,
            light_intensity: light_illuminance_outer_space,

            pad: [0.0, 0.0, 0.0],
        };

        self.resources
            .update_buffer(&self.uniforms.atmosphere_buffers[frame_index], &atmospheric_sample_data);

        Ok(())
    }

    fn update_command_buffers(&self) -> Result<Vec<vk::CommandBuffer>> {
        let frame_index = self.frame_index;
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.swapchain.extent.width as f32,
            height: self.swapchain.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain.extent,
        };

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
            },
        ];

        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass.handle)
            .framebuffer(
                self.render_pass
                    .frame_buffer(frame_index)
                    .ok_or_else(|| anyhow!("Frame buffer index out of bounds: {frame_index}"))?,
            )
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent,
            })
            .clear_values(&clear_values);

        //ToDo: pass in gathered objects rather than keep a reference to the world
        let world = self
            .world
            .as_ref()
            .ok_or_else(|| anyhow!("Invalid world reference"))?
            .read()
            .map_err(|_| anyhow!("Poisoned world reference"))?;

        let planet = world
            .get_entities()
            .first()
            .ok_or_else(|| anyhow!("Empty world"))?;

        let planet_matrix = planet.transform.matrix();
        //

        let cmd = self
            .commands
            .begin_frame(&self.context.device, frame_index)?;

        let device = &self.context.device;

        unsafe {
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);

            device.cmd_begin_render_pass(cmd, &render_pass_info, SubpassContents::INLINE);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.handle);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &[self.per_frame_descriptor_sets[frame_index]],
                &[],
            );

            device.cmd_push_constants(
                cmd,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                std::slice::from_raw_parts(&planet_matrix as *const Matrix4x4 as *const u8, size_of::<Matrix4x4>()),
            );

            let vertex_buffer = self
                .temp_buffers
                .planet_vertex_buffer
                .as_ref()
                .unwrap()
                .handle;
            device.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer], &[0]);

            let index_buffer = self
                .temp_buffers
                .planet_index_buffer
                .as_ref()
                .unwrap()
                .handle;
            device.cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT16);

            device.cmd_draw_indexed(cmd, INDICES.len() as u32, 1, 0, 0, 0);

            device.cmd_end_render_pass(cmd);
        }

        self.commands.end_frame(&self.context.device, cmd)?;

        Ok(vec![])
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        if !self.is_destroyed {
            self.destroy();
        }
    }
}
