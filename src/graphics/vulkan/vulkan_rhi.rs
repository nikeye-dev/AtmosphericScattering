use std::mem::size_of;
use std::ptr::copy_nonoverlapping;
use std::slice;
use std::sync::{Arc, RwLock};

use anyhow::{anyhow, Result};
use cgmath::{vec3, vec4, Deg, Euler, Quaternion, Rotation};
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::{DeviceV1_0, KhrSwapchainExtensionDeviceCommands};
use winit::window::Window;

use crate::config::config::GraphicsConfig;
use crate::graphics::rhi::RHI;
use crate::graphics::vulkan::atmopsheric_scattering::{AtmosphereSampleData, ScatteringMedium};
use crate::graphics::vulkan::push_constants::PushConstants;
use crate::graphics::vulkan::transformation::{Matrix4x4, Transformation};
use crate::graphics::vulkan::vertex::Vertex;
use crate::graphics::vulkan::view_state::ViewState;
use crate::graphics::vulkan::vulkan_commands::VulkanCommands;
use crate::graphics::vulkan::vulkan_context::VulkanContext;
use crate::graphics::vulkan::vulkan_descriptors::VulkanDescriptors;
use crate::graphics::vulkan::vulkan_pipeline::{PipelineData, PipelineDataBuilder};
use crate::graphics::vulkan::vulkan_pipeline2::{
    BlendMode, GraphicsShaderStage, VulkanGraphicsPipelineBuilder, VulkanPipeline,
};
use crate::graphics::vulkan::vulkan_render_pass::VulkanRenderPass;
use crate::graphics::vulkan::vulkan_resources::VulkanResources;
use crate::graphics::vulkan::vulkan_rhi_data::{VulkanRHIData, VulkanRHIDataBuilder};
use crate::graphics::vulkan::vulkan_swapchain::{SwapchainData, SwapchainDataBuilder};
use crate::graphics::vulkan::vulkan_swapchain2::VulkanSwapchain;
use crate::graphics::vulkan::vulkan_sync_objects::SyncObjects;
use crate::graphics::vulkan::vulkan_utils::{
    perspective_matrix, RHIDestroy, INDICES, PERSPECTIVE_CORRECTION, VALIDATION_ENABLED,
};
use crate::utils::math::VECTOR3_FORWARD;
use crate::world::transform::OwnedTransform;
use crate::world::world::World;

pub struct RHIVulkan {
    max_frames_in_flight: usize,

    is_destroyed: bool,
    config: GraphicsConfig,
    frame_index: usize,
    world: Option<Arc<RwLock<World>>>,

    data: VulkanRHIData,
    swapchain_data: SwapchainData,
    pipeline_data: PipelineData,

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
}

impl RHI for RHIVulkan {
    fn initialize(&mut self, world: Arc<RwLock<World>>) -> Result<()> {
        self.world = Some(world);
        Ok(())
    }
    fn update(&mut self) {
        todo!()
    }

    fn render(&mut self, window: &Window) -> Result<()> {
        let fence = self.sync_objects.in_flight_fences[self.frame_index];

        unsafe {
            self.data
                .logical_device
                .wait_for_fences(&[fence], true, u64::MAX)?;
        }

        let next_image_result = unsafe {
            self.data.logical_device.acquire_next_image_khr(
                self.swapchain_data.swapchain,
                u64::MAX,
                self.sync_objects.image_available_semaphores[self.frame_index],
                vk::Fence::null(),
            )
        };

        let image_index = match next_image_result {
            Ok((image_index, _)) => image_index,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        if !self.sync_objects.images_in_flight[image_index as usize].is_null() {
            unsafe {
                self.data.logical_device.wait_for_fences(
                    &[self.sync_objects.images_in_flight[image_index as usize]],
                    true,
                    u64::MAX,
                )?;
            }
        }

        self.sync_objects
            .set_image_fence(image_index as usize, fence);

        self.update_command_buffers(image_index as usize)?;
        self.update_uniform_buffers(image_index as usize)?;

        let wait_semaphores = &[self.sync_objects.image_available_semaphores[self.frame_index]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.pipeline_data.primary_command_buffers[image_index as usize]];
        let signal_semaphores = &[self.sync_objects.render_finished_semaphores[self.frame_index]];
        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores)
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages);

        unsafe {
            self.data.logical_device.reset_fences(&[fence])?;
            self.data.logical_device.queue_submit(
                self.data.graphics_queue,
                &[submit_info],
                self.sync_objects.in_flight_fences[self.frame_index],
            )?;
        }

        let swapchains = &[self.swapchain_data.swapchain];
        let image_indices = &[image_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let present_result = unsafe {
            self.data
                .logical_device
                .queue_present_khr(self.data.present_queue, &present_info)
        };

        //ToDo: Handle swapchain invalidation better - resize, minimize etc
        if present_result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || present_result == Err(vk::ErrorCode::OUT_OF_DATE_KHR)
        {
            self.recreate_swapchain(window)?;
        } else if let Err(e) = present_result {
            return Err(anyhow!(e));
        }

        self.frame_index = (self.frame_index + 1) % self.max_frames_in_flight;

        Ok(())
    }

    fn destroy(&mut self) {
        self.is_destroyed = true;

        unsafe { self.data.logical_device.device_wait_idle() }.unwrap();

        self.sync_objects.destroy(&self.data);
        self.pipeline_data.destroy(&self.data);
        self.swapchain_data.destroy(&self.data);
        self.data.destroy();
    }

    fn get_width(&self) -> u32 {
        todo!()
    }

    fn get_height(&self) -> u32 {
        todo!()
    }
}

impl RHIVulkan {
    pub fn new(window: &Window, config: GraphicsConfig) -> Result<Self> {
        //ToDo: Delete
        let app_info = vk::ApplicationInfo::builder()
            .application_version(vk::make_version(0, 1, 0))
            .api_version(vk::make_version(1, 0, 0))
            .engine_version(vk::make_version(1, 0, 0))
            .application_name(b"Test Name")
            .engine_name(b"Test Engine")
            .build();

        let rhi_data = VulkanRHIDataBuilder::default()
            .application_info(app_info)
            .config(config)
            .validation(VALIDATION_ENABLED)
            .build(window)?;

        let swapchain_data = SwapchainDataBuilder::default().build(window, &rhi_data)?;

        let pipeline_data = PipelineDataBuilder::new(&rhi_data, &swapchain_data)
            .shader(
                vk::ShaderStageFlags::VERTEX,
                "./resources/shaders/compiled/basic_vert.spv",
            )
            .shader(
                vk::ShaderStageFlags::FRAGMENT,
                "./resources/shaders/compiled/atmosphere_frag.spv",
            )
            .build()?;
        //

        let loader = unsafe { LibloadingLoader::new(LIBRARY) }?;
        let entry = unsafe { Entry::new(loader) }.unwrap();
        let context = VulkanContext::new(window, window, &entry, &config)?;

        let commands = VulkanCommands::new(&context, config.max_frames_in_flight)?;
        let resources = VulkanResources::new(&context)?;
        let swapchain = VulkanSwapchain::new(&context, window.inner_size().into(), None)?;
        let render_pass = VulkanRenderPass::new(&context, &resources, &swapchain)?;

        let push_constant_ranges = vec![vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<PushConstants>() as u32)
            .build()];

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
            .descriptor_set_layouts(vec![Self::create_descriptor_set_layout(&context.device)?])
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

        let descriptors =
            VulkanDescriptors::new(&context.device, config.max_frames_in_flight as u32, 1, 0)?;

        let sync_objects = SyncObjects::create(
            &context.device,
            swapchain.image_count(),
            config.max_frames_in_flight,
        )?;

        Ok(Self {
            max_frames_in_flight: config.max_frames_in_flight,
            is_destroyed: false,
            config,
            frame_index: 0,
            world: None,
            data: rhi_data,
            swapchain_data,
            pipeline_data,
            sync_objects,
            context,
            commands,
            resources,
            swapchain,
            render_pass,
            pipeline,
            descriptors,
        })
    }

    fn create_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
        let ubo0_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

        let ubo1_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

        let ubo2_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let ubo3_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = &[ubo0_binding, ubo1_binding, ubo2_binding, ubo3_binding];
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

        let layout = unsafe { device.create_descriptor_set_layout(&info, None) }?;
        Ok(layout)
    }

    fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        unsafe {
            self.data.logical_device.device_wait_idle()?;
        }

        self.swapchain_data.destroy(&self.data);

        //ToDo: Reuse what can be reused (e.g. command buffers)
        self.pipeline_data.destroy(&self.data);

        self.sync_objects.destroy(&self.data);

        self.swapchain_data = SwapchainDataBuilder::default().build(window, &self.data)?;

        self.pipeline_data = PipelineDataBuilder::new(&self.data, &self.swapchain_data)
            .shader(
                vk::ShaderStageFlags::VERTEX,
                "../../../resources/shaders/compiled/basic_vert.spv",
            )
            .shader(
                vk::ShaderStageFlags::FRAGMENT,
                "../../../resources/shaders/compiled/basic_frag.spv",
            )
            .build()?;

        self.sync_objects = SyncObjects::create(
            &self.data.logical_device,
            self.swapchain_data.swapchain_images.len(),
            self.max_frames_in_flight,
        )?;

        Ok(())
    }

    //ToDo: Add transforms and move from here
    fn update_uniform_buffers(&self, image_index: usize) -> Result<()> {
        let world = self.world.as_ref().unwrap().read().unwrap();
        let camera = world.active_camera();
        let view = camera.view_matrix();

        let camera_pos = camera.transform().location();
        let projection = PERSPECTIVE_CORRECTION
            * perspective_matrix(
                camera.view().fov,
                self.swapchain_data.swapchain_extent.width as f32,
                self.swapchain_data.swapchain_extent.height as f32,
                camera.view().near,
                camera.view().far,
            );

        let transformation = Transformation::new(view, projection);

        unsafe {
            let buffer_memory = self.pipeline_data.uniform_buffers_memory[image_index][0];
            let memory = self.data.logical_device.map_memory(
                buffer_memory,
                0,
                size_of::<Transformation>() as u64,
                vk::MemoryMapFlags::empty(),
            )?;

            copy_nonoverlapping(&transformation, memory.cast(), 1);
            self.data.logical_device.unmap_memory(buffer_memory)
        };

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

        unsafe {
            let buffer_memory = self.pipeline_data.uniform_buffers_memory[image_index][1];
            let memory = self.data.logical_device.map_memory(
                buffer_memory,
                0,
                size_of::<ViewState>() as u64,
                vk::MemoryMapFlags::empty(),
            )?;

            copy_nonoverlapping(&view_state, memory.cast(), 1);
            self.data.logical_device.unmap_memory(buffer_memory)
        };

        let unit_scale = 0.2;
        let scattering_ray = vec3(0.175287, 0.409607, 1.0);
        let medium = ScatteringMedium::new(0.2, scattering_ray);

        unsafe {
            let buffer_memory = self.pipeline_data.uniform_buffers_memory[image_index][2];
            let memory = self.data.logical_device.map_memory(
                buffer_memory,
                0,
                size_of::<ScatteringMedium>() as u64,
                vk::MemoryMapFlags::empty(),
            )?;

            copy_nonoverlapping(&medium, memory.cast(), 1);
            self.data.logical_device.unmap_memory(buffer_memory)
        };

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

        unsafe {
            let buffer_memory = self.pipeline_data.uniform_buffers_memory[image_index][3];
            let memory = self.data.logical_device.map_memory(
                buffer_memory,
                0,
                size_of::<AtmosphereSampleData>() as u64,
                vk::MemoryMapFlags::empty(),
            )?;

            copy_nonoverlapping(&atmospheric_sample_data, memory.cast(), 1);
            self.data.logical_device.unmap_memory(buffer_memory)
        };

        Ok(())
    }

    fn update_command_buffers(&mut self, image_index: usize) -> Result<()> {
        let command_pool = self.pipeline_data.command_pools[image_index];
        unsafe {
            self.data
                .logical_device
                .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
        }?;

        let command_buffer = self
            .pipeline_data
            .primary_command_buffers
            .get(image_index)
            .unwrap();
        self.update_command_buffer(image_index, *command_buffer)?;

        Ok(())
    }

    fn update_command_buffer(
        &mut self,
        image_index: usize,
        command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        let command_buffer_inheritance_info = vk::CommandBufferInheritanceInfo::builder();

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .inheritance_info(&command_buffer_inheritance_info);

        let logical_device = &self.data.logical_device;

        let render_area = vk::Rect2D::builder()
            .extent(self.swapchain_data.swapchain_extent)
            .offset(vk::Offset2D::default());

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };

        let clear_values = &[color_clear_value, depth_clear_value];
        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.pipeline_data.render_pass)
            .framebuffer(self.pipeline_data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        unsafe {
            logical_device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;
            logical_device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
            );
        }

        let secondary_command_buffer = self.pipeline_data.get_or_allocate_secondary_buffer(
            image_index,
            0,
            &self.data.logical_device,
        );
        self.update_secondary_command_buffer(secondary_command_buffer, image_index)?;

        unsafe {
            logical_device.cmd_execute_commands(command_buffer, &[secondary_command_buffer]);

            logical_device.cmd_end_render_pass(command_buffer);
            logical_device.end_command_buffer(command_buffer)?;
        }

        Ok(())
    }

    //ToDo: Make async and parallelize
    fn update_secondary_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) -> Result<()> {
        //ToDo:
        let world = self.world.as_ref().unwrap().read().unwrap();
        let entities = world.get_entities();

        let model = entities[0].transform.matrix();
        let model_bytes = unsafe {
            slice::from_raw_parts(
                &model as *const Matrix4x4 as *const u8,
                size_of::<Matrix4x4>(),
            )
        };

        // let command_buffer = self.get_or_add_secondary_buffer(&image_index, buffer_index);

        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.pipeline_data.render_pass)
            .subpass(0)
            .framebuffer(self.pipeline_data.framebuffers[image_index]);

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        let logical_device = &self.data.logical_device;
        unsafe {
            logical_device.begin_command_buffer(command_buffer, &info)?;
            logical_device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_data.pipeline,
            );

            logical_device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[self.pipeline_data.vertex_buffer],
                &[0],
            );
            logical_device.cmd_bind_index_buffer(
                command_buffer,
                self.pipeline_data.index_buffer,
                0,
                vk::IndexType::UINT16,
            );

            logical_device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_data.pipeline_layout,
                0,
                &[self.pipeline_data.descriptor_sets[image_index]],
                &[],
            );

            logical_device.cmd_push_constants(
                command_buffer,
                self.pipeline_data.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                model_bytes,
            );

            logical_device.cmd_draw_indexed(command_buffer, INDICES.len() as u32, 1, 0, 0, 0);

            logical_device.end_command_buffer(command_buffer)?
        }

        Ok(())
    }
}

impl Drop for RHIVulkan {
    fn drop(&mut self) {
        if !self.is_destroyed {
            self.destroy();
        }
    }
}
