use std::sync::{Arc, RwLock};

use anyhow::Result;
use log::info;
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalSize, Size};
use winit::error::EventLoopError;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey, PhysicalKey};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
use winit::window::{Window, WindowId};

use crate::config::config::{Config, GraphicsApiType};
use crate::graphics::renderer::Renderer;
use crate::graphics::vulkan::vulkan_renderer::VulkanRenderer;
use crate::world::game_object::GameObject;
use crate::world::world::World;

pub struct App {
    config: Config,
    window: Option<Window>,
    graphics: Option<VulkanRenderer>,
    world_ref: Arc<RwLock<World>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut attributes = Window::default_attributes();
        attributes.title = "Rust - Vulkan".to_string();
        attributes.inner_size = Some(Size::Physical(PhysicalSize::new(1536, 1152)));

        self.window = Some(event_loop.create_window(attributes).unwrap());

        if self.graphics.is_none() {
            info!("Creating graphics...");
            let mut api = VulkanRenderer::new(
                self.window.as_ref().unwrap(),
                self.config
                    .graphics
                    .get(&GraphicsApiType::Vulkan)
                    .cloned()
                    .expect("Failed to read graphics config"),
            )
            .expect("Failed to create Vulkan renderer");

            api.initialize(self.world_ref.clone()).unwrap();

            self.graphics = Some(api);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        //ToDo: Add input handling in engine
        match event {
            WindowEvent::CloseRequested => {
                info!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in AboutToWait, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.

                {
                    let mut world = self.world_ref.write().unwrap();
                    world.update(0.0);
                }

                // Draw.
                self.render().unwrap();

                // Queue a RedrawRequested event.
                //
                // You only need to call this if you've determined that you need to redraw in
                // applications which do not always need to. Applications that redraw continuously
                // can render here instead.
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::KeyboardInput {
                device_id,
                event,
                is_synthetic,
            } => {
                if let Key::Named(named_key) = event.key_without_modifiers() {
                    match named_key {
                        NamedKey::Escape if event.state == ElementState::Pressed => {
                            event_loop.exit();
                        }
                        _ => (),
                    }
                }
            }
            _ => (),
        }
    }

    fn device_event(&mut self, event_loop: &ActiveEventLoop, device_id: DeviceId, event: DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                let mut world = self.world_ref.write().unwrap();

                // let (x, y) = (delta.0.clamp(-1.0, 1.0), delta.1.clamp(-1.0, 1.0));
                // debug!("Mouse delta(mod): {:?}, {:?}", x, y);
                //
                // world.active_camera_mut().transform_mut().rotate(y as f32, x as f32, 0.0);

                world.main_camera.handle_mouse_move(delta);
            }
            DeviceEvent::Key(key) => {
                if let PhysicalKey::Code(key_code) = key.physical_key {
                    let mut world = self.world_ref.write().unwrap();
                    world.main_camera.handle_input_key(key_code, key.state);
                }
            }
            _ => (),
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        info!("Destroying app");

        match self.graphics.as_mut() {
            Some(x) => x.destroy(),
            None => (),
        }
    }
}

impl App {
    pub(crate) fn new(config: Config) -> Self {
        let world = World::new();

        Self {
            config,
            window: None,
            graphics: None,
            world_ref: Arc::new(RwLock::new(world)),
        }
    }

    pub(crate) fn run(&mut self) -> Result<(), EventLoopError> {
        let event_loop = EventLoop::new().unwrap();
        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.run_app(self)
    }

    fn render(&mut self) -> Result<()> {
        if self.graphics.is_some() {
            return self
                .graphics
                .as_mut()
                .unwrap()
                .render(self.window.as_ref().unwrap());
        }

        Ok(())
    }
}
