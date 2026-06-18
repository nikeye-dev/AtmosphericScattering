use std::sync::{Arc, RwLock};

use anyhow::Result;
use winit::window::Window;

use crate::world::world::World;

pub trait Renderer {
    fn initialize(&mut self, world: Arc<RwLock<World>>) -> Result<()>;
    fn update(&mut self);
    fn render(&mut self, window: &Window) -> Result<()>;
    fn destroy(&mut self);

    fn get_width(&self) -> u32;
    fn get_height(&self) -> u32;
}
