use cgmath::InnerSpace;
use std::time::Instant;

use crate::camera::camera::Camera;
use crate::camera::orbit_camera::OrbitCamera;
use crate::utils::math::{Vector3, Zero};
use crate::world::entity::Entity;
use crate::world::game_object::GameObject;
use crate::world::transform::{OwnedTransform, Transform};

pub struct World {
    start_time: Instant,
    last_frame_time: f32,
    //ToDo: Make private and handle input
    pub main_camera: OrbitCamera,
    entities: Vec<Entity>,
}

impl World {
    pub fn new() -> Self {
        let test_entity = Entity {
            id: 1,
            name: "test".into(),
            transform: Transform::new(
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(33.0, 33.0, 33.0),
            ),
        };

        let entities = vec![test_entity];

        let mut orbit_camera = OrbitCamera::default();
        orbit_camera
            .camera_mut()
            .transform_mut()
            .set_location_xyz(0.0, 0.0, -105.0);

        let start_time = Instant::now();

        Self {
            start_time,
            last_frame_time: start_time.elapsed().as_secs_f32(),
            main_camera: orbit_camera,
            entities,
        }
    }

    pub fn active_camera(&self) -> &Camera {
        self.main_camera.camera()
    }

    pub fn active_camera_mut(&mut self) -> &mut Camera {
        self.main_camera.camera_mut()
    }

    pub fn start_time(&self) -> Instant {
        self.start_time
    }

    pub fn get_entities(&self) -> &[Entity] {
        &self.entities
    }
}

impl GameObject for World {
    fn start(&mut self) {
        todo!()
    }

    fn update(&mut self, _: f32) {
        let frame_time = self.start_time.elapsed().as_secs_f32();
        let delta_time = frame_time - self.last_frame_time;
        self.last_frame_time = frame_time;

        self.main_camera.update(delta_time);

        let entity = self.entities.get_mut(0).unwrap();
        let mut location = entity.transform.location();
        location = location + Vector3::new(1.0, 1.0, 1.0).normalize() * delta_time;
        if location.magnitude() > 5.0 {
            location = Vector3::zero();
        }

        // entity.transform.set_location(location);
    }

    fn destroy(&mut self) {
        todo!()
    }
}
