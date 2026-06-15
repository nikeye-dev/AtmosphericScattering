use crate::config::config::GraphicsConfig;
use crate::graphics::vulkan::vulkan_swapchain::SwapchainSupport;
use crate::graphics::vulkan::vulkan_utils::{
    debug_callback, CompatibilityError, QueueFamilyIndices, DEVICE_EXTENSIONS, PORTABILITY_MACOS_VERSION, VALIDATION_LAYER,
};
use anyhow::{anyhow, Result};
use log::{info, warn};
use std::collections::HashSet;
pub use vulkanalia::prelude::v1_2::*;
use vulkanalia::vk::ExtDebugUtilsExtensionInstanceCommands;
use vulkanalia::window;
use winit::dpi::PhysicalSize;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Default, Hash)]
pub struct Size<T> {
    pub width: T,
    pub height: T,
}

impl From<PhysicalSize<u32>> for Size<u32> {
    fn from(size: PhysicalSize<u32>) -> Self {
        Self {
            width: size.width,
            height: size.height,
        }
    }
}

pub struct VulkanContext {
    messenger: vk::DebugUtilsMessengerEXT,

    pub instance: Instance,
    pub device: Device,
    pub physical_device: vk::PhysicalDevice,
    pub surface: vk::SurfaceKHR,
    pub family_indices: QueueFamilyIndices,
}

impl VulkanContext {
    pub fn new(window: &dyn HasWindowHandle, display: &dyn HasDisplayHandle, entry: &Entry, config: &GraphicsConfig) -> Result<Self> {
        let validation_layers = Self::validation_layers(entry, config.validation_enabled)?;
        let instance = Self::create_instance(window, entry, config, &validation_layers)?;
        let messenger = Self::create_debug_messenger(&instance, config)?;
        let surface = unsafe { window::create_surface(&instance, display, window) }?;

        let physical_device = Self::pick_physical_device(&instance, surface)?;
        let family_indices = QueueFamilyIndices::get(&instance, physical_device, surface)?;
        let device = Self::create_logical_device(&instance, physical_device, &family_indices, &validation_layers)?;

        Ok(Self {
            instance,
            device,
            physical_device,
            surface,
            family_indices,
            messenger,
        })
    }

    fn create_instance(
        window: &dyn HasWindowHandle,
        entry: &Entry,
        config: &GraphicsConfig,
        validation_layers: &[*const std::ffi::c_char],
    ) -> Result<Instance> {
        let mut extensions = window::get_required_instance_extensions(window)
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();

        //Enable compatibility extensions
        // Required by Vulkan SDK on macOS since 1.3.216.
        let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
            info!("Enabling extensions for macOS portability.");
            extensions.push(
                vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION
                    .name
                    .as_ptr(),
            );
            extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };
        //

        let app_info = vk::ApplicationInfo::builder()
            .application_version(vk::make_version(0, 1, 0))
            .api_version(vk::make_version(1, 0, 0))
            .engine_version(vk::make_version(1, 0, 0))
            .application_name(b"Test Name")
            .engine_name(b"Test Engine")
            .build();

        let instance_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_layer_names(&validation_layers)
            .enabled_extension_names(&extensions)
            .flags(flags);

        //Debug
        let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder();
        if config.validation_enabled {
            debug_info
                .message_severity(config.log_level.into())
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                .user_callback(Some(debug_callback));

            instance_info.push_next(&mut debug_info);
            info!("Added debug callback to Vulkan with level {:?}", config.log_level);
        }
        //

        let result = unsafe { entry.create_instance(&instance_info, None) }?;
        Ok(result)
    }

    fn create_debug_messenger(instance: &Instance, config: &GraphicsConfig) -> Result<vk::DebugUtilsMessengerEXT> {
        let mut messenger = vk::DebugUtilsMessengerEXT::default();
        if config.validation_enabled {
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(config.log_level.into())
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                .user_callback(Some(debug_callback));

            messenger = unsafe { instance.create_debug_utils_messenger_ext(&debug_info, None) }?;
        }

        Ok(messenger)
    }

    fn pick_physical_device(instance: &Instance, surface: vk::SurfaceKHR) -> Result<vk::PhysicalDevice> {
        for physical_device in unsafe { instance.enumerate_physical_devices()? } {
            let properties = unsafe { instance.get_physical_device_properties(physical_device) };

            match Self::check_physical_device_compatibility(instance, physical_device, surface) {
                Ok(_) => {
                    info!("Selected physical device (`{}`).", properties.device_name);
                    return Ok(physical_device);
                }
                Err(error) => warn!("Skipping physical device (`{}`): {}", properties.device_name, error),
            }
        }

        Err(anyhow!(CompatibilityError("Failed to find compatible physical device")))
    }

    fn check_physical_device_compatibility(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<()> {
        let _ = QueueFamilyIndices::get(instance, physical_device, surface)?;
        let _ = Self::check_physical_device_extensions(instance, physical_device)?;

        let support = SwapchainSupport::get(instance, physical_device, surface)?;
        if support.formats.is_empty() || support.present_modes.is_empty() {
            return Err(anyhow!(CompatibilityError("Insufficient swapchain support.")));
        }

        Ok(())
    }

    fn check_physical_device_extensions(instance: &Instance, physical_device: vk::PhysicalDevice) -> Result<()> {
        let extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device, None)?
                .iter()
                .map(|e| e.extension_name)
                .collect::<HashSet<_>>()
        };
        //Check for graphics commands
        let is_supported = DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e));
        if is_supported {
            Ok(())
        } else {
            Err(anyhow!(CompatibilityError("Missing required queue family extensions.")))
        }
    }

    fn create_logical_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        family_indices: &QueueFamilyIndices,
        validation_layers: &[*const std::ffi::c_char],
    ) -> Result<Device> {
        let indices = HashSet::from([family_indices.graphics, family_indices.present, family_indices.transfer]);

        let extensions = DEVICE_EXTENSIONS
            .iter()
            .map(|n| n.as_ptr())
            .collect::<Vec<_>>();

        let features = vk::PhysicalDeviceFeatures::builder();

        let queue_priorities = &[1.0];
        let queue_infos = indices
            .iter()
            .map(|i| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(*i)
                    .queue_priorities(queue_priorities)
            })
            .collect::<Vec<_>>();

        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_layer_names(&validation_layers)
            .enabled_extension_names(&extensions)
            .enabled_features(&features);

        let device = unsafe { instance.create_device(physical_device, &device_info, None) }?;
        Ok(device)
    }

    fn validation_layers(entry: &Entry, validation_enabled: bool) -> Result<Vec<*const std::ffi::c_char>> {
        let mut layers = Vec::new();
        if validation_enabled {
            let available_layers = unsafe { entry.enumerate_instance_layer_properties() }?
                .iter()
                .map(|l| l.layer_name)
                .collect::<HashSet<_>>();

            if !available_layers.contains(&VALIDATION_LAYER) {
                return Err(anyhow!("Validation layer requested but not supported."));
            }

            layers.push(VALIDATION_LAYER.as_ptr());
        }

        Ok(layers)
    }
}
