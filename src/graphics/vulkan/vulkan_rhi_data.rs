use crate::config::config::{GraphicsConfig, LogLevel};
use crate::graphics::vulkan::vulkan_swapchain::SwapchainSupport;
use crate::graphics::vulkan::vulkan_utils::{debug_callback, CompatibilityError, QueueFamilyIndices, DEVICE_EXTENSIONS, PORTABILITY_MACOS_VERSION, VALIDATION_LAYER};
use anyhow::anyhow;
use anyhow::Result;
use log::{info, warn};
use std::collections::HashSet;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::vk::{ApplicationInfo, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT, DebugUtilsMessengerCreateInfoEXT, DebugUtilsMessengerEXT, DeviceCreateInfo, DeviceQueueCreateInfo, DeviceV1_0, EntryV1_0, ExtDebugUtilsExtension, HasBuilder, InstanceV1_0, KhrSurfaceExtension, PhysicalDevice, PhysicalDeviceFeatures, Queue, SurfaceKHR};
use vulkanalia::window as vk_window;
use vulkanalia::window::create_surface;
use vulkanalia::{vk, Device, Entry, Instance};
use winit::window::Window;

pub struct VulkanRHIData {
    pub(crate) entry: Entry,
    pub(crate) instance: Instance,
    pub(crate) messenger: DebugUtilsMessengerEXT,
    pub(crate) physical_device: PhysicalDevice,
    pub(crate) logical_device: Device,

    pub(crate) graphics_queue: Queue,
    pub(crate) present_queue: Queue,
    pub(crate) surface: SurfaceKHR,
}

impl VulkanRHIData {
    pub fn destroy(&self) {
        unsafe {
            self.logical_device.destroy_device(None);
            self.instance.destroy_debug_utils_messenger_ext(self.messenger, None);
            self.instance.destroy_surface_khr(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Default)]
pub struct VulkanRHIDataBuilder {
    config: GraphicsConfig,
    validation_enabled: bool,
    application_info: ApplicationInfo
}

impl VulkanRHIDataBuilder {
    pub fn validation(mut self, on: bool) -> Self {
        self.validation_enabled = on;
        self
    }

    pub fn config(mut self, config: GraphicsConfig) -> Self {
        self.config = config;
        self
    }

    pub fn application_info(mut self, application_info: ApplicationInfo) -> Self {
        self.application_info = application_info;
        self
    }

    pub fn build(self, window: &Window) -> Result<VulkanRHIData> {
        let loader = unsafe { LibloadingLoader::new(LIBRARY) }?;
        let entry = unsafe { Entry::new(loader) }.unwrap();
        let instance = self.create_instance(window, &entry, &self.config)?;

        let mut messenger = DebugUtilsMessengerEXT::default();
        if self.validation_enabled {
            let severity = match self.config.log_level {
                LogLevel::Verbose => DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
                LogLevel::Info => DebugUtilsMessageSeverityFlagsEXT::INFO,
                LogLevel::Warning => DebugUtilsMessageSeverityFlagsEXT::WARNING,
                LogLevel::Error => DebugUtilsMessageSeverityFlagsEXT::ERROR
            };

            let debug_info = DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(severity)
                .message_type(DebugUtilsMessageTypeFlagsEXT::all())
                .user_callback(Some(debug_callback));

            messenger = unsafe { instance.create_debug_utils_messenger_ext(&debug_info, None) }?;
        }

        let surface = unsafe { create_surface(&instance, &window, &window) }?;

        let physical_device = self.pick_physical_device(&instance, surface)?;
        let logical_device = self.create_logical_device(&instance, physical_device, surface)?;

        let queue_family_indices = QueueFamilyIndices::get(&instance, physical_device, surface)?;

        let graphics_queue = unsafe { logical_device.get_device_queue(queue_family_indices.graphics, 0) };
        let present_queue = unsafe { logical_device.get_device_queue(queue_family_indices.present, 0) };

        Ok(VulkanRHIData {
            entry,
            instance,
            messenger,
            physical_device,
            logical_device,
            graphics_queue,
            present_queue,
            surface
        })
    }

    fn create_instance(&self, window: &Window, entry: &Entry, config: &GraphicsConfig) -> Result<Instance> {
        // let app_info = vk::ApplicationInfo::builder()
        //     .application_version(vk::make_version(0, 1, 0))
        //     .api_version(vk::make_version(1, 0, 0))
        //     .engine_version(vk::make_version(1, 0, 0))
        //     .application_name(b"Test Name")
        //     .engine_name(b"Test Engine")
        //     .build();

        let mut extensions = vk_window::get_required_instance_extensions(window)
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();

        //Enable compatibility extensions
        // Required by Vulkan SDK on macOS since 1.3.216.
        let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION
        {
            info!("Enabling extensions for macOS portability.");
            extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name.as_ptr());
            extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };
        //

        //Validation layers
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }?
            .iter()
            .map(|l| l.layer_name)
            .collect::<HashSet<_>>();

        if self.validation_enabled {
            if !available_layers.contains(&VALIDATION_LAYER) {
                return Err(anyhow!("Validation layer requested but not supported."));
            }

            extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
        }

        let layers = if self.validation_enabled {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            Vec::new()
        };
        //

        let instance_info = vk::InstanceCreateInfo::builder()
            .application_info(&self.application_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .flags(flags);

        //Debug
        let mut debug_info = DebugUtilsMessengerCreateInfoEXT::builder();
        if self.validation_enabled {
            let severity = match config.log_level {
                LogLevel::Verbose => DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
                LogLevel::Info => DebugUtilsMessageSeverityFlagsEXT::INFO,
                LogLevel::Warning => DebugUtilsMessageSeverityFlagsEXT::WARNING,
                LogLevel::Error => DebugUtilsMessageSeverityFlagsEXT::ERROR
            };
            println!("Severity: {:?}", severity);
            debug_info.message_severity(severity)
                .message_type(DebugUtilsMessageTypeFlagsEXT::all())
                .user_callback(Some(debug_callback));

            instance_info.push_next(&mut debug_info);

            info!("Added debug callback to Vulkan with level {:?}", severity);
        }
        //

        let result = unsafe { entry.create_instance(&instance_info, None) }?;
        Ok(result)
    }

    fn pick_physical_device(&self, instance: &Instance, surface: SurfaceKHR) -> Result<PhysicalDevice> {
        for physical_device in unsafe {instance.enumerate_physical_devices()?} {
            let properties = unsafe { instance.get_physical_device_properties(physical_device) };

            match Self::check_physical_device_compatibility(instance, physical_device, surface) {
                Ok(_) => {
                    info!("Selected physical device (`{}`).", properties.device_name);
                    return Ok(physical_device);
                }
                Err(error) => warn!("Skipping physical device (`{}`): {}", properties.device_name, error)
            }
        }

        Err(anyhow!(CompatibilityError("Failed to find compatible physical device")))
    }

    fn check_physical_device_compatibility(instance: &Instance, physical_device: PhysicalDevice, surface: SurfaceKHR) -> Result<()> {
        let _ = QueueFamilyIndices::get(instance, physical_device, surface)?;
        let _ = Self::check_physical_device_extensions(instance, physical_device)?;

        let support = SwapchainSupport::get(instance, physical_device, surface)?;
        if support.formats.is_empty() || support.present_modes.is_empty() {
            return Err(anyhow!(CompatibilityError("Insufficient swapchain support.")));
        }

        Ok(())
    }

    fn check_physical_device_extensions(instance: &Instance, physical_device: PhysicalDevice) -> Result<()> {
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device, None)?.iter().map(|e| e.extension_name).collect::<HashSet<_>>() };
        //Check for graphics commands
        let is_supported = DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e));
        if is_supported {
            Ok(())
        }
        else {
            Err(anyhow!(CompatibilityError("Missing required queue family extensions.")))
        }
    }

    //ToDo: Maybe instance method - e.g. initialize?
    fn create_logical_device(&self, instance: &Instance, physical_device: PhysicalDevice, surface: SurfaceKHR) -> Result<Device> {
        //ToDo: Make this use the already resolved indices from previous step. The current implementation may cause discrepancies
        let queue_family_indices = QueueFamilyIndices::get(instance, physical_device, surface)?;
        let mut indices = HashSet::new();

        indices.insert(queue_family_indices.graphics);
        indices.insert(queue_family_indices.present);

        let layers = if self.validation_enabled {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            vec![]
        };

        let extensions = DEVICE_EXTENSIONS
            .iter()
            .map(|n| n.as_ptr())
            .collect::<Vec<_>>();

        let features = PhysicalDeviceFeatures::builder();

        let queue_priorities = &[1.0];
        let queue_infos =
            indices.iter().map(|i| {
                DeviceQueueCreateInfo::builder()
                    .queue_family_index(*i)
                    .queue_priorities(queue_priorities)
            })
                .collect::<Vec<_>>();

        let device_info = DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .enabled_features(&features);

        let device = unsafe { instance.create_device(physical_device, &device_info, None) }.unwrap();

        Ok(device)
    }
}