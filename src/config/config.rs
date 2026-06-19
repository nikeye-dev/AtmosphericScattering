use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};

#[derive(Default, Serialize, Deserialize, Debug, Hash, Eq, PartialEq)]
#[serde(rename_all(serialize="lowercase", deserialize="lowercase"))]
pub enum GraphicsApiType {
    #[default]
    Vulkan
}

#[derive(Default, Serialize_repr, Deserialize_repr, Debug, PartialEq, PartialOrd, Copy, Clone)]
#[repr(u8)]
pub enum LogLevel {
    #[default]
    Verbose = 1,
    Info = 2,
    Warning = 3,
    Error = 4
}

#[derive(Default, Serialize, Deserialize, Debug, Copy, Clone)]
pub struct GraphicsConfig {
    pub log_level: LogLevel,
    pub validation_enabled: bool,
    pub max_frames_in_flight: usize,
}

#[derive(Default, Serialize, Deserialize, Debug)]
pub struct Config {
    pub graphics: HashMap<GraphicsApiType, GraphicsConfig>
}

pub async fn load_config() -> Result<Config> {
    let path = Path::new("./resources/config/default_config.json");

    let config_json = tokio::fs::read_to_string(path).await?;
    Ok(serde_json::from_str::<Config>(&config_json)?)
}
