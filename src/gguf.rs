use std::fs::File;
use std::io::{Read, Seek};
use std::collections::HashMap;
use anyhow::{Result, bail};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;

#[derive(Debug, Clone, PartialEq)]
pub enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GgufType {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(GgufType::Uint8),
            1 => Ok(GgufType::Int8),
            2 => Ok(GgufType::Uint16),
            3 => Ok(GgufType::Int16),
            4 => Ok(GgufType::Uint32),
            5 => Ok(GgufType::Int32),
            6 => Ok(GgufType::Float32),
            7 => Ok(GgufType::Bool),
            8 => Ok(GgufType::String),
            9 => Ok(GgufType::Array),
            10 => Ok(GgufType::Uint64),
            11 => Ok(GgufType::Int64),
            12 => Ok(GgufType::Float64),
            _ => bail!("Invalid GGUF type: {}", value),
        }
    }
}

pub struct GgufHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub n_tensors: u64,
    pub n_kv: u64,
}

pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

pub struct GgufContext {
    pub header: GgufHeader,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub data: Mmap,
    pub data_offset: u64,
}

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufContext {
    pub fn load(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = std::io::BufReader::new(&file);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            bail!("Invalid GGUF magic");
        }

        let version = reader.read_u32::<LittleEndian>()?;
        if version != 3 {
             bail!("Unsupported GGUF version: {}", version);
        }

        let n_tensors = reader.read_u64::<LittleEndian>()?;
        let n_kv = reader.read_u64::<LittleEndian>()?;

        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let key = read_string(&mut reader)?;
            let val_type = GgufType::try_from(reader.read_u32::<LittleEndian>()?)?;
            let value = read_value(&mut reader, &val_type)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::new();
        for _ in 0..n_tensors {
            let name = read_string(&mut reader)?;
            let n_dims = reader.read_u32::<LittleEndian>()?;
            let mut dimensions = Vec::new();
            for _ in 0..n_dims {
                dimensions.push(reader.read_u64::<LittleEndian>()?);
            }
            let ggml_type = reader.read_u32::<LittleEndian>()?;
            let offset = reader.read_u64::<LittleEndian>()?;
            tensors.push(GgufTensorInfo {
                name,
                dimensions,
                ggml_type,
                offset,
            });
        }

        let current_pos = reader.stream_position()?;
        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::Uint32(a)) => *a as u64,
            _ => 32,
        };
        
        let data_offset = (current_pos + alignment - 1) / alignment * alignment;
        
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(GgufContext {
            header: GgufHeader {
                magic,
                version,
                n_tensors,
                n_kv,
            },
            metadata,
            tensors,
            data: mmap,
            data_offset,
        })
    }

    pub fn get_tensor_data(&self, name: &str) -> Result<&[u8]> {
        let tensor = self.tensors.iter().find(|t| t.name == name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found", name))?;
        
        let start = (self.data_offset + tensor.offset) as usize;
        
        let mut sorted_offsets: Vec<u64> = self.tensors.iter().map(|t| t.offset).collect();
        sorted_offsets.sort_unstable();
        
        let end = if let Some(next_offset) = sorted_offsets.iter().find(|&&o| o > tensor.offset) {
            (self.data_offset + next_offset) as usize
        } else {
            self.data.len()
        };

        if start >= self.data.len() || end > self.data.len() || start > end {
            bail!("Tensor data out of bounds for {}", name);
        }

        Ok(&self.data[start..end])
    }

    pub fn get_tensor_type(&self, name: &str) -> Result<u32> {
        let tensor = self.tensors.iter().find(|t| t.name == name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found", name))?;
        Ok(tensor.ggml_type)
    }
}

fn read_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

fn read_value<R: Read>(reader: &mut R, val_type: &GgufType) -> Result<GgufValue> {
    match val_type {
        GgufType::Uint8 => Ok(GgufValue::Uint8(reader.read_u8()?)),
        GgufType::Int8 => Ok(GgufValue::Int8(reader.read_i8()?)),
        GgufType::Uint16 => Ok(GgufValue::Uint16(reader.read_u16::<LittleEndian>()?)),
        GgufType::Int16 => Ok(GgufValue::Int16(reader.read_i16::<LittleEndian>()?)),
        GgufType::Uint32 => Ok(GgufValue::Uint32(reader.read_u32::<LittleEndian>()?)),
        GgufType::Int32 => Ok(GgufValue::Int32(reader.read_i32::<LittleEndian>()?)),
        GgufType::Float32 => Ok(GgufValue::Float32(reader.read_f32::<LittleEndian>()?)),
        GgufType::Bool => Ok(GgufValue::Bool(reader.read_u8()? != 0)),
        GgufType::String => Ok(GgufValue::String(read_string(reader)?)),
        GgufType::Uint64 => Ok(GgufValue::Uint64(reader.read_u64::<LittleEndian>()?)),
        GgufType::Int64 => Ok(GgufValue::Int64(reader.read_i64::<LittleEndian>()?)),
        GgufType::Float64 => Ok(GgufValue::Float64(reader.read_f64::<LittleEndian>()?)),
        GgufType::Array => {
            let sub_type = GgufType::try_from(reader.read_u32::<LittleEndian>()?)?;
            let len = reader.read_u64::<LittleEndian>()? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_value(reader, &sub_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
    }
}
