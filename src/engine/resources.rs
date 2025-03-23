use std::collections::HashMap;

pub struct GlobalResources {
    pub meshes: Vec<Mesh>,

    pub loaded_shaders: HashMap<String, wgpu::ShaderModule>,
}

impl GlobalResources {
    pub fn new() -> Self {
        GlobalResources::default()
    }

    pub fn add_mesh(&mut self, m: Mesh) {
        self.meshes.push(m)
    }

    pub fn shaders(&self) -> &HashMap<String, wgpu::ShaderModule> {
        &self.loaded_shaders
    }

    pub fn meshes(&self) -> &Vec<Mesh> {
        &self.meshes
    }
}

impl Default for GlobalResources {
    fn default() -> Self {
        Self {
            meshes: Vec::new(),
            loaded_shaders: HashMap::new(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pos: [f32; 3],
    tex_coord: [f32; 2],
    normals: [f32; 3],
}

impl Vertex {
    pub fn new(pos: [f32; 3], tex_coord: [f32; 2], normals: [f32; 3]) -> Self {
        Vertex {
            pos,
            tex_coord,
            normals
        }
    }

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress, // Size of a single element
            step_mode: wgpu::VertexStepMode::Vertex, // per-vertex or per-instance data
            attributes: &[
                wgpu::VertexAttribute { // Vertex position
                    offset: 0,
                    format: wgpu::VertexFormat::Float32x3,
                    shader_location: 0,
                },
                wgpu::VertexAttribute { // UV coordinates
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    format: wgpu::VertexFormat::Float32x2,
                    shader_location: 1,
                },
                wgpu::VertexAttribute { // Normals
                    offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f32; 2]>()) as wgpu::BufferAddress,
                    format: wgpu::VertexFormat::Float32x3,
                    shader_location: 2,
                }
            ],
        }
    }
}

pub struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    //bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,

    shader_name: String,
    texture: Option<TextureHandle>,
    //shader: Arc<wgpu::ShaderModule>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>, vertex_buffer: wgpu::Buffer, index_buffer: wgpu::Buffer, shader_name: String, texture: Option<TextureHandle>) -> Self {
        Mesh {
            vertices,
            indices,

            vertex_buffer,
            index_buffer,

            shader_name,
            texture
        }
    }

    pub fn vertices(&self) -> &Vec<Vertex> {
        &self.vertices
    }

    pub fn indices(&self) -> &Vec<u32> {
        &self.indices
    }

    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertex_buffer
    }

    pub fn index_buffer(&self) -> &wgpu::Buffer {
        &self.index_buffer
    }

    pub fn shader_name(&self) -> &String {
        &self.shader_name
    }

    pub fn texture(&self) -> &Option<TextureHandle> {
        &self.texture
    }
}

pub struct TextureHandle {
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    texture_sampler: wgpu::Sampler,

    bind_group: wgpu::BindGroup,
    bind_layout: wgpu::BindGroupLayout,
}

impl TextureHandle {
    pub fn new(texture: wgpu::Texture, texture_view: wgpu::TextureView, texture_sampler: wgpu::Sampler, bind_group: wgpu::BindGroup, bind_layout: wgpu::BindGroupLayout) -> Self {
        TextureHandle {
            texture,
            texture_view,
            texture_sampler,

            bind_group,
            bind_layout
        }
    }
}

impl TextureHandle {
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn bind_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_layout
    }
}