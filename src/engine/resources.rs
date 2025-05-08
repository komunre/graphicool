use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};
use cgmath::{One, SquareMatrix, Zero};
use wgpu::util::DeviceExt;

use crate::engine::OPENGL_TO_WGPU_MATRIX;

#[repr(C)]
#[derive(Debug)]
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
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct Transform {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3]
}

impl Transform {
    pub fn identity() -> Self {
        Self {
            translation: cgmath::Point3::<f32>::new(0.0, 0.0, 0.0).into(),
            rotation: cgmath::Quaternion::<f32>::one().into(),
            scale: cgmath::Vector3::<f32>::new(1.0, 1.0, 1.0).into()
        }
    }

    pub fn new(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> Self {
        Self {
            translation,
            rotation,
            scale
        }
    }

    pub fn build_model_projection_matrix(&self) -> cgmath::Matrix4::<f32> {
        let translation_mat = cgmath::Matrix4::from_translation(self.translation.into());
        let scale_mat = cgmath::Matrix4::from_nonuniform_scale(self.scale[0], self.scale[1], self.scale[2]);
        // OH GOD HELP ME QUATERNIONS
        let rotation_mat = cgmath::Matrix4::new(
            2.0 * (self.rotation[3] * self.rotation[3] + self.rotation[0] * self.rotation[0]) - 1.0,
            2.0 * (self.rotation[0] * self.rotation[1] + self.rotation[3] * self.rotation[2]),
            2.0 * (self.rotation[0] * self.rotation[2] - self.rotation[3] * self.rotation[1]),
            0.0,

            2.0 * (self.rotation[0] * self.rotation[2]),
            2.0 * (self.rotation[3] * self.rotation[3] + self.rotation[1] * self.rotation[1]) - 1.0,
            2.0 * (self.rotation[1] * self.rotation[2] + self.rotation[3] * self.rotation[0]),
            0.0,

            2.0 * (self.rotation[0] * self.rotation[2] + self.rotation[3] * self.rotation[1]),
            2.0 * (self.rotation[1] * self.rotation[2] - self.rotation[3] * self.rotation[0]),
            2.0 *  (self.rotation[3] * self.rotation[3] + self.rotation[2] * self.rotation[2]) - 1.0,
            0.0,

            0.0,
            0.0,
            0.0,
            1.0
        );

        translation_mat * rotation_mat * scale_mat
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct TransformUniform {
    pub model_proj: [[f32; 4]; 4]
}

impl TransformUniform {
    pub fn new() -> Self {
        Self {
            model_proj: cgmath::Matrix4::identity().into()
        }
    }

    pub fn update_model_projection(&mut self, transform: &Transform) {
        self.model_proj = transform.build_model_projection_matrix().into();
    }
}

#[derive(Debug)]
pub struct TransformHandle {
    buffer: wgpu::Buffer,
}

impl TransformHandle {
    pub fn new(transform: &Transform, device: &wgpu::Device) -> Self {
        let mut transform_uniform = TransformUniform::new();
        transform_uniform.update_model_projection(transform);

        let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera buffer"),
            contents: bytemuck::cast_slice(&[transform_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
        });

        TransformHandle {
            buffer: transform_buffer,
        }   
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn update_handle(&self, transform: &Transform, queue: &wgpu::Queue) {
        let mut transform_uniform = TransformUniform::new();
        transform_uniform.update_model_projection(transform);

        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[transform_uniform]));
    }

    pub fn binding_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None
                },
            ],
            label: Some("transform_uniform_buffer_binding_layout")
        })
    }
}


#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4]
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity().into()
        }
    }

    pub fn update_view_projection(&mut self, camera: &CameraView) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

#[derive(Debug)]
pub struct CameraHandle {
    buffer: wgpu::Buffer,
}

impl CameraHandle {
    pub fn new(camera: &CameraView, device: &wgpu::Device) -> Self {
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_projection(camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
        });

        CameraHandle {
            buffer: camera_buffer,
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn update_handle(&self, camera: &CameraView, queue: &wgpu::Queue) {
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_projection(camera);

        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[camera_uniform]));
    }

    pub fn binding_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None
                }
            ],
            label: Some("camera_uniform_buffer_binding_layout")
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CameraView {
    pub eye: cgmath::Point3<f32>,
    pub target: cgmath::Point3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Default for CameraView {
    fn default() -> Self {
        Self {
            eye: cgmath::Point3::new(0.0, 0.0, 1.0),
            target: cgmath::Point3::new(0.0, 0.0, 0.0),
            up: cgmath::Vector3::new(0.0, 1.0, 0.0),
            aspect: 16.0_f32 / 9.0_f32,
            fovy: 60.0,
            znear: 0.01,
            zfar: 1000.0,
        }
    }
}

impl CameraView {
    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);

        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        OPENGL_TO_WGPU_MATRIX * proj * view
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub tex_coord: [f32; 2],
    pub normals: [f32; 3],
}

// Use Vertex::zeroed() instead
/*impl Default for Vertex {
    fn default() -> Self {
        Vertex { pos: [0.0;3], tex_coord: [0.0;2], normals: [0.0;3] }
    }
}*/

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

#[derive(Debug)]
pub struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    //bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,

    shader_name: String,
    texture: Option<TextureHandle>,
    transform: Transform,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>, vertex_buffer: wgpu::Buffer, index_buffer: wgpu::Buffer, shader_name: String, texture: Option<TextureHandle>, transform: Transform) -> Self {
        Mesh {
            vertices,
            indices,

            vertex_buffer,
            index_buffer,

            shader_name,
            texture,
            transform
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

    pub fn transform(&self) -> &Transform {
        &self.transform
    }

    pub fn set_texture(&mut self, texture: TextureHandle) {
        self.texture = Some(texture);
    }

    pub fn set_transform(&mut self, transform: Transform) {
        self.transform = transform;
    }
}

#[derive(Debug, Clone)]
pub struct TextureHandle {
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    texture_sampler: wgpu::Sampler,

    bind_group: Option<wgpu::BindGroup>,
    bind_layout: Option<wgpu::BindGroupLayout>,
}

impl TextureHandle {
    pub fn new(texture: wgpu::Texture, texture_view: wgpu::TextureView, texture_sampler: wgpu::Sampler, bind_group: Option<wgpu::BindGroup>, bind_layout: Option<wgpu::BindGroupLayout>) -> Self {
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
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub fn textuer_view(&self) -> &wgpu::TextureView {
        &self.texture_view
    }

    pub fn texture_sampler(&self) -> &wgpu::Sampler {
        &self.texture_sampler
    }

    pub fn bind_group(&self) -> &Option<wgpu::BindGroup> {
        &self.bind_group
    }

    pub fn bind_layout(&self) -> &Option<wgpu::BindGroupLayout> {
        &self.bind_layout
    }

    /*pub fn write_to_texture(&self, offset: u64, data: &[u8], queue: wgpu::Queue) {
        queue.write_texture(wgpu::TexelCopyTextureInfo {

        },
        data,
        wgpu::TexelCopyBufferLayout { offset: (), bytes_per_row: (), rows_per_image: () })
    }*/
}


pub mod provider {
    use vecmath::Vector2;
    use std::sync::{Arc, RwLock};
    use crate::engine::resources::{TextureHandle, Mesh, Vertex};
    use crate::math::convert::*;
    use std::fs::File;
    use std::io::Read;
    use std::borrow::Cow;
    use wgpu::util::DeviceExt; // to access create_buffer_init
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::str::FromStr; // for PathBuf from string
    use bytemuck::Zeroable;

    use super::GlobalResources;

    pub struct DefaultResourceProvider {

    }

    // After some consideration it was decided to use &T instead of Arc<T> to save on performance
    // since atomic operations take 20ns + put traffic on CPU interconnect, which
    // will tank multithreaded performance
    // So Arc<wgpu::Device> is, instead, &wgpu::Device

    pub trait BillboardProvider {
        fn create_billboard(&self, dimensions: Vector2<f32>, texture: Option<TextureHandle>, device: &wgpu::Device) -> Mesh;
    }
    
    pub trait Texture2DProvider {
        fn create_texture_2d(&self, dimensions: (u32, u32), usage: wgpu::TextureUsages, tex_dimension: wgpu::TextureDimension, tex_format: wgpu::TextureFormat, device: &wgpu::Device) -> (wgpu::Texture, wgpu::Extent3d);
    }

    pub trait DepthBufferProvider {
        fn create_depth_buffer(&self, dimensions: (u32, u32), device: &wgpu::Device) -> (TextureHandle, wgpu::Extent3d);
    }

    pub trait TextureFromImageProvider {
        fn load_texture_from_image(&self, path: &str, device: &wgpu::Device, queue: &wgpu::Queue) -> TextureHandle;
    }

    pub trait GLTFModelProvider {
        fn load_model(&self, path: &str, device: &wgpu::Device) -> Result<Vec<Mesh>, gltf::Error>;
    }

    pub trait ShaderProvider {
        fn load_shader_into_resources(&self, source: &str, label: String, device: &wgpu::Device, resources: Arc<RwLock<GlobalResources>>);
    }

    impl ShaderProvider for DefaultResourceProvider {
        fn load_shader_into_resources(&self, source: &str, label: String, device: &wgpu::Device, resources: Arc<RwLock<GlobalResources>>) {
            let mut file = File::open(source).unwrap();
            let mut data: Vec<u8> = Vec::new();
            file.read_to_end(&mut data).unwrap();

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&label),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&String::from_utf8(data).unwrap()))
            });
    
            match resources.write() {
                Ok(mut resources) => _ = resources.loaded_shaders.insert(label, shader),
                Err(e) => {
                    println!("[ERR] An error in acquiring resource lock has occured trying to load shader!");
                    println!("[ERR] {:?}", e);
                }
            }
        }
    }

    impl Texture2DProvider for DefaultResourceProvider {
        fn create_texture_2d(&self, dimensions: (u32, u32), usage: wgpu::TextureUsages, tex_dimension: wgpu::TextureDimension, tex_format: wgpu::TextureFormat, device: &wgpu::Device) -> (wgpu::Texture, wgpu::Extent3d) {
            let texture_size = wgpu::Extent3d {
                width: dimensions.0,
                height: dimensions.1,
                depth_or_array_layers: 1,
            };
    
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: texture_size,
                mip_level_count: 1, // Mip levels // TODO: create mip level global settings
                sample_count: 1, // Sample count (for animations, probably?)
                dimension: tex_dimension,
                format: tex_format,
                // TEXTURE_BINDING tells wgpu that we plan to use it as a texture in shaders
                // COPY_DST means that we want to copy data to this texture
                // RENDER_ATTACHMENT tells that we plan to render to texture
                usage: usage,
                
                // This is the same as with the SurfaceConfig. It
                // specifies what texture formats can be used to
                // create TextureViews for this texture. The base
                // texture format (Rgba8UnormSrgb in this case) is
                // always supported. Note that using a different
                // texture format is not supported on the WebGL2
                // backend.
                view_formats: &[],
            });
    
            (texture, texture_size)
        }
    }

    impl TextureFromImageProvider for DefaultResourceProvider {
        fn load_texture_from_image(&self, path: &str, device: &wgpu::Device, queue: &wgpu::Queue) -> TextureHandle {
            let mut file = File::open(path).unwrap();
            let mut texture_bytes: Vec<u8> = Vec::new();
            file.read_to_end(&mut texture_bytes);
    
            let texture_image = image::load_from_memory(&texture_bytes).unwrap();
            let texture_rgba = texture_image.to_rgba8();
    
            use image::GenericImageView;
            let dimensions = texture_image.dimensions();
    
            let texture = self.create_texture_2d(dimensions, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, wgpu::TextureDimension::D2, wgpu::TextureFormat::Rgba8UnormSrgb, device);

            queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture.0,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &texture_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1)
            },
            texture.1,
            );
    
            let texture_view = texture.0.create_view(&wgpu::TextureViewDescriptor::default());
            let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
    
            let texture_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float {filterable: true},
                            view_dimension: wgpu::TextureViewDimension::D2
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    }
                ]
            });
    
            let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &texture_bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_view)
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&texture_sampler)
                    }
                ]
            });
    
            TextureHandle::new(
                texture.0,
                texture_view,
                texture_sampler,
                Some(texture_bind_group),
                Some(texture_bind_layout),
            )
        }
    }

    impl BillboardProvider for DefaultResourceProvider {
        fn create_billboard(&self, dimensions: Vector2<f32>, texture: Option<TextureHandle>, device: &wgpu::Device) -> Mesh {
            let verts = vec![
                Vertex::new(
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0, -1.0]
                ),
                Vertex::new(
                    [dimensions[0], 0.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 0.0, -1.0]
                ),
                Vertex::new(
                    [dimensions[0], dimensions[1], 0.0],
                    [1.0, 0.0],
                    [0.0, 0.0, -1.0]
                ),
                Vertex::new(
                    [0.0, dimensions[1], 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0, -1.0]
                ),
            ];
    
            let indices = vec![
                0,
                1,
                2,
    
                0,
                2,
                3
            ];
    
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX
            });
            
            Mesh::new(
                verts,
                indices,
    
                vertex_buffer,
                index_buffer,
                "billboard".to_string(),
                texture,
                super::Transform::identity(),
            )
        }
    }

    impl DepthBufferProvider for DefaultResourceProvider {
        fn create_depth_buffer(&self, dimensions: (u32, u32), device: &wgpu::Device) -> (TextureHandle, wgpu::Extent3d) {
            let texture = self.create_texture_2d(dimensions, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, wgpu::TextureDimension::D2, wgpu::TextureFormat::Depth32Float, device);

            let view = texture.0.create_view(&wgpu::TextureViewDescriptor::default());
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                compare: Some(wgpu::CompareFunction::LessEqual),
                lod_min_clamp: 0.0,
                lod_max_clamp: 100.0,
                ..Default::default()
            });

            

            (TextureHandle {
                texture: texture.0,
                texture_view: view,
                texture_sampler: sampler,
                bind_group: None,
                bind_layout: None
            }, texture.1)
        }
    }

    impl GLTFModelProvider for DefaultResourceProvider {
        // GLTF is hard.
        fn load_model(&self, path: &str, device: &wgpu::Device) -> Result<Vec<Mesh>, gltf::Error> {
            println!("Loading GLTF model by path of {}", path);

            let gltf_path = Path::new(path);
            let file_seek_path = gltf_path.parent().unwrap_or(Path::new("./"));
            let gltf = gltf::Gltf::open(gltf_path)?;

            let binary_payload = gltf.blob.clone();

            fn access_buffer(view: &gltf::buffer::View, accessor: &gltf::Accessor, buffer_cache: &HashMap<usize, Vec<u8>>, binary_payload: &Option<Vec<u8>>) -> Vec<u8> {
                let buffer = view.buffer();

                let accessor_offset = accessor.offset();
                let view_offset = view.offset();
                let full_offset = accessor_offset + view_offset;

                let view_length = view.length();

                match buffer.source() {
                    gltf::buffer::Source::Uri(uri) => {
                        let cached_data = buffer_cache.get(&buffer.index());
                        match cached_data {
                            Some(v) => {
                                let slice = v[full_offset..full_offset+view_length].to_vec();
                                return slice;
                            }
                            None => return Vec::new()
                        }
                    },
                    gltf::buffer::Source::Bin => {
                        // TODO: return from bin
                        match binary_payload {
                            Some(payload) => {
                                let slice = payload[full_offset..full_offset+view_length].to_vec();
                                //println!("Returning binary slice of: {:?}", slice);
                                return slice;
                            }
                            None => panic!("No binary blob present despite GLTF containing binary data reference")
                        }
                    }
                }
            }

            fn cache_buffer(buffer: &gltf::Buffer, buffer_cache: &mut HashMap<usize, Vec<u8>>, file_seek_path: &Path) {
                let buffer_length = buffer.length();

                let mut buffer_data: Vec<u8> = Vec::with_capacity(buffer_length);

                match buffer.source() {
                    gltf::buffer::Source::Uri(uri) => {

                        let mut path_uri = PathBuf::from_str(uri).unwrap();
                        if path_uri.is_relative() {
                            path_uri = file_seek_path.join(path_uri);
                        }

                        let file = File::open(&path_uri);
                        if let Err(_) = file {
                            buffer_cache.insert(buffer.index(), Vec::with_capacity(buffer_length));
                            return;
                        }
                        let mut file = file.unwrap();
                        let bytes_read = file.read_to_end(&mut buffer_data).unwrap_or_else(|_| {
                            println!("Failed to read referenced by GLTF file: {}", &path_uri.to_str().unwrap_or("NON-UTF-8 PATH NAME"));
                            for _ in 0..buffer_length {
                                buffer_data.push(0);
                            }
                            buffer_length
                        });
                        if bytes_read < buffer_length {
                            println!("Read number of bytes is smaller than buffer length!");
                        }

                        buffer_cache.insert(buffer.index(), buffer_data);
                    },
                    gltf::buffer::Source::Bin => {
                        return;
                    }
                }
            }

            fn is_buffer_cached(buffer: &gltf::Buffer, buffer_cache: &HashMap<usize, Vec<u8>>) -> bool {
                buffer_cache.contains_key(&buffer.index())
            }

            fn read_acessor(accessor: &gltf::Accessor<'_>, file_seek_path: &Path, binary_payload: &Option<Vec<u8>>, buffer_cache: &mut HashMap<usize, Vec<u8>>) -> Vec<u8> {
                // Get accessor parameters
                //let index = accessor.index();
                //let size = accessor.size();

                let accessor_offset = accessor.offset();
                let count = accessor.count();

                let component_type = accessor.data_type();
                let element_type = accessor.dimensions();

                // Sparse accessor
                let sparse = accessor.sparse();
                
                // Get view buffer
                let view_buffer = accessor.view();

                match view_buffer {
                    Some(view) => {
                        // Get buffer view parameters
                        let offset = view.offset();
                        let length = view.length();
                        let stride = view.stride();
                        let target = view.target();
                        
                        let buffer = view.buffer();
                        let buffer_length: usize = buffer.length();
                        if buffer_length < length {
                            panic!("!!! buffer length is smaller than specified view length !!!")
                        }

                        let full_offset = offset+accessor_offset;

                        let mut slice: &[u8];

                        if !is_buffer_cached(&buffer, buffer_cache) {
                            cache_buffer(&buffer, buffer_cache, file_seek_path);
                        }

                        let buffer_data = access_buffer(&view, accessor, buffer_cache, binary_payload);

                        buffer_data
                    }
                    None => panic!("No view buffer in accessor...")
                }
            }

            fn process_node(node: &gltf::Node, file_seek_path: &Path, binary_payload: &Option<Vec<u8>>, buffer_cache: &mut HashMap<usize, Vec<u8>>, device: &wgpu::Device, mesh_list: &mut Vec<Mesh>) {
                let transform = node.transform();
                // transform.matrix() [[f32;4];4] is available instead of decomposed
                // translation, rotation, scale
                let decomposed_transform = transform.decomposed();

                let mut vertex_list = Vec::<Vertex>::new();
                let mut index_list = Vec::<u32>::new();

                // Node may not have a mesh.
                match node.mesh() {
                    Some(m) => {
                        let primitive_count = m.primitives().count();
                        
                        // Primitives are submodels.
                        for primitive in m.primitives() {
                            for attribute in primitive.attributes() {
                                // Get accessor parameters
                                let index = attribute.1.index();
                                let size = attribute.1.size();

                                let accessor_offset = attribute.1.offset();
                                let count = attribute.1.count();

                                let component_type = attribute.1.data_type();
                                let element_type = attribute.1.dimensions();

                                // Sparse accessor
                                let sparse = attribute.1.sparse();
                                
                                // Get view buffer
                                let view_buffer = attribute.1.view();
                                match view_buffer {
                                    Some(view) => {
                                        // Get buffer view parameters
                                        let offset = view.offset();
                                        let length = view.length();
                                        let stride = view.stride();
                                        let target = view.target();
                                        
                                        let mut accessor_data = read_acessor(&attribute.1, file_seek_path, binary_payload, buffer_cache);
                                        let mut aligned_data = Vec::with_capacity(accessor_data.len());

                                        let bytes_per_element: usize;
                                        match component_type {
                                            gltf::accessor::DataType::F32 | gltf::accessor::DataType::U32 => bytes_per_element = 4,
                                            gltf::accessor::DataType::I16 | gltf::accessor::DataType::U16 => bytes_per_element = 2,
                                            gltf::accessor::DataType::I8  | gltf::accessor::DataType::U8 => bytes_per_element = 1,
                                        }
                                        
                                        match stride {
                                            Some(stride) => {
                                                //println!("Stride of {}", stride);
                                                let mut byte_index: usize = 0;
                                                while byte_index < accessor_data.len() {
                                                    for i in 0..bytes_per_element {
                                                        aligned_data.push(accessor_data[byte_index + i])
                                                    }
                                                    byte_index += stride;
                                                }
                                            }
                                            None => {
                                                //println!("No stride");
                                                aligned_data = accessor_data.to_vec()
                                            }
                                        }
                                        //println!("Aligned data total elements: {}", aligned_data.len());
                                    
                                        // Semantics
                                        match attribute.0 {
                                            gltf::Semantic::Positions => {
                                                println!("Adding positions...");
                                                match component_type {
                                                    // Positions are always floats
                                                    gltf::accessor::DataType::F32 => {
                                                        let transmuted = u8vec_to_f32vec(aligned_data);
                                                        //println!("Positions vector: {:?}", transmuted);
                                                        if transmuted.len() < count * 3 {
                                                            panic!("Not enough values in position view buffer!");
                                                        }
                                                    for i in 0..count {
                                                            let x = transmuted[i * 3];
                                                            let y = transmuted[i * 3 + 1];
                                                            let z = transmuted[i * 3 + 2];
                                                            let xyz = [x, y, z];

                                                            if vertex_list.len() <= i {
                                                                vertex_list.push(Vertex::zeroed());
                                                            }
                                                            vertex_list.get_mut(i).unwrap().pos = xyz;
                                                    }
                                                    }
                                                    _ => panic!("GLTF position accessor is something other than f32. That's against GLTF 2.0 specification.")
                                                }
                                                
                                            },
                                            gltf::Semantic::Normals => {
                                                println!("Adding normals...");
                                                match component_type {
                                                    // Normals are always floats
                                                    gltf::accessor::DataType::F32 => {
                                                        let transmuted = u8vec_to_f32vec(aligned_data);
                                                        if transmuted.len() < count * 3 {
                                                            panic!("Not enough values in position view buffer!");
                                                        }
                                                    for i in 0..count {
                                                            let x = transmuted[i * 3];
                                                            let y = transmuted[i * 3 + 1];
                                                            let z = transmuted[i * 3 + 2];
                                                            let xyz: [f32; 3] = [x, y, z];

                                                            if vertex_list.len() < i {
                                                                vertex_list.push(Vertex::zeroed());
                                                            }
                                                            vertex_list.get_mut(i).unwrap().normals = xyz;
                                                    }
                                                    }
                                                    _ => panic!("GLTF normal accessor is something other than f32. That's against GLTF 2.0 specification.")
                                                }
                                            },
                                            gltf::Semantic::TexCoords(_v) => {
                                                println!("Adding texcoord...");
                                                // It can be floats, unsigned byte normalized and unsigned short normalized
                                                match component_type {
                                                    gltf::accessor::DataType::F32 => {
                                                        let transmuted = u8vec_to_f32vec(aligned_data);
                                                        if transmuted.len() < count * 2 {
                                                            panic!("Not enough values in texcoord view buffer!")
                                                        }
                                                        for i in 0..count {
                                                            let x = transmuted[i * 2];
                                                            let y = transmuted[i * 2 + 1];
                                                            let xy = [x, y];

                                                            if vertex_list.len() < i {
                                                                vertex_list.push(Vertex::zeroed());
                                                            }
                                                            vertex_list.get_mut(i).unwrap().tex_coord = xy;
                                                        }
                                                    }
                                                    gltf::accessor::DataType::U16 => {
                                                        let transmuted = u8vec_to_u16vec(aligned_data);
                                                        if transmuted.len() < count * 2 {
                                                            panic!("Not enough values in texcoord view buffer!")
                                                        }
                                                        for i in 0..count {
                                                            let x = transmuted[i * 2];
                                                            let y = transmuted[i * 2 + 1];
                                                            let xy = [x as f32, y as f32];
                                                            
                                                            if vertex_list.len() < i {
                                                                vertex_list.push(Vertex::zeroed());
                                                            }
                                                            vertex_list.get_mut(i).unwrap().tex_coord = xy;
                                                        }
                                                    }
                                                    gltf::accessor::DataType::U8 => {
                                                        // aligned_data is already u8.
                                                        if aligned_data.len() < count * 2 {
                                                            panic!("Not enough values in texcoord view buffer!")
                                                        }
                                                        for i in 0..count {
                                                            let x = aligned_data[i * 2];
                                                            let y = aligned_data[i * 2 + 1];
                                                            let xy = [x as f32, y as f32];
                                                            
                                                            if vertex_list.len() < i {
                                                                vertex_list.push(Vertex::zeroed());
                                                            }
                                                            vertex_list.get_mut(i).unwrap().tex_coord = xy;
                                                        }
                                                    }
                                                    _ => panic!("GLTF texcoord accessor is something other than f32, u16 or u8. That's against GLTF 2.0 specification.")
                                                }
                                            },
                                            gltf::Semantic::Weights(_v) => {
                                                // Ignore
                                                // TODO: read
                                            },
                                            gltf::Semantic::Joints(_v) => {
                                                // Ignore
                                                // TODO: read
                                            },
                                            // Vertex colors are not planned to be supported
                                            gltf::Semantic::Colors(_) => (),
                                            _ => (),
                                        }
                                    }
                                    None => ()
                                }

                                let min = attribute.1.min();
                                let max = attribute.1.max();
                            }
                            let bounding_box = primitive.bounding_box();
                            let indices = primitive.indices();
                            match indices {
                                Some(accessor) => {
                                    let view = accessor.view();
                                    if let None = view {
                                        panic!("FUCK!!!! No indices view!")
                                    }
                                    let view = view.unwrap();

                                    let buffer = view.buffer();

                                    if !is_buffer_cached(&buffer, buffer_cache) {
                                        cache_buffer(&buffer, buffer_cache, file_seek_path);
                                    }

                                    let indices_data = read_acessor(&accessor, file_seek_path, binary_payload, buffer_cache);

                                    let component_type = accessor.data_type();

                                    let stride = view.stride();
                                    let bytes_per_element: usize;
                                    match component_type {
                                        gltf::accessor::DataType::F32 | gltf::accessor::DataType::U32 => bytes_per_element = 4,
                                        gltf::accessor::DataType::I16 | gltf::accessor::DataType::U16 => bytes_per_element = 2,
                                        gltf::accessor::DataType::I8  | gltf::accessor::DataType::U8 => bytes_per_element = 1,
                                    }

                                    let mut aligned_data = Vec::with_capacity(indices_data.len());

                                    match stride {
                                        Some(stride) => {
                                            let mut byte_index: usize = 0;
                                            while byte_index < indices_data.len() {
                                                for i in 0..bytes_per_element {
                                                    aligned_data.push(indices_data[byte_index + i])
                                                }
                                                byte_index += stride;
                                            }
                                        }
                                        None => aligned_data = indices_data.to_vec(),
                                    }

                                    match component_type {
                                        gltf::accessor::DataType::U32  => {
                                            let aligned_data = u8vec_to_u32vec(aligned_data);
                                            for v in aligned_data {
                                                index_list.push(v);
                                            }
                                        }
                                        gltf::accessor::DataType::U16 | gltf::accessor::DataType::I16 => {
                                            let aligned_data = u8vec_to_u16vec(aligned_data);
                                            for v in aligned_data {
                                                index_list.push(v.into());
                                            }
                                        }
                                        gltf::accessor::DataType::U8 | gltf::accessor::DataType::I8 => {
                                            for v in aligned_data {
                                                index_list.push(v.into());
                                            }
                                        }
                                        _ => ()
                                    }
                                },
                                None => {
                                    for i in 0..primitive.attributes().count() {
                                        index_list.push(i as u32);
                                    }
                                }
                            }
                        }

                        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: bytemuck::cast_slice(&vertex_list),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: bytemuck::cast_slice(&index_list),
                            usage: wgpu::BufferUsages::INDEX
                        });

                        let mesh = Mesh::new(vertex_list, index_list, vertex_buffer, index_buffer, "default".to_string(), None, super::Transform::identity());
                        mesh_list.push(mesh);
                    },
                    None => (),
                }

                // Full tree processing
                // Recursion
                for child in node.children() {
                    process_node(&child, file_seek_path, binary_payload, buffer_cache, device, mesh_list);
                }
            }

            let mut buffer_cache = HashMap::<usize, Vec<u8>>::new();

            let mut mesh_list = Vec::<Mesh>::new();

            // Scenes
            for scene in gltf.scenes() {
                // Nodes - i.e. meshes, cameras, lights, etc.
                for node in scene.nodes() {
                    process_node(&node, file_seek_path, &binary_payload, &mut buffer_cache, device, &mut mesh_list);
                }
            }

            // let buffer_layout = Vertex::layout();

            Ok(mesh_list)
        }
    }
}