use std::{borrow::Cow, collections::HashMap, fs::File, io::{BufReader, Read}, mem::transmute, path::{Path, PathBuf}, rc::Rc, str::FromStr, sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard}};

use bytemuck::Zeroable;
use wgpu::{naga::proc::index, util::{align_to, DeviceExt}, RenderPass};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId}
};
use vecmath::*;

pub mod engine;

use crate::engine::resources::{GlobalResources, Mesh, Vertex, TextureHandle};

struct WindowState {
    instance: wgpu::Instance,

    adapter: wgpu::Adapter,
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
}

impl WindowState {
    async fn new(window: Arc<Window>) -> Self {
        // InstanceDescriptor is used to specify backends, backend options and flags to be used in the instance
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = instance.
            request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance, // Request high performance adapter
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor::default(),
                None,
            )
            .await
            .unwrap();

        let size = window.inner_size();

        let surface = instance.create_surface(window.clone()).unwrap();
        let cap = surface.get_capabilities(&adapter);
        let surface_format = cap.formats[0];

        let mut state = Self {
            instance,
            adapter,
            window,
            device,
            queue,
            size,
            surface,
            surface_format,
        };
        
        state.configure_surface();

        state
    }

    fn configure_surface(&self) {
        let surface_config = wgpu::SurfaceConfiguration {
            // https://gpuweb.github.io/gpuweb/#texture-usage
            // Relevant to the average rendering usages:
            // RENDER_ATTACHMENT - color/depth/stencil or other attachment in a render pass
            // TEXTURE_BINDING - can be used as a sampled texture in a shader
            // STORAGE_BINDING - can be used as a storage texture in a shader
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, // attachment in a render pass
            format: self.surface_format,
            // Request compatibility with the sRGB-format texture view we're going to create later
            view_formats: vec![self.surface_format.add_srgb_suffix()],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: self.size.width,
            height: self.size.height,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::AutoNoVsync, // Cab be used to set Vsync / NoVsync mode
        };

        // Surface config can be set at any time.
        // Use it to set window-related rendering options!!!
        // Should be used to set resolution!
        // Can be used to set Vsync / NoVsync Mode!!!
        // Also can be used for changing view formats and alpha modes.
        self.surface.configure(&self.device, &surface_config); 
    }

    fn get_window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        
        self.configure_surface();
    }

    fn render(&mut self, resources: Arc<RwLock<GlobalResources>>) {
        // Get surface texture (for texture view)
        let surface_texture = self
            .surface
            .get_current_texture()
            .expect("Failed to acquire next swapchain texture");

        // Create texture view
        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.surface_format.add_srgb_suffix()),
                ..Default::default()
            });

        // Create command encoder for our render pass
        let mut encoder = self.device.create_command_encoder(&Default::default());
        // Create render pass that will clear the screen
        let mut renderpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view, // Set render pass view to the texture view we just created
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.6, g: 0.2, b: 1.0, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                }
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Draw calls here
        let resource_read =  resources.read().unwrap();
        for mesh in resource_read.meshes() {
            let name = mesh.shader_name();
            let shader: &wgpu::ShaderModule = resource_read.shaders().get(name).unwrap();
            self.render_mesh(shader, &mut renderpass, mesh);
        }
        //}
        // End the renderpass
        drop(renderpass);

        // Submit the command in the queue to execute
        self.queue.submit([encoder.finish()]); // Submits the commands
        self.window.pre_present_notify();
        surface_texture.present(); // Puts texture on screen
    }

    fn render_mesh(&mut self, shader: &wgpu::ShaderModule, render_pass: &mut RenderPass, mesh: &Mesh) {
        let pipeline_layout = match mesh.texture() {
            Some(texture) => self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[texture.bind_layout()],
                push_constant_ranges: &[]
            }),
            None => self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[],
                push_constant_ranges: &[]
            })
        };

        let swapchain_capabilities = self.surface.get_capabilities(&self.adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        let render_pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { 
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    Vertex::layout()
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(swapchain_format.into())]
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None, // TODO: A toggle between u16 and u32 // u32 indices list for extended indice count maximum (more memory usage)
                front_face: wgpu::FrontFace::Ccw, // Counter-clockwise front
                cull_mode: Some(wgpu::Face::Back), // Some(wgpu::Face::Back), // Cull back --- None // Cull nothing
                unclipped_depth: false, // When set to false clips depth to 0-1. Unclipped depth requires `Features::DEPTH_CLIP_CONTROL` to be enabled
                polygon_mode: wgpu::PolygonMode::Fill, // PolygonMode::Fill means rasterization
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(), // Texture multisample (MSAA)
            multiview: None,
            cache: None,
        });

        render_pass.set_pipeline(&render_pipeline);

        // Rendering
        // Vertices and indices
        render_pass.set_vertex_buffer(0, mesh.vertex_buffer().slice(..));
        render_pass.set_index_buffer(mesh.index_buffer().slice(..), wgpu::IndexFormat::Uint32);
        // Texture binds
        match mesh.texture() {
            Some(texture) => {
                render_pass.set_bind_group(0, texture.bind_group(), &[]);
            }
            None => ()
        }
        
        render_pass.draw_indexed(0..mesh.indices().len() as u32, 0, 0..1);
        //render_pass.draw(0..3, 0..1);
    }
}

#[derive(Default)]
struct App {
    state: Option<WindowState>,

    resources: Arc<RwLock<GlobalResources>>,
    
    initialized: bool,
}

/*impl App {
    pub fn new() -> Self {

        app
    }
}*/

fn u8vec_to_f32vec(input: Vec<u8>) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len() / 4);
    let mut i = 0;
    while i < input.len() {
        let arr = [input[i], input[i + 1], input[i + 2], input[i + 3]];
        let v = f32::from_le_bytes(arr);
        i += 4;
        output.push(v);
    }
    return output;
}

fn u8vec_to_u32vec(input: Vec<u8>) -> Vec<u32> {
    let mut output = Vec::with_capacity(input.len() / 4);
    let mut i = 0;
    while i < input.len() {
        let arr = [input[i], input[i + 1], input[i + 2], input[i + 3]];
        let v = u32::from_le_bytes(arr);
        i += 4;
        output.push(v);
    }
    return output;
}

fn u8vec_to_u16vec(input: Vec<u8>) -> Vec<u16> {
    let mut output = Vec::with_capacity(input.len() / 2);
    let mut i = 0;
    while i < input.len() {
        let arr = [input[i], input[i + 1]];
        let v = u16::from_le_bytes(arr);
        i += 2;
        output.push(v);
    }
    return output;
}

impl App {
    pub fn create_billboard(&mut self, dimensions: Vector2<f32>, texture: Option<TextureHandle>) -> Mesh {
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

        let device = &self.state.as_ref().expect("Can not create billboard before state is initialized").device;
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
        )
    }

    pub fn create_texture_bind(&mut self) {

    }

    pub fn create_texture_2d(&mut self, dimensions: (u32, u32)) -> (wgpu::Texture, wgpu::Extent3d) {
        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let device = &self.state.as_ref().unwrap().device;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: texture_size,
            mip_level_count: 1, // Mip levels // TODO: create mip level global settings
            sample_count: 1, // Sample count (for animations, probably?)
            dimension: wgpu::TextureDimension::D2,
            // srgb
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            // TEXTURE_BINDING tells wgpu that we plan to use it as a texture in shaders
            // COPY_DST means that we want to copy data to this texture
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            
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

    pub fn load_texture_from_image(&mut self, path: &str) -> TextureHandle {
        let device = &self.state.as_ref().unwrap().device;

        let mut file = File::open(path).unwrap();
        let mut texture_bytes: Vec<u8> = Vec::new();
        file.read_to_end(&mut texture_bytes);

        let texture_image = image::load_from_memory(&texture_bytes).unwrap();
        let texture_rgba = texture_image.to_rgba8();

        use image::GenericImageView;
        let dimensions = texture_image.dimensions();

        let _ = device; // Free mutable borrow to use immutable borrow on the next line
        let texture = self.create_texture_2d(dimensions);

        let device = &self.state.as_ref().unwrap().device;

        let queue = &self.state.as_ref().unwrap().queue;
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
            mipmap_filter: wgpu::FilterMode::Nearest,
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
            texture_bind_group,
            texture_bind_layout,
        )
    }
    
    // GLTF is hard.
    pub fn load_model(&mut self, path: &str) -> Result<Vec<Mesh>, gltf::Error> {
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

                    let mesh = Mesh::new(vertex_list, index_list, vertex_buffer, index_buffer, "default".to_string(), None);
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

        let device = &self.state.as_ref().expect("Can not create billboard before state is initialized").device;
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

    pub fn load_shader(&mut self, source: &str, label: String) {
        let mut file = File::open(source).unwrap();
        let mut data: Vec<u8> = Vec::new();
        file.read_to_end(&mut data).unwrap();
        
        let device = &self.state.as_ref().expect("Can not add module before state is initialized").device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&label),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&String::from_utf8(data).unwrap()))
        });

        match self.resources.write() {
            Ok(mut resources) => _ = resources.loaded_shaders.insert(label, shader),
            Err(e) => {
                println!("[ERR] An error in acquiring resource lock has occured trying to load shader!");
                println!("[ERR] {:?}", e);
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut window_attributes = Window::default_attributes();
        //window_attributes.transparent = true;
        let window = Arc::new(
            event_loop
                .create_window(window_attributes)
                .unwrap()
        );

        let state = pollster::block_on(WindowState::new(window.clone()));
        self.state = Some(state);

        if !self.initialized {
            self.load_shader("shaders/triangle.wgsl", "triangle".to_string());
            self.load_shader("shaders/default.wgsl", "default".to_string());
            self.load_shader("shaders/billboard.wgsl", "billboard".to_string());

            let texture = self.load_texture_from_image("example/test.png");
            let billboard = self.create_billboard([0.4, 0.4], Some(texture));
            self.resources.write().unwrap().add_mesh(billboard);

            let mut model_meshes = self.load_model("res/test_model2.glb").unwrap();
            let mut mesh = model_meshes.remove(0);
            //println!("Mesh vertices: {:?}", mesh.vertices());
            let texture = self.load_texture_from_image("example/test.png");
            mesh.set_texture(texture);
            self.resources.write().unwrap().add_mesh(mesh);

            self.initialized = true;
        }

        window.request_redraw();
    }

    fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            window_id: WindowId,
            event: WindowEvent,
        ) {
        let state = self.state.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                println!("Window close requested. Quitting.");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                state.render(self.resources.clone());

                // Emits new redraw request after finishing last one.
                state.get_window().request_redraw();
            }
            WindowEvent::Resized(size) => {
                state.resize(size);
            }
            _ => ()
        }
    }
}

fn main() {
    println!("Hello, world!");

    // wgpu uses `log` for all of our logging, so we initialize a logger with the `env_logger` crate.
    //
    // To change the log level, set the `RUST_LOG` environment variable. See the `env_logger`
    // documentation for more information.
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();

    // When the current loop iteration finishes, immediately begin a new
    // iteration regardless of whether or not new events are available to
    // process. Preferred for applications that want to render as fast as
    // possible, like games.
    event_loop.set_control_flow(ControlFlow::Poll);
    // ControlFlow::Wait will suspend the thread until
    // another event arrives. Helps to keep CPU utilization low if nothing
    // is happening

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();

}
