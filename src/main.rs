use std::{fs::File, io::Read, str::FromStr, sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard}, borrow::Cow, collections::HashMap};

use wgpu::{RenderPass, util::DeviceExt};
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
        let Ok(name) = String::from_str("billboard");
        let shader = resources.read().unwrap().shaders().get(&name).unwrap().clone();
        for mesh in resources.read().unwrap().meshes() {
            self.render_mesh(&shader, &mut renderpass, mesh);
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
                topology: wgpu::PrimitiveTopology::TriangleStrip, // wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(wgpu::IndexFormat::Uint32), // TODO: A toggle between u16 and u32 // u32 indices list for extended indice count maximum (more memory usage)
                front_face: wgpu::FrontFace::Ccw, // Counter-clockwise front
                cull_mode: None, // Some(wgpu::Face::Back), // Cull back
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
        
        render_pass.draw_indexed(0..6, 0, 0..1);
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
    pub fn load_model(&mut self, path: &str) -> Result<Mesh, gltf::Error> {
        //self.state.as_ref().unwrap().device.create_bind_group()

        let gltf = gltf::Gltf::open(path)?;

        let verts: Vec<Vertex> = Vec::new();

        // Scenes
        for scene in gltf.scenes() {
            // Nodes - i.e. meshes, cameras, lights, etc.
            for node in scene.nodes() {
                // Node may not have a mesh.
                match node.mesh() {
                    Some(m) => {
                        // Primitives are triangles, lines, etc.
                        for primitive in m.primitives() {
                            // Primitive Attributes contain many vertex attributes like positions, normals, UVs, etc.
                            for attribute in primitive.attributes() {
                                let index = attribute.1.index();
                                let size = attribute.1.size();
                                let view_buffer = attribute.1.view();
                                let offset = attribute.1.offset();
                                let count = attribute.1.count();

                                let data_type = attribute.1.data_type();
                                let dimensions = attribute.1.dimensions();

                                match attribute.0 {
                                    gltf::Semantic::Positions => {
                                        
                                        //verts.push(attribute.1);
                                    },
                                    gltf::Semantic::Normals => {

                                    },
                                    gltf::Semantic::TexCoords(v) => {

                                    },
                                    gltf::Semantic::Weights(v) => {

                                    },
                                    _ => {}
                                }
                            }
                            let bounding_box = primitive.bounding_box();
                            let indices = primitive.indices();
                        }
                    },
                    None => (),
                }
            }
        }

        let device = &self.state.as_ref().expect("Can not load model before state is initialized").device;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[0; 1],
            usage: wgpu::BufferUsages::INDEX,
        });

        // let buffer_layout = Vertex::layout();

        Ok(Mesh::new(
            verts,
            Vec::new(),
            vertex_buffer,
            index_buffer,

            "default".to_string(),
            None
        ))
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
