use crate::engine::{
    is_texture_hdr,
    resources::{GlobalResources, Mesh, Transform, TransformHandle, Vertex},
};
use std::sync::{Arc, RwLock};
use wgpu::{Operations, RenderPassDepthStencilAttachment};
use winit::window::Window;

use super::resources::{
    provider::{DefaultResourceProvider, DepthBufferProvider},
    CameraHandle, CameraView, TextureHandle,
};

pub struct WindowState {
    instance: Arc<wgpu::Instance>,
    adapter: Arc<wgpu::Adapter>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    window: Arc<Window>,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,

    depth_buffer: TextureHandle,
    camera_view: CameraView,
    camera_handle: CameraHandle,
    camera_bind_group: wgpu::BindGroup,

    is_hdr: bool,
}

impl WindowState {
    pub fn new(
        window: Arc<Window>,
        instance: Arc<wgpu::Instance>,
        adapter: Arc<wgpu::Adapter>,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Self {
        let size = window.inner_size();

        let surface = instance.create_surface(window.clone()).unwrap();
        let cap = surface.get_capabilities(&adapter);
        let surface_format = match cap
            .formats
            .iter()
            .find(|f| return **f == wgpu::TextureFormat::Rgba32Float)
        {
            Some(f) => *f,
            None => match cap
                .formats
                .iter()
                .find(|f2| return **f2 == wgpu::TextureFormat::Rgba16Float)
            {
                Some(f2) => *f2,
                None => cap.formats[0],
            },
        };
        println!(
            "Selected {:?} surface format for {} window",
            surface_format, "unspecified"
        );

        // TODO: Replace with generic
        let depth_buffer = DefaultResourceProvider {}
            .create_depth_buffer((size.width, size.height), &device)
            .0; // PLACEHOLDER!

        let mut default_camera_view = CameraView::default();
        default_camera_view.aspect = size.width as f32 / size.height as f32;
        //default_camera_view.eye = cgmath::Point3::<f32>::new(3.0, 3.0, 2.0);
        let camera_handle = CameraHandle::new(&default_camera_view, &device);

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &CameraHandle::binding_layout(&device),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_handle.buffer().as_entire_binding(),
            }],
        });

        let state = Self {
            instance,
            adapter,
            window,
            device,
            queue,
            size,
            surface,
            surface_format,

            depth_buffer,
            camera_view: default_camera_view,
            camera_handle,
            camera_bind_group,

            is_hdr: is_texture_hdr(surface_format),
        };

        state.configure_surface();

        state
    }

    pub fn configure_surface(&self) {
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

    pub fn get_window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;

        self.configure_surface();

        // TODO: Replace with generic (if needed)
        self.set_depth_buffer(&DefaultResourceProvider {});
    }

    pub fn set_depth_buffer<T: DepthBufferProvider>(&mut self, provider: &T) {
        self.depth_buffer = provider
            .create_depth_buffer((self.size.width, self.size.height), &self.device)
            .0
    }

    pub fn set_camera_view(&mut self, camera_view: CameraView) {
        self.camera_view = camera_view;
        self.camera_handle
            .update_handle(&self.camera_view, &self.queue);
    }

    pub fn render(&mut self, resources: Arc<RwLock<GlobalResources>>) {
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
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view, // Set render pass view to the texture view we just created
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.6,
                        g: 0.2,
                        b: 1.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: self.depth_buffer.textuer_view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Camera view projection matrix bind
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

        // Draw calls here
        let resource_read = resources.read().unwrap();
        for mesh in resource_read.meshes() {
            let name = mesh.shader_name();
            let shader: &wgpu::ShaderModule = match resource_read.shaders().get(name) {
                Some(shader) => shader,
                None => resource_read
                    .shaders()
                    .get("default")
                    .expect("Fallback shader expected"),
            };
            self.render_mesh(shader, &mut render_pass, mesh);
        }
        //}
        // End the renderpass
        drop(render_pass);

        // Submit the command in the queue to execute
        self.queue.submit([encoder.finish()]); // Submits the commands
        self.window.pre_present_notify();
        surface_texture.present(); // Puts texture on screen
    }

    fn render_mesh(
        &mut self,
        shader: &wgpu::ShaderModule,
        render_pass: &mut wgpu::RenderPass,
        mesh: &Mesh,
    ) {
        fn default_bind_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &CameraHandle::binding_layout(device),
                    &TransformHandle::binding_layout(device),
                ],
                push_constant_ranges: &[],
            })
        }

        let pipeline_layout = match mesh.texture() {
            Some(texture) => match texture.bind_layout() {
                Some(layout) => {
                    self.device
                        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[
                                &CameraHandle::binding_layout(&self.device),
                                &TransformHandle::binding_layout(&self.device),
                                layout,
                            ],
                            push_constant_ranges: &[],
                        })
                }
                None => default_bind_layout(&self.device),
            },
            None => default_bind_layout(&self.device),
        };

        let swapchain_capabilities = self.surface.get_capabilities(&self.adapter);
        let swapchain_format = match swapchain_capabilities
            .formats
            .iter()
            .find(|f| return **f == wgpu::TextureFormat::Rgba32Float)
        {
            Some(f) => *f,
            None => match swapchain_capabilities
                .formats
                .iter()
                .find(|f2| return **f2 == wgpu::TextureFormat::Rgba16Float)
            {
                Some(f2) => *f2,
                None => swapchain_capabilities.formats[0],
            },
        };
        if swapchain_format != self.surface_format {
            panic!("Swapchain and surface format mismatch!");
        }

        let render_pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(swapchain_format.into())],
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
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float, // TODO: replace with actual texture format from texture,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(), // Texture multisample (MSAA)
                multiview: None,
                cache: None,
            });

        render_pass.set_pipeline(&render_pipeline);

        // Rendering
        // Vertices and indices
        render_pass.set_vertex_buffer(0, mesh.vertex_buffer().slice(..));
        render_pass.set_index_buffer(mesh.index_buffer().slice(..), wgpu::IndexFormat::Uint32);

        // Model matrix bind
        let transform_handle = TransformHandle::new(mesh.transform(), &self.device);

        let transform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transform_bind_group"),
            layout: &TransformHandle::binding_layout(&self.device),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: transform_handle.buffer().as_entire_binding(),
            }],
        });

        render_pass.set_bind_group(1, Some(&transform_bind_group), &[]);

        // Texture binds
        match mesh.texture() {
            Some(texture) => {
                render_pass.set_bind_group(2, texture.bind_group(), &[]);
            }
            None => (),
        }

        render_pass.draw_indexed(0..mesh.indices().len() as u32, 0, 0..1);
        //render_pass.draw(0..3, 0..1);
    }
}
