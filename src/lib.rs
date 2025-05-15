use std::sync::{Arc, RwLock};

use engine::resources::{CameraView, Transform};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop},
    window::{Window, WindowId}
};

pub mod engine;
pub mod math;

use crate::engine::resources::{GlobalResources, provider::*};
use crate::engine::window_state::WindowState;


pub struct GraphicalApplication<T: Texture2DProvider + TextureFromImageProvider + BillboardProvider + ShaderProvider + GLTFModelProvider> {
    state: Option<WindowState>,
    resource_provider: T,

    // Wgpu stuff
    instance: Arc<wgpu::Instance>,
    adapter: Arc<wgpu::Adapter>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    resources: Arc<RwLock<GlobalResources>>,
    
    initialized: bool,

    time_passed: f64,
    time_delta: f64,
    last_frame: std::time::Instant,
}

impl<T: Texture2DProvider + TextureFromImageProvider + BillboardProvider + ShaderProvider + GLTFModelProvider> GraphicalApplication<T> {
    pub async fn new(resource_provider: T) -> Self {
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

        GraphicalApplication {
            instance: Arc::new(instance),
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
            state: Option::<WindowState>::default(),
            resources: Arc::<RwLock::<GlobalResources>>::default(),
            resource_provider,
            initialized: false,
            time_passed: 0.0,
            time_delta: 0.0,
            last_frame: std::time::Instant::now(),
        }
    }
}

impl<T: Texture2DProvider + TextureFromImageProvider + BillboardProvider + ShaderProvider + GLTFModelProvider> ApplicationHandler for GraphicalApplication<T> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes();

        let window = Arc::new(
            event_loop
                .create_window(window_attributes)
                .unwrap()
        );

        let state = WindowState::new(window.clone(), self.instance.clone(), self.adapter.clone(), self.device.clone(), self.queue.clone());
        self.state = Some(state);

        if !self.initialized {
            self.resource_provider.load_shader_into_resources("shaders/triangle.wgsl", "triangle".to_string(), &self.device, self.resources.clone());
            self.resource_provider.load_shader_into_resources("shaders/default.wgsl", "default".to_string(), &self.device, self.resources.clone());
            self.resource_provider.load_shader_into_resources("shaders/billboard.wgsl", "billboard".to_string(), &self.device, self.resources.clone());
            self.resource_provider.load_shader_into_resources("shaders/default_srgb.wgsl", "default_srgb".to_string(), &self.device, self.resources.clone());
            self.resource_provider.load_shader_into_resources("shaders/billboard_srgb.wgsl", "billboard_srgb".to_string(), &self.device, self.resources.clone());

            let texture = self.resource_provider.load_texture_from_image("example/test.png", &self.device, &self.queue);
            let billboard = self.resource_provider.create_billboard([0.4, 0.4], Some(texture.clone()), &self.device);
            self.resources.write().unwrap().add_mesh(billboard);

            let mut model_meshes = self.resource_provider.load_model("res/cube_test.glb", &self.device).unwrap();
            let mut mesh = model_meshes.remove(0);
            //println!("Mesh vertices: {:?}", mesh.vertices());
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
                // Time logic
                let time_now = std::time::Instant::now();
                self.time_delta = (time_now - self.last_frame).as_secs_f64();
                self.time_passed += self.time_delta;
                self.last_frame = time_now;

                // Funny logic
                let mut new_camera_view = CameraView::default();
                new_camera_view.eye = cgmath::Point3::new((self.time_passed / 5.0).sin() as f32 * 0.6, 0.15, (self.time_passed / 5.0).cos() as f32 * 0.6);
                state.set_camera_view(new_camera_view);
                
                let resources = self.resources.try_write();
                if let Ok(mut resources) = resources {
                    resources.meshes[1].set_transform(Transform::new([0.0, (self.time_passed).sin() as f32 * 0.2, 0.0], cgmath::Quaternion::new(0.0, 0.0, 0.0, 1.0).into(), [1.0, 1.0, 1.0]));
                }

                // Render
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