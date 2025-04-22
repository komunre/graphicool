use std::{borrow::Cow, collections::HashMap, default, fs::File, io::{BufReader, Read}, mem::transmute, path::{Path, PathBuf}, rc::Rc, str::FromStr, sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard}};

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
pub mod math;

use crate::engine::resources::{GlobalResources, Mesh, Vertex, TextureHandle, provider::*};
use crate::engine::window_state::WindowState;
use crate::math::convert::*;

struct App<T: Texture2DProvider + TextureFromImageProvider + BillboardProvider + ShaderProvider + GLTFModelProvider> {
    state: Option<WindowState>,
    resource_provider: T,

    // Wgpu stuff
    instance: Arc<wgpu::Instance>,
    adapter: Arc<wgpu::Adapter>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    resources: Arc<RwLock<GlobalResources>>,
    
    initialized: bool,
}

impl<T: Texture2DProvider + TextureFromImageProvider + BillboardProvider + ShaderProvider + GLTFModelProvider> App<T> {
    async fn new(resource_provider: T) -> Self {
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

        App {
            instance: Arc::new(instance),
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
            state: Option::<WindowState>::default(),
            resources: Arc::<RwLock::<GlobalResources>>::default(),
            resource_provider,
            initialized: false,
        }
    }
}

impl<T: Texture2DProvider + TextureFromImageProvider + BillboardProvider + ShaderProvider + GLTFModelProvider> ApplicationHandler for App<T> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut window_attributes = Window::default_attributes();
        //window_attributes.transparent = true;
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

            let texture = self.resource_provider.load_texture_from_image("example/test.png", &self.device, &self.queue);
            let billboard = self.resource_provider.create_billboard([0.4, 0.4], Some(texture), &self.device);
            self.resources.write().unwrap().add_mesh(billboard);

            let mut model_meshes = self.resource_provider.load_model("res/cube_test.glb", &self.device).unwrap();
            let mut mesh = model_meshes.remove(0);
            //println!("Mesh vertices: {:?}", mesh.vertices());
            let texture = self.resource_provider.load_texture_from_image("example/test.png", &self.device, &self.queue);
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

    let mut app = pollster::block_on(App::new(DefaultResourceProvider{}));
    event_loop.run_app(&mut app).unwrap();

}
