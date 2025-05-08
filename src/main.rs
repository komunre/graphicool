
use winit::
    event_loop::{ControlFlow, EventLoop}
;

use graphicool::engine::resources::{provider::*};
use graphicool::GraphicalApplication;

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

    let mut app = pollster::block_on(GraphicalApplication::new(DefaultResourceProvider{}));
    event_loop.run_app(&mut app).unwrap();

}
