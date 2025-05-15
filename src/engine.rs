pub mod resources;
pub mod window_state;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

pub fn is_texture_hdr(texture_format: wgpu::TextureFormat) -> bool {
    return texture_format == wgpu::TextureFormat::Rgba32Float
        || texture_format == wgpu::TextureFormat::Rgba16Float;
}
