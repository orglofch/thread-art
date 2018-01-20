extern crate gl;

use std::mem::size_of;
use std::os::raw::c_void;
use std::ptr;

pub struct RenderConfig {
    /// Enables rendering of pegs.
    render_pegs: bool,
}

impl RenderConfig {
    pub fn new() -> RenderConfig {
        RenderConfig::default()
    }

    pub fn set_render_pegs(&mut self, enabled: bool) -> &RenderConfig {
        self.render_pegs = enabled;
        self
    }
}

impl Default for RenderConfig {
    fn default() -> RenderConfig {
        RenderConfig {
            render_pegs: true,
        }
    }
}


/// Vertex information used in rendering the thread art.
#[derive(Clone, Debug)]
pub(crate) struct Vertex {
    /// The `(x, y)` position of the peg in view space (E.g. ((-1, 1), (-1, 1))).
    pos: (f32, f32),

    colour: (f32, f32, f32),
}

impl Vertex {
    pub(crate) fn new(peg: (f32, f32), thread: (f32, f32, f32)) -> Vertex {
        Vertex {
            pos: peg,
            colour: thread,
        }
    }
}

/// Buffer `Vertex` information to the GPU.
pub(crate) unsafe fn buffer_to_gpu(
    vertices: &Vec<Vertex>,
    indices: &Vec<u32>,
    vao: u32,
    vbo: u32,
    ebo: u32,
) {
    gl::BindVertexArray(vao);

    gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
    let size = (vertices.len() * size_of::<Vertex>()) as isize;
    let data = &vertices[0] as *const Vertex as *const c_void;
    gl::BufferData(gl::ARRAY_BUFFER, size, data, gl::STATIC_DRAW);

    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
    let size = (indices.len() * size_of::<u32>()) as isize;
    let data = &indices[0] as *const u32 as *const c_void;
    gl::BufferData(gl::ELEMENT_ARRAY_BUFFER, size, data, gl::STATIC_DRAW);

    let size = size_of::<Vertex>() as i32;
    gl::EnableVertexAttribArray(0);
    gl::VertexAttribPointer(
        0,
        2,
        gl::FLOAT,
        gl::FALSE,
        size,
        offset_of!(Vertex, pos) as *const c_void,
    );
    gl::EnableVertexAttribArray(1);
    gl::VertexAttribPointer(
        1,
        3,
        gl::FLOAT,
        gl::FALSE,
        size,
        offset_of!(Vertex, colour) as *const c_void,
    );

    gl::BindVertexArray(0);
}
