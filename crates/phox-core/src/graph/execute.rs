use super::*;
use phox_rhi::{BufferBarrier, ColorTarget, DepthTarget, RenderingInfo, SubmitTicket, TextureBarrier};
use std::fmt::Write;

impl<'g> GraphBuilder<'g> {
    /// Records every live pass (barriers, then its commands, inside rendering when it has targets) into one
    /// command list and submits it once. Needs a successful [`GraphBuilder::compile`].
    pub fn execute(&mut self) -> GraphResult<SubmitTicket> {
        if !self.data.compiled {
            return Err(GraphError::NotCompiled);
        }
        let device = self.graph.device.clone();
        let timestamps = self.timestamps && device.capabilities().has(Feature::Timestamps);
        let mut cmd = device.begin_commands()?;
        let data = &mut self.data;
        for i in 0..data.passes.len() {
            if !data.passes[i].live {
                continue;
            }
            let (texture_barriers, buffer_barriers) = (&data.passes[i].texture_barriers, &data.passes[i].buffer_barriers);
            if !texture_barriers.is_empty() || !buffer_barriers.is_empty() {
                cmd.barrier(texture_barriers, buffer_barriers);
            }
            let name = data.passes[i].decl.name.clone();
            cmd.begin_debug_label(&name);
            if timestamps {
                cmd.write_timestamp(&name);
            }

            let decl = &data.passes[i].decl;
            let rendering = !decl.colors.is_empty() || decl.depth.is_some();
            if rendering {
                let mut info = RenderingInfo::default();
                for c in &decl.colors {
                    info.colors.push(ColorTarget { texture: data.textures[c.id as usize].handle, clear: c.clear, clear_color: c.value });
                }
                if let Some(d) = &decl.depth {
                    info.depth = Some(DepthTarget {
                        texture: data.textures[d.id as usize].handle,
                        clear: d.clear,
                        clear_depth: d.value,
                        read_only: d.read_only,
                    });
                }
                let size_id = decl.colors.first().map(|c| c.id).or(decl.depth.as_ref().map(|d| d.id)).unwrap_or(0);
                info.width = data.textures[size_id as usize].desc.width;
                info.height = data.textures[size_id as usize].desc.height;
                cmd.begin_rendering(&info);
            }
            if let Some(execute) = data.passes[i].execute.take() {
                let used_textures: Vec<u32> = data.passes[i].decl.textures.iter().map(|u| u.id).collect();
                let used_buffers: Vec<u32> = data.passes[i].decl.buffers.iter().map(|u| u.id).collect();
                let mut context = PassContext {
                    cmd: cmd.as_mut(),
                    device: device.as_ref(),
                    pass: &name,
                    textures: &data.textures,
                    buffers: &data.buffers,
                    used_textures: &used_textures,
                    used_buffers: &used_buffers,
                };
                execute(&mut context)?;
            }
            if rendering {
                cmd.end_rendering();
            }
            if timestamps {
                cmd.write_timestamp(&format!("{name}/end"));
            }
            cmd.end_debug_label();
        }
        if !data.final_texture_barriers.is_empty() || !data.final_buffer_barriers.is_empty() {
            cmd.barrier(&data.final_texture_barriers, &data.final_buffer_barriers);
        }
        Ok(device.submit(cmd)?)
    }

    /// Passes in order, whether each is live or culled, and the barriers before it. Used by tests.
    pub fn dump(&self) -> String {
        let data = &self.data;
        let mut out = String::new();
        for (i, pass) in data.passes.iter().enumerate() {
            let state = if !data.compiled { "pending" } else if pass.live { "live" } else { "culled" };
            let _ = writeln!(out, "pass {i} \"{}\" {state}", pass.decl.name);
            if data.compiled && pass.live {
                write_barriers(&mut out, data, &pass.texture_barriers, &pass.buffer_barriers);
            }
        }
        if data.compiled {
            out.push_str("final\n");
            write_barriers(&mut out, data, &data.final_texture_barriers, &data.final_buffer_barriers);
        }
        out
    }
}

fn write_barriers(out: &mut String, data: &GraphData<'_>, textures: &[TextureBarrier], buffers: &[BufferBarrier]) {
    for b in textures {
        let index = data.textures.iter().position(|t| t.handle == b.texture).unwrap_or(data.textures.len());
        let _ = write!(out, "  barrier texture#{index}");
        if let Some(t) = data.textures.get(index).filter(|t| !t.name.is_empty()) {
            let _ = write!(out, " '{}'", t.name);
        }
        let _ = writeln!(out, " {} -> {}", b.from.name(), b.to.name());
    }
    for b in buffers {
        let index = data.buffers.iter().position(|t| t.handle == b.buffer).unwrap_or(data.buffers.len());
        let _ = write!(out, "  barrier buffer#{index}");
        if let Some(t) = data.buffers.get(index).filter(|t| !t.name.is_empty()) {
            let _ = write!(out, " '{}'", t.name);
        }
        let _ = writeln!(out, " {} -> {}", b.from.name(), b.to.name());
    }
}
