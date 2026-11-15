use super::*;
use phox_rhi::{BufferBarrier, TextureBarrier};

/// Fills `texture_barriers` / `buffer_barriers` of every live pass, in declaration order, plus the final
/// barriers that put imported resources into their declared `after` state. Fails when one pass needs a
/// resource in two different states.
pub(crate) fn plan_barriers(data: &mut GraphData<'_>) -> GraphResult<()> {
    let mut textures: Vec<ResourceState> = data.textures.iter().map(|t| if t.imported { t.before } else { ResourceState::Undefined }).collect();
    let mut buffers: Vec<ResourceState> = data.buffers.iter().map(|b| if b.imported { b.before } else { ResourceState::Undefined }).collect();

    for pass in &mut data.passes {
        pass.texture_barriers.clear();
        pass.buffer_barriers.clear();
        if !pass.live {
            continue;
        }

        // The state this pass needs each resource in; one pass cannot need two states of one resource.
        let mut texture_states: Vec<(u32, ResourceState)> = Vec::new();
        for u in &pass.decl.textures {
            match texture_states.iter().find(|(id, _)| *id == u.id) {
                Some(&(_, state)) if state != u.state => {
                    return Err(GraphError::Declaration(format!(
                        "pass '{}' uses texture '{}' as both {} and {}",
                        pass.decl.name,
                        plain_label(&data.textures[u.id as usize].name, "texture", u.id),
                        state.name(),
                        u.state.name()
                    )));
                }
                Some(_) => {}
                None => texture_states.push((u.id, u.state)),
            }
        }
        let mut buffer_states: Vec<(u32, ResourceState)> = Vec::new();
        for u in pass.decl.buffers.iter().filter(|u| !u.host) {
            match buffer_states.iter().find(|(id, _)| *id == u.id) {
                Some(&(_, state)) if state != u.state => {
                    return Err(GraphError::Declaration(format!(
                        "pass '{}' uses buffer '{}' as both {} and {}",
                        pass.decl.name,
                        plain_label(&data.buffers[u.id as usize].name, "buffer", u.id),
                        state.name(),
                        u.state.name()
                    )));
                }
                Some(_) => {}
                None => buffer_states.push((u.id, u.state)),
            }
        }

        for (id, state) in texture_states {
            let tracked = &mut textures[id as usize];
            if *tracked != state || state.is_write() {
                pass.texture_barriers.push(TextureBarrier { texture: data.textures[id as usize].handle, from: *tracked, to: state });
            }
            *tracked = state;
        }
        for (id, state) in buffer_states {
            let tracked = &mut buffers[id as usize];
            if *tracked != state || state.is_write() {
                pass.buffer_barriers.push(BufferBarrier { buffer: data.buffers[id as usize].handle, from: *tracked, to: state });
            }
            *tracked = state;
        }
    }

    data.final_texture_barriers.clear();
    data.final_buffer_barriers.clear();
    for (i, node) in data.textures.iter().enumerate() {
        // `after == Undefined` means the caller does not care what state the resource is left in.
        if node.imported && node.after != ResourceState::Undefined && textures[i] != node.after {
            data.final_texture_barriers.push(TextureBarrier { texture: node.handle, from: textures[i], to: node.after });
        }
    }
    for (i, node) in data.buffers.iter().enumerate() {
        if node.imported && node.after != ResourceState::Undefined && buffers[i] != node.after {
            data.final_buffer_barriers.push(BufferBarrier { buffer: node.handle, from: buffers[i], to: node.after });
        }
    }
    Ok(())
}

fn plain_label(name: &str, kind: &str, id: u32) -> String {
    if name.is_empty() { format!("{kind}#{id}") } else { name.to_string() }
}
