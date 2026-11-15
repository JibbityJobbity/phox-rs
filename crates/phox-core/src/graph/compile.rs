use super::*;
use phox_rhi::{BufferDesc, BufferUsage, TextureDesc, TextureUsage};

impl<'g> GraphBuilder<'g> {
    /// Checks the device's features, validates the declarations, culls passes nothing needs, derives each
    /// transient's usage, takes transients from the pool and plans the barriers. Fails with a
    /// [`GraphError`] naming the pass or the missing features.
    pub fn compile(&mut self) -> GraphResult<()> {
        self.data.compiled = false;

        // 1. Capabilities
        let caps = self.graph.device.capabilities();
        let missing: Vec<Feature> = self.data.required.iter().copied().filter(|&f| !caps.has(f)).collect();
        if !missing.is_empty() {
            let names: Vec<&str> = missing.iter().map(|f| f.name()).collect();
            return Err(GraphError::MissingFeatures {
                message: format!(
                    "render graph requires features that device '{}' does not support: {}",
                    caps.device_name,
                    names.join(" ")
                ),
                features: missing,
            });
        }

        // 2. Validation, 3. Culling
        self.validate()?;
        self.cull();

        // 4. Usage derivation and transient resources
        self.graph.frame += 1;
        self.acquire_resources()?;

        // 5. Barriers
        barriers::plan_barriers(&mut self.data)?;

        self.evict_stale();
        self.data.compiled = true;
        Ok(())
    }

    fn validate(&self) -> GraphResult<()> {
        let data = &self.data;
        let mut texture_written = vec![false; data.textures.len()];
        let mut buffer_written = vec![false; data.buffers.len()];
        let fail = |message: String| Err(GraphError::Declaration(message));

        for pass in &data.passes {
            let name = &pass.decl.name;
            // Reads must see an earlier write, unless the resource comes from outside the graph.
            for u in &pass.decl.textures {
                let node = &data.textures[u.id as usize];
                let label = label(&node.name, "texture", u.id);
                if !u.write && !node.imported && !texture_written[u.id as usize] {
                    return fail(format!("pass '{name}' reads texture {label} that no earlier pass writes"));
                }
                if pass.decl.textures.iter().any(|o| o.id == u.id && o.write != u.write) {
                    return fail(format!("pass '{name}' both reads and writes texture {label}; use two passes"));
                }
            }
            for u in &pass.decl.buffers {
                let node = &data.buffers[u.id as usize];
                let label = label(&node.name, "buffer", u.id);
                if !u.write && !node.imported && !buffer_written[u.id as usize] {
                    return fail(format!("pass '{name}' reads buffer {label} that no earlier pass writes"));
                }
                if pass.decl.buffers.iter().any(|o| o.id == u.id && o.write != u.write) {
                    return fail(format!("pass '{name}' both reads and writes buffer {label}; use two passes"));
                }
            }

            // A target that keeps its old contents needs an earlier writer too.
            let check_load = |id: u32, clear: bool| {
                let node = &data.textures[id as usize];
                if !clear && !node.imported && !texture_written[id as usize] {
                    return Err(GraphError::Declaration(format!(
                        "pass '{name}' loads texture {} that no earlier pass writes",
                        label(&node.name, "texture", id)
                    )));
                }
                Ok(())
            };
            for c in &pass.decl.colors {
                check_load(c.id, c.clear)?;
            }
            if let Some(d) = &pass.decl.depth {
                // A read-only depth attachment is a read: it was checked with the uses above.
                if !d.read_only {
                    check_load(d.id, d.clear)?;
                }
            }

            // All attachments of one pass share a size.
            let mut extent: Option<(u32, u32)> = None;
            let ids = pass.decl.colors.iter().map(|c| c.id).chain(pass.decl.depth.iter().map(|d| d.id));
            for id in ids {
                let desc = &data.textures[id as usize].desc;
                match extent {
                    None => extent = Some((desc.width, desc.height)),
                    Some(e) if e != (desc.width, desc.height) => {
                        return fail(format!("pass '{name}' has render targets of different sizes"));
                    }
                    Some(_) => {}
                }
            }

            for u in &pass.decl.textures {
                texture_written[u.id as usize] |= u.write;
            }
            for u in &pass.decl.buffers {
                buffer_written[u.id as usize] |= u.write;
            }
        }
        Ok(())
    }

    fn cull(&mut self) {
        let data = &mut self.data;
        let mut texture_needed = vec![false; data.textures.len()];
        let mut buffer_needed = vec![false; data.buffers.len()];
        for pass in data.passes.iter_mut().rev() {
            let mut live = pass.decl.side_effect;
            for u in &pass.decl.textures {
                live |= u.write && (data.textures[u.id as usize].imported || texture_needed[u.id as usize]);
            }
            for u in &pass.decl.buffers {
                live |= u.write && (data.buffers[u.id as usize].imported || buffer_needed[u.id as usize]);
            }
            pass.live = live;
            if !live {
                continue;
            }
            for u in &pass.decl.textures {
                texture_needed[u.id as usize] |= !u.write;
            }
            for u in &pass.decl.buffers {
                buffer_needed[u.id as usize] |= !u.write;
            }
            for c in &pass.decl.colors {
                texture_needed[c.id as usize] |= !c.clear; // a load needs what was there before
            }
            if let Some(d) = &pass.decl.depth {
                texture_needed[d.id as usize] |= !d.clear;
            }
        }
    }

    fn acquire_resources(&mut self) -> GraphResult<()> {
        let device = self.graph.device.clone();
        let frame = self.graph.frame;
        for t in &mut self.graph.textures {
            t.in_use = false;
        }
        for b in &mut self.graph.buffers {
            b.in_use = false;
        }

        // Usage = every access of every live pass, OR-ed together.
        let data = &mut self.data;
        let mut texture_usage = vec![TextureUsage::NONE; data.textures.len()];
        let mut buffer_usage = vec![BufferUsage::NONE; data.buffers.len()];
        let mut texture_used = vec![false; data.textures.len()];
        let mut buffer_used = vec![false; data.buffers.len()];
        for pass in data.passes.iter().filter(|p| p.live) {
            for u in &pass.decl.textures {
                texture_usage[u.id as usize] |= u.usage;
                texture_used[u.id as usize] = true;
            }
            for u in &pass.decl.buffers {
                buffer_usage[u.id as usize] |= u.usage;
                buffer_used[u.id as usize] = true;
            }
        }

        for (i, node) in data.textures.iter_mut().enumerate() {
            if node.imported || !texture_used[i] {
                continue;
            }
            let key = TextureDesc { usage: texture_usage[i], debug_name: String::new(), ..node.desc.clone() };
            let pool = &mut self.graph.textures;
            if let Some(entry) = pool.iter_mut().find(|e| {
                !e.in_use
                    && e.desc.width == key.width
                    && e.desc.height == key.height
                    && e.desc.mips == key.mips
                    && e.desc.format == key.format
                    && e.desc.usage == key.usage
            }) {
                entry.in_use = true;
                entry.last_used = frame;
                node.handle = entry.handle;
                continue;
            }
            let handle = device.create_texture(&TextureDesc { debug_name: node.name.clone(), ..key.clone() })?;
            pool.push(PooledTexture { desc: key, handle, last_used: frame, in_use: true });
            node.handle = handle;
        }

        for (i, node) in data.buffers.iter_mut().enumerate() {
            if node.imported || !buffer_used[i] {
                continue;
            }
            let mut key = BufferDesc { usage: buffer_usage[i], debug_name: String::new(), ..node.desc.clone() };
            if key.usage.is_empty() {
                key.usage = BufferUsage::TRANSFER_DST; // only written by the host and never read: any usage will do
            }
            let pool = &mut self.graph.buffers;
            if let Some(entry) = pool.iter_mut().find(|e| {
                !e.in_use
                    && e.desc.size == key.size
                    && e.desc.usage == key.usage
                    && e.desc.memory == key.memory
                    && e.desc.device_address == key.device_address
            }) {
                entry.in_use = true;
                entry.last_used = frame;
                node.handle = entry.handle;
                continue;
            }
            let handle = device.create_buffer(&BufferDesc { debug_name: node.name.clone(), ..key.clone() })?;
            pool.push(PooledBuffer { desc: key, handle, last_used: frame, in_use: true });
            node.handle = handle;
        }
        Ok(())
    }

    fn evict_stale(&mut self) {
        let frame = self.graph.frame;
        let device = self.graph.device.clone();
        self.graph.textures.retain(|e| {
            let stale = !e.in_use && frame - e.last_used >= POOL_RETIRE_FRAMES;
            if stale {
                device.destroy_texture(e.handle);
            }
            !stale
        });
        self.graph.buffers.retain(|e| {
            let stale = !e.in_use && frame - e.last_used >= POOL_RETIRE_FRAMES;
            if stale {
                device.destroy_buffer(e.handle);
            }
            !stale
        });
    }
}
