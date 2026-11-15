use super::*;
use std::fmt::Write;

fn escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

impl GraphBuilder<'_> {
    /// The compiled graph as Graphviz DOT (`dot -Tsvg graph.dot -o graph.svg`): a box per pass (dashed and
    /// marked "(culled)" when nothing needs it), an ellipse per resource a live pass uses, and an edge for
    /// each access labelled with the state the resource is in for it.
    pub fn to_dot(&self) -> GraphResult<String> {
        let data = &self.data;
        if !data.compiled {
            return Err(GraphError::NotCompiled);
        }
        let mut out = String::from("digraph rendergraph {\n  rankdir=LR;\n  node [fontname=\"sans-serif\"];\n");
        for (i, pass) in data.passes.iter().enumerate() {
            let name = escape(&pass.decl.name);
            if pass.live {
                let _ = writeln!(out, "  p{i} [shape=box, label=\"{name}\"];");
            } else {
                let _ = writeln!(out, "  p{i} [shape=box, style=dashed, label=\"{name}\\n(culled)\"];");
            }
        }
        let mut texture_drawn = vec![false; data.textures.len()];
        let mut buffer_drawn = vec![false; data.buffers.len()];
        for (i, pass) in data.passes.iter().enumerate() {
            for u in &pass.decl.textures {
                let id = u.id as usize;
                if !texture_drawn[id] && pass.live {
                    texture_drawn[id] = true;
                    let node = &data.textures[id];
                    let name = if node.name.is_empty() { format!("texture#{id}") } else { node.name.clone() };
                    let kind = if node.imported { "imported" } else { "transient" };
                    let _ = writeln!(out, "  t{id} [shape=ellipse, label=\"{}\\n{kind}\"];", escape(&name));
                }
                self.edge(&mut out, i, "t", id, u.write, u.state, pass.live);
            }
            for u in pass.decl.buffers.iter().filter(|u| !u.host) {
                let id = u.id as usize;
                if !buffer_drawn[id] && pass.live {
                    buffer_drawn[id] = true;
                    let node = &data.buffers[id];
                    let name = if node.name.is_empty() { format!("buffer#{id}") } else { node.name.clone() };
                    let kind = if node.imported { "imported" } else { "transient" };
                    let _ = writeln!(out, "  b{id} [shape=ellipse, label=\"{}\\n{kind}\"];", escape(&name));
                }
                self.edge(&mut out, i, "b", id, u.write, u.state, pass.live);
            }
        }
        out.push_str("}\n");
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn edge(&self, out: &mut String, pass: usize, prefix: &str, id: usize, write: bool, state: ResourceState, live: bool) {
        if !live {
            return;
        }
        let (from, to) = if write { (format!("p{pass}"), format!("{prefix}{id}")) } else { (format!("{prefix}{id}"), format!("p{pass}")) };
        let _ = writeln!(out, "  {from} -> {to} [label=\"{}\"];", state.name());
    }
}
