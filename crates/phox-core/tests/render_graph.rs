//! Render graph logic, checked against the null device, which records what it is asked to do. No GPU needed.
//! Ported from test/source/graph_test.cpp (the RHI graph) and the cases of render_graph_test.cpp that apply
//! to it (hazard barriers, culling, import transitions, depth read-only, DOT export).

use phox_core::graph::*;
use phox_null::NullDevice;
use phox_rhi::*;
use std::sync::Arc;

fn texture(name: &str, size: u32) -> TextureDesc {
    // The usage is ignored for transient textures: the graph derives it from the accesses.
    TextureDesc::new(size, size, Format::RGBA8Unorm, TextureUsage::SAMPLED, name)
}

fn setup() -> (Arc<NullDevice>, RenderGraph) {
    let device = Arc::new(NullDevice::new());
    let graph = RenderGraph::new(device.clone());
    (device, graph)
}

/// The text of the dump belonging to one pass (or to "final").
fn pass_block(dump: &str, name: &str) -> String {
    let header = if name == "final" { "final\n".to_string() } else { format!("\"{name}\" ") };
    let start = dump.find(&header).unwrap_or_else(|| panic!("no '{name}' in:\n{dump}"));
    let line_end = dump[start..].find('\n').map_or(dump.len(), |i| start + i);
    let end = [dump[line_end..].find("\npass "), dump[line_end..].find("\nfinal\n")].into_iter().flatten().min().map_or(dump.len(), |i| line_end + i);
    dump[start..end].to_string()
}

fn noop(_: &mut PassContext<'_>) -> GraphResult<()> {
    Ok(())
}

fn draw3(c: &mut PassContext<'_>) -> GraphResult<()> {
    c.cmd.draw(3, 1, 0);
    Ok(())
}

fn dispatch1(c: &mut PassContext<'_>) -> GraphResult<()> {
    c.cmd.dispatch(1, 1, 1);
    Ok(())
}

#[test]
fn barriers_between_passes() {
    let (device, mut rg) = setup();
    let mut b = rg.begin();
    let t = b.create_texture(&texture("T", 64));
    b.add_pass("A", |p| p.color_target(t), draw3).unwrap();
    b.add_pass(
        "B",
        |p| {
            p.read_texture(t, TextureAccess::ShaderRead)?;
            p.side_effect();
            Ok(())
        },
        dispatch1,
    )
    .unwrap();
    b.compile().unwrap();

    let dump = b.dump();
    assert!(pass_block(&dump, "A").contains("Undefined -> ColorAttachment"));
    assert!(pass_block(&dump, "B").contains("ColorAttachment -> ShaderRead"));
    assert!(!pass_block(&dump, "B").contains("Undefined"));

    // The transient's usage is derived from what the passes do with it.
    let created = device.describe_texture(b.resolve_texture(t).unwrap()).unwrap();
    assert!(created.usage.contains(TextureUsage::COLOR_ATTACHMENT) && created.usage.contains(TextureUsage::SAMPLED));
    assert!(!created.usage.contains(TextureUsage::STORAGE));

    b.execute().unwrap();
    assert_eq!(
        device.last_recording(),
        [
            "barrier",
            "beginDebugLabel:A",
            "beginRendering",
            "draw",
            "endRendering",
            "endDebugLabel",
            "barrier",
            "beginDebugLabel:B",
            "dispatch",
            "endDebugLabel"
        ]
    );
}

#[test]
fn culling() {
    let (device, mut rg) = setup();
    let mut b = rg.begin();
    let unused = b.create_texture(&texture("unused", 64));
    b.add_pass("X", |p| p.color_target(unused), draw3).unwrap();
    b.add_pass(
        "Y",
        |p| {
            p.side_effect();
            Ok(())
        },
        dispatch1,
    )
    .unwrap();
    b.compile().unwrap();

    assert!(pass_block(&b.dump(), "X").contains("culled"));
    assert!(pass_block(&b.dump(), "Y").contains("live"));
    assert!(!b.resolve_texture(unused).unwrap().is_valid()); // only a culled pass uses it, so it was never created
    b.execute().unwrap();
    let recording = device.last_recording();
    assert!(!recording.contains(&"draw".to_string()));
    assert!(recording.contains(&"dispatch".to_string()));
}

#[test]
fn culling_follows_dependencies() {
    let (device, mut rg) = setup();
    let sink = device.create_buffer(&BufferDesc::new(64, BufferUsage::STORAGE, MemoryKind::Readback, "sink")).unwrap();
    let mut b = rg.begin();
    let mid = b.create_buffer(&BufferDesc::new(64, BufferUsage::NONE, MemoryKind::GpuOnly, "mid"));
    let imported = b.import_buffer(sink, ResourceState::Undefined, ResourceState::Undefined).unwrap();
    let orphan = b.create_buffer(&BufferDesc::new(64, BufferUsage::NONE, MemoryKind::GpuOnly, "orphan"));
    b.add_pass("produce", |p| p.write_buffer(mid, BufferAccess::ShaderWrite), noop).unwrap();
    b.add_pass(
        "consume",
        |p| {
            p.read_buffer(mid, BufferAccess::ShaderRead)?;
            p.write_buffer(imported, BufferAccess::ShaderWrite)
        },
        noop,
    )
    .unwrap();
    b.add_pass("orphan", |p| p.write_buffer(orphan, BufferAccess::ShaderWrite), noop).unwrap();
    b.compile().unwrap();

    let dump = b.dump();
    assert!(pass_block(&dump, "produce").contains("live")); // feeds a pass that writes an imported resource
    assert!(pass_block(&dump, "consume").contains("live")); // writes an imported resource
    assert!(pass_block(&dump, "orphan").contains("culled")); // writes something nobody needs
}

#[test]
fn a_chain_nothing_consumes_is_culled() {
    // a -> b -> (nothing): a producer is live only if a live pass consumes what it made.
    let (_device, mut rg) = setup();
    let mut b = rg.begin();
    let (ta, tb) = (b.create_texture(&texture("a", 64)), b.create_texture(&texture("b", 64)));
    b.add_pass("pa", |p| p.color_target(ta), noop).unwrap();
    b.add_pass(
        "pb",
        |p| {
            p.read_texture(ta, TextureAccess::TransferSrc)?;
            p.write_texture(tb, TextureAccess::TransferDst)
        },
        noop,
    )
    .unwrap();
    b.compile().unwrap();
    assert!(pass_block(&b.dump(), "pa").contains("culled") && pass_block(&b.dump(), "pb").contains("culled"));
}

#[test]
fn read_of_unwritten_transient() {
    let (_device, mut rg) = setup();
    {
        let mut b = rg.begin();
        let t = b.create_texture(&texture("never-written", 64));
        b.add_pass(
            "reader",
            |p| {
                p.read_texture(t, TextureAccess::ShaderRead)?;
                p.side_effect();
                Ok(())
            },
            noop,
        )
        .unwrap();
        let message = b.compile().unwrap_err().to_string();
        assert!(message.contains("reader") && message.contains("no earlier pass writes"), "{message}");
    }
    // Loading a target (clear = false) is a read too.
    let mut b = rg.begin();
    let t = b.create_texture(&texture("loaded", 64));
    b.add_pass(
        "loader",
        |p| {
            p.color_target_load(t)?;
            p.side_effect();
            Ok(())
        },
        noop,
    )
    .unwrap();
    assert!(b.compile().unwrap_err().to_string().contains("loads texture"));
}

/// Not a real technique. It exists to prove that a component requiring a GPU feature is rejected at
/// `compile()` time on devices without that feature, and accepted on devices that have it.
struct MeshletStub;

impl Component for MeshletStub {
    type In = ();
    type Out = ();
    fn name(&self) -> &str {
        "MeshletStub"
    }
    fn setup<'g>(&mut self, b: &mut GraphBuilder<'g>, _: &()) -> GraphResult<()> {
        b.require(Feature::MeshShader);
        b.add_pass(
            "meshlet",
            |p| {
                p.side_effect();
                Ok(())
            },
            noop,
        )
    }
}

#[test]
fn capability_gating() {
    let (device, mut rg) = setup(); // no features
    let mut stub = MeshletStub;
    {
        let mut b = rg.begin();
        stub.setup(&mut b, &()).unwrap();
        let error = b.compile().unwrap_err();
        assert_eq!(error.missing_features(), [Feature::MeshShader]);
        assert!(error.to_string().contains("MeshShader"));
    }
    device.set_features([Feature::MeshShader]);
    {
        let mut b = rg.begin();
        stub.setup(&mut b, &()).unwrap();
        b.compile().unwrap();
        b.execute().unwrap(); // does not fail
    }
    // Every missing feature is reported, once each, in declaration order.
    device.set_features([]);
    let mut b = rg.begin();
    b.require(Feature::RayTracing);
    b.require(Feature::MeshShader);
    b.require(Feature::RayTracing);
    assert_eq!(b.compile().unwrap_err().missing_features(), [Feature::RayTracing, Feature::MeshShader]);
}

struct StubTechnique;

impl Technique for StubTechnique {
    fn name(&self) -> &str {
        "stub"
    }
    fn requirements(&self) -> Vec<Feature> {
        vec![Feature::MeshShader, Feature::Atomic64]
    }
    fn init(&mut self, _: &dyn Device) -> GraphResult<()> {
        Ok(())
    }
    fn shutdown(&mut self, _: &dyn Device) {}
    fn build<'g>(&mut self, _: &mut GraphBuilder<'g>) -> GraphResult<()> {
        Ok(())
    }
}

#[test]
fn technique_support() {
    let device = NullDevice::new();
    let technique = StubTechnique;
    assert!(!technique.is_supported(&device.capabilities()));
    device.set_features([Feature::MeshShader]);
    assert!(!technique.is_supported(&device.capabilities()));
    device.set_features([Feature::MeshShader, Feature::Atomic64]);
    assert!(technique.is_supported(&device.capabilities()));
}

fn build_frame(rg: &mut RenderGraph, execute: bool) -> TextureHandle {
    let mut b = rg.begin();
    let t = b.create_texture(&texture("T", 64));
    b.add_pass("A", |p| p.color_target(t), draw3).unwrap();
    b.add_pass(
        "B",
        |p| {
            p.read_texture(t, TextureAccess::ShaderRead)?;
            p.side_effect();
            Ok(())
        },
        noop,
    )
    .unwrap();
    b.compile().unwrap();
    if execute {
        b.execute().unwrap();
    }
    b.resolve_texture(t).unwrap()
}

fn empty_frame(rg: &mut RenderGraph) {
    rg.begin().compile().unwrap();
}

#[test]
fn pool_reuse() {
    let (device, mut rg) = setup();
    let first = build_frame(&mut rg, true);
    assert!(first.is_valid());

    // The same frame again reuses the pooled texture: no new resource is created.
    assert_eq!(build_frame(&mut rg, false), first);

    // Two frames that do not use it keep it alive...
    empty_frame(&mut rg);
    empty_frame(&mut rg);
    assert_eq!(build_frame(&mut rg, false), first);

    // ...and the third unused frame retires it. The next frame gets a fresh resource.
    for _ in 0..3 {
        empty_frame(&mut rg);
    }
    let recreated = build_frame(&mut rg, false);
    assert_ne!(recreated, first);
    assert_eq!(device.describe_texture(recreated).unwrap().width, 64);
    assert!(device.describe_texture(first).is_err()); // the retired one was destroyed

    // A different size is a different pool entry.
    let mut b = rg.begin();
    let big = b.create_texture(&texture("big", 128));
    b.add_pass("A", |p| p.color_target(big), noop).unwrap();
    b.add_pass(
        "B",
        |p| {
            p.read_texture(big, TextureAccess::ShaderRead)?;
            p.side_effect();
            Ok(())
        },
        noop,
    )
    .unwrap();
    b.compile().unwrap();
    assert_eq!(device.describe_texture(b.resolve_texture(big).unwrap()).unwrap().width, 128);
}

#[test]
fn imported_final_state() {
    let (device, mut rg) = setup();
    let target = device
        .create_texture(&TextureDesc::new(64, 64, Format::RGBA8Unorm, TextureUsage::COLOR_ATTACHMENT | TextureUsage::TRANSFER_SRC, "target"))
        .unwrap();
    {
        let mut b = rg.begin();
        let t = b.import_texture(target, ResourceState::Undefined, ResourceState::TransferSrc).unwrap();
        b.add_pass("draw", |p| p.color_target(t), draw3).unwrap();
        b.compile().unwrap();

        let dump = b.dump();
        assert!(pass_block(&dump, "draw").contains("Undefined -> ColorAttachment"));
        assert!(pass_block(&dump, "final").contains("ColorAttachment -> TransferSrc"));
        assert_eq!(b.resolve_texture(t).unwrap(), target); // imports keep their own handle

        b.execute().unwrap();
        assert_eq!(device.last_recording().last().map(String::as_str), Some("barrier"));
    }
    // `after = Undefined` means "leave it wherever it ends up": no final barrier.
    {
        let mut b = rg.begin();
        let t = b.import_texture(target, ResourceState::Undefined, ResourceState::Undefined).unwrap();
        b.add_pass("draw", |p| p.color_target(t), noop).unwrap();
        b.compile().unwrap();
        assert!(!pass_block(&b.dump(), "final").contains("barrier"));
    }
    // An imported resource that no pass touches still ends up in its declared `after` state.
    let mut b = rg.begin();
    b.import_texture(target, ResourceState::ShaderRead, ResourceState::TransferSrc).unwrap();
    b.compile().unwrap();
    assert!(pass_block(&b.dump(), "final").contains("ShaderRead -> TransferSrc"));
}

#[test]
fn write_after_write_gets_a_barrier() {
    let (device, mut rg) = setup();
    let buffer = device.create_buffer(&BufferDesc::new(64, BufferUsage::STORAGE, MemoryKind::GpuOnly, "buf")).unwrap();
    let mut b = rg.begin();
    let buf = b.import_buffer(buffer, ResourceState::Undefined, ResourceState::Undefined).unwrap();
    b.add_pass("first", |p| p.write_buffer(buf, BufferAccess::ShaderWrite), noop).unwrap();
    b.add_pass("second", |p| p.write_buffer(buf, BufferAccess::ShaderWrite), noop).unwrap();
    b.add_pass("reader", |p| p.read_buffer(buf, BufferAccess::ShaderRead), noop).unwrap();
    b.add_pass("reader2", |p| p.read_buffer(buf, BufferAccess::ShaderRead), noop).unwrap();
    b.compile().unwrap();

    let dump = b.dump();
    assert!(pass_block(&dump, "first").contains("Undefined -> ShaderWrite"));
    assert!(pass_block(&dump, "second").contains("ShaderWrite -> ShaderWrite")); // a write hazard: must wait
    // The readers do not write, and no live pass uses their result, so they are culled.
    assert!(pass_block(&dump, "reader").contains("culled"));
}

#[test]
fn read_after_read_needs_no_barrier() {
    let (_device, mut rg) = setup();
    let mut b = rg.begin();
    let t = b.create_texture(&texture("T", 64));
    b.add_pass("write", |p| p.color_target(t), noop).unwrap();
    for name in ["read1", "read2"] {
        b.add_pass(
            name,
            |p| {
                p.read_texture(t, TextureAccess::ShaderRead)?;
                p.side_effect();
                Ok(())
            },
            noop,
        )
        .unwrap();
    }
    b.compile().unwrap();
    let dump = b.dump();
    assert!(pass_block(&dump, "read1").contains("barrier"));
    assert!(!pass_block(&dump, "read2").contains("barrier"));
}

#[test]
fn host_writes_need_no_barrier() {
    let (device, mut rg) = setup();
    let target = device.create_texture(&TextureDesc::new(64, 64, Format::RGBA8Unorm, TextureUsage::COLOR_ATTACHMENT, "target")).unwrap();
    let mut b = rg.begin();
    let out = b.import_texture(target, ResourceState::Undefined, ResourceState::Undefined).unwrap();
    let vertices = b.create_buffer(&BufferDesc::new(32, BufferUsage::NONE, MemoryKind::Upload, "vertices"));
    b.add_pass(
        "upload",
        |p| p.write_buffer(vertices, BufferAccess::HostWrite),
        move |c| {
            let buffer = c.buffer(vertices)?;
            c.device.write_buffer(buffer, 0, &[0x7f; 32])?;
            Ok(())
        },
    )
    .unwrap();
    b.add_pass(
        "draw",
        |p| {
            p.color_target(out)?;
            p.read_buffer(vertices, BufferAccess::Vertex)
        },
        draw3,
    )
    .unwrap();
    b.compile().unwrap();

    let dump = b.dump();
    assert!(!pass_block(&dump, "upload").contains("barrier")); // host writes need no barrier
    assert!(pass_block(&dump, "draw").contains("-> VertexOrIndex"));
    let desc = device.describe_buffer(b.resolve_buffer(vertices).unwrap()).unwrap();
    assert!(desc.usage.contains(BufferUsage::VERTEX));
    assert_eq!(desc.memory, MemoryKind::Upload);

    b.execute().unwrap();
    let mut byte = [0u8];
    device.read_buffer(b.resolve_buffer(vertices).unwrap(), 0, &mut byte).unwrap();
    assert_eq!(byte[0], 0x7f);
}

#[test]
fn validation() {
    let (_device, mut rg) = setup();
    // One pass cannot read and write the same resource.
    {
        let mut b = rg.begin();
        let t = b.create_texture(&texture("T", 64));
        b.add_pass("seed", |p| p.color_target(t), noop).unwrap();
        b.add_pass(
            "both",
            |p| {
                p.read_texture(t, TextureAccess::ShaderRead)?;
                p.write_texture(t, TextureAccess::ShaderWrite)?;
                p.side_effect();
                Ok(())
            },
            noop,
        )
        .unwrap();
        assert!(b.compile().unwrap_err().to_string().contains("both reads and writes"));
    }
    // Render targets of one pass share a size.
    {
        let mut b = rg.begin();
        let (small, large) = (b.create_texture(&texture("small", 32)), b.create_texture(&texture("large", 64)));
        b.add_pass(
            "mismatch",
            |p| {
                p.color_target(small)?;
                p.color_target(large)?;
                p.side_effect();
                Ok(())
            },
            noop,
        )
        .unwrap();
        assert!(b.compile().unwrap_err().to_string().contains("different sizes"));
    }
    // read() and write() reject the wrong kind of access; a failing setup is rolled back.
    {
        let mut b = rg.begin();
        let t = b.create_texture(&texture("T", 64));
        let wrong_read = b.add_pass("bad", |p| p.read_texture(t, TextureAccess::ShaderWrite), noop).unwrap_err();
        assert!(wrong_read.to_string().contains("read()"), "{wrong_read}");
        let wrong_write = b.add_pass("bad", |p| p.write_texture(t, TextureAccess::ShaderRead), noop).unwrap_err();
        assert!(wrong_write.to_string().contains("write()"), "{wrong_write}");
        let invalid = b.add_pass("bad", |p| p.read_texture(GraphTexture::default(), TextureAccess::ShaderRead), noop).unwrap_err();
        assert!(invalid.to_string().contains("invalid texture handle"), "{invalid}");
        assert!(!b.dump().contains("pass "));
    }
    // Two states of one resource in one pass are a conflict the barrier planner reports.
    {
        let mut b = rg.begin();
        let t = b.create_texture(&texture("T", 64));
        b.add_pass("seed", |p| p.color_target(t), noop).unwrap();
        b.add_pass(
            "conflict",
            |p| {
                p.read_texture(t, TextureAccess::ShaderRead)?;
                p.read_texture(t, TextureAccess::TransferSrc)?;
                p.side_effect();
                Ok(())
            },
            noop,
        )
        .unwrap();
        assert!(b.compile().unwrap_err().to_string().contains("as both"));
        // execute() needs a successful compile().
        assert!(matches!(b.execute(), Err(GraphError::NotCompiled)));
    }
}

#[test]
fn headless_frame_records_in_order() {
    // scene -> color -> readback into a host buffer, as the renderer does without a window.
    let (device, mut rg) = setup();
    let staging = device.create_buffer(&BufferDesc::new(64 * 64 * 4, BufferUsage::TRANSFER_DST, MemoryKind::Readback, "staging")).unwrap();
    let mut b = rg.begin();
    let color = b.create_texture(&texture("color", 64));
    let staging_node = b.import_buffer(staging, ResourceState::Undefined, ResourceState::Undefined).unwrap();
    b.add_pass("scene", |p| p.color_target(color), noop).unwrap();
    b.add_pass(
        "readback",
        |p| {
            p.read_texture(color, TextureAccess::TransferSrc)?;
            p.write_buffer(staging_node, BufferAccess::TransferDst)
        },
        move |c| {
            let (src, dst) = (c.texture(color)?, c.buffer(staging_node)?);
            c.cmd.copy_texture_to_buffer(src, dst);
            Ok(())
        },
    )
    .unwrap();
    b.compile().unwrap();
    b.execute().unwrap();
    assert_eq!(
        device.last_recording(),
        [
            "barrier",
            "beginDebugLabel:scene",
            "beginRendering",
            "endRendering",
            "endDebugLabel",
            "barrier",
            "beginDebugLabel:readback",
            "copyTextureToBuffer",
            "endDebugLabel",
        ]
    );
    let desc = device.describe_texture(b.resolve_texture(color).unwrap()).unwrap();
    assert_eq!(desc.usage, TextureUsage::COLOR_ATTACHMENT | TextureUsage::TRANSFER_SRC);
    let dump = b.dump();
    assert!(pass_block(&dump, "readback").contains("ColorAttachment -> TransferSrc"));
    assert!(pass_block(&dump, "readback").contains("Undefined -> TransferDst"));
}

#[test]
fn present_frame_ends_in_present() {
    let (device, mut rg) = setup();
    let backbuffer = device
        .create_texture(&TextureDesc::new(64, 64, Format::BGRA8Unorm, TextureUsage::TRANSFER_DST | TextureUsage::COLOR_ATTACHMENT, "backbuffer"))
        .unwrap();
    let mut b = rg.begin();
    let color = b.create_texture(&texture("color", 64));
    let swap = b.import_texture(backbuffer, ResourceState::Undefined, ResourceState::Present).unwrap();
    b.add_pass("scene", |p| p.color_target(color), noop).unwrap();
    b.add_pass(
        "blit",
        |p| {
            p.read_texture(color, TextureAccess::TransferSrc)?;
            p.write_texture(swap, TextureAccess::TransferDst)
        },
        noop,
    )
    .unwrap();
    b.compile().unwrap();
    let dump = b.dump();
    assert!(pass_block(&dump, "blit").contains("'backbuffer' Undefined -> TransferDst"));
    assert!(pass_block(&dump, "final").contains("'backbuffer' TransferDst -> Present"));
    assert_eq!(b.resolve_texture(swap).unwrap(), backbuffer); // never pooled, so never acquired or released
}

#[test]
fn buffer_hazards_across_pass_types() {
    // A compute pass fills a buffer that a graphics pass then draws from.
    let (device, mut rg) = setup();
    let mut b = rg.begin();
    let vertices = b.create_buffer(&BufferDesc::new(1024, BufferUsage::NONE, MemoryKind::GpuOnly, "vertices"));
    let color = b.create_texture(&texture("color", 64));
    b.add_pass("generate", |p| p.write_buffer(vertices, BufferAccess::ShaderWrite), dispatch1).unwrap();
    b.add_pass(
        "draw",
        |p| {
            p.read_buffer(vertices, BufferAccess::Vertex)?;
            p.color_target(color)?;
            p.side_effect();
            Ok(())
        },
        draw3,
    )
    .unwrap();
    b.compile().unwrap();
    assert!(pass_block(&b.dump(), "draw").contains("buffer#0 'vertices' ShaderWrite -> VertexOrIndex"));
    let usage = device.describe_buffer(b.resolve_buffer(vertices).unwrap()).unwrap().usage;
    assert_eq!(usage, BufferUsage::STORAGE | BufferUsage::VERTEX);
}

#[test]
fn indirect_arguments_are_a_state_of_their_own() {
    // A compute pass writes draw arguments; a graphics pass consumes them with an indirect draw.
    let (device, mut rg) = setup();
    device.set_features([Feature::DrawIndirectCount]);
    let mut b = rg.begin();
    let args = b.create_buffer(&BufferDesc::new(64, BufferUsage::NONE, MemoryKind::GpuOnly, "args"));
    let color = b.create_texture(&texture("color", 64));
    b.add_pass("cull", |p| p.write_buffer(args, BufferAccess::ShaderWrite), dispatch1).unwrap();
    b.add_pass(
        "draw",
        |p| {
            p.read_buffer(args, BufferAccess::Indirect)?;
            p.color_target(color)?;
            p.side_effect();
            Ok(())
        },
        move |c| {
            let buffer = c.buffer(args)?;
            c.cmd.draw_indexed_indirect_count(buffer, 0, buffer, 0, 16, 20);
            Ok(())
        },
    )
    .unwrap();
    b.compile().unwrap();
    assert!(pass_block(&b.dump(), "draw").contains("ShaderWrite -> IndirectArgs"));
    assert!(device.describe_buffer(b.resolve_buffer(args).unwrap()).unwrap().usage.contains(BufferUsage::INDIRECT));
    b.execute().unwrap();
    assert!(device.last_recording().contains(&"drawIndexedIndirectCount".to_string()));
}

#[test]
fn depth_read_only_after_depth_write() {
    let (_device, mut rg) = setup();
    let mut b = rg.begin();
    let depth = b.create_texture(&TextureDesc::new(64, 64, Format::D32Float, TextureUsage::NONE, "depth"));
    let color = b.create_texture(&texture("color", 64));
    b.add_pass("prepass", |p| p.depth_target(depth), noop).unwrap();
    b.add_pass(
        "shade",
        |p| {
            p.depth_read(depth)?;
            p.color_target(color)?;
            p.side_effect();
            Ok(())
        },
        noop,
    )
    .unwrap();
    b.compile().unwrap();
    let dump = b.dump();
    assert!(pass_block(&dump, "shade").contains("'depth' DepthWrite -> DepthRead"));
    assert!(pass_block(&dump, "prepass").contains("live")); // the read-only use keeps the prepass alive
}

#[test]
fn timestamps_bracket_each_live_pass() {
    let (device, mut rg) = setup();
    device.set_features([Feature::Timestamps]);
    let mut b = rg.begin();
    b.set_timestamps(true);
    b.add_pass(
        "Y",
        |p| {
            p.side_effect();
            Ok(())
        },
        dispatch1,
    )
    .unwrap();
    b.compile().unwrap();
    let ticket = b.execute().unwrap();
    let stamps = device.read_timestamps(ticket).unwrap();
    assert_eq!(stamps.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["Y", "Y/end"]);
}

#[test]
fn dot_shows_passes_and_culling() {
    let (_device, mut rg) = setup();
    let mut b = rg.begin();
    let color = b.create_texture(&texture("color", 64));
    let unused = b.create_texture(&texture("unused", 64));
    let staging = b.create_buffer(&BufferDesc::new(64, BufferUsage::NONE, MemoryKind::Readback, "staging"));
    b.add_pass("dead", |p| p.color_target(unused), noop).unwrap();
    b.add_pass("scene", |p| p.color_target(color), noop).unwrap();
    b.add_pass(
        "readback",
        |p| {
            p.read_texture(color, TextureAccess::TransferSrc)?;
            p.write_buffer(staging, BufferAccess::TransferDst)?;
            p.side_effect();
            Ok(())
        },
        noop,
    )
    .unwrap();
    assert!(matches!(b.to_dot(), Err(GraphError::NotCompiled))); // not compiled yet
    b.compile().unwrap();
    let dot = b.to_dot().unwrap();
    assert!(dot.starts_with("digraph"));
    for needle in ["scene", "readback", "dead\\n(culled)", "color", "TransferSrc", "staging"] {
        assert!(dot.contains(needle), "missing '{needle}' in:\n{dot}");
    }
    assert!(!dot.contains("unused")); // only used by a culled pass: not drawn
}
