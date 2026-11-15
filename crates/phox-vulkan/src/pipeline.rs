//! Descriptor set layouts, pipeline layouts and pipelines.

use crate::convert;
use crate::device::{Inner, PipelineRes, State, backend};
use ash::vk;
use phox_rhi::*;

pub(crate) fn descriptor_type(t: BindingType) -> vk::DescriptorType {
    match t {
        BindingType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
        BindingType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        BindingType::SampledTexture | BindingType::BindlessTextures => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        BindingType::StorageTexture => vk::DescriptorType::STORAGE_IMAGE,
    }
}

fn type_id(t: BindingType) -> u32 {
    t as u32
}

impl Inner {
    /// The layout of one set. Layouts are cached by what they contain, so equal sets share one object.
    fn set_layout(&self, st: &mut State, bindings: &[&Binding]) -> Result<vk::DescriptorSetLayout> {
        let key: crate::device::SetLayoutKey =
            bindings.iter().map(|b| (b.binding, type_id(b.ty), b.stages.bits(), b.count, b.sampler)).collect();
        if let Some(&layout) = st.set_layouts.get(&key) {
            return Ok(layout);
        }
        // Samplers are baked into the layout, so a descriptor write needs only a view.
        let mut samplers: Vec<Vec<vk::Sampler>> = Vec::new();
        for b in bindings {
            samplers.push(if b.ty == BindingType::SampledTexture { vec![self.sampler(st, &b.sampler)?; b.count as usize] } else { Vec::new() });
        }
        let vk_bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = bindings
            .iter()
            .zip(&samplers)
            .map(|(b, s)| {
                let mut binding = vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(descriptor_type(b.ty))
                    .descriptor_count(b.count)
                    .stage_flags(convert::shader_stages(b.stages));
                if !s.is_empty() {
                    binding = binding.immutable_samplers(s);
                }
                binding
            })
            .collect();
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_bindings);
        let layout = unsafe { self.device.create_descriptor_set_layout(&info, None) }.map_err(backend)?;
        st.set_layouts.insert(key, layout);
        Ok(layout)
    }

    /// The pipeline layout for `bindings` (sets 0..=max; a set without bindings is empty, and a set with a
    /// bindless binding is the device's texture table) and push constant ranges.
    fn pipeline_layout(
        &self,
        st: &mut State,
        bindings: &[Binding],
        push_constants: &[PushConstantRange],
    ) -> Result<(vk::PipelineLayout, Vec<Vec<Binding>>, Vec<vk::DescriptorSetLayout>)> {
        let set_count = bindings.iter().map(|b| b.set + 1).max().unwrap_or(0) as usize;
        let mut set_bindings: Vec<Vec<Binding>> = vec![Vec::new(); set_count];
        for b in bindings {
            set_bindings[b.set as usize].push(*b);
        }
        let mut layouts = Vec::with_capacity(set_count);
        for set in &set_bindings {
            let layout = if set.iter().any(|b| b.ty == BindingType::BindlessTextures) {
                st.bindless.as_ref().ok_or_else(|| RhiError::Unsupported("BindlessDescriptors".into()))?.layout
            } else {
                let refs: Vec<&Binding> = set.iter().collect();
                self.set_layout(st, &refs)?
            };
            layouts.push(layout);
        }
        let ranges: Vec<vk::PushConstantRange> = push_constants
            .iter()
            .map(|r| vk::PushConstantRange { stage_flags: convert::shader_stages(r.stages), offset: r.offset, size: r.size })
            .collect();
        let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges);
        let layout = unsafe { self.device.create_pipeline_layout(&info, None) }.map_err(backend)?;
        Ok((layout, set_bindings, layouts))
    }

    pub(crate) fn create_graphics_pipeline(&self, desc: &GraphicsPipelineDesc) -> Result<PipelineHandle> {
        let mut st = self.lock();
        let mut stages = Vec::new();
        let (mut vertex, mut mesh, mut fragment) = (false, false, false);
        let mut seen: Vec<ShaderStage> = Vec::new();
        let mut entries = Vec::new();
        for &handle in &desc.shaders {
            let shader = st.shaders.get(handle)?;
            if seen.contains(&shader.stage) {
                return Err(RhiError::InvalidDesc("create_graphics_pipeline: duplicate shader stage".into()));
            }
            seen.push(shader.stage);
            vertex |= shader.stage == ShaderStage::Vertex;
            mesh |= shader.stage == ShaderStage::Mesh;
            fragment |= shader.stage == ShaderStage::Fragment;
            if shader.stage == ShaderStage::Compute {
                return Err(RhiError::InvalidDesc("create_graphics_pipeline: compute shader in a graphics pipeline".into()));
            }
            entries.push((shader.module, shader.stage, shader.entry.clone()));
        }
        if !fragment || (!vertex && !mesh) {
            return Err(RhiError::InvalidDesc("create_graphics_pipeline: needs a Fragment shader and a Vertex or Mesh shader".into()));
        }
        if mesh && !self.info.mesh_shaders {
            return Err(RhiError::Unsupported("MeshShader".into()));
        }
        for (module, stage, entry) in &entries {
            stages.push(vk::PipelineShaderStageCreateInfo::default().stage(convert::shader_stage(*stage)).module(*module).name(entry));
        }

        let (layout, set_bindings, set_layouts) = self.pipeline_layout(&mut st, &desc.bindings, &desc.push_constants)?;

        let vertex_bindings = [vk::VertexInputBindingDescription::default().binding(0).stride(desc.vertex_layout.stride).input_rate(vk::VertexInputRate::VERTEX)];
        let attributes: Vec<vk::VertexInputAttributeDescription> = desc
            .vertex_layout
            .attributes
            .iter()
            .map(|a| vk::VertexInputAttributeDescription::default().location(a.location).binding(0).format(convert::vertex_format(a.format)).offset(a.offset))
            .collect();
        let vertex_input = if attributes.is_empty() {
            vk::PipelineVertexInputStateCreateInfo::default()
        } else {
            vk::PipelineVertexInputStateCreateInfo::default().vertex_binding_descriptions(&vertex_bindings).vertex_attribute_descriptions(&attributes)
        };
        let topology = match desc.topology {
            Topology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
            Topology::LineList => vk::PrimitiveTopology::LINE_LIST,
            Topology::PointList => vk::PrimitiveTopology::POINT_LIST,
        };
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(topology);
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let cull = match desc.cull {
            CullMode::None => vk::CullModeFlags::NONE,
            CullMode::Back => vk::CullModeFlags::BACK,
            CullMode::Front => vk::CullModeFlags::FRONT,
        };
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(cull)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_format = convert::format(desc.depth_format);
        let depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(desc.depth_test)
            .depth_write_enable(desc.depth_write)
            .depth_compare_op(convert::compare(desc.depth_compare));
        let write_mask = vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A;
        let blend_state = if desc.blend.is_some_and(|b| b.enabled) {
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(write_mask)
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
        } else {
            vk::PipelineColorBlendAttachmentState::default().color_write_mask(write_mask)
        };
        let blends = vec![blend_state; desc.color_formats.len()];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let color_formats: Vec<vk::Format> = desc.color_formats.iter().map(|&f| convert::format(f)).collect();
        let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
        if desc.depth_format != Format::Undefined {
            rendering = rendering.depth_attachment_format(depth_format);
            if desc.depth_format == Format::D24UnormS8Uint {
                rendering = rendering.stencil_attachment_format(depth_format);
            }
        }
        let mut info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering);
        if !mesh {
            info = info.vertex_input_state(&vertex_input);
        }
        let pipelines = unsafe { self.device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }.map_err(|(_, e)| {
            unsafe { self.device.destroy_pipeline_layout(layout, None) };
            backend(format!("failed to create graphics pipeline '{}': {e}", desc.debug_name))
        })?;
        self.set_name(pipelines[0], &desc.debug_name);
        Ok(st.pipelines.create(PipelineRes { pipeline: pipelines[0], layout, bind_point: vk::PipelineBindPoint::GRAPHICS, set_bindings, set_layouts }))
    }

    pub(crate) fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> Result<PipelineHandle> {
        let mut st = self.lock();
        let shader = st.shaders.get(desc.shader)?;
        if shader.stage != ShaderStage::Compute {
            return Err(RhiError::InvalidDesc("create_compute_pipeline: shader is not a Compute shader".into()));
        }
        let (module, entry) = (shader.module, shader.entry.clone());
        let (layout, set_bindings, set_layouts) = self.pipeline_layout(&mut st, &desc.bindings, &desc.push_constants)?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&entry);
        let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
        let pipelines = unsafe { self.device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) }.map_err(|(_, e)| {
            unsafe { self.device.destroy_pipeline_layout(layout, None) };
            backend(format!("failed to create compute pipeline '{}': {e}", desc.debug_name))
        })?;
        self.set_name(pipelines[0], &desc.debug_name);
        Ok(st.pipelines.create(PipelineRes { pipeline: pipelines[0], layout, bind_point: vk::PipelineBindPoint::COMPUTE, set_bindings, set_layouts }))
    }
}
