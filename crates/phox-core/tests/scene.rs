//! The CPU half of test/source/scene_test.cpp: the change log and the sample scenes. (The renderer-reuse
//! half needs a GPU renderer and is ported with the GPU-driven scene.)

use phox_core::math::{Quat, Vec3};
use phox_core::sample_scenes::*;
use phox_core::scene::*;
use std::sync::Arc;

#[test]
fn change_log_tracks_objects() {
    let mut scene = Scene::new();
    let cube = Arc::new(Mesh::cube(1.0));
    let with_cube = || ObjectDesc { mesh: Some(cube.clone()), ..Default::default() };
    let [a, b] = scene.add_objects([with_cube(), with_cube()]).unwrap();
    let mut cursor = ChangeCursor::default();
    assert!(scene.changes(&mut cursor).is_none()); // a new reader looks at everything
    assert!(scene.changes(&mut cursor).unwrap().is_empty());

    // Changed twice, logged once; reading logs nothing.
    scene.at_mut(a).unwrap().transform_mut().position.x = 1.0;
    scene.at_mut(a).unwrap().material_mut().roughness = 0.2;
    assert_eq!(scene.at(b).unwrap().transform().position.x, 0.0);
    assert_eq!(scene.changes(&mut cursor).unwrap(), vec![a.index]);
    // Once read, a change is logged again.
    scene.at_mut(a).unwrap().set_mesh(None);
    assert_eq!(scene.changes(&mut cursor).unwrap(), vec![a.index]);

    // Removal moves b into a's place in the object array, but b keeps its index; both are logged.
    assert!(scene.remove_object(a));
    let c = scene.add_object(with_cube()).unwrap();
    assert_eq!(scene.changes(&mut cursor).unwrap(), vec![a.index, c.index]);
    assert!(c.index == a.index && std::ptr::eq(scene.object_at(c.index).unwrap(), scene.at(c).unwrap()));
    scene.at_mut(b).unwrap().transform_mut().position.y = 2.0;
    assert_eq!(scene.changes(&mut cursor).unwrap(), vec![b.index]);
    assert!(scene.object_at(99).is_none());

    // A moved scene keeps its log, and its objects still log to it.
    let mut moved = scene;
    moved.at_mut(b).unwrap().transform_mut().position.y = 3.0;
    assert_eq!(moved.changes(&mut cursor).unwrap().len(), 1);
    // A copy has a log of its own: its objects log there, and a cursor from the original reads it whole.
    let mut copy = moved.clone();
    copy.at_mut(b).unwrap().transform_mut().position.y = 4.0;
    assert!(moved.changes(&mut cursor).unwrap().is_empty());
    let mut copy_cursor = cursor;
    assert!(copy.changes(&mut copy_cursor).is_none());

    // Unread, the log starts over once it outgrows the scene; a reader that fell behind looks at everything.
    let mut behind = ChangeCursor::default();
    copy.changes(&mut behind);
    for i in 0..3000 {
        copy.at_mut(b).unwrap().transform_mut().position.z = i as f32;
        copy.changes(&mut copy_cursor); // keeps reading, so every change is logged
    }
    assert!(copy.changes(&mut behind).is_none());
    assert!(copy.changes(&mut copy_cursor).unwrap().is_empty());
}

#[test]
fn guards_bump_revisions_and_dropped_guards_log() {
    let mut scene = Scene::new();
    let id = scene.add_object(ObjectDesc::default()).unwrap();
    let mut cursor = ChangeCursor::default();
    scene.changes(&mut cursor);
    let (t, m) = (scene.at(id).unwrap().transform_revision(), scene.at(id).unwrap().material_revision());
    {
        let mut guard = scene.at_mut(id).unwrap().transform_mut();
        guard.position = Vec3::ONE;
    }
    assert_eq!(scene.at(id).unwrap().transform_revision(), t + 1);
    assert_eq!(scene.at(id).unwrap().material_revision(), m);
    scene.at_mut(id).unwrap().material_mut().metallic = 1.0;
    assert_eq!(scene.at(id).unwrap().material_revision(), m + 1);
    assert_eq!(scene.changes(&mut cursor).unwrap(), vec![id.index]);
}

#[test]
fn sample_scenes_are_built_and_named() {
    assert_eq!(SAMPLE_SCENES.len(), 6);
    for id in SAMPLE_SCENES {
        let sample = make_sample_scene(id).unwrap();
        assert!(sample.scene.object_count() > 0);
        assert!(sample.name == sample_scene_name(id) && parse_sample_scene(&sample.name) == Some(id));
    }
    assert!(parse_sample_scene("nope").is_none());
    assert_eq!(make_sample_scene(SampleSceneId::Field).unwrap().scene.object_count(), 24 * 24);
    let mut cubes = make_sample_scene(SampleSceneId::Cubes).unwrap();
    assert_eq!(cubes.scene.object_count(), 6);
    let red = cubes.scene.object_ids()[1];
    assert!(cubes.scene.contains(red));
    let before: Quat = cubes.scene.at(red).unwrap().transform().rotation;
    (cubes.animate)(&mut cubes.scene, 1.0);
    assert_ne!(cubes.scene.at(red).unwrap().transform().rotation, before);
}
