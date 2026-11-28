//! The framework through its public API, without a GPU.

use std::time::Duration;

use phox_ui::prelude::*;
use phox_ui::testing::{find, Tester};
use phox_ui::{KeyCode, Role};

// --- A counter: messages, update, view -----------------------------------------------------------------

#[derive(Default)]
struct Counter {
    count: i32,
}

#[derive(Clone, Debug)]
enum CounterMsg {
    Inc,
    Dec,
}

impl App for Counter {
    type Message = CounterMsg;
    fn update(&mut self, message: CounterMsg, _: &mut UpdateCx<CounterMsg>) {
        match message {
            CounterMsg::Inc => self.count += 1,
            CounterMsg::Dec => self.count -= 1,
        }
    }
    fn view(&self) -> Element<CounterMsg> {
        Column::new()
            .gap(8.0)
            .child(Text::new(format!("COUNT {}", self.count)))
            .child(Row::new().gap(8.0).child(Button::new("-").on_press(CounterMsg::Dec)).child(Button::new("+").on_press(CounterMsg::Inc)))
            .into()
    }
}

#[test]
fn tapping_buttons_sends_messages_and_rebuilds_the_view() {
    let mut t = Tester::new(Counter::default(), Size::new(200.0, 100.0));
    t.tap(find::text("+"));
    t.tap(find::text("+"));
    t.tap(find::text("-"));
    assert_eq!(t.app().count, 1);
    assert!(t.exists(find::text("COUNT 1")));
    assert!(!t.exists(find::text("COUNT 0")));
}

#[test]
fn messages_sent_from_outside_the_view_update_it() {
    let mut t = Tester::new(Counter::default(), Size::new(200.0, 100.0));
    t.runtime().send(CounterMsg::Inc);
    t.runtime().send(CounterMsg::Inc);
    assert_eq!(t.app().count, 2);
    assert!(t.exists(find::text("COUNT 2")));
}

#[test]
fn a_press_that_leaves_the_button_does_not_count() {
    let mut t = Tester::new(Counter::default(), Size::new(200.0, 100.0));
    t.drag(find::text("+"), vec2(0.0, 60.0));
    assert_eq!(t.app().count, 0);
}

#[test]
fn touch_works_like_the_mouse() {
    let mut t = Tester::new(Counter::default(), Size::new(200.0, 100.0));
    t.touch(find::text("+"));
    assert_eq!(t.app().count, 1);
}

#[test]
fn the_layout_snapshot_shows_every_widget() {
    let mut t = Tester::new(Counter::default(), Size::new(200.0, 100.0));
    // Text: 7 glyphs at size 10 (capital height 6 units, advance 5.6): (6 * 5.6 + 4) * 10 / 6 = 62.67 wide,
    // line height 16. Buttons: a 6.67-wide glyph plus 2 * 14 padding, 28 high.
    let expected = "\
Column @ 0,0 200x100
  Text \"COUNT 0\" @ 0,0 62.67x16
  Row @ 0,24 200x28
    Button \"-\" @ 0,24 34.67x28
    Button \"+\" @ 42.67,24 34.67x28
";
    assert_eq!(t.layout_snapshot(), expected);
}

// --- Layout --------------------------------------------------------------------------------------------

struct Static(fn() -> Element<()>);

impl App for Static {
    type Message = ();
    fn update(&mut self, _: (), _: &mut UpdateCx<()>) {}
    fn view(&self) -> Element<()> {
        (self.0)()
    }
}

fn boxed(w: f32, h: f32, key: &str) -> Element<()> {
    Element::from(SizedBox::new(w, h)).key(key)
}

#[test]
fn expanded_children_share_the_free_space_by_flex() {
    let mut t = Tester::new(
        Static(|| {
            Row::new()
                .gap(10.0)
                .child(boxed(40.0, 10.0, "fixed"))
                .child(Element::from(Expanded::new(boxed(0.0, 10.0, "one"))))
                .child(Element::from(Expanded::new(boxed(0.0, 10.0, "two")).flex(2)))
                .into()
        }),
        Size::new(250.0, 50.0),
    );
    // 250 - 40 - 2 gaps of 10 = 190 free, split 1:2.
    assert_eq!(t.rect(find::key("fixed")), Rect::new(0.0, 0.0, 40.0, 10.0));
    let one = t.rect(find::key("one"));
    let two = t.rect(find::key("two"));
    assert!((one.w - 190.0 / 3.0).abs() < 1e-3 && (two.w - 2.0 * 190.0 / 3.0).abs() < 1e-3, "{one:?} {two:?}");
    assert!((one.x - 50.0).abs() < 1e-3 && (two.x - (60.0 + 190.0 / 3.0)).abs() < 1e-3);
}

#[test]
fn main_and_cross_axis_alignment() {
    let mut t = Tester::new(
        Static(|| {
            Column::new()
                .main_axis(MainAxis::SpaceBetween)
                .cross_axis(CrossAxis::Center)
                .child(boxed(20.0, 10.0, "a"))
                .child(boxed(40.0, 10.0, "b"))
                .child(boxed(60.0, 10.0, "c"))
                .into()
        }),
        Size::new(100.0, 100.0),
    );
    assert_eq!(t.rect(find::key("a")), Rect::new(40.0, 0.0, 20.0, 10.0));
    assert_eq!(t.rect(find::key("b")), Rect::new(30.0, 45.0, 40.0, 10.0));
    assert_eq!(t.rect(find::key("c")), Rect::new(20.0, 90.0, 60.0, 10.0));
}

#[test]
fn containers_pad_align_and_stack() {
    let mut t = Tester::new(
        Static(|| {
            Stack::new()
                .child(Container::new().padding(10.0).child(boxed(30.0, 20.0, "padded")))
                .child(Container::new().align(Alignment::BOTTOM_RIGHT).child(boxed(10.0, 10.0, "corner")))
                .into()
        }),
        Size::new(100.0, 80.0),
    );
    assert_eq!(t.rect(find::key("padded")), Rect::new(10.0, 10.0, 30.0, 20.0));
    assert_eq!(t.rect(find::key("corner")), Rect::new(90.0, 70.0, 10.0, 10.0));
}

// --- Controls ------------------------------------------------------------------------------------------

#[derive(Default)]
struct Controls {
    gain: f32,
    cutoff: f32,
    on: bool,
    name: String,
    submitted: Vec<String>,
}

#[derive(Clone, Debug)]
enum Ctl {
    Gain(f32),
    Cutoff(f32),
    On(bool),
    Name(String),
    Submit,
}

impl App for Controls {
    type Message = Ctl;
    fn update(&mut self, message: Ctl, _: &mut UpdateCx<Ctl>) {
        match message {
            Ctl::Gain(v) => self.gain = v,
            Ctl::Cutoff(v) => self.cutoff = v,
            Ctl::On(v) => self.on = v,
            Ctl::Name(s) => self.name = s,
            Ctl::Submit => self.submitted.push(self.name.clone()),
        }
    }
    fn view(&self) -> Element<Ctl> {
        Column::new()
            .gap(8.0)
            .child(Element::from(Slider::new(self.gain, 0.0, 10.0).width(120.0).on_change(Ctl::Gain)).key("gain"))
            .child(Element::from(Knob::new(self.cutoff, 0.0, 1.0).label("CUTOFF").on_change(Ctl::Cutoff)).key("cutoff"))
            .child(Toggle::new(self.on).label("BYPASS").on_toggle(Ctl::On))
            .child(Element::from(TextField::new(self.name.clone()).width(150.0).on_change(Ctl::Name).on_submit(Ctl::Submit)).key("name"))
            .into()
    }
}

#[test]
fn sliders_follow_the_pointer_and_clamp() {
    let mut t = Tester::new(Controls::default(), Size::new(300.0, 300.0));
    let r = t.rect(find::key("gain"));
    // The track runs from the left thumb radius (7) to the right one.
    t.tap_at(vec2(r.x + 7.0 + (r.w - 14.0) * 0.25, r.center().y));
    assert!((t.app().gain - 2.5).abs() < 1e-3, "{}", t.app().gain);
    t.drag(find::key("gain"), vec2(1000.0, 0.0));
    assert_eq!(t.app().gain, 10.0, "dragged far past the end");
}

#[test]
fn the_wheel_steps_knobs_and_sliders() {
    let mut t = Tester::new(Controls::default(), Size::new(300.0, 300.0));
    t.scroll(find::key("cutoff"), vec2(0.0, -1.0)); // a notch up
    assert!((t.app().cutoff - 0.01).abs() < 1e-6);
    t.scroll(find::key("gain"), vec2(0.0, -3.0));
    assert!((t.app().gain - 0.1).abs() < 1e-5, "{}", t.app().gain);
}

#[test]
fn knobs_turn_with_vertical_drags() {
    let mut t = Tester::new(Controls::default(), Size::new(300.0, 300.0));
    t.drag(find::key("cutoff"), vec2(0.0, -100.0)); // half of the 200-pixel range
    assert!((t.app().cutoff - 0.5).abs() < 1e-3, "{}", t.app().cutoff);
    t.drag(find::key("cutoff"), vec2(0.0, 500.0));
    assert_eq!(t.app().cutoff, 0.0);
}

#[test]
fn toggles_flip() {
    let mut t = Tester::new(Controls::default(), Size::new(300.0, 300.0));
    t.tap(find::text("BYPASS"));
    assert!(t.app().on);
    t.tap(find::text("BYPASS"));
    assert!(!t.app().on);
}

#[test]
fn text_fields_take_typing_editing_and_enter() {
    let mut t = Tester::new(Controls::default(), Size::new(300.0, 300.0));
    t.type_text("ignored"); // nothing has focus yet
    assert_eq!(t.app().name, "");
    t.tap(find::key("name"));
    t.type_text("SAW");
    t.press_key(KeyCode::Left);
    t.type_text("X");
    assert_eq!(t.app().name, "SAXW");
    t.press_key(KeyCode::End);
    t.press_key(KeyCode::Backspace);
    t.press_key(KeyCode::Home);
    t.press_key(KeyCode::Delete);
    assert_eq!(t.app().name, "AX");
    t.press_key(KeyCode::Enter);
    assert_eq!(t.app().submitted, ["AX"]);
}

#[test]
fn text_fields_select_copy_cut_and_paste() {
    let mut t = Tester::new(Controls::default(), Size::new(300.0, 300.0));
    t.paste("ignored"); // nothing has focus yet
    assert_eq!(t.app().name, "");
    t.tap(find::key("name"));
    t.paste("PK_LIVE_123\n");
    assert_eq!(t.app().name, "PK_LIVE_123", "a trailing line break isn't pasted");
    let shift = phox_ui::Modifiers { shift: true, ..Default::default() };
    for _ in 0..3 {
        t.press_key_with(KeyCode::Left, shift);
    }
    t.press_key(KeyCode::Copy);
    assert_eq!(t.clipboard().as_deref(), Some("123"));
    t.type_text("9");
    assert_eq!(t.app().name, "PK_LIVE_9", "typing replaces the selection");
    t.press_key(KeyCode::SelectAll);
    t.press_key(KeyCode::Cut);
    assert_eq!(t.clipboard().as_deref(), Some("PK_LIVE_9"));
    assert_eq!(t.app().name, "");
    t.paste("ABCD");
    t.press_key(KeyCode::Left);
    t.press_key_with(KeyCode::Home, shift);
    t.press_key(KeyCode::Backspace);
    assert_eq!(t.app().name, "D", "Backspace deletes the selection");
    t.press_key(KeyCode::Copy);
    assert_eq!(t.clipboard().as_deref(), Some("PK_LIVE_9"), "copy with nothing selected leaves the clipboard alone");
}

#[test]
fn update_can_copy_to_the_clipboard() {
    struct Copier;
    impl App for Copier {
        type Message = ();
        fn update(&mut self, _: (), cx: &mut UpdateCx<()>) {
            cx.set_clipboard("http://example.com/callback");
        }
        fn view(&self) -> Element<()> {
            Button::new("COPY").on_press(()).into()
        }
    }
    let mut t = Tester::new(Copier, Size::new(200.0, 100.0));
    assert_eq!(t.clipboard(), None);
    t.tap(find::text("COPY"));
    assert_eq!(t.clipboard().as_deref(), Some("http://example.com/callback"));
}

#[test]
fn dragging_across_a_text_field_selects() {
    let mut t = Tester::new(Controls::default(), Size::new(300.0, 300.0));
    t.tap(find::key("name"));
    t.paste("HELLO");
    let r = t.rect(find::key("name"));
    let adv = 10.0 * 5.6 / 6.0; // the default theme's text size
    t.drag_from(vec2(r.x + 8.0 + 0.2, r.y + r.h / 2.0), vec2(adv * 2.0, 0.0));
    t.press_key(KeyCode::Copy);
    assert_eq!(t.clipboard().as_deref(), Some("HE"));
    // A long value scrolls so the caret stays in view.
    t.press_key(KeyCode::End);
    t.paste(&"X".repeat(60));
    let before = t.pump().vertices().len();
    assert!(before > 0);
}

#[test]
fn tab_moves_focus_and_keys_drive_the_focused_control() {
    let mut t = Tester::new(Controls::default(), Size::new(300.0, 300.0));
    t.press_key(KeyCode::Tab); // the slider
    t.press_key(KeyCode::End);
    assert_eq!(t.app().gain, 10.0);
    t.press_key(KeyCode::Tab); // the knob
    t.press_key(KeyCode::Up);
    assert!((t.app().cutoff - 0.01).abs() < 1e-6);
    t.press_key(KeyCode::Tab); // the toggle
    t.press_key(KeyCode::Space);
    assert!(t.app().on);
    t.press_key_with(KeyCode::Tab, phox_ui::Modifiers { shift: true, ..Default::default() }); // back to the knob
    t.press_key(KeyCode::Home);
    assert_eq!(t.app().cutoff, 0.0);
}

// --- Scrolling and clipping ----------------------------------------------------------------------------

#[test]
fn scroll_areas_move_their_content_and_clamp() {
    let mut t = Tester::new(
        Static(|| {
            Element::from(Scroll::new(Column::new().children((0..10).map(|i| boxed(50.0, 30.0, &format!("row{i}")))))).key("list")
        }),
        Size::new(100.0, 100.0),
    );
    assert_eq!(t.rect(find::key("row1")).y, 30.0);
    t.scroll(find::key("list"), vec2(0.0, 45.0));
    assert_eq!(t.rect(find::key("row1")).y, -15.0);
    t.scroll(find::key("list"), vec2(0.0, 1000.0));
    assert_eq!(t.rect(find::key("row9")).y, 70.0, "clamped: the last row ends at the bottom");
    t.scroll(find::key("list"), vec2(0.0, -1000.0));
    assert_eq!(t.rect(find::key("row0")).y, 0.0);
}

#[test]
fn widgets_scrolled_out_of_view_cannot_be_hit() {
    struct Buttons(Vec<u32>);
    impl App for Buttons {
        type Message = u32;
        fn update(&mut self, i: u32, _: &mut UpdateCx<u32>) {
            self.0.push(i);
        }
        fn view(&self) -> Element<u32> {
            Column::new()
                .child(Element::from(SizedBox::height(50.0).child(Scroll::new(Column::new().children((0..5).map(|i| Button::new(format!("B{i}")).on_press(i)))))))
                .into()
        }
    }
    let mut t = Tester::new(Buttons(Vec::new()), Size::new(100.0, 200.0));
    t.tap(find::text("B0"));
    assert_eq!(t.app().0, [0], "visible buttons work");
    // B2 is laid out at y = 56..84, below the 50-pixel viewport: a click there hits nothing.
    let b2 = t.rect(find::text("B2"));
    assert!(b2.y >= 50.0);
    t.tap_at(b2.center());
    assert_eq!(t.app().0, [0], "B2 is clipped away: the tap hits nothing");
    let viewport = t.rect(find::role(Role::Scroll));
    assert_eq!(viewport.h, 50.0);
    let list = t.pump().clone();
    assert!(list.batches().iter().any(|b| b.clip == Some(viewport)), "the scroll area clips its content");
}

// --- Components, keys, timers, frames -------------------------------------------------------------------

/// A component with its own message type, used inside a parent app with `Element::map`.
#[derive(Clone, Debug)]
enum Child {
    Clicked,
}

fn child_view(label: &str) -> Element<Child> {
    Button::new(label).on_press(Child::Clicked).into()
}

#[derive(Default)]
struct Parent {
    left: u32,
    right: u32,
}

#[derive(Clone, Debug)]
enum ParentMsg {
    Left(Child),
    Right(Child),
}

impl App for Parent {
    type Message = ParentMsg;
    fn update(&mut self, message: ParentMsg, _: &mut UpdateCx<ParentMsg>) {
        match message {
            ParentMsg::Left(Child::Clicked) => self.left += 1,
            ParentMsg::Right(Child::Clicked) => self.right += 1,
        }
    }
    fn view(&self) -> Element<ParentMsg> {
        Row::new().child(child_view("L").map(ParentMsg::Left)).child(child_view("R").map(ParentMsg::Right)).into()
    }
}

#[test]
fn mapped_components_send_their_parents_messages() {
    let mut t = Tester::new(Parent::default(), Size::new(200.0, 50.0));
    t.tap(find::text("R"));
    t.tap(find::text("R"));
    t.tap(find::text("L"));
    assert_eq!((t.app().left, t.app().right), (1, 2));
}

/// Knob state (a drag in progress) belongs to a keyed widget and survives reordering.
struct Reorder {
    swapped: bool,
    values: [f32; 2],
}

#[derive(Clone)]
enum ReorderMsg {
    Set(usize, f32),
    Swap,
}

impl App for Reorder {
    type Message = ReorderMsg;
    fn update(&mut self, message: ReorderMsg, _: &mut UpdateCx<ReorderMsg>) {
        match message {
            ReorderMsg::Set(i, v) => self.values[i] = v,
            ReorderMsg::Swap => self.swapped = !self.swapped,
        }
    }
    fn view(&self) -> Element<ReorderMsg> {
        let knob = |i: usize| Element::from(Knob::new(self.values[i], 0.0, 1.0).on_change(move |v| ReorderMsg::Set(i, v))).key(format!("k{i}"));
        let order = if self.swapped { [1, 0] } else { [0, 1] };
        Row::new().child(knob(order[0])).child(knob(order[1])).child(Button::new("SWAP").on_press(ReorderMsg::Swap)).into()
    }
}

#[test]
fn keyed_widgets_keep_their_identity_when_reordered() {
    let mut t = Tester::new(Reorder { swapped: false, values: [0.0, 0.0] }, Size::new(300.0, 100.0));
    let before = t.get(find::key("k0")).id;
    t.tap(find::text("SWAP"));
    let after = t.get(find::key("k0"));
    assert_eq!(after.id, before, "same node, same state");
    assert!(after.rect.x > 0.0, "now second in the row");
    t.drag(find::key("k0"), vec2(0.0, -40.0));
    assert!((t.app().values[0] - 0.2).abs() < 1e-3);
    assert_eq!(t.app().values[1], 0.0);
}

struct Blink {
    on: bool,
}

#[derive(Clone)]
struct Tick;

impl App for Blink {
    type Message = Tick;
    fn update(&mut self, _: Tick, cx: &mut UpdateCx<Tick>) {
        self.on = !self.on;
        cx.after(Duration::from_millis(500), Tick);
    }
    fn view(&self) -> Element<Tick> {
        Column::new().child(Text::new(if self.on { "ON" } else { "OFF" })).child(Button::new("START").on_press(Tick)).into()
    }
}

#[test]
fn timers_follow_the_fake_clock() {
    let mut t = Tester::new(Blink { on: false }, Size::new(100.0, 100.0));
    t.tap(find::text("START"));
    assert!(t.exists(find::text("ON")));
    t.advance(Duration::from_millis(499));
    assert!(t.exists(find::text("ON")), "not due yet");
    t.advance(Duration::from_millis(1));
    assert!(t.exists(find::text("OFF")));
    t.advance(Duration::from_millis(1000)); // two more ticks
    assert!(t.exists(find::text("OFF")));
    assert_eq!(t.runtime().next_timer(), Some(Duration::from_millis(2000)));
}

#[test]
fn frames_happen_only_when_something_changed() {
    let mut t = Tester::new(Counter::default(), Size::new(200.0, 100.0));
    assert!(t.needs_frame(), "the first frame");
    t.pump();
    assert!(!t.needs_frame());
    // Moving the mouse over empty space changes nothing.
    t.runtime().event(phox_ui::Event::mouse(phox_ui::Phase::Move, vec2(190.0, 90.0)));
    assert!(!t.needs_frame());
    // Hovering a button repaints (its hover colour); clicking it changes the view.
    t.hover(find::text("+"));
    assert!(t.needs_frame());
    t.pump();
    t.tap(find::text("+"));
    assert!(t.needs_frame());
    let drawn = t.pump().vertices().len();
    assert!(drawn > 0);
    assert!(!t.needs_frame());
}

#[test]
fn the_same_state_and_input_draw_the_same_frame() {
    let run = || {
        let mut t = Tester::new(Controls::default(), Size::new(300.0, 300.0));
        t.drag(find::key("gain"), vec2(30.0, 0.0));
        t.tap(find::key("name"));
        t.type_text("AB");
        t.pump().vertices().to_vec()
    };
    assert_eq!(run(), run());
}

#[test]
#[should_panic(expected = "2 widgets match")]
fn ambiguous_finders_fail_clearly() {
    let mut t = Tester::new(Counter::default(), Size::new(200.0, 100.0));
    t.tap(find::role(Role::Button));
}
