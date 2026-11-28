# phox-ui: design

A declarative UI framework on top of `phox-immediate`. You write apps in the Elm style: state, `update`,
`view`. The widgets and layout feel like Flutter, and you build them with builder calls rather than an HTML-like
definition. The framework can live in a window it creates itself, or embed in a window someone else owns, such as
an audio plugin host. Either way the app code is the same.

Status: **the core is built** (`crates/phox-ui`), steps 1 to 4 of the build order below: clip rects in
`phox-immediate` (with a new `set_scissor` command in the RHI), the Elm runtime, the persistent widget tree,
Flutter-style layout, Text, Container, Padding, SizedBox, Align, Expanded, Stack, Row, Column, Scroll, Button,
Toggle, Slider, Knob and TextField, keyboard focus, timers, `Element::map`, and `testing::Tester`.
`tests/widgets.rs` covers them without a GPU; `tests/render.rs` draws a panel on the GPU.

Not built yet: Dropdown (it needs an overlay layer), touch-drag scrolling and the example app (step 6).

## Goals

- **Declarative.** `view(&state)` returns a tree of widgets. Change happens only through messages passed to
  `update`.
- **Testable without a GPU.** Building the view, layout, input handling and painting into a `DrawList` are plain
  CPU code. Only submitting the `DrawList` touches a device.
- **Deterministic.** The same state, the same input and the same clock reading always give the same layout and
  the same draw list.
- **Embeddable.** A host can hand the framework a parent view, and a helper makes the app feel as if it owns the
  window.
- **Idle costs nothing.** A frame runs only when something has changed.

Out of scope for the MVP: images and textures, real fonts and text shaping, animation, accessibility output,
markup or macro syntax, hot reload (the toolchain's job), and anything specific to audio plugins.

## Principles

1. **Widgets never touch the GPU.** They paint into a `DrawList`. `ImmediateRenderer` is the only thing that
   submits work.
2. **No global mutable state.** Several UIs can run in one process, for example several plugin instances. The
   GPU device is shared as an `Arc<dyn Device>` passed in by the caller.
3. **Time is an input.** The runtime receives the clock reading with each frame and never reads the system
   clock itself. Tests drive a fake clock.
4. **Everything happens on one thread.** UI state is not `Send`. The window host calls the runtime from the
   thread that owns the view, which on macOS is the main thread.
5. **Each widget has a stable identity and a semantic role**, such as button, slider or text, from the start.
   That gives tests something reliable to find, and leaves room for AccessKit later without a redesign.
6. **Only builders for now.** Every widget is a struct with chainable setters. A macro layer can come later.

## Crates

| Crate | Contents | Depends on a device? |
|---|---|---|
| `phox-ui` | Widgets, layout, input, the Elm runtime, theme, and the `testing` module | No (only `phox-immediate::DrawList`) |
| `phox-ui-host` | Window helper: a standalone window, or embedding in a parent view; turns OS events into `Input`; schedules frames; calls `ImmediateRenderer` | Yes |

The `testing` module is part of `phox-ui` itself, so apps can use it from their own tests with no extra crate.

## The app model

```rust
pub trait App {
    type Message: Clone + 'static;

    fn update(&mut self, message: Self::Message, ctx: &mut UpdateCtx);
    fn view(&self) -> Element<Self::Message>;
    fn theme(&self) -> Theme { Theme::default() }
}
```

`UpdateCtx` lets `update` ask for a redraw, change focus, or set a timer. Timers are how time-driven updates work
without animation in the MVP. Messages from outside the view (data from another thread, a host's parameter
changes) go in through `Runtime::send`, the way Elm's subscriptions do.

A counter app:

```rust
fn view(&self) -> Element<Msg> {
    Column::new()
        .gap(8.0)
        .padding(16.0)
        .child(Text::new(format!("COUNT {}", self.count)))
        .child(Row::new()
            .gap(8.0)
            .child(Button::new("-").on_press(Msg::Dec))
            .child(Button::new("+").on_press(Msg::Inc)))
        .into()
}
```

## Widgets

```rust
pub trait Widget: 'static {
    fn role(&self) -> Role;                            // Button, Slider, Text, Container...
    fn label(&self) -> Option<&str>;                   // What finders match by text
    fn children(&self) -> &[AnyElement];
    fn flex(&self) -> u32;                             // Share of a Row or Column's free space
    fn interactive(&self) -> bool;                     // Receives pointer events
    fn focusable(&self) -> bool;                       // Visited by Tab
    fn clips(&self) -> bool;                           // Clips its children (Scroll)
    fn new_state(&self) -> Box<dyn Any>;               // Internal state, kept between views
    fn layout(&self, cx: &mut LayoutCx, c: Constraints) -> Size;
    fn event(&self, cx: &mut EventCx, e: &Event);      // May emit messages and change its own state
    fn paint(&self, cx: &mut PaintCx);                 // Draws into the DrawList through cx
    fn map_message(&self, m: Box<dyn Any>) -> Box<dyn Any>;
}
// Every method but layout has a default. Keys are set on the element: `Element::from(w).key("cutoff")`.
```

**Internal state** covers things like hover, pressed, focus, scroll offset, and the cursor in a text field. It
lives in a persistent tree, not in the widget values that `view` creates. After each `view`, the new tree is
matched against the old one by position and key, as in iced's `Tree` or Flutter's element tree, so internal
state survives rebuilds. App state stays in the `App`, as Elm intends.

**Widgets are not generic over the message type.** They emit messages type-erased (`Box<dyn Any>`), and the
typed `Element<M>` wrapper keeps the builders type-safe. That is what makes `Element::map` work: a `Map` node
converts every message that bubbles up through it, so a component written against its own message type can
sit inside any app. The runtime downcasts to `App::Message` at the top, and a mismatch is a panic that names
`Element::map`.

**MVP widgets:** Text, Button, Toggle, Slider, Knob, TextField, Dropdown, Row, Column, Stack, Padding, Align,
SizedBox, Expanded, Container (background, border, radius) and Scroll.

## Layout

Layout follows Flutter's box protocol: constraints go down, sizes come up, and the parent sets each child's
position.

- `Constraints { min: Size, max: Size }`. A widget returns a size within its constraints.
- Row and Column use Flutter's flex rules. Fixed-size children are measured first. `Expanded` and `Flexible`
  children share whatever space is left, by flex factor. The main-axis and cross-axis alignments match Flutter's.
- Scroll gives its child unbounded height (or width) along the scroll axis, and clips to its own bounds.
- Results are cached per subtree, keyed by the incoming constraints plus a dirty flag that a message, a change
  to the widget's internal state, or a new view sets.
- Sizes are in logical pixels. The host supplies the scale factor, and it is passed to `DrawList::begin`.

## Input

```rust
pub enum Event {
    Pointer { id: PointerId, kind: PointerKind /* Mouse, Touch, Pen */, phase: Phase /* Down, Move, Up, Cancel */,
              pos: Vec2, buttons: Buttons },
    Scroll { pos: Vec2, delta: Vec2 },
    Key { key: Key, phase: KeyPhase, modifiers: Modifiers },
    Text(String),
    Focus(bool),   // The window gained or lost focus
}
```

- **Hit-testing** walks the laid-out tree from front to back.
- **Capture:** a pointer down on a widget captures that pointer until it is released, so dragging a slider or
  knob keeps working outside the widget.
- **Touch:** each pointer id is tracked separately. MVP widgets respond to one pointer at a time.
- **Focus:** Tab and Shift+Tab move through focusable widgets in tree order. Keys go to the focused widget
  first, then bubble up to its parents.
- **Clipboard:** widgets never touch the system clipboard. The host turns the platform's shortcuts into
  `KeyCode::SelectAll`, `Copy` and `Cut`, and on paste reads the clipboard and sends `Event::Paste(text)` to the
  focused widget. A widget copies with `EventCx::set_clipboard`, and the host collects that with
  `Runtime::take_clipboard()` after each event. An app's `update` can copy too (`UpdateCx::set_clipboard`, for "copy" buttons). `Tester::paste` and `Tester::clipboard` stand in for the system.
  TextField supports selection (Shift+arrows, Home/End, mouse drag, select-all), copy, cut and paste, and scrolls
  long text to keep the caret visible.

## When frames happen

The runtime keeps one `dirty` flag. It is set when:

- `update` handles a message;
- an event changes some widget's internal state, such as hover or press;
- the size or scale changes;
- a timer fires; or
- something outside calls `request_redraw()`.

When the flag is set, a frame runs `view` (only if a message arrived), then layout, then paint into the
`DrawList`, then submit. With nothing dirty, the host doesn't schedule a frame at all. Layout currently
reruns for the whole tree when anything needs it; caching per subtree can come later if profiles ask for it.

When the clock jumps past several timers at once, each one is delivered with the clock reading its own due
time, so timers that `update` sets again stay on schedule.

The draw list is always rebuilt in full when a frame runs, which is cheap at UI scale. Painting only parts of
the window is not planned.

## Changes needed in phox-immediate

1. **Clip rects:** `push_clip(Rect)` and `pop_clip()`. Vertices are grouped into draw ranges, each with a
   scissor rectangle. This needs a scissor command in the RHI's `CommandList`, or a clip rect sent to the shader
   that discards fragments outside it.
2. A text-measuring function that agrees exactly with `text`, including line height. `text_width` exists.
3. Lower-case letters draw as capitals with the stroke font. That is accepted for the MVP.

## Hosting (`phox-ui-host`, `phox-ui-nih`)

`phox-ui-host` shows an app in a native window. It is built on baseview (the windowing library audio plugins
use), draws with `phox-metal`, and is macOS only for now; on other platforms the crate holds only its traits.

```rust
// Standalone: a window of its own, until it closes.
phox_ui_host::run("Title", (800.0, 600.0), || MyApp::new());

// Embedded: a child view in a window someone else owns. Dropping the handle closes the view.
let handle = phox_ui_host::open_parented(&parent, "Title", (800.0, 600.0), move || MyApp::new());
```

The app implements `HostedApp`, which has only defaults:

- `poll(&self) -> Option<Message>` is called every display frame, for news from outside the view (meters, a
  worker thread, a parameter the plugin host moved);
- `changed(&self) -> bool` is called every display frame and rebuilds the view when it returns true; and
- `command(&mut self, Command) -> bool` handles window-wide shortcuts (Undo and Redo). If the app does not use
  a shortcut, the key goes on to the window's owner, so a DAW's own undo keeps working.

The host:

- turns baseview's mouse, wheel and key events into `Event`s. Typed keys go to the window's owner unless a
  widget has focus;
- keeps the runtime's clock, for timers;
- hands copy and paste between widgets and the system clipboard (`arboard`); and
- draws only when the runtime needs a frame.

`phox-ui-nih` makes this an nih-plug `Editor`: `PhoxEditor::new(title, size, |context| Panel::new(...))`. It
adds `Binding` (a panel's link to its plugin's parameters, with begin/set/end gestures and change detection),
controls bound to a parameter (`knob`, `switch`, `choices`), a `Look` for a plugin's palette and sections, and
`offscreen::write_png` for drawing a panel without a host. Every plugin built on it must use the same nih-plug
revision as `phox-ui-nih` (or `[patch]` that URL), so that there is one `nih_plug` crate.

Still to come: resizing an embedded view, Windows and Linux, and a CVDisplayLink-driven frame schedule (baseview
calls `on_frame` at about 60 Hz, and the host skips frames that have nothing to draw).

## Testing

```rust
let mut t = Tester::new(Counter::default(), Size::new(200.0, 100.0));
t.tap(find::text("+"));
t.pump();                                    // Handles queued messages and runs one frame
assert_eq!(t.app().count, 1);
assert_eq!(t.rect(find::role(Role::Button).nth(0)), Rect::new(16.0, 40.0, 40.0, 24.0));
t.drag(find::key("cutoff"), vec2(0.0, -50.0));
t.type_text("440");
t.advance(Duration::from_millis(500));       // The fake clock; fires any due timers
assert_eq!(t.layout_snapshot(), include_str!("snapshots/counter.txt"));
```

- **Finders** match by text, key, role, type, or their combinations and `nth`. `t.find` returns every match, and
  any action fails clearly unless exactly one widget matches.
- **Actions** (`tap`, `drag`, `scroll`, `press_key`, `type_text`, `resize`) send real `Event`s through the
  runtime. Nothing goes around hit-testing.
- **Layout snapshot:** the laid-out tree as indented text (role, key, rect), compared against a checked-in file.
- **Draw snapshot:** `t.draw_list()` exposes the vertices for low-level checks.
- **Image tests:** `t.render(&mut ImmediateRenderer)` returns pixels. Golden images are deferred for now. When
  they come, they follow `CLAUDE.md`: never raise a tolerance.

## Build order

1. ✅ Clip rects in `phox-immediate` (plus a scissor command in the RHI).
2. ✅ `phox-ui` core: the `App` trait, `Element`, the persistent state tree, `Constraints`, Text, Container,
   Padding, Row, Column, Stack, SizedBox, Expanded and Align, plus `Tester` and layout snapshots.
3. ✅ Input (except Dropdown): pointer capture, Button, Toggle, Slider and Knob, then keyboard focus, TextField and Dropdown.
4. ✅ Scroll, which needs the clip rects.
5. ✅ `phox-ui-host` on macOS (standalone and embedded), and `phox-ui-nih` for nih-plug editors.
6. An example app.
