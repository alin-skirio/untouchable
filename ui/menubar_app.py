"""Menu bar status item with a native camera panel.

AppKit owns the main thread; the vision loop lives in HandTracker on a worker
thread. A single timer refreshes the icon and, only while the panel is on
screen, converts the newest annotated frame into the image view.
"""

from __future__ import annotations

import time
from typing import Optional

import objc
import AppKit
from AppKit import (
    NSAlert,
    NSAlertFirstButtonReturn,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSButton,
    NSColor,
    NSFont,
    NSImage,
    NSImageAlignCenter,
    NSImageScaleProportionallyUpOrDown,
    NSImageView,
    NSMenu,
    NSMenuItem,
    NSPanel,
    NSScreen,
    NSSlider,
    NSStatusBar,
    NSStatusWindowLevel,
    NSTextField,
    NSVariableStatusItemLength,
    NSView,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskNonactivatingPanel,
    NSWindowStyleMaskTitled,
    NSWindowStyleMaskUtilityWindow,
    NSWindowTitleHidden,
)
from Foundation import NSMakePoint, NSMakeRect, NSObject, NSTimer

from gestures.cursor import SENSITIVITY_MAX, SENSITIVITY_MIN
from .nsimage import nsimage_from_bgr

# Constants renamed across macOS SDKs; fall back to the legacy spelling.
NSBezelStyleRounded = getattr(AppKit, "NSBezelStyleRounded", getattr(AppKit, "NSRoundedBezelStyle", 1))
NSEventMaskLeftMouseUp = getattr(AppKit, "NSEventMaskLeftMouseUp", 1 << 2)
NSEventMaskRightMouseUp = getattr(AppKit, "NSEventMaskRightMouseUp", 1 << 4)
NSEventTypeRightMouseUp = getattr(AppKit, "NSEventTypeRightMouseUp", 4)
NSEventModifierFlagControl = getattr(AppKit, "NSEventModifierFlagControl", 1 << 18)
NSBackingStoreBuffered = getattr(AppKit, "NSBackingStoreBuffered", 2)

PANEL_WIDTH = 480.0
PAD = 14.0
CONTENT_WIDTH = PANEL_WIDTH - 2 * PAD
VIDEO_HEIGHT = round(CONTENT_WIDTH * 9 / 16)
BUTTON_HEIGHT = 32.0
BUTTON_GAP = 10.0
PANEL_HEIGHT = 410.0

ICON_IDLE = "hand.point.up.left"
ICON_ENGAGED = "hand.point.up.left.fill"
ICON_ERROR = "exclamationmark.triangle"
FRAME_INTERVAL = 1.0 / 30.0
SENSITIVITY_PERSIST_DELAY = 0.75


def _label(rect, text: str, size: float, dim: bool = False) -> NSTextField:
    field = NSTextField.alloc().initWithFrame_(rect)
    field.setStringValue_(text)
    field.setBezeled_(False)
    field.setDrawsBackground_(False)
    field.setEditable_(False)
    field.setSelectable_(False)
    field.setFont_(NSFont.systemFontOfSize_(size))
    field.setTextColor_(
        NSColor.secondaryLabelColor() if dim else NSColor.labelColor()
    )
    return field


class MenuBarApp(NSObject):
    """Status item, panel, and the timer that drives both.

    Helpers are marked @objc.python_method so PyObjC does not try to expose
    names like _build_status_item as selectors.
    """

    @objc.python_method
    def setup(self, tracker) -> None:
        self.tracker = tracker
        self.panel = None
        self.timer = None
        self._icon_symbol: Optional[str] = None
        self._sensitivity_dirty_at = 0.0
        self._last_status = ""
        self._last_reference = None

        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        app.setDelegate_(self)

        self._build_status_item()
        self._build_panel()

        self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            FRAME_INTERVAL, self, "tick:", None, True
        )

    # --- construction ---------------------------------------------------

    @objc.python_method
    def _build_status_item(self) -> None:
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            NSVariableStatusItemLength
        )
        button = self.status_item.button()
        button.setTarget_(self)
        button.setAction_("statusItemClicked:")
        button.sendActionOn_(NSEventMaskLeftMouseUp | NSEventMaskRightMouseUp)
        self._apply_icon(ICON_IDLE)

    @objc.python_method
    def _symbol_image(self, symbol: str):
        maker = getattr(NSImage, "imageWithSystemSymbolName_accessibilityDescription_", None)
        if maker is None:
            return None
        image = maker(symbol, "Hand Control")
        if image is not None:
            image.setTemplate_(True)
        return image

    @objc.python_method
    def _apply_icon(self, symbol: str) -> None:
        if symbol == self._icon_symbol:
            return
        button = self.status_item.button()
        image = self._symbol_image(symbol)
        if image is not None:
            button.setImage_(image)
            button.setTitle_("")
        else:
            # Pre-Big Sur, or SF Symbols unavailable: fall back to a glyph.
            button.setImage_(None)
            button.setTitle_("✋" if symbol != ICON_ERROR else "⚠")
        self._icon_symbol = symbol

    @objc.python_method
    def _build_panel(self) -> None:
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskUtilityWindow
            | NSWindowStyleMaskNonactivatingPanel
        )
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, PANEL_WIDTH, PANEL_HEIGHT), style, NSBackingStoreBuffered, False
        )
        panel.setTitle_("Hand Control")
        panel.setTitleVisibility_(NSWindowTitleHidden)
        panel.setTitlebarAppearsTransparent_(True)
        panel.setLevel_(NSStatusWindowLevel)
        panel.setHidesOnDeactivate_(False)
        panel.setFloatingPanel_(True)
        panel.setBecomesKeyOnlyIfNeeded_(True)
        panel.setMovableByWindowBackground_(True)
        panel.setReleasedWhenClosed_(False)
        panel.setDelegate_(self)

        content = NSView.alloc().initWithFrame_(
            NSMakeRect(0, 0, PANEL_WIDTH, PANEL_HEIGHT)
        )

        video_y = PANEL_HEIGHT - PAD - VIDEO_HEIGHT
        self.image_view = NSImageView.alloc().initWithFrame_(
            NSMakeRect(PAD, video_y, CONTENT_WIDTH, VIDEO_HEIGHT)
        )
        self.image_view.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        self.image_view.setImageAlignment_(NSImageAlignCenter)
        self.image_view.setWantsLayer_(True)
        layer = self.image_view.layer()
        if layer is not None:
            layer.setBackgroundColor_(
                NSColor.blackColor().colorWithAlphaComponent_(0.9).CGColor()
            )
        content.addSubview_(self.image_view)

        status_y = video_y - 8 - 18
        self.status_label = _label(
            NSMakeRect(PAD, status_y, CONTENT_WIDTH, 18), "Starting camera…", 12.0
        )
        content.addSubview_(self.status_label)

        ref_y = status_y - 4 - 18
        self.reference_label = _label(
            NSMakeRect(PAD, ref_y, CONTENT_WIDTH, 18), "", 11.0, dim=True
        )
        content.addSubview_(self.reference_label)

        slider_y = PAD + BUTTON_HEIGHT + 14
        self.slider = NSSlider.alloc().initWithFrame_(
            NSMakeRect(PAD, slider_y, CONTENT_WIDTH - 120, 22)
        )
        self.slider.setMinValue_(SENSITIVITY_MIN)
        self.slider.setMaxValue_(SENSITIVITY_MAX)
        self.slider.setDoubleValue_(self.tracker.sensitivity)
        self.slider.setContinuous_(True)
        self.slider.setTarget_(self)
        self.slider.setAction_("sensitivityChanged:")
        content.addSubview_(self.slider)

        self.sensitivity_label = _label(
            NSMakeRect(PANEL_WIDTH - PAD - 112, slider_y + 2, 112, 18),
            f"Desk {self.tracker.sensitivity:.2f}x",
            11.0,
            dim=True,
        )
        content.addSubview_(self.sensitivity_label)

        width = (CONTENT_WIDTH - 2 * BUTTON_GAP) / 3
        for i, (title, action) in enumerate(
            (
                ("Set Reference", "setReference:"),
                ("Place Cursor", "placeCursor:"),
                ("Add Face", "addFace:"),
            )
        ):
            button = NSButton.alloc().initWithFrame_(
                NSMakeRect(PAD + i * (width + BUTTON_GAP), PAD, width, BUTTON_HEIGHT)
            )
            button.setTitle_(title)
            button.setBezelStyle_(NSBezelStyleRounded)
            button.setTarget_(self)
            button.setAction_(action)
            content.addSubview_(button)

        panel.setContentView_(content)
        self.panel = panel

    @objc.python_method
    def _build_menu(self) -> NSMenu:
        menu = NSMenu.alloc().init()
        for title, action in (
            ("Show Panel", "showPanel:"),
            ("Add Face…", "addFace:"),
            ("List Profiles", "listFaces:"),
        ):
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, "")
            item.setTarget_(self)
            menu.addItem_(item)
        menu.addItem_(NSMenuItem.separatorItem())
        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit Hand Control", "quitApp:", "q"
        )
        quit_item.setTarget_(self)
        menu.addItem_(quit_item)
        return menu

    # --- status item ----------------------------------------------------

    def statusItemClicked_(self, sender) -> None:
        app = NSApplication.sharedApplication()
        event = app.currentEvent()
        right_click = event is not None and (
            event.type() == NSEventTypeRightMouseUp
            or bool(event.modifierFlags() & NSEventModifierFlagControl)
        )
        if right_click:
            menu = self._build_menu()
            self.status_item.setMenu_(menu)
            self.status_item.button().performClick_(None)
            self.status_item.setMenu_(None)
            return
        self.togglePanel_(sender)

    def togglePanel_(self, _sender) -> None:
        if self.panel.isVisible():
            self._hide_panel()
        else:
            self.showPanel_(None)

    def showPanel_(self, _sender) -> None:
        self._position_panel()
        self.tracker.set_preview_enabled(True)
        self.panel.orderFrontRegardless()

    @objc.python_method
    def _hide_panel(self) -> None:
        self.tracker.set_preview_enabled(False)
        self.panel.orderOut_(None)
        self.image_view.setImage_(None)

    @objc.python_method
    def _position_panel(self) -> None:
        """Anchor the panel just under the status item, right-aligned to the icon."""
        button = self.status_item.button()
        window = button.window()
        if window is None:
            return
        rect = window.convertRectToScreen_(button.frame())
        x = rect.origin.x + rect.size.width - PANEL_WIDTH
        y = rect.origin.y - 6

        screen = window.screen() or NSScreen.mainScreen()
        if screen is not None:
            visible = screen.visibleFrame()
            max_x = visible.origin.x + visible.size.width - PANEL_WIDTH - 8
            x = min(max(x, visible.origin.x + 8), max_x)
        self.panel.setFrameTopLeftPoint_(NSMakePoint(x, y))

    # --- panel delegate -------------------------------------------------

    def windowShouldClose_(self, _sender) -> bool:
        self._hide_panel()
        return False

    # --- controls -------------------------------------------------------

    def sensitivityChanged_(self, sender) -> None:
        value = float(sender.doubleValue())
        self.tracker.set_sensitivity(value, persist=False)
        self.sensitivity_label.setStringValue_(f"Desk {value:.2f}x")
        self._sensitivity_dirty_at = time.monotonic()

    def setReference_(self, _sender) -> None:
        self.tracker.request("set_reference")

    def placeCursor_(self, _sender) -> None:
        self.tracker.request("place_cursor")

    def listFaces_(self, _sender) -> None:
        self.tracker.request("list_faces")

    def addFace_(self, _sender) -> None:
        alert = NSAlert.alloc().init()
        alert.setMessageText_("Add face profile")
        alert.setInformativeText_(
            "Name this face, then follow the guided poses in the panel."
        )
        alert.addButtonWithTitle_("Start")
        alert.addButtonWithTitle_("Cancel")

        field = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 240, 24))
        alert.setAccessoryView_(field)
        alert.window().setInitialFirstResponder_(field)

        # Text entry needs focus, so this is the one place we activate.
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        if alert.runModal() != NSAlertFirstButtonReturn:
            return
        name = str(field.stringValue()).strip()
        if not name:
            return
        self.tracker.request("add_face", name=name)
        if not self.panel.isVisible():
            self.showPanel_(None)

    def quitApp_(self, _sender) -> None:
        NSApplication.sharedApplication().terminate_(None)

    # --- timer ----------------------------------------------------------

    def tick_(self, _timer) -> None:
        tracker = self.tracker

        if tracker.camera_lost:
            self._apply_icon(ICON_ERROR)
        elif tracker.engaged:
            self._apply_icon(ICON_ENGAGED)
        else:
            self._apply_icon(ICON_IDLE)

        if self._sensitivity_dirty_at:
            if time.monotonic() - self._sensitivity_dirty_at > SENSITIVITY_PERSIST_DELAY:
                tracker.persist_sensitivity()
                self._sensitivity_dirty_at = 0.0

        if not self.panel.isVisible():
            return

        latest = tracker.latest_frame()
        if latest is None:
            if tracker.error:
                # open_camera's message is multi-line; the label holds one.
                self._set_status(tracker.error.splitlines()[0])
            elif tracker.camera_lost:
                self._set_status("Lost the camera feed.")
            else:
                self._set_status("Starting camera…")
            return

        frame, lines = latest
        image = nsimage_from_bgr(frame, point_width=CONTENT_WIDTH)
        if image is not None:
            self.image_view.setImage_(image)

        self._set_status(lines[0] if lines else "Tracking")

        reference = tracker.has_reference
        if reference != self._last_reference:
            self.reference_label.setStringValue_(
                "⌘T reference set" if reference else "⌘T to set reference"
            )
            self._last_reference = reference

    @objc.python_method
    def _set_status(self, text: str) -> None:
        if text != self._last_status:
            self.status_label.setStringValue_(text)
            self._last_status = text

    # --- app delegate ---------------------------------------------------

    def applicationShouldTerminateAfterLastWindowClosed_(self, _app) -> bool:
        return False

    def applicationWillTerminate_(self, _notification) -> None:
        if self.timer is not None:
            self.timer.invalidate()
            self.timer = None
        if self._sensitivity_dirty_at:
            self.tracker.persist_sensitivity()
            self._sensitivity_dirty_at = 0.0
        self.tracker.stop()
