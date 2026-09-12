-- One-shot top-center welcome pill. argv 1 is the full label, e.g. "Welcome, Alex".
use framework "AppKit"
use scripting additions

on run argv
	if (count of argv) < 1 then return
	set theText to item 1 of argv
	if theText is "" then return
	set theFont to current application's NSFont's systemFontOfSize:17 weight:0.3
	set padX to 22
	set padY to 11
	set field to current application's NSTextField's alloc()'s initWithFrame:(current application's NSMakeRect(0, 0, 800, 40))
	field's setBezeled:false
	field's setBordered:false
	field's setDrawsBackground:false
	field's setEditable:false
	field's setSelectable:false
	field's setAlignment:2
	field's setFont:theFont
	field's setTextColor:(current application's NSColor's colorWithWhite:1 alpha:0.96)
	field's setStringValue:theText
	field's sizeToFit()
	set {{fx, fy}, {fw, fh}} to field's frame()
	set cardW to fw + padX + padX
	set cardH to fh + padY + padY
	if cardW < 96 then set cardW to 96
	if cardH < 36 then set cardH to 36
	field's setFrame:(current application's NSMakeRect((cardW - fw) / 2, (cardH - fh) / 2, fw, fh))
	set {{xs, ys}, {sw, sh}} to current application's NSScreen's mainScreen()'s visibleFrame()
	set x to xs + (sw - cardW) / 2
	set y to ys + sh - cardH - 14
	set win to current application's NSPanel's alloc()'s initWithContentRect:(current application's NSMakeRect(x, y, cardW, cardH)) styleMask:128 backing:2 defer:false
	win's setOpaque:false
	win's setBackgroundColor:(current application's NSColor's clearColor())
	win's setLevel:3
	win's setIgnoresMouseEvents:true
	win's setHasShadow:true
	win's setHidesOnDeactivate:false
	win's setFloatingPanel:true
	win's setBecomesKeyOnlyIfNeeded:true
	win's setAlphaValue:0
	set hud to current application's NSVisualEffectView's alloc()'s initWithFrame:(current application's NSMakeRect(0, 0, cardW, cardH))
	hud's setMaterial:10
	hud's setState:1
	hud's setBlendingMode:0
	hud's setWantsLayer:true
	hud's layer()'s setCornerRadius:(cardH / 2)
	hud's layer()'s setMasksToBounds:true
	hud's addSubview:field
	win's setContentView:hud
	win's orderFront:(missing value)
	repeat with i from 1 to 10
		win's setAlphaValue:(i / 10)
		delay 0.016
	end repeat
	delay 2.8
	repeat with i from 10 to 0 by -1
		win's setAlphaValue:(i / 10)
		delay 0.018
	end repeat
	win's orderOut:(missing value)
	win's |close|()
end run
