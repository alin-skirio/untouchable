-- Compact frost HUD. Welcome is a centered card; chips and bars hug their text.
-- argv 1 = state text file, argv 2 = event file (optional)
use framework "AppKit"
use scripting additions

on kv(raw, keyName)
	set needle to keyName & "="
	repeat with ln in paragraphs of raw
		set lineText to ln as text
		if lineText starts with needle then
			if (count of lineText) < ((count of needle) + 1) then return ""
			return text ((count of needle) + 1) thru -1 of lineText
		end if
	end repeat
	return ""
end kv

on readFile(posixPath)
	try
		return do shell script "cat " & quoted form of posixPath
	on error
		return ""
	end try
end readFile

on writeEvent(posixPath, theType)
	if posixPath is "" then return
	try
		do shell script "printf %s " & quoted form of ("type=" & theType) & " > " & quoted form of posixPath
	end try
end writeEvent

on deskBounds()
	set {{xs, ys}, {sw, sh}} to current application's NSScreen's mainScreen()'s visibleFrame()
	return {xs, ys, sw, sh}
end deskBounds

on measureText(theText, theSize, theWeight)
	set theFont to current application's NSFont's systemFontOfSize:theSize weight:theWeight
	set attrs to current application's NSDictionary's dictionaryWithObject:theFont forKey:(current application's NSFontAttributeName)
	set sz to (current application's NSString's stringWithString:theText)'s sizeWithAttributes:attrs
	return {(sz's |width|) as real, (sz's |height|) as real}
end measureText

on bigger(a, b)
	if a > b then return a
	return b
end bigger

on makeField(theText, theSize, theWeight, theAlpha)
	set field to current application's NSTextField's alloc()'s initWithFrame:(current application's NSMakeRect(0, 0, 800, 40))
	field's setBezeled:false
	field's setBordered:false
	field's setDrawsBackground:false
	field's setEditable:false
	field's setSelectable:false
	field's setAlignment:2
	field's setFont:(current application's NSFont's systemFontOfSize:theSize weight:theWeight)
	field's setTextColor:(current application's NSColor's colorWithWhite:1 alpha:theAlpha)
	field's setStringValue:theText
	return field
end makeField

on makeCard(x, y, cardW, cardH, radius, clickable)
	set win to current application's NSPanel's alloc()'s initWithContentRect:(current application's NSMakeRect(x, y, cardW, cardH)) styleMask:128 backing:2 defer:false
	win's setOpaque:false
	win's setBackgroundColor:(current application's NSColor's clearColor())
	win's setLevel:3
	win's setIgnoresMouseEvents:(not clickable)
	win's setHasShadow:true
	win's setHidesOnDeactivate:false
	win's setFloatingPanel:true
	win's setBecomesKeyOnlyIfNeeded:true
	set fx to current application's NSVisualEffectView's alloc()'s initWithFrame:(current application's NSMakeRect(0, 0, cardW, cardH))
	fx's setMaterial:10
	fx's setState:1
	fx's setBlendingMode:0
	fx's setWantsLayer:true
	if radius > 0 then
		fx's layer()'s setCornerRadius:radius
		fx's layer()'s setMasksToBounds:true
	end if
	win's setContentView:fx
	return {win, fx}
end makeCard

on placeWindow(win, view, x, y, w, h, radius)
	win's setFrame:(current application's NSMakeRect(x, y, w, h)) display:true animate:false
	view's setFrame:(current application's NSMakeRect(0, 0, w, h))
	if radius > 0 then
		view's layer()'s setCornerRadius:radius
		view's layer()'s setMasksToBounds:true
	end if
end placeWindow

on layoutWelcome(win, view, welcome, deskReady, clickHint, theName)
	set title to "Welcome"
	if theName is not "" then set title to "Welcome, " & theName
	welcome's setStringValue:title
	set {tw, titleH} to my measureText(title, 28, -0.2)
	set {sw1, sh1} to my measureText("Desk ready", 11, 0.2)
	set {hw, hh} to my measureText("Click or Esc", 11, 0)
	set padX to 28
	set padY to 16
	set gap to 5
	set cardW to my bigger(my bigger(tw, sw1), hw) + padX * 2
	if cardW < 228 then set cardW to 228
	set cardH to padY + titleH + gap + sh1 + gap + hh + padY
	set {xs, ys, dw, dh} to my deskBounds()
	set x to xs + (dw - cardW) / 2
	set y to ys + (dh - cardH) / 2
	my placeWindow(win, view, x, y, cardW, cardH, 20)
	set yHint to padY
	set ySub to yHint + hh + gap
	set yTitle to ySub + sh1 + gap
	clickHint's setFrame:(current application's NSMakeRect(0, yHint, cardW, hh + 2))
	deskReady's setFrame:(current application's NSMakeRect(0, ySub, cardW, sh1 + 2))
	welcome's setFrame:(current application's NSMakeRect(0, yTitle, cardW, titleH + 4))
end layoutWelcome

on layoutPill(win, view, field, theText, theSize, theWeight, padX, cardH, x, y)
	field's setStringValue:theText
	set {tw, textH} to my measureText(theText, theSize, theWeight)
	set cardW to tw + padX * 2
	if cardW < 72 then set cardW to 72
	my placeWindow(win, view, x, y, cardW, cardH, cardH / 2)
	set fieldH to textH + 2
	field's setFrame:(current application's NSMakeRect(0, (cardH - fieldH) / 2, cardW, fieldH))
	return cardW
end layoutPill

on pointIn(mx, mouseY, win)
	set {{wx, wy}, {ww, wh}} to win's frame()
	return (mx ≥ wx and mx ≤ (wx + ww) and mouseY ≥ wy and mouseY ≤ (wy + wh))
end pointIn

on run argv
	if (count of argv) < 1 then return
	set statePath to item 1 of argv
	set eventPath to ""
	if (count of argv) ≥ 2 then set eventPath to item 2 of argv
	
	set {xs, ys, sw, sh} to my deskBounds()
	
	set introParts to my makeCard(xs + sw / 2 - 114, ys + sh / 2 - 48, 228, 96, 20, true)
	set introWin to item 1 of introParts
	set introView to item 2 of introParts
	set welcome to my makeField("Welcome", 28, -0.2, 0.94)
	set deskReady to my makeField("Desk ready", 11, 0.2, 0.42)
	set clickHint to my makeField("Click or Esc", 11, 0, 0.32)
	introView's addSubview:welcome
	introView's addSubview:deskReady
	introView's addSubview:clickHint
	my layoutWelcome(introWin, introView, welcome, deskReady, clickHint, "")
	introWin's orderOut:(missing value)
	
	set chipParts to my makeCard(xs + sw / 2 - 48, ys + sh - 38, 96, 28, 14, false)
	set chipWin to item 1 of chipParts
	set chipView to item 2 of chipParts
	set chipText to my makeField("Unlocked", 12, 0.2, 0.9)
	chipView's addSubview:chipText
	chipWin's orderOut:(missing value)
	
	set barParts to my makeCard(xs + sw / 2 - 200, ys + 12, 400, 32, 16, false)
	set barWin to item 1 of barParts
	set barView to item 2 of barParts
	set barLabel to "Pointer  open → fist → gun    Clicker  thumb + middle    Scroll  index / pinky    Switcher  pinch + flap    Voice  hey Grok"
	set barText to my makeField(barLabel, 11, 0, 0.86)
	barView's addSubview:barText
	barWin's orderOut:(missing value)
	
	set togParts to my makeCard(xs + sw - 140, ys + 12, 128, 28, 14, true)
	set togWin to item 1 of togParts
	set togView to item 2 of togParts
	set togText to my makeField("H · HUD", 11, 0.1, 0.62)
	togView's addSubview:togText
	togWin's orderOut:(missing value)
	
	set lastSig to ""
	set lastButtons to 0
	repeat
		set raw to my readFile(statePath)
		if my kv(raw, "quit") is "1" then exit repeat
		set introOn to (my kv(raw, "intro") is "1")
		set hudOn to (my kv(raw, "hud") is "1")
		set lockedOn to (my kv(raw, "locked") is "1")
		set theName to my kv(raw, "name")
		set voiceOn to (my kv(raw, "voice") is "1")
		set pointerOn to (my kv(raw, "pointer") is "1")
		set sig to (my kv(raw, "intro")) & "|" & (my kv(raw, "hud")) & "|" & (my kv(raw, "locked")) & "|" & theName & "|" & (my kv(raw, "voice")) & "|" & (my kv(raw, "pointer"))
		if sig is not lastSig then
			set lastSig to sig
			my layoutWelcome(introWin, introView, welcome, deskReady, clickHint, theName)
			if lockedOn then
				set chipLabel to "Commands off · unfamiliar face"
			else
				set chipLabel to "Unlocked"
				if theName is not "" then set chipLabel to "Unlocked · " & theName
				if voiceOn then set chipLabel to chipLabel & "  ·  Voice"
				if pointerOn then set chipLabel to chipLabel & "  ·  Pointer"
			end if
			set {xs, ys, sw, sh} to my deskBounds()
			set chipW to my layoutPill(chipWin, chipView, chipText, chipLabel, 12, 0.2, 14, 28, xs, ys + sh - 38)
			my placeWindow(chipWin, chipView, xs + (sw - chipW) / 2, ys + sh - 38, chipW, 28, 14)
			set barW to my layoutPill(barWin, barView, barText, barLabel, 11, 0, 16, 32, xs, ys + 12)
			my placeWindow(barWin, barView, xs + (sw - barW) / 2, ys + 12, barW, 32, 16)
			if hudOn then
				set togLabel to "H · HUD  ON"
			else
				set togLabel to "H · HUD"
			end if
			set togW to my layoutPill(togWin, togView, togText, togLabel, 11, 0.1, 12, 28, xs, ys + 12)
			my placeWindow(togWin, togView, xs + sw - togW - 12, ys + 12, togW, 28, 14)
			if introOn then
				introWin's orderFront:(missing value)
			else
				introWin's orderOut:(missing value)
			end if
			if (hudOn and not introOn) or (lockedOn and not introOn) then
				chipWin's orderFront:(missing value)
			else
				chipWin's orderOut:(missing value)
			end if
			if hudOn and not introOn and not lockedOn then
				barWin's orderFront:(missing value)
			else
				barWin's orderOut:(missing value)
			end if
			if introOn then
				togWin's orderOut:(missing value)
			else
				togWin's orderFront:(missing value)
			end if
		end if
		try
			set buttons to (current application's NSEvent's pressedMouseButtons()) as integer
			if buttons > 0 and lastButtons is 0 then
				set mouseLoc to current application's NSEvent's mouseLocation()
				set mx to x of mouseLoc
				set mouseY to y of mouseLoc
				if introOn then
					if my pointIn(mx, mouseY, introWin) then my writeEvent(eventPath, "dismiss_intro")
				else
					if my pointIn(mx, mouseY, togWin) then my writeEvent(eventPath, "toggle_hud")
				end if
			end if
			set lastButtons to buttons
		on error
			set lastButtons to 0
		end try
		delay 0.12
	end repeat
	
	introWin's orderOut:(missing value)
	chipWin's orderOut:(missing value)
	barWin's orderOut:(missing value)
	togWin's orderOut:(missing value)
	introWin's |close|()
	chipWin's |close|()
	barWin's |close|()
	togWin's |close|()
end run
