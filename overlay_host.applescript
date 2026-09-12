-- Persistent frost HUD. Same AppKit path as the old welcome pill.
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
	win's setHasShadow:false
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

on run argv
	if (count of argv) < 1 then return
	set statePath to item 1 of argv
	set eventPath to ""
	if (count of argv) ≥ 2 then set eventPath to item 2 of argv
	
	set {{xs, ys}, {sw, sh}} to current application's NSScreen's mainScreen()'s frame()
	
	set introParts to my makeCard(xs, ys, sw, sh, 0, true)
	set introWin to item 1 of introParts
	set introView to item 2 of introParts
	set welcome to my makeField("Welcome", 54, -0.4, 0.94)
	set deskReady to my makeField("DESK READY", 11, 0.2, 0.28)
	set clickHint to my makeField("Click or Esc to continue", 12, 0, 0.28)
	welcome's setFrame:(current application's NSMakeRect(0, sh / 2 - 8, sw, 70))
	deskReady's setFrame:(current application's NSMakeRect(0, sh / 2 - 40, sw, 20))
	clickHint's setFrame:(current application's NSMakeRect(0, 28, sw, 18))
	introView's addSubview:welcome
	introView's addSubview:deskReady
	introView's addSubview:clickHint
	introWin's orderOut:(missing value)
	
	set chipW to 560
	set chipH to 36
	set chipParts to my makeCard(xs + (sw - chipW) / 2, ys + sh - chipH - 22, chipW, chipH, 18, false)
	set chipWin to item 1 of chipParts
	set chipView to item 2 of chipParts
	set chipText to my makeField("Unlocked", 13, 0.2, 0.9)
	chipText's setFrame:(current application's NSMakeRect(10, 4, chipW - 20, 28))
	chipView's addSubview:chipText
	chipWin's orderOut:(missing value)
	
	set barW to 740
	set barH to 48
	set barParts to my makeCard(xs + (sw - barW) / 2, ys + 22, barW, barH, 20, false)
	set barWin to item 1 of barParts
	set barView to item 2 of barParts
	set barText to my makeField("Pointer  open → fist → gun     Clicker  thumb + middle     Scroll  index / pinky     Switcher  pinch + flap     Voice  hey Grok", 12, 0, 0.86)
	barText's setFrame:(current application's NSMakeRect(12, 8, barW - 24, 32))
	barView's addSubview:barText
	barWin's orderOut:(missing value)
	
	set togW to 188
	set togH to 32
	set togParts to my makeCard(xs + sw - togW - 18, ys + 16, togW, togH, 16, true)
	set togWin to item 1 of togParts
	set togView to item 2 of togParts
	set togText to my makeField("H · toggle HUD", 11, 0.1, 0.5)
	togText's setFrame:(current application's NSMakeRect(8, 4, togW - 16, 24))
	togView's addSubview:togText
	togWin's orderFront:(missing value)
	
	set lastSig to ""
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
			if theName is "" then
				welcome's setStringValue:"Welcome"
			else
				welcome's setStringValue:("Welcome " & theName)
			end if
			if lockedOn then
				chipText's setStringValue:"Commands off · unfamiliar face"
			else
				set bits to "Unlocked"
				if voiceOn then set bits to bits & "     Voice listening"
				if pointerOn then set bits to bits & "     Pointer armed"
				chipText's setStringValue:bits
			end if
			if hudOn then
				togText's setStringValue:"H · toggle HUD   ON"
			else
				togText's setStringValue:"H · toggle HUD"
			end if
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
			togWin's orderFront:(missing value)
		end if
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
