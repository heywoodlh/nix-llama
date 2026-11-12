# Hyprland configuration

## bind

Keyboard bindings, written as `bind = MODS, key, dispatcher, params`. For
example `bind = SUPER, Q, exec, kitty` launches a terminal.

## bindm

Mouse bindings. `bindm = SUPER, mouse:272, movewindow` drags a window while the
super key is held. Distinct from `bind`, which handles keyboard input.

## How do I enable Hyprland on NixOS?

Set `programs.hyprland.enable = true;` in your NixOS configuration. Add
`programs.hyprland.xwayland.enable = true;` for X11 clients.
