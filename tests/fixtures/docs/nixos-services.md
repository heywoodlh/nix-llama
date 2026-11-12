# Enabling services in NixOS

## services.openssh.enable

Whether to enable the OpenSSH secure shell daemon, which allows secure remote
logins over an encrypted channel. Type: `boolean`. Default: `false`.

```nix
{
  services.openssh = {
    enable = true;
    settings.PasswordAuthentication = false;
    settings.PermitRootLogin = "no";
  };
}
```

## How do I apply a configuration change?

Run `nixos-rebuild switch` to build the new system configuration and activate it
immediately, or `nixos-rebuild boot` to activate it at the next reboot. Both
commands replace the running system's service set, so expect running services to
restart.

## services.nginx.enable

Whether to enable the nginx web server and reverse proxy. Type: `boolean`.
