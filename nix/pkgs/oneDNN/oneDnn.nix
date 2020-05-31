let
  pkgs = import <nixpkgs> { };
in
with pkgs; callPackage ./default.nix { inherit cmake; }
