{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
    rust-overlay.url = "github:oxalica/rust-overlay";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, devenv, systems, rust-overlay, utils } @ inputs: with utils.lib;
    (eachSystem [ system.x86_64-linux ]
      (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          rust-bin = pkgs.rust-bin.stable.latest.default.override {
            extensions = [ "rust-analyzer" "clippy" ];
          };
        in
        {
          devShells.default = devenv.lib.mkShell {
            inherit inputs pkgs;
            modules = [
              ({ pkgs, ... }: {
                # https://devenv.sh/reference/options/
                packages = with pkgs; [
                  rust-bin
                  gcc
                  pkg-config
                ];

                pre-commit.hooks = {
                  nixpkgs-fmt.enable = true;
                  # lint shell scripts
                  shellcheck.enable = true;
                  # format rust code
                  rustfmt.enable = true;
                  # format TOML
                  taplo.enable = true;
                };
              })
            ];
          };
        })
    );
}
