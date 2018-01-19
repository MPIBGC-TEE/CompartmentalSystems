with import <nixpkgs> {};
#python36.withPackages (ps: with ps; [ pip setuptools matplotlib numpy])
python36.buildEnv.override {
  extraLibs = [ pkgs.python36Packages.tkinter pkgs.python36Packages.matplotlib pkgs.python36Packages.pip];
  ignoreCollisions = true;
}
