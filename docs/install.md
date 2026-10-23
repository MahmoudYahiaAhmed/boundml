# Installing BoundML

BoundML is one self-contained executable, `boundml`. It has no runtime dependencies:
no Python, no ML framework, no network access.

| Platform | Prebuilt archive | Status |
|---|---|---|
| Windows x86_64 (`x86_64-pc-windows-msvc`) | `boundml-<version>-x86_64-pc-windows-msvc.zip` | built and smoke-tested by `scripts/release.py` on Windows 11 |
| Linux x86_64 (`x86_64-unknown-linux-gnu`) | `boundml-<version>-x86_64-unknown-linux-gnu.tar.gz` | built and smoke-tested in CI (`package-smoke`); not yet attached to a release |
| macOS arm64 (`aarch64-apple-darwin`) | `boundml-<version>-aarch64-apple-darwin.tar.gz` | built and smoke-tested in CI (`package-smoke`); not yet attached to a release |
| Anything with Rust 1.94+ | build from source (below) | tested in CI on Windows x86_64, Linux x86_64 and macOS arm64 |

Archives are attached to this repository's [Releases](../../../releases) page. Each
archive has a `.sha256` file, and the release has a combined `SHA256SUMS`.
The archive includes the docs and example sources for reading; building the Rust
examples (`examples/embed`, `examples/power-app`) needs a checkout of the repository.

## Windows (PowerShell)

```powershell
$v = "0.1.0"; $t = "x86_64-pc-windows-msvc"
(Get-FileHash ".\boundml-$v-$t.zip" -Algorithm SHA256).Hash -eq (Get-Content ".\boundml-$v-$t.zip.sha256").Split(" ")[0].ToUpper()   # True = match
Expand-Archive ".\boundml-$v-$t.zip" -DestinationPath "$env:LOCALAPPDATA\Programs"
& "$env:LOCALAPPDATA\Programs\boundml-$v-$t\boundml.exe" --version
```

To run `boundml` from any folder, add that folder to your user `PATH`:

```powershell
$dir = "$env:LOCALAPPDATA\Programs\boundml-$v-$t"
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";$dir", "User")
```

Open a new terminal afterwards. Paths that contain spaces work. The release smoke
test runs from a folder named `My Apps`.

## Linux and macOS (bash/zsh)

```bash
v=0.1.0; t=x86_64-unknown-linux-gnu   # or aarch64-apple-darwin
sha256sum -c "boundml-$v-$t.tar.gz.sha256"   # macOS: shasum -a 256 -c "boundml-$v-$t.tar.gz.sha256"
tar xzf "boundml-$v-$t.tar.gz"
install -m 755 "boundml-$v-$t/boundml" ~/.local/bin/boundml
boundml --version
```

`~/.local/bin` must be on your `PATH`. On macOS, Gatekeeper may block an unsigned
download. `xattr -d com.apple.quarantine ~/.local/bin/boundml` removes the quarantine
flag; the binaries are not code-signed.

## From source (all platforms)

Requires Rust 1.94 or newer (the repository pins 1.94.0 in `rust-toolchain.toml`).
From a checkout of this repository:

```bash
cargo install --path crates/boundml-cli --locked
```

This installs `boundml` into `~/.cargo/bin` (Windows: `%USERPROFILE%\.cargo\bin`).
A release build took about 35 s on an 8-core machine with crate downloads already cached; the first build also downloads about 50 crates. To put it
somewhere else, add `--root <dir>`.

BoundML is **not yet published on crates.io**. `cargo install boundml-cli` only works
once it is.

## Python package

From a checkout, with Rust 1.94 and Python ≥ 3.9:

```bash
pip install maturin
pip install ./python
python -c "import boundml; print(boundml.__version__)"
```

This builds one wheel for every Python ≥ 3.9 on your platform (abi3). It is **not
yet published on PyPI**. Usage: [python/README.md](../python/README.md).

## Checking the version

```bash
boundml --version
```

prints `boundml 0.1.0 (revision <git revision>, <target triple>)`. `boundml version`
adds the supported package and checkpoint format versions, and `boundml --json version`
prints the same as JSON.

## Updating

Replace the executable (or rerun `cargo install --path crates/boundml-cli --locked
--force`). Packages and checkpoints are separate files and are not touched. After an
update, run `boundml verify <package>` for each package you deploy. That re-checks
integrity, format compatibility and the bundled self-tests under the new runtime.

## Removing

- Archive install: delete the extracted folder (or `~/.local/bin/boundml`) and remove
  it from `PATH` if you added it.
- Source install: `cargo uninstall boundml-cli`.

BoundML writes no configuration, cache or registry entries. Its only files are the
ones you name on the command line.
