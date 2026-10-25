# Installing BoundML

BoundML 0.1.0 is an early public beta. The `boundml` CLI is one self-contained
executable with no runtime dependencies: no Python, no ML framework, no network
access.

| Platform | Prebuilt CLI archive (from the [v0.1.0 release](https://github.com/MahmoudYahiaAhmed/boundml/releases/tag/v0.1.0)) | Python wheel (`pip install boundml-rs`) |
|---|---|---|
| Windows x86_64 | `boundml-0.1.0-x86_64-pc-windows-msvc.zip` | Python 3.11–3.14 |
| Linux x86_64, glibc 2.34+ (Ubuntu 22.04+, Debian 12+) | `boundml-0.1.0-x86_64-unknown-linux-gnu.tar.gz` | Python 3.11–3.14 |
| macOS on Apple silicon (arm64) | `boundml-0.1.0-aarch64-apple-darwin.tar.gz` | Python 3.11–3.14 |

Each combination above is built and tested in CI. Other platforms (Intel macOS, ARM
Linux, musl, 32-bit) are not built; on them, `cargo install boundml-cli` may work but
is untested.

Every archive has a `.sha256` file, and the release has a combined `SHA256SUMS`;
check it before extracting (below). The archive also contains the docs, a real
example dataset (UCI breast cancer) with its configs, and the example sources.

## Windows (PowerShell)

```powershell
$v = "0.1.0"; $t = "x86_64-pc-windows-msvc"
Invoke-WebRequest "https://github.com/MahmoudYahiaAhmed/boundml/releases/download/v0.1.0/boundml-$v-$t.zip" -OutFile "boundml-$v-$t.zip"
Invoke-WebRequest "https://github.com/MahmoudYahiaAhmed/boundml/releases/download/v0.1.0/boundml-$v-$t.zip.sha256" -OutFile "boundml-$v-$t.zip.sha256"
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
curl -LO "https://github.com/MahmoudYahiaAhmed/boundml/releases/download/v0.1.0/boundml-$v-$t.tar.gz" -LO "https://github.com/MahmoudYahiaAhmed/boundml/releases/download/v0.1.0/boundml-$v-$t.tar.gz.sha256"
sha256sum -c "boundml-$v-$t.tar.gz.sha256"   # macOS: shasum -a 256 -c "boundml-$v-$t.tar.gz.sha256"
tar xzf "boundml-$v-$t.tar.gz"
install -m 755 "boundml-$v-$t/boundml" ~/.local/bin/boundml
boundml --version
```

`~/.local/bin` must be on your `PATH`. On macOS, Gatekeeper may block an unsigned
download. `xattr -d com.apple.quarantine ~/.local/bin/boundml` removes the quarantine
flag; the binaries are not code-signed.

## From crates.io (all platforms)

Requires Rust 1.94 or newer.

```bash
cargo install boundml-cli --locked
```

This builds and installs `boundml` into `~/.cargo/bin` (Windows:
`%USERPROFILE%\.cargo\bin`). To build from a checkout instead:
`cargo install --path crates/boundml-cli --locked`. To embed packages in your own
program, add the library: `cargo add boundml-artifact@0.1`.

## Python package

```bash
pip install boundml-rs
python -c "import boundml_rs; print(boundml_rs.__version__)"
```

Wheels exist for the three platforms above and Python 3.11–3.14. The distribution is
`boundml-rs`; the import name is `boundml_rs` (the project named `boundml` on PyPI is
unrelated). Use a virtual environment (`python -m venv .venv`). Usage:
[python/README.md](../python/README.md).

## Checking the version

```bash
boundml --version
```

prints `boundml 0.1.0 (revision <git revision>, <target triple>)`. A build without
`.git`, such as `cargo install boundml-cli` from crates.io, prints `revision unknown`;
release archives print the tagged commit (`3c79dd9570` for 0.1.0). `boundml version`
adds the supported package and checkpoint format versions, and `boundml --json version`
prints the same as JSON.

## Updating

Replace the executable, rerun `cargo install boundml-cli --locked --force`, or
`pip install --upgrade boundml-rs`. Packages and checkpoints are separate files and are not touched. After an
update, run `boundml verify <package>` for each package you deploy. That re-checks
integrity, format compatibility and the bundled self-tests under the new runtime.

## Removing

- Archive install: delete the extracted folder (or `~/.local/bin/boundml`) and remove
  it from `PATH` if you added it.
- Cargo install: `cargo uninstall boundml-cli`.
- Python: `pip uninstall boundml-rs`.

BoundML writes no configuration, cache or registry entries. Its only files are the
ones you name on the command line.
