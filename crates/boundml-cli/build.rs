//! Embeds the build target and (when building from a git checkout) the source
//! revision into `boundml --version`. Falls back to "unknown" outside a checkout,
//! e.g. when building from a crates.io package.

use std::process::Command;

fn main() {
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".into());
    let revision = std::env::var("BOUNDML_BUILD_REVISION").ok().or_else(|| {
        Command::new("git")
            .args(["rev-parse", "--short=10", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|s| !s.is_empty())
    });
    println!("cargo:rustc-env=BOUNDML_BUILD_TARGET={target}");
    println!(
        "cargo:rustc-env=BOUNDML_BUILD_REVISION={}",
        revision.unwrap_or_else(|| "unknown".into())
    );
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-env-changed=BOUNDML_BUILD_REVISION");
}
