//! Library side of the `boundml` command-line tool (commands, server, import).

pub mod commands;
pub mod import;
pub mod init;
pub mod serve;

/// Process exit codes (documented in `docs/errors.md`).
pub mod exit {
    pub const OK: i32 = 0;
    pub const INTERNAL: i32 = 1;
    pub const USAGE: i32 = 2;
    pub const INVALID_INPUT: i32 = 3;
    pub const BUDGET: i32 = 4;
    pub const ARTIFACT: i32 = 5;
    pub const CHECK_FAILED: i32 = 6;
}

/// Map an error message (which starts with its `BML-xxxx` code) to an exit code.
pub fn exit_code_for(message: &str) -> i32 {
    let code = serve::code_of(message);
    match code.as_str() {
        "BML-C005" => exit::BUDGET,
        "BML-A002" | "BML-A003" | "BML-A004" | "BML-A005" | "BML-A006" => exit::ARTIFACT,
        "BML-A007" => exit::CHECK_FAILED,
        c if c.starts_with("BML-C") || c.starts_with("BML-D") || c.starts_with("BML-T") => {
            exit::INVALID_INPUT
        }
        "BML-A001" | "BML-A008" | "BML-A009" | "BML-M001" => exit::INVALID_INPUT,
        _ => exit::INTERNAL,
    }
}
