//! Logging: the `log` facade with env_logger, honouring `PHOX_LOG` (debug|info|warn|error|off).

use std::sync::Once;

/// Installs the logger once. The level starts at `info`, or at `PHOX_LOG` when set.
pub fn init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let env = env_logger::Env::new().filter_or("PHOX_LOG", "info");
        let _ = env_logger::Builder::from_env(env)
            .format(|f, r| {
                use std::io::Write;
                writeln!(f, "[phox] {}: {}", r.level().as_str().to_lowercase(), r.args())
            })
            .try_init();
    });
}

/// Raises the log level to debug (what `--debug` does), unless `PHOX_LOG` is set.
pub fn enable_debug() {
    if std::env::var_os("PHOX_LOG").is_none() {
        log::set_max_level(log::LevelFilter::Debug);
    }
}
