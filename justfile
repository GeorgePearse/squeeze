default:
    @just --list

build:
    cargo build

test:
    cargo test --all-targets

check:
    cargo fmt --check
    cargo clippy --all-targets
    cargo test --all-targets

fix:
    cargo fmt
    cargo clippy --fix --allow-dirty --allow-staged --all-targets

benchmark:
    cargo bench

docs:
    cargo doc --no-deps --open

clean:
    cargo clean
