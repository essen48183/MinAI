# MinAI — prebuilt binaries

These are ready-to-run copies of MinAI. You don't need a compiler.
Pick your platform, download the file inside, and run it from a terminal.

| Folder | Platform | How to run |
|---|---|---|
| `macos-universal/` | macOS on Apple Silicon (M1/M2/M3/M4) **and** Intel Macs | `./minai` from a terminal |
| `linux-x86_64/` | 64-bit Linux | `./minai` from a terminal |
| `windows-x86_64/` | 64-bit Windows | double-click `minai.exe` (or run from cmd/PowerShell) |

If your platform isn't listed, or you want to modify the code, see the main
[README](../README.md) — building from source is one `make` (macOS/Linux) or
one `cmake --build` (any platform) away.

## First-run on macOS: the Gatekeeper warning

macOS may refuse to launch the binary the first time because it isn't
signed by Apple. To allow it:

1. Try to run `./minai`; macOS will block it and say "cannot be opened".
2. Go to **System Settings → Privacy & Security**, scroll down, click
   **"Open Anyway"** next to the blocked file.
3. Run `./minai` again. It will work from then on.

Or, from a terminal:

```bash
xattr -d com.apple.quarantine ./minai
./minai
```

## What they do

Each binary is the compiled MinAI program — the exact `minai.cpp` from this
repository. Run with `./minai` for the default demo, or `./minai --help`
for all flags. See [../TRAINER.md](../TRAINER.md) for the flag reference
and [../ARITHMETICOFINTELLIGENCE.md](../ARITHMETICOFINTELLIGENCE.md) for the textbook walkthrough.

## How these are built

- **macOS universal**: `c++ -std=c++17 -O2 -arch arm64 -arch x86_64 -o bin/macos-universal/minai minai.cpp`
- **Linux** and **Windows**: produced by the GitHub Actions CI workflow
  (`.github/workflows/build.yml`) on every push to main, then committed
  manually to this folder.
