# Setup on WSL and Linux

The lessons run on CPU. No NVIDIA driver, CUDA toolkit, Apple Silicon, GPU
passthrough, Jupyter server, or cloud account is required.

The recommended target is **WSL 2 with Ubuntu 24.04 on an Intel/AMD x86_64 PC**.
The same Linux instructions work on native Ubuntu 24.04. Other glibc-based
x86_64 distributions need equivalent prerequisites. The provided Linux wheel
profile requires glibc 2.28 or newer; it is not an Alpine/musl or ARM64 profile.

Use **Python 3.12**, or Python 3.13 if you already have it. Python 3.14 and the
older default Python in some Linux distributions are outside the project's
declared range. Around 8 GB of RAM and a few GB of free disk space are a
comfortable starting point, not hard minimums. Keep the documented small image
subsets for CPU experiments; full-dataset ResNet training costs much more.

## 1. Windows only: prepare WSL 2

Skip this section on native Linux. Reuse an existing suitable WSL 2 distribution
if you have one; there is no need to install a second copy.

WSL's installation commands require Windows 11 or Windows 10 version 2004+
(build 19041+), virtualization support, and permission to enable the Windows
features. On a managed machine, ask IT to enable/install WSL if this is blocked.

In **Windows PowerShell as administrator**, install Ubuntu if needed:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart Windows if requested, open Ubuntu, and create its Linux username and
password. Check the distribution's version from PowerShell:

```powershell
wsl --list --verbose
```

The `VERSION` column should say `2`. If this distribution is still on WSL 1,
convert it with `wsl --set-version Ubuntu-24.04 2`, using its actual listed name.
This does not require any GPU configuration.

All remaining commands, except the explicitly labeled offline PowerShell
example, run in the **Ubuntu/Linux terminal**, not Windows PowerShell.

## 2. Install Linux prerequisites

On Ubuntu 24.04:

```sh
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip ca-certificates libgomp1 libstdc++6
python3.12 --version
uname -m
```

Expect Python 3.12 and `x86_64`. Use your distribution's approved equivalent
packages if it is not Ubuntu. Python must include `venv` and pip; `libgomp` and
the C++ runtime support the prebuilt PyTorch libraries. No C/C++ compiler,
source build of PyTorch, or system-wide Python package installation is needed.

If your default `python3` is 3.10 or 3.11, do not replace the system interpreter
or assume it meets the course requirement. Install an approved Python 3.12/3.13
alongside it, or use Ubuntu 24.04. If using Python 3.13, substitute `python3.13`
for `python3.12` when creating the virtual environment.

## 3. Bring over the source, not the Mac environment

Transfer the updated course through your normal approved Git or folder-copy
workflow. The examples below assume it is at `~/src/pytorch-course`; use your
existing checkout's location instead if different.

In WSL, prefer the Linux filesystem (`~/src/...`) rather than running from
`/mnt/c/...`, a Windows OneDrive directory, or a Windows-created virtual
environment. This avoids filesystem overhead and environment/symlink problems.
The Windows copy can be the transfer source, but run the course from Linux.

Keep `README.md`, this guide, `CURRICULUM.md`, `pyproject.toml`, `requirements/`,
`course/`, `lessons/`, `scripts/`, `tests/`, and the workflow if using GitHub.
Do **not** transfer `.venv/`, `.python/`, `.bootstrap/`, `.cache/`,
`__pycache__/`, or `.DS_Store`. You do not need `artifacts/` or `.artifacts/`:
running lessons in order recreates their checkpoints. Transfer optional data
and weights separately when needed.

The Mac virtual environment and macOS ARM64 wheel bundle cannot run on Linux.
A Windows virtual environment and Windows wheels cannot run inside WSL either.
Create the Linux environment below rather than trying to repair a copied one.

## 4. Create a local, CPU-only environment

From the Linux course folder, with no copied `.venv` present:

```sh
cd ~/src/pytorch-course
python3.12 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu -r requirements/linux-cpu.txt
.venv/bin/python -m pip check
```

Use the download command only if that package source is approved on your
machine. For a restricted network, use the approved mirror/offline procedure
below instead. No lesson itself downloads packages or data.

The [Linux CPU profile](requirements/linux-cpu.txt) pins `torch==2.8.0+cpu`.
That release meets the project's `torch>=2.8,<3` requirement, has official
Python 3.12/3.13 Linux wheels, and has a matching optional torchvision release.
The `+cpu` requirement prevents accidentally selecting a CUDA build, even when
an index contains both. This is a separate Linux setup profile, not a request
to downgrade the existing Mac environment. Transitive package versions can
still vary; these requirement files are not a complete hashed lockfile.

**Do not use the existing `uv.lock` as the Linux CPU installation profile.**
It records the original corporate package source and includes CUDA dependencies
for Linux. `uv sync --locked` can therefore use an unavailable registry or pull
unneeded accelerator packages. The pip commands above deliberately use the
Linux requirements instead; do not run `uv sync` over that environment afterward.
Keep the lockfile for its original environment rather than rewriting it blindly.

Activation is optional. `.venv/bin/python` explicitly selects the right
interpreter on both Linux and macOS. Do not use `sudo pip`, Windows `python.exe`,
or Windows `py` to run lessons inside WSL.

## 5. Verify and start learning

```sh
.venv/bin/python -c "import sys, torch; print(sys.executable); print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); assert torch.version.cuda is None and torch.version.hip is None, 'Expected a CPU-only build'"
.venv/bin/python lessons/01-learn-a-line/train.py
.venv/bin/python -m lessons.02-inspect-gradients.train
```

The executable should be your Linux project's `.venv/bin/python`; PyTorch
should report `2.8.0+cpu` and `CUDA build: None`. The first lesson should learn
approximately weight `2`, bias `1`, and predictions `[7, 9]`.

You may see a PyTorch warning about missing NumPy in the torch-only setup.
Those lessons do not require it. Optional torchvision installs its NumPy/Pillow
dependencies when you reach real transfer learning; do not hide warnings or
alter system Python to remove this one.

Run the whole offline course with:

```sh
.venv/bin/python scripts/check_course.py
```

This runs lessons in order, including saved-model prerequisites. Image and
transfer lessons use explicitly synthetic `--smoke` paths in this command;
passing it does not establish real Fashion-MNIST or pretrained-model quality.
See [all focused check commands](README.md#course-checks) for additional contracts.
On a slow or resource-constrained machine, `--timeout 300` raises the checker's
per-lesson deadline without changing the training configuration.

The [Linux CPU workflow](.github/workflows/linux-cpu.yml) runs the same offline
course and contracts on Ubuntu 24.04 with Python 3.12 and 3.13 when this project
is hosted on GitHub with Actions enabled. It also checks that the optional
torchvision wheel imports with its matching CPU PyTorch build. It does not
download datasets or pretrained weights.

For VS Code on Windows, use Microsoft's WSL extension and open the folder
**in WSL** (for example, `code .` from the Linux course folder). Install the
Python extension in that WSL context and select `.venv/bin/python` as the
interpreter, not an interpreter installed on Windows.

## Optional real transfer learning

Sessions 18-25 read local Fashion-MNIST files directly and do not require
torchvision. All synthetic smoke paths also work with the base profile.

For the **real ResNet** path in Sessions 30-33, add the
[matching Linux vision profile](requirements/linux-vision-cpu.txt):

```sh
.venv/bin/python -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu -r requirements/linux-vision-cpu.txt
.venv/bin/python -m pip check
.venv/bin/python -c "import torch, torchvision; print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)"
```

This pins `torchvision==0.23.0+cpu` alongside `torch==2.8.0+cpu`; pip resolves
its required NumPy/Pillow dependencies. Do not install an arbitrary latest
torchvision alongside a differently versioned PyTorch.

Supply the approved local dataset files described in
[Session 18](lessons/18-images-flat/README.md) and the trusted ResNet weights
and digest described in [Session 30](lessons/30-transfer-contract/README.md).
The version/profile install does not provide these resources or permission to
use them. The lesson commands and CPU-sized subsets are the same on Linux and
macOS; paths such as `data/fashion` are relative to the course folder.

## Restricted networks and offline installation

An IT-approved mirror can replace the CPU index URL in the install commands,
but it must contain the exact `+cpu` wheel versions and all their dependencies.
Stop and ask IT if a required host or artifact is blocked. Do not disable TLS
verification, use `--trusted-host`, or try to work around network restrictions.

Alternatively, on an approved connected Linux x86_64 machine using the same
Python minor version as the destination, download a complete wheel set:

```sh
python3.12 -m pip download --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu -r requirements/linux-vision-cpu.txt --dest wheelhouse-linux
```

Using the vision profile also includes the base profile and its dependencies.
Use `requirements/linux-cpu.txt` instead if only the base course is needed.

If the connected machine is **Windows**, it can download Linux wheels without
running them. With Python 3.12 installed and the updated course as the current
folder, run this in **PowerShell**; the long download command is one line:

```powershell
py -3.12 -m pip download --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 --python-version 3.12 --implementation cp --abi cp312 -r requirements/linux-vision-cpu.txt --dest wheelhouse-linux
if ($LASTEXITCODE -ne 0) { throw "Download failed; do not use an incomplete wheel set." }
(Resolve-Path .\wheelhouse-linux).Path
```

The final line prints the full download folder path. These commands target
Python 3.12 on Linux x86_64, **not** Windows or macOS. Use matching Python 3.12
on the download machine too so dependency markers are evaluated consistently.
For a Python 3.13 destination, use a Python 3.13 downloader and change the
target version and ABI to `3.13` and `cp313`. An ARM64 destination needs a
different, explicitly matched wheel profile; do not use these x86_64 files.

Transfer the complete `wheelhouse-linux` directory through an approved method.
This generated package directory is ignored by Git; do not commit its wheels.
After installing the Linux OS prerequisites and creating `.venv`, install
without contacting any package index:

```sh
.venv/bin/python -m pip install --no-index --find-links /path/to/wheelhouse-linux -r requirements/linux-cpu.txt
.venv/bin/python -m pip check
```

Replace `/path/to/wheelhouse-linux` with the actual transferred directory.
For real transfer learning, use `requirements/linux-vision-cpu.txt` instead.
If pip reports a missing or incompatible wheel, the set does not match the
target or is incomplete; obtain the correct files rather than allowing a
network fallback or source build.

## References

- [Microsoft: install WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
- [Microsoft: filesystems and WSL performance](https://learn.microsoft.com/en-us/windows/wsl/filesystems)
- [PyTorch: version-specific CPU installation commands](https://pytorch.org/get-started/previous-versions/)
- [Torchvision: version compatibility](https://github.com/pytorch/vision#installation)
