## Wafer VSCode Extension – NCU & PTX/SASS Internals

This document traces the **actual** Wafer VSCode extension bundled in
`wafer.wafer-0.2.92.vsix` (unpacked into `wafer/extension`) and then explains
how to implement the same workflows **without** using the extension.

All links below are **relative** to this file so they are clickable from VSCode.

---

## 1. Key Files & Function Index

### 1.1 Key Files

- Extension manifest: [`wafer/extension/package.json`](wafer/extension/package.json)
- Main extension bundle: [`wafer/extension/dist/extension.js`](wafer/extension/dist/extension.js)
- NCU helper tool (Python):
  [`wafer/extension/resources/ncu-tool/ncu_tool.py`](wafer/extension/resources/ncu-tool/ncu_tool.py)
- Compiler Explorer / PTX+SASS tool (Python):
  [`wafer/extension/resources/compiler-explorer-tool/compiler_explorer_tool.py`](wafer/extension/resources/compiler-explorer-tool/compiler_explorer_tool.py)
- Webview UI (React SPA used by all tools):
  [`wafer/extension/dist/webview/panel.js`](wafer/extension/dist/webview/panel.js)
  and sources under
  [`wafer/extension/webview-ui/src`](wafer/extension/webview-ui/src)

### 1.2 Key Functions Index

| Function / Class                                      | File                                                                                                   | Purpose |
|------------------------------------------------------|--------------------------------------------------------------------------------------------------------|---------|
| `activate` / `Xa` (“WaferViewProvider”)              | [`wafer/extension/dist/extension.js#L1501`](wafer/extension/dist/extension.js#L1501)                   | Activates Wafer and drives the sidebar webview + commands. |
| `xa` (“NCU Profiler” webview factory)                | [`wafer/extension/dist/extension.js#L1312`](wafer/extension/dist/extension.js#L1312)                   | Creates the NCU Profiler webview with initial React state. |
| `Aa` (“Compiler Explorer” webview factory)           | [`wafer/extension/dist/extension.js#L1312`](wafer/extension/dist/extension.js#L1312)                   | Creates the Compiler Explorer webview with PTX/SASS view. |
| `wr` (“NCUHandler”)                                  | [`wafer/extension/dist/extension.js#L1341`](wafer/extension/dist/extension.js#L1341)                   | Wraps `ncu_tool.py`, runs/parses NCU locally or over SSH. |
| `Xa.executeNcuProfile`                               | [`wafer/extension/dist/extension.js#L1501`](wafer/extension/dist/extension.js#L1501)                   | Kicks off NCU profiling and displays results in the UI. |
| `Xa.selectNcuFileAndAnalyze`                         | [`wafer/extension/dist/extension.js#L1501`](wafer/extension/dist/extension.js#L1501)                   | Lets user pick an `.ncu-rep` file and analyze it. |
| `cmd_run`                                            | [`wafer/extension/resources/ncu-tool/ncu_tool.py#L326`](wafer/extension/resources/ncu-tool/ncu_tool.py#L326) | Runs `ncu` on a command, writes `.ncu-rep`/CSV report. |
| `cmd_parse` + `parse_ncu_output`                     | [`ncu_tool.py#L713`](wafer/extension/resources/ncu-tool/ncu_tool.py#L713)                              | Imports `.ncu-rep`, parses text details into structured kernels + recommendations. |
| `nr` (“CompilerExplorerHandler”)                     | [`wafer/extension/dist/extension.js#L1351`](wafer/extension/dist/extension.js#L1351)                   | Wraps `compiler_explorer_tool.py`, compiles CUDA to PTX/SASS. |
| `nr.compile`                                         | [`wafer/extension/dist/extension.js#L1351`](wafer/extension/dist/extension.js#L1351)                   | Calls Python tool to compile and return PTX/SASS and mappings. |
| `cmd_check`, `cmd_detect_arch`, `cmd_compile`        | [`compiler_explorer_tool.py#L370`](wafer/extension/resources/compiler-explorer-tool/compiler_explorer_tool.py#L370) | Checks NVCC, detects GPU arch, compiles CUDA to PTX/SASS. |
| `parse_ptx_line_mapping`, `parse_sass_line_mapping`  | [`compiler_explorer_tool.py#L455`](wafer/extension/resources/compiler-explorer-tool/compiler_explorer_tool.py#L455) | Build bidirectional source↔PTX/SASS line maps. |

> Note: `extension.js` is a large bundled file; class names like `Xa`, `wr`,
> and `nr` come from the bundler. They’re treated here as:
> - `Xa` → main Wafer view/controller
> - `wr` → NCUHandler
> - `nr` → CompilerExplorerHandler

---

## 2. Architecture Overview

At a high level, Wafer implements GPU‑profiling features as three layers:

```text
React webview UI  <-->  VSCode extension (Xa, wr, nr)  <-->  Python tools  <-->  ncu / nvcc / nvdisasm
```

- The **VSCode extension** is described in
  [`wafer/extension/package.json`](wafer/extension/package.json) and
  implemented in [`dist/extension.js`](wafer/extension/dist/extension.js).
  It contributes a Wafer activity bar icon and a single webview view
  `wafer.mainView`.
- The **webview UI** is a React SPA (`panel.js`) that switches between
  tools (“Home”, “NCU Profiler”, “Compiler Explorer”, etc.) based on an
  `__TOOL_ID__` field in the initial state.
- For heavy GPU tooling Wafer delegates to **Python helper scripts**:
  - `ncu_tool.py` for Nsight Compute.
  - `compiler_explorer_tool.py` for NVCC + PTX/SASS generation.
- On remote/SSH workspaces, the extension uploads these Python scripts to the
  remote machine and executes them there, rather than locally.

### 2.1 View/Tool Wiring

The extension manifest declares the Wafer container and view:

```jsonc
// wafer/extension/package.json
"activationEvents": [
  "onView:wafer.mainView"
],
"main": "./dist/extension.js",
"contributes": {
  "viewsContainers": {
    "activitybar": [
      { "id": "wafer", "title": "Wafer", "icon": "media/icon.svg" }
    ]
  },
  "views": {
    "wafer": [
      { "type": "webview", "id": "wafer.mainView", "name": "Wafer" }
    ]
  }
}
```

When `wafer.mainView` is opened, `activate` (in `extension.js`) creates a
`WaferViewProvider` (`Xa`) that loads the webview and responds to messages.

The individual tools (Daily Kernel, NCU, Compiler Explorer, etc.) share the
same React SPA; different helper functions provide the initial HTML:

```ts
// wafer/extension/dist/extension.js:1312
function xa(r, e) {
  return ht({
    extensionUri: r,
    webview: e,
    scriptPath: "panel.js",
    stylePath: "main.css",
    title: "NCU Profiler",
    initialState: { __TOOL_ID__: "ncu" },
    apiUrl: H.get().getApiUrl(),
  });
}

function Aa(r, e, t) {
  return ht({
    extensionUri: r,
    webview: e,
    scriptPath: "panel.js",
    stylePath: "main.css",
    title: "Compiler Explorer",
    initialState: { __TOOL_ID__: "compiler-explorer", compiledState: t || null },
    apiUrl: H.get().getApiUrl(),
  });
}
```

Here `ht(...)` is the shared HTML template; the `__TOOL_ID__` tells the React
app which tool to render.

### 2.2 High‑Level Component Diagram

```text
+---------------------------+          +-----------------------------+
| VSCode Extension (Xa)    |          | React Webview (panel.js)    |
|  - commands              |<-------->|  - shows tools (NCU, CE)    |
|  - message routing       |   post   |  - sends messages (compile, |
|  - NCUHandler (wr)       | messages |    profile, select file)    |
|  - CompilerHandler (nr)  |          +-----------------------------+
|  - Wevin, workspaces ... |
+---------------+-----------+
                |
                | spawns Python
                v
+---------------------------+
| Python tools              |
|  ncu_tool.py             |
|  compiler_explorer_tool.py|
+---------------+-----------+
                |
                | subprocess
                v
+---------------------------+
| ncu / nvcc / nvdisasm     |
+---------------------------+
```

---

## 3. NCU Profiling Flow (Wafer NCU Profiler)

This section traces the path for **“view NCU profiles in VSCode”**.

### 3.1 User → Webview → Xa

When the user selects the NCU tool from the Wafer home UI, the webview sends a
message with `toolId: "ncu"`. That is handled by a message router in
`extension.js` (simplified here):

```ts
// wafer/extension/dist/extension.js:1489 (inside rf(...) handler)
case "selectTool":
  if (e.toolId === "ncu") {
    r._currentTool = "ncu";
    r.checkNcuInstallation();
    r.refresh();
  }
```

- `r` is the `WaferViewProvider` instance (`Xa`).
- `checkNcuInstallation()` updates the NCU status (installed + version).
- `refresh()` rebuilds the webview HTML, now with `__TOOL_ID__:"ncu"`.

The NCU panel itself is created via `openToolPanel("ncu", ...)`, which calls
`xa(...)` and then attaches panel‑specific message handlers.

### 3.2 Xa → NCUHandler (`wr`)

`Xa` owns an `ncuHandler` instance:

```ts
// wafer/extension/dist/extension.js:1501
Xa = class {
  _view;
  _currentTool = null;
  ncuHandler;
  ncuInstalled = !1;
  ncuVersion = "";
  parsedSummary;
  // ...
  constructor() {
    this.ncuHandler = new wr;
    this.wevinHandler = new qr;
  }

  async checkNcuInstallation() {
    await af(this.ncuHandler, (installed, version) => {
      this.ncuInstalled = installed;
      this.ncuVersion = version;
      this._view &&
        this._view.webview.postMessage({
          type: "ncuInstallationStatus",
          installed,
          version,
        });
    });
  }

  async executeNcuProfile(config) {
    await cf(
      config,
      {
        ncuHandler: this.ncuHandler,
        parsedSummary: this.parsedSummary,
        setParsedSummary: (s) => { this.parsedSummary = s; },
        _view: this._view,
      },
      (panel) => this.setupNcuPanel(panel),
      (reportPath, outputDir, onStatus, isSsh) => {
        lf(
          reportPath,
          outputDir,
          onStatus,
          this.ncuHandler,
          (summary) => { this.parsedSummary = summary; },
          isSsh
        );
      }
    );
  }
};
```

Key points:

- `af(...)` is a helper that calls `ncuHandler.checkInstallation()` and feeds
  the result back into the webview.
- `cf(...)` orchestrates running a profile (either locally or on Wafer’s GPU
  service), then calls `lf(...)` to analyze the resulting `.ncu-rep` with
  `ncuHandler`.
- `parsedSummary` caches the last NCU analysis so the UI can reuse it.

### 3.3 NCUHandler (`wr`) → `ncu_tool.py`

The NCUHandler class is defined in `extension.js` and points at
`resources/ncu-tool/ncu_tool.py`:

```ts
// wafer/extension/dist/extension.js:1341
wr = class {
  scriptPath;
  venvManager;

  constructor() {
    this.scriptPath = ee.resolve(
      M.get().extensionPath,
      "resources",
      "ncu-tool",
      "ncu_tool.py"
    );
    this.venvManager = new Ss;
  }

  async checkInstallation() {
    bs("NCU Detection");
    m(`[Wafer NCU Detection] Script path: ${this.scriptPath}`);
    try {
      const result = await Vt.findNCU();
      return result;
    } catch (err) {
      // On error, surface a VSCode message and return a structured error.
      const message = err instanceof Error ? err.message : String(err);
      ze.window
        .showErrorMessage(
          'NCU detection failed. Check "Wafer" output channel for details.',
          "Open Output Channel"
        )
        .then((choice) => {
          if (choice === "Open Output Channel") _s();
        });
      return { installed: !1, install_command: `Error checking NCU: ${message}` };
    }
  }

  async parseReport(reportPath, outputDir, onProgress, isSsh) {
    const useSsh = isSsh ?? dt();
    m(
      `[Wafer NCU] parseReport called with reportPath=${reportPath}, isSsh=${isSsh}, useSsh=${useSsh}`
    );
    return useSsh
      ? this.runPythonViaSsh(["parse", reportPath, "--output-dir", outputDir], onProgress)
      : this.runPython(["parse", reportPath, "--output-dir", outputDir], onProgress);
  }

  async exportCSV(reportPath, isSsh) {
    return (isSsh ?? dt())
      ? this.runPythonViaSsh(["csv-export", reportPath])
      : this.runPython(["csv-export", reportPath]);
  }
};
```

The interesting logic is in `runPythonViaSsh` (remote) and an analogous
`runPython` (local) method:

```ts
// wafer/extension/dist/extension.js:1341 (excerpt)
async runPythonViaSsh(args, onProgress) {
  m(`[Wafer NCU] runPythonRemote called with args: ${args.join(" ")}`);

  // 1. Read ncu_tool.py from the local extension install
  const scriptUri = ze.Uri.file(this.scriptPath);
  const contentBytes = await ze.workspace.fs.readFile(scriptUri);
  const scriptText = Buffer.from(contentBytes).toString("utf8");

  // 2. Upload to the remote SSH workspace under .wafer/tmp/
  const workspace = V.get();
  const remotePath = workspace.getRemotePath();
  const remoteDir = `${remotePath}/.wafer/tmp`;
  const remoteScript = `${remoteDir}/wafer_ncu_tool_${Date.now()}.py`;
  // ... create directory and write scriptText to remoteScript ...

  // 3. Execute via remote shell
  const quotedArgs = args.map(a =>
    a.includes(" ") || a.includes('"') || a.includes("'")
      ? `"${a.replace(/"/g, '\\"')}"`
      : a
  );
  const cmd = `python3 "${remoteScript}" ${quotedArgs.join(" ")}`;
  const output = await Vt.executeRemoteCommand(cmd, 120000, 1);

  // 4. Parse JSON from stdout and return it to the caller
  // (onProgress is used to stream log lines back into the webview)
}
```

The local `runPython(...)` has a similar shape but calls `python3` directly on
`this.scriptPath` using `child_process.spawn`, accumulating stdout and then
`JSON.parse`‑ing the result.

### 3.4 `ncu_tool.py` – Running & Parsing NCU

The Python helper provides a CLI that the extension calls with subcommands.

#### 3.4.1 Command line entrypoint

```py
# wafer/extension/resources/ncu-tool/ncu_tool.py:1022
def main():
    parser = argparse.ArgumentParser(
        description="NCU Tool - NVIDIA Nsight Compute analysis for GPU optimization"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("check", help="Check if NCU is installed")

    parse_parser = subparsers.add_parser("parse", help="Parse an .ncu-rep file")
    parse_parser.add_argument("file", help="Path to .ncu-rep file")
    parse_parser.add_argument("--output-dir", "-o", help="Output directory for analysis files")

    run_parser = subparsers.add_parser("run", help="Run NCU profiler on a command")
    run_parser.add_argument("run_cmd", help="Command to profile")
    run_parser.add_argument("--output-file", "-f", required=True, help="Output file name")
    run_parser.add_argument("--output-dir", "-d", default=".wafer/ncu-tool", help="Output directory")
    run_parser.add_argument("--format", choices=["ncu-rep", "csv"], default="ncu-rep")
    run_parser.add_argument("--extra-args", "-e", help="Extra NCU arguments")

    # csv-export & details-export omitted for brevity

    args = parser.parse_args()

    if args.command == "check":
        result = cmd_check()
    elif args.command == "parse":
        result = cmd_parse(args.file, args.output_dir)
    elif args.command == "run":
        result = cmd_run(args.run_cmd, args.output_file, args.output_dir, args.format, args.extra_args)
    ...

    print(json.dumps(result, indent=2))
```

The extension always expects **JSON on stdout**, which is why the JS wrapper
searches stdout for a JSON object containing `"success"`.

#### 3.4.2 `cmd_run`: capturing an `.ncu-rep`

```py
# ncu_tool.py:326
def cmd_run(command: str, output_file: str, output_dir: str,
            output_format: str, extra_args: Optional[str] = None) -> dict:
    """Run NCU profiler on a command and save the report."""
    ncu_path = find_ncu()
    if not ncu_path:
        return {"success": False, "error": "NCU not installed. Run 'check' command for install instructions."}

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ext = ".csv" if output_format == "csv" else ".ncu-rep"
    output_file_path = out_path / f"{output_file}{ext}"

    ncu_cmd = [ncu_path]
    if extra_args:
        for arg in extra_args.replace('\n', ' ').split():
            if arg.strip():
                ncu_cmd.append(arg.strip())

    ncu_cmd.extend(["-o", str(output_file_path)])
    ncu_cmd.extend(command.split())

    result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=600)
    ...
```

This is how Wafer **generates** `.ncu-rep` files when asked to profile a
command from the UI.

#### 3.4.3 `cmd_parse`: turning `.ncu-rep` into structured data

```py
# ncu_tool.py:713
def cmd_parse(file_path: str, output_dir: Optional[str] = None) -> dict:
    """Parse an .ncu-rep file and extract metrics."""
    path = Path(file_path)
    if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}
    if path.suffix != ".ncu-rep":
        return {"success": False, "error": f"Expected .ncu-rep file, got: {path.suffix}"}

    ncu_path = find_ncu()
    if not ncu_path:
        return {"success": False, "error": "NCU not installed. Run 'check' command for install instructions."}

    file_size_bytes = path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    timeout_seconds = max(120, int(file_size_mb * 2))

    # Phase 1: session info (GPU name)
    session_result = subprocess.run(
        [ncu_path, "--import", str(path), "--page", "session"],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )

    # Phase 2: detailed metrics & recommendations
    details_result = subprocess.run(
        [ncu_path, "--import", str(path), "--page", "details"],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )

    summary = parse_ncu_output(session_result.stdout, details_result.stdout)
    txt_path = (Path(output_dir) if output_dir else Path(".wafer") / "ncu-tool") / \
               f"{path.stem}-{datetime.now().strftime('%Y%m%d_%H%M%S')}" / "analysis.txt"
    # (directory setup omitted here for brevity)

    llm_text = generate_llm_text(path.name, summary)
    txt_path.write_text(llm_text)

    return {
        "success": True,
        "output_file": str(txt_path),
        "summary": summary,
        "recommendations": summary.get("recommendations", []),
    }
```

`parse_ncu_output(...)` then parses the human‑readable `details` page into a
structured `summary`:

- Per‑kernel fields: duration, occupancy, memory throughput, SM throughput,
  registers/thread, block/grid size.
- Per‑kernel plus global **recommendations** parsed from Nsight’s textual
  diagnostic sections.

### 3.5 NCU Sequence Diagram

Putting the pieces together:

```text
User (NCU tool)             Webview (React)           VSCode Extension (Xa, wr)          ncu_tool.py                 ncu
-----------------           -----------------         -----------------------------       ------------------------     ------------------
Clicks "NCU"          ->    postMessage(selectTool) -> rf(...) sets _currentTool="ncu"
                                                      Xa.checkNcuInstallation()
                                                      Xa.refresh()
                                                                                          (no NCU yet)

Clicks "Profile" or        postMessage(executeNcu)   -> Xa.executeNcuProfile(cfg)
selects .ncu-rep                                         cf(...) orchestrates run/parse
                                                      -> wr.parseReport(reportPath,...)
                                                          (local or SSH)
                                                      -> wr.runPython(...)
                                                                                     -> main(): cmd_parse(...)
                                                                                     -> subprocess ncu --import --page session/details
                                                                                     -> parse_ncu_output(...)
                                                                                     <- JSON { success, summary, recommendations }
                                                      <- JSON summary
                                                      -> cache parsedSummary
                                                      -> webview.postMessage(parseResult,...)

Sees metrics, timeline    <- render NCU React view  <- receives parseResult
and recommendations
```

---

## 4. PTX / SASS Correlation (Compiler Explorer)

Wafer’s **Compiler Explorer** tool handles CUDA compilation to PTX/SASS and
correlates instructions with source lines.

### 4.1 Panel setup (`Aa` + `Zn`)

The Compiler Explorer webview is created with `Aa(...)` (see §2.1) and then
wired up by `Zn(panel)`:

```ts
// wafer/extension/dist/extension.js:1434 (excerpt)
function Zn(panel) {
  const handler = new nr(); // CompilerExplorerHandler
  const panelId = panel.__waferPanelId || panel.viewColumn?.toString() || String(Date.now());
  const { panelHandlers } = (Gr(), re(qa));

  if (panelHandlers) {
    panelHandlers.has(panelId) || panelHandlers.set(panelId, {});
    const entry = panelHandlers.get(panelId);
    entry.compilerExplorerHandler = { checkInstallation: () => handler.checkInstallation() };
    // ...
  }

  // On panel open, immediately check nvcc and maybe detect arch
  (async () => {
    const status = await handler.checkInstallation();
    panel.webview.postMessage({ type: "nvccStatus", ...status });
    if (status.installed) {
      const archInfo = await handler.detectArchitecture();
      panel.webview.postMessage({ type: "archDetected", ...archInfo });
    }
  })();

  panel.webview.onDidReceiveMessage(async (msg) => {
    switch (msg.command) {
      case "selectFile":
        // prompts for a .cu file and posts fileSelected(content)
        break;
      case "compileSingleFile":
        // invokes handler.compile(...) and posts compileResult/compileOutput messages
        break;
      // other UI commands omitted
    }
  });
}
```

The React UI (in `panel.js` / `webview-ui`) turns these `nvccStatus`,
`archDetected`, `fileSelected`, and `compileResult` messages into the
interactive PTX/SASS view.

### 4.2 CompilerExplorerHandler (`nr`) → `compiler_explorer_tool.py`

The handler wraps the Python tool and knows how to run it locally or via SSH:

```ts
// wafer/extension/dist/extension.js:1351 (excerpt)
nr = class {
  scriptPath;

  constructor() {
    const extPath = M.get().extensionPath;
    this.scriptPath = J.resolve(
      extPath,
      "resources",
      "compiler-explorer-tool",
      "compiler_explorer_tool.py"
    );
  }

  async checkInstallation() {
    bs("NVCC Detection");
    m(`[Wafer NVCC Detection] Script path: ${this.scriptPath}`);
    try {
      const res = await Vt.findNVCC();
      return {
        installed: res.installed,
        path: res.path,
        version: res.version,
        install_command: res.install_command,
        nvdisasm_available: !1,
        isSsh: res.isSsh,
      };
    } catch (err) {
      // ... show VSCode error message and return structured error ...
    }
  }

  async detectArchitecture(isSsh) {
    return isSsh ?? dt()
      ? this.runPythonViaSsh(["detect-arch"])
      : this.runPython(["detect-arch"]);
  }

  async compile(filePath, arch, outputDir, abortSignal, onLine, isSshOverride) {
    if (filePath.endsWith(".cuh")) {
      return { success: !1, error: "Header files (.cuh) cannot be compiled directly. Please select a .cu source file instead." };
    }

    const args = ["compile", filePath, "--arch", arch];
    if (outputDir) args.push("--output-dir", outputDir);

    const useSsh = isSshOverride ?? dt();
    m(`[Wafer NVCC] compile called with filePath=${filePath}, arch=${arch}, isSsh=${isSshOverride}, useSsh=${useSsh}`);

    try {
      const nvccInfo = await Vt.findNVCC();
      if (nvccInfo.installed && nvccInfo.path) {
        args.push("--nvcc-path", nvccInfo.path);
      }
    } catch {}

    return useSsh
      ? this.runPythonViaSsh(args, 60000, onLine, abortSignal)
      : this.runPython(args, 30000, onLine, abortSignal);
  }
};
```

- `runPythonViaSsh` mirrors the NCU SSH flow: upload the script into
  `.wafer/tmp/wafer_compiler_tool_*.py`, execute it via `python3`, parse JSON.
- `runPython` (local) uses `child_process.spawn` with an augmented `PATH` that
  includes common CUDA install locations (see the code around
  `buildEnhancedPath()` near the `[CompilerExplorerHandler]` logs).

### 4.3 `compiler_explorer_tool.py` – PTX/SASS + Line Mappings

The Python tool is responsible for:

1. Finding `nvcc` & `nvdisasm`.
2. Detecting GPU architecture.
3. Compiling a CUDA file into PTX and SASS with line information.
4. Building source↔assembly line mappings from `.loc` and `//## File` markers.

#### 4.3.1 Entry point

```py
# compiler_explorer_tool.py:966
def main():
    parser = argparse.ArgumentParser(description="CUDA PTX/SASS Compiler Explorer")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    subparsers.add_parser("check", help="Check if nvcc is installed")
    subparsers.add_parser("detect-arch", help="Detect GPU architecture")

    compile_parser = subparsers.add_parser("compile", help="Compile CUDA to PTX/SASS")
    compile_parser.add_argument("file", help="CUDA source file")
    compile_parser.add_argument("--arch", default="sm_80", help="Target architecture (e.g., sm_80, sm_90, sm_100)")
    compile_parser.add_argument("--output-dir", help="Output directory")
    compile_parser.add_argument("--nvcc-path", help="Path to nvcc executable (overrides auto-detection)")

    args = parser.parse_args()

    if args.command == "check":
        result = cmd_check()
    elif args.command == "detect-arch":
        result = cmd_detect_arch()
    elif args.command == "compile":
        nvcc_path = getattr(args, 'nvcc_path', None)
        result = cmd_compile(args.file, args.arch, args.output_dir, nvcc_path)
    else:
        parser.print_help()
        sys.exit(1)

    print(json.dumps(result))
```

#### 4.3.2 `cmd_check` & `cmd_detect_arch`

```py
# compiler_explorer_tool.py:370
def cmd_check() -> Dict[str, Any]:
    nvcc = find_nvcc()
    nvdisasm = find_nvdisasm()
    if nvcc:
        version = get_nvcc_version(nvcc)
        return {
            "installed": True,
            "path": nvcc,
            "version": version,
            "nvdisasm_available": nvdisasm is not None,
            "nvdisasm_path": nvdisasm,
        }
    else:
        return {
            "installed": False,
            "install_command": get_install_command(),
        }
```

`cmd_detect_arch` calls `nvidia-smi --query-gpu=compute_cap,name` and converts
the compute capability (e.g. `8.9`) into an `sm_89` architecture string and a
human‑friendly `gpu_type` (L4, H100, B200, etc.).

#### 4.3.3 `cmd_compile`: PTX + SASS with line info

```py
# compiler_explorer_tool.py:760
def cmd_compile(file_path: str, arch: str, output_dir: Optional[str] = None,
                nvcc_path: Optional[str] = None) -> Dict[str, Any]:
    # Choose nvcc (override or auto-detect)
    if nvcc_path and os.path.isfile(nvcc_path) and os.access(nvcc_path, os.X_OK):
        nvcc = nvcc_path
    else:
        nvcc = find_nvcc()
    # ... check CUDA version & architecture support ...

    path = Path(file_path)
    if not path.exists():
        return {"success": False, "error": f"❌ File Not Found\n\nThe file does not exist: {file_path}"}

    result = {
        "success": True,
        "ptx": None,
        "sass": None,
        "ptx_mapping": None,
        "sass_mapping": None,
        "arch_used": arch,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        compile_path = path
        if path.suffix == '.cuh':
            # Create a wrapper .cu that #includes the header
            wrapper_file = Path(tmpdir) / f"{path.stem}_wrapper.cu"
            wrapper_file.write_text(f'#include "{path}"\n')
            compile_path = wrapper_file

        ptx_path = Path(tmpdir) / f"{path.stem}.ptx"
        cubin_path = Path(tmpdir) / f"{path.stem}.cubin"

        include_paths = find_include_paths(path)
        include_flags = []
        for inc in include_paths:
            include_flags.extend(["-I", inc])

        # PTX with line information
        ptx_cmd = [
            nvcc, "-ptx",
            f"-arch={arch}",
            "--generate-line-info",
            *include_flags,
            str(compile_path),
            "-o", str(ptx_path),
        ]
        ptx_result = subprocess.run(ptx_cmd, capture_output=True, text=True, timeout=120)
        if ptx_result.returncode != 0:
            # Error reporting (Python.h / PyTorch hints, etc.)
            ...
        ptx_content = ptx_path.read_text()
        result["ptx"] = ptx_content
        result["ptx_mapping"] = parse_ptx_line_mapping(ptx_content)

        # SASS via cubin + nvdisasm -g (if nvdisasm is available)
        if nvdisasm:
            cubin_cmd = [
                nvcc, "-cubin",
                f"-arch={arch}",
                "--generate-line-info",
                *include_flags,
                str(compile_path),
                "-o", str(cubin_path),
            ]
            cubin_result = subprocess.run(cubin_cmd, capture_output=True, text=True, timeout=120)
            if cubin_result.returncode == 0:
                sass_cmd = [nvdisasm, "-c", "-g", str(cubin_path)]
                sass_result = subprocess.run(sass_cmd, capture_output=True, text=True, timeout=60)
                if sass_result.returncode == 0:
                    result["sass"] = sass_result.stdout
                    result["sass_mapping"] = parse_sass_line_mapping(sass_result.stdout)

    return result
```

The important flags:

- `--generate-line-info` ensures PTX and SASS include source line information.
- PTX line info comes from `.loc` directives.
- SASS line info comes from `//## File "...", line N` comments emitted by
  `nvdisasm -g`.

#### 4.3.4 Building source↔assembly mappings

```py
# compiler_explorer_tool.py:455
def parse_ptx_line_mapping(ptx_content: str) -> Dict[str, Any]:
    """
    Parse .loc directives from PTX to build source->PTX line mapping.
    .loc format: .loc <file_id> <line_number> <column>
    """
    mapping = {"source_to_asm": {}, "asm_to_source": {}}
    lines = ptx_content.split('\n')
    current_source_line = None

    for ptx_line_num, line in enumerate(lines, 1):
        loc_match = re.match(r'\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)', line)
        if loc_match:
            source_line = int(loc_match.group(2))
            current_source_line = source_line
            continue

        stripped = line.strip()
        if current_source_line and stripped and not stripped.startswith('//'):
            if stripped and (not stripped.startswith('.') or stripped.startswith('.pragma')):
                mapping["asm_to_source"][ptx_line_num] = current_source_line
                mapping["source_to_asm"].setdefault(current_source_line, [])
                if ptx_line_num not in mapping["source_to_asm"][current_source_line]:
                    mapping["source_to_asm"][current_source_line].append(ptx_line_num)

    return mapping
```

```py
# compiler_explorer_tool.py:495
def parse_sass_line_mapping(sass_content: str) -> Dict[str, Any]:
    """
    Parse nvdisasm output for source file correlation.
    nvdisasm with -g flag includes source lines as comments like:
    //## File "/path/to/file.cu", line 6
    """
    mapping = {"source_to_asm": {}, "asm_to_source": {}}
    lines = sass_content.split('\n')
    current_source_line = None

    for sass_line_num, line in enumerate(lines, 1):
        source_match = re.search(r'//##.*line\s+(\d+)', line, re.IGNORECASE)
        if source_match:
            current_source_line = int(source_match.group(1))
            continue

        stripped = line.strip()
        if current_source_line and stripped and not stripped.startswith('//'):
            if re.match(r'/\*[0-9a-fA-Fx]+\*/', stripped):
                mapping["asm_to_source"][sass_line_num] = current_source_line
                mapping["source_to_asm"].setdefault(current_source_line, [])
                if sass_line_num not in mapping["source_to_asm"][current_source_line]:
                    mapping["source_to_asm"][current_source_line].append(sass_line_num)

    return mapping
```

These mappings are what the React UI uses to highlight PTX/SASS instructions
when the user hovers or clicks a line in the CUDA source view and vice‑versa.

### 4.4 Compiler Explorer Sequence Diagram

```text
User (Compiler Explorer)   Webview (React)          VSCode Extension (Xa, nr)          compiler_explorer_tool.py   nvcc / nvdisasm
------------------------   -----------------        ------------------------------      --------------------------   ----------------------
Clicks "Compiler Explorer" -> selectTool("compiler-explorer")
                           -> openToolPanel(...)  -> Aa(...) creates CE webview
                                                    Zn(panel) wires handlers, checks nvcc

Selects .cu file          -> postMessage(selectFile)
                           <- fileSelected(content) from Zn (reads file)

Clicks "Compile"          -> postMessage(compileSingleFile, { filePath, arch })
                                                    -> nr.compile(filePath, arch, ...)
                                                       (local or SSH)
                                                    -> runPython / runPythonViaSsh
                                                                                     -> main(): cmd_compile(...)
                                                                                     -> nvcc -ptx --generate-line-info
                                                                                     -> nvcc -cubin --generate-line-info
                                                                                     -> nvdisasm -c -g
                                                                                     -> parse_ptx_line_mapping(...)
                                                                                     -> parse_sass_line_mapping(...)
                                                                                     <- JSON { ptx, sass, ptx_mapping, sass_mapping }
                                                    <- JSON result
                           <- compileResult(...)    <- webview.postMessage

User clicks source line   -> React uses source_to_asm / asm_to_source maps
to highlight instructions    to highlight PTX/SASS rows in side panel.
```

---

## 5. Implementing These Features Without the Wafer Extension

Wafer’s implementation gives a clear blueprint for how to build NCU and
PTX/SASS tooling yourself. You can replicate the **core capabilities** without
installing the Wafer extension by combining standard CUDA tools with thin
scripts (much like `ncu_tool.py` and `compiler_explorer_tool.py`).

### 5.1 NCU Profiles in VSCode Without Wafer

The key operations that Wafer performs via `ncu_tool.py` are:

1. **Locate `ncu`** (`find_ncu()`).
2. **Run a profile** (`cmd_run`):
   - Build `ncu_cmd = [ncu_path, "-o", output_file, ...command...]`.
   - Execute, producing `.ncu-rep` or CSV.
3. **Parse `.ncu-rep`** (`cmd_parse`):
   - `ncu --import <file> --page session` → GPU info.
   - `ncu --import <file> --page details` → per‑kernel metrics.
   - Extract durations, occupancy, throughput, recommendations into JSON.

You can reproduce this in your own repo by:

- Adding a small Python script modeled on
  [`ncu_tool.py`](wafer/extension/resources/ncu-tool/ncu_tool.py) that exposes
  just `run` and `parse`.
- Wiring it into VSCode **tasks** in `.vscode/tasks.json`:

```jsonc
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "profile:tma_test:ncu",
      "type": "shell",
      "command": "python tools/ncu_tool_min.py run './build/tma_test_exe' -f tma_test -d .wafer/ncu-tool"
    },
    {
      "label": "analyze:tma_test:ncu",
      "type": "shell",
      "command": "python tools/ncu_tool_min.py parse .wafer/ncu-tool/tma_test.ncu-rep -o .wafer/ncu-tool"
    }
  ]
}
```

Then open the generated text/JSON files directly in VSCode as you iterate on
kernels like `experiments/tma_test.cu`.

### 5.2 PTX / SASS Correlation Without Wafer

The important bits from `compiler_explorer_tool.py` are:

1. Compile with line info:

   ```bash
   nvcc -std=c++17 -O3 --generate-line-info \
        -ptx -arch=sm_90 \
        -I/path/to/cutlass/include \
        experiments/tma_test.cu \
        -o build/tma_test.ptx
   ```

2. Disassemble SASS with line info:

   ```bash
   nvcc -std=c++17 -O3 --generate-line-info \
        -cubin -arch=sm_90 \
        experiments/tma_test.cu \
        -o build/tma_test.cubin

   nvdisasm -c -g build/tma_test.cubin > build/tma_test.sass
   ```

3. Parse mappings:

   - For PTX: scan `.loc <file_id> <line> <col>` directives and associate all
     following instructions with that source line.
   - For SASS: scan for `//## File "...", line N` comments in `nvdisasm` output
     and associate subsequent instructions until the next marker.

You can take the `parse_ptx_line_mapping` and `parse_sass_line_mapping`
implementations almost verbatim (or simplified) into your own script
`tools/ptx_sass_map.py`, then output a JSON mapping:

```jsonc
{
  "source_to_asm": { "42": [120, 121, 122] },
  "asm_to_source": { "120": 42, "121": 42, "122": 42 }
}
```

Once you have that mapping, there are two easy ways to integrate with VSCode
without a full extension:

- Generate a **static HTML report** (like a simple version of Wafer’s webview)
  that shows the source on the left and PTX/SASS on the right with basic
  hyperlinks. Open that HTML file in VSCode.
- Use a **notebook** (`.ipynb`) or markdown file where you embed source and
  assembly blocks with line references and rely on VSCode’s outline and search
  to jump between them.

### 5.3 How This Relates to Your Repo

For kernels in this repo such as:

- `experiments/tma_test.cu`
- `experiments/tma_load.cu`
- CUTLASS/CuTe kernels under
  [`3rdparty/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp`](3rdparty/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp)

you can:

- Add `-lineinfo` / `--generate-line-info` to your existing `Makefile`
  recipes in `experiments/Makefile`.
- Use a **minimal** variant of `compiler_explorer_tool.py` to build PTX/SASS
  (or just copy the command lines).
- Use the mapping logic from §4.3.4 to quickly see which PTX/SASS instructions
  correspond to a given line in `tma_test.cu`.

This reproduces the core Wafer experience—NCU profiles and source‑correlated
PTX/SASS—without relying on the extension, while the extension’s own
implementation serves as a concrete reference design for how the pieces fit
together.

Use `ncu` on the command line to profile a specific kernel test executable. For example, if you build a test binary via `experiments/Makefile`, you might have:

```bash
ncu --set full \
    --kernel-name-base demangled \
    --target-processes all \
    --export profile.ncu-rep \
    ./build/tma_test_exe
```

Key points:

- `--export profile.ncu-rep` writes a binary report file.
- You can scope the profile further with `--kernel-name` or `--launch-skip` / `--launch-count` if the binary runs multiple kernels.

### 4.2 Step 2 – Convert `.ncu-rep` into Text/CSV

The `.ncu-rep` file is best viewed using `ncu` itself. From a terminal:

```bash
ncu --import profile.ncu-rep --page summary
ncu --import profile.ncu-rep --page details --csv > profile_details.csv
```

You can then open the text or CSV output directly in VSCode:

- `profile_details.csv` – open as a spreadsheet‑like table using VSCode’s CSV extensions, or just as text.
- `profile.ncu-rep` – keep as the canonical raw profile; convert on demand.

### 4.3 Step 3 – Wire Into VSCode Tasks

To make this convenient, define tasks in `.vscode/tasks.json` that:

1. Build your CUDA test binary.
2. Run `ncu` with the appropriate arguments.
3. Optionally run the `--import` step to produce a `.csv` file next to the binary.

Example `tasks.json` snippet:

```jsonc
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "build:tma_test",
      "type": "shell",
      "command": "make -C experiments tma_test"
    },
    {
      "label": "profile:tma_test",
      "type": "shell",
      "dependsOn": "build:tma_test",
      "command": "cd experiments && ncu --set full --export tma_test.ncu-rep ./tma_test && ncu --import tma_test.ncu-rep --page summary > tma_test_summary.txt"
    }
  ]
}
```

Run `Tasks: Run Task` → `profile:tma_test` from the command palette. VSCode will:

- Build your kernel test.
- Launch NCU.
- Produce `experiments/tma_test_summary.txt`, which you can open in the editor.

This gives you a text‑based profile view in VSCode similar in spirit to Wafer’s integrated UI, though without a custom webview.

---

## 5. PTX / SASS Correlation to Source (No Wafer)

The Wafer extension’s “PTX/SASS correlated with source” feature can be approximated by:

1. Compiling CUDA kernels with **line information**.
2. Dumping PTX and SASS with that line info preserved.
3. Using `.loc` / `#line` directives in the generated assembly to map back to source lines.

### 5.1 Compile with Line Info

For a file like `experiments/tma_test.cu`, compile with `-lineinfo`:

```bash
nvcc -std=c++17 -O3 -g -lineinfo \
     -c experiments/tma_test.cu \
     -o build/tma_test.o
```

Add this flag into your existing build in `experiments/Makefile` so all kernels get line mappings.

### 5.2 Generate PTX

```bash
nvcc -std=c++17 -O3 -g -lineinfo \
     -ptx experiments/tma_test.cu \
     -o build/tma_test.ptx
```

Open `build/tma_test.ptx` in VSCode. You will see sections like:

```ptx
.file   1 "experiments/tma_test.cu"
.loc    1 42 0
        ld.global.f32  %f1, [%rd1];
.loc    1 43 0
        fadd.rn.f32    %f2, %f1, %f3;
```

Interpretation:

- `.file 1` defines the source file index.
- `.loc 1 42 0` means “this instruction corresponds to line 42 in `tma_test.cu`”.

This already lets you correlate PTX instructions to their originating CUDA source lines by eye.

### 5.3 Generate SASS

First build a cubin or full executable, then disassemble:

```bash
nvcc -std=c++17 -O3 -g -lineinfo \
     experiments/tma_test.cu \
     -o build/tma_test_exe

nvdisasm --print-line-info build/tma_test_exe > build/tma_test.sass
```

`nvdisasm` embeds line numbers, for example:

```text
/* 42 */ LDG.E.SYS R4, [R2.64];
/* 43 */ FADD R5, R4, R6;
```

Open `build/tma_test.sass` alongside `experiments/tma_test.cu` in VSCode and you can visually align instructions to source using the comments.

### 5.4 Light Automation: Simple Correlation Script

To get closer to Wafer’s auto‑highlight behavior without writing a full extension, you can:

1. Write a small Python script (e.g. `tools/correlate_ptx.py`) that:
   - Parses `.loc` directives in `.ptx` or `/* line */` comments in `.sass`.
   - Produces a JSON mapping from `{"file": "experiments/tma_test.cu", "line": N} -> [instruction strings]`.
2. Use that JSON in a simple HTML report opened in VSCode.

Example pseudo‑code for the parser:

```python
current_file = None
current_line = None
mapping = {}

for line in ptx_lines:
    if line.startswith(".file"):
        # parse index -> path
    elif line.startswith(".loc"):
        # update current_file, current_line
    elif is_instruction(line):
        key = (current_file, current_line)
        mapping.setdefault(key, []).append(line.strip())
```

You can render this mapping as HTML with `<pre>` blocks organized by source line and open it in VSCode with the built‑in HTML viewer.

### 5.5 VSCode Workflow Without a Custom Extension

You can combine the above into a smooth loop:

1. Add a `tasks.json` task `build+ptx:tma_test` that runs:
   - `nvcc` with `-lineinfo` to produce `.ptx`.
   - `nvdisasm` to produce `.sass`.
   - `python tools/correlate_ptx.py build/tma_test.ptx > build/tma_test_ptx_map.html`.
2. Use `problemMatcher` or `openIn` scripts to automatically open:
   - `experiments/tma_test.cu`
   - `build/tma_test_ptx_map.html`

This is less polished than Wafer’s integrated PTX/SASS explorer but achieves the same technical goal: *viewing assembly with clear line‑level mapping back to CUDA source*.

---

## 6. Summary

- The Wafer VSCode extension appears to be a closed‑source tool that:
  - Wraps Nsight Compute and CUDA disassembly tools.
  - Adds VSCode commands, custom editors, and webviews to show NCU profiles and PTX/SASS correlated with source.
- Even without Wafer, you can replicate the core workflows by:
  - Running `ncu` from VSCode tasks and viewing text/CSV reports in the editor.
  - Compiling with `-lineinfo` and using `nvcc`, `nvdisasm`, and/or `cuobjdump` to generate PTX/SASS annotated with source line numbers.
  - Optionally adding small helper scripts to build source↔assembly maps and HTML views that VSCode can render.

If you’d like, the next step would be to add concrete `tasks.json` and helper scripts in this repo to wire these commands directly into your `experiments/tma_test.cu` workflow.
