/**
 * ComfyUI-GaussianPack - Gaussian Splat Preview Widget
 * Interactive 3D Gaussian Splatting viewer using gsplat.js
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Auto-detect extension folder name (so this copy doesn't collide with upstream comfyui-PlyPreview).
const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI-GaussianPack";
})();

console.log("[GaussianPack] Loading extension...");

// Build a small WASD / arrows / Shift+Space+C keypad that lights up as keys
// are pressed (driven by KEY_STATE messages the viewer iframe forwards). Handy
// for screen-recording navigation. Returns { el, set(code,on), reset() }.
function makeKeyPad() {
    const codes = {};
    const wrap = document.createElement("div");
    Object.assign(wrap.style, {
        display: "flex", gap: "10px", alignItems: "center",
        padding: "4px 10px", flexShrink: "0", background: "#1a1a1a",
        borderTop: "1px solid #444", borderLeft: "1px solid #333",
    });
    const keyEl = (label, code, w = 18) => {
        const k = document.createElement("div");
        k.textContent = label;
        Object.assign(k.style, {
            minWidth: w + "px", height: "18px", lineHeight: "16px", textAlign: "center",
            fontSize: "10px", fontFamily: "monospace", color: "#aaa",
            background: "#2a2a2a", border: "1px solid #444", borderRadius: "3px",
            padding: "0 3px", boxSizing: "border-box", transition: "all 0.04s",
        });
        codes[code] = k;
        return k;
    };
    const grid = (rows) => {
        const g = document.createElement("div");
        Object.assign(g.style, { display: "grid", gridTemplateColumns: "repeat(3, 18px)", gap: "2px" });
        rows.forEach(r => r.forEach(cell => g.appendChild(cell || document.createElement("div"))));
        return g;
    };
    const wasd = grid([
        [null, keyEl("W", "KeyW"), null],
        [keyEl("A", "KeyA"), keyEl("S", "KeyS"), keyEl("D", "KeyD")],
    ]);
    const arrows = grid([
        [null, keyEl("↑", "ArrowUp"), null],
        [keyEl("←", "ArrowLeft"), keyEl("↓", "ArrowDown"), keyEl("→", "ArrowRight")],
    ]);
    const extra = document.createElement("div");
    Object.assign(extra.style, { display: "flex", gap: "2px", alignItems: "center" });
    const shift = keyEl("⇧", "ShiftLeft", 24);
    codes["ShiftRight"] = shift;             // both shifts light the same key
    extra.appendChild(shift);
    extra.appendChild(keyEl("Spc", "Space", 26));
    extra.appendChild(keyEl("C", "KeyC"));
    wrap.append(wasd, arrows, extra);

    const set = (code, on) => {
        const el = codes[code];
        if (!el) return;
        el.style.background = on ? "#00e0ff" : "#2a2a2a";
        el.style.color = on ? "#002b30" : "#aaa";
        el.style.borderColor = on ? "#00e0ff" : "#444";
    };
    const reset = () => Object.keys(codes).forEach(c => set(c, false));
    return { el: wrap, set, reset };
}

app.registerExtension({
    name: "gaussianpack.previewgaussians",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Auto-refresh PLY dropdown list for the file selector node
        if (nodeData.name === "PlyPreviewLoadGaussianPLYEnhance") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const widget = (this.widgets || []).find(w => w.name === "ply_file");

                const refreshList = async () => {
                    try {
                        const resp = await api.fetchApi("/plypreview/files");
                        const json = await resp.json();
                        const files = Array.isArray(json.files) && json.files.length > 0 ? json.files : ["No PLY files found"];
                        if (widget) {
                            // Update dropdown choices and keep current selection if still present
                            widget.options = { ...(widget.options || {}), values: files };
                            if (!files.includes(widget.value)) {
                                widget.value = files[0];
                                widget.callback?.(widget.value);
                            }
                            // Force UI redraw
                            widget.computeSize?.();
                            this.setDirtyCanvas(true);
                            app.graph.setDirtyCanvas(true, true);
                        }
                    } catch (e) {
                        console.warn("[GaussianPack] Failed to refresh PLY list", e);
                    }
                };

                this.refreshPlyList = refreshList;
                refreshList();

                // Hint label: user must right-click "Reload Node" to refresh PLY list
                const hintEl = document.createElement("div");
                hintEl.style.cssText = "font-size:10px;color:#888;text-align:center;padding:4px 8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
                hintEl.textContent = "Right-click -> Refresh Node to load new PLY files";
                this.addDOMWidget("ply_hint", "PLY_HINT", hintEl, {
                    serialize: false,
                    hideOnZoom: false,
                });

                return r;
            };
        }

        if (nodeData.name === "PreviewGaussians") {
            // Spectate is now a camera_mode of PreviewGaussians; the dedicated
            // PreviewGaussianSpectate node was removed. Kept as a const so the
            // downstream "spectate" branches still read cleanly.
            const _isSpectate = false;
            console.log("[GaussianPack] Registering Preview Gaussians node");

            // After ComfyUI applies serialized widgets_values, defend
            // against stale workflow schemas. Two shapes have shipped
            // to users over time:
            //
            //   (a) Pre-renderer-widget era: widgets_values was
            //       [fov, w, h, "<camera-state JSON>"] — only 4
            //       entries, with the camera state glued onto the end.
            //   (b) Post-renderer-widget era: widgets_values is
            //       [fov, w, h, renderer, transport_format, ""].
            //
            // When (a) is loaded against today's 5-widget schema,
            // LiteGraph applies the JSON string to widgets[3] which is
            // now `renderer` (a combo) — the prompt validator refuses
            // because "<JSON>" is not in ["spark","playcanvas"]. The
            // walk below rescues that JSON back to
            // node.properties["Camera Config"] and resets the widget.
            //
            // onConfigure runs AFTER configure() has stamped the saved
            // values onto the widgets, so this is where the migration
            // has to live (onNodeCreated runs before and only sees
            // defaults).
            const _origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                // Build a name -> value map from the saved metadata BEFORE
                // calling configure. ComfyUI's positional widget pairing
                // breaks when the live frontend renders a widget that
                // wasn't a widget at save time (most commonly a
                // `forceInput: True` STRING input — saved as a connection
                // socket only, but rendered today as widget[0]). Saved
                // `inputs[]` carries a `widget: {name}` annotation only on
                // inputs that had widgets, so it's the authoritative
                // ordering for widgets_values.
                const r = _origOnConfigure ? _origOnConfigure.apply(this, arguments) : undefined;

                // ComfyUI's widgetInputs extension strips one entry from
                // info.widgets_values for `forceInput: True` inputs like
                // ply_path. The strip silently corrupts the
                // value->widget pairing (fov ends up with image_width's
                // value, etc.). The disk file is correct; the in-memory
                // info we receive is not.
                //
                // To survive that, we mirror every named widget into
                // node.properties["Widget Values"] (same pattern as
                // "Camera Config") on every change. On load we trust
                // properties over info.widgets_values when available.
                //
                // For old workflows without "Widget Values", we also fall
                // back to "Camera Config".fov which captures the fov the
                // user was actually using.
                this.properties = this.properties || {};
                const saved = this.properties["Widget Values"] || {};
                const camCfg = this.properties["Camera Config"] || {};
                if (typeof camCfg.fov === "number" && saved.fov_degrees === undefined) {
                    saved.fov_degrees = camCfg.fov;
                }
                for (const w of (this.widgets || [])) {
                    if (w && w.name && w.name in saved) {
                        const v = saved[w.name];
                        if (v !== undefined) w.value = v;
                    }
                }

                for (const w of (this.widgets || [])) {
                    const optsValues = w.options && Array.isArray(w.options.values);
                    const isCombo    = w.type === "combo" || optsValues;

                    // (1) Stranded camera-state JSON in any widget value.
                    if (typeof w.value === "string"
                        && w.value.length > 20
                        && w.value[0] === "{"
                        && w.value.includes('"pos"')) {
                        try {
                            const parsed = JSON.parse(w.value);
                            if (Array.isArray(parsed.pos) && Array.isArray(parsed.target)) {
                                this.properties = this.properties || {};
                                const cfg = this.properties["Camera Config"] || {};
                                cfg.cameraType = cfg.cameraType || "perspective";
                                cfg.fov   = (typeof parsed.fov === "number") ? parsed.fov : (cfg.fov ?? 50);
                                cfg.state = cfg.state || w.value;
                                this.properties["Camera Config"] = cfg;
                                console.warn("[GaussianPack] rescued stranded camera state from widget",
                                             w.name, "-> properties['Camera Config']");
                                w.value = w.options?.default
                                          ?? (optsValues ? w.options.values[0] : "");
                                continue;
                            }
                        } catch (_) { /* not JSON; fall through */ }
                    }

                    // (2) Combo widget value isn't a valid option.
                    if (isCombo && optsValues && !w.options.values.includes(w.value)) {
                        const def = (w.options.default !== undefined)
                                    ? w.options.default
                                    : w.options.values[0];
                        console.warn("[GaussianPack] resetting combo widget",
                                     w.name, "(value", JSON.stringify(w.value), "not in options) ->", def);
                        w.value = def;
                        continue;
                    }

                    // (3) Numeric widgets given a string (legacy shift).
                    if ((w.type === "number" || w.type === "INT" || w.type === "FLOAT")
                        && typeof w.value === "string") {
                        const def = w.options?.default;
                        console.warn("[GaussianPack] resetting corrupted widget",
                                     w.name, "(was string) ->", def);
                        w.value = (def !== undefined) ? def : 0;
                    }

                    // (4) Out-of-range numerics (e.g. fov_degrees=512
                    //     from a saved-then-shifted workflow).
                    if (typeof w.value === "number" && w.options) {
                        const { min, max, default: def } = w.options;
                        if ((typeof min === "number" && w.value < min) ||
                            (typeof max === "number" && w.value > max)) {
                            console.warn("[GaussianPack] resetting out-of-range widget",
                                         w.name, w.value, "->", def);
                            w.value = (def !== undefined) ? def : (min ?? 0);
                        }
                    }
                }
                return r;
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Mirror every named widget into node.properties["Widget
                // Values"] so we survive ComfyUI's widget_values stripping
                // (see onConfigure comment above). Apply current values
                // immediately + wrap each widget's callback to keep the
                // mirror up to date.
                const TRACKED = new Set([
                    "fov_degrees", "camera_mode",
                    "renderer", "transport_format",
                ]);
                this.properties = this.properties || {};
                if (!this.properties["Widget Values"]) {
                    this.properties["Widget Values"] = {};
                }
                const propVals = this.properties["Widget Values"];
                for (const w of (this.widgets || [])) {
                    if (!w || !TRACKED.has(w.name)) continue;
                    if (propVals[w.name] === undefined) propVals[w.name] = w.value;
                    const origCb = w.callback;
                    w.callback = function(v) {
                        propVals[w.name] = v;
                        return origCb ? origCb.apply(this, arguments) : undefined;
                    };
                }

                // Create container for viewer + info panel
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.backgroundColor = "#1a1a1a";
                container.style.overflow = "hidden";
                container.style.position = "relative";   // anchor for loading bar overlay

                // Loading-progress overlay (download + parse). Sits over the
                // top edge of the iframe so the user sees activity during the
                // long PLY fetch + parse window.
                const loadBar = document.createElement("div");
                loadBar.style.cssText =
                    "position:absolute;top:0;left:0;right:0;height:4px;" +
                    "background:rgba(0,0,0,0.35);z-index:100;" +
                    "transition:opacity 0.3s ease;pointer-events:none;opacity:0;";
                const loadFill = document.createElement("div");
                loadFill.style.cssText =
                    "height:100%;width:0%;background:linear-gradient(90deg,#3a8,#6c6);" +
                    "transition:width 0.15s linear;";
                loadBar.appendChild(loadFill);
                const loadLabel = document.createElement("div");
                loadLabel.style.cssText =
                    "position:absolute;top:6px;left:8px;" +
                    "font:10px monospace;color:#cfc;" +
                    "text-shadow:0 0 4px rgba(0,0,0,0.8);pointer-events:none;";
                loadBar.appendChild(loadLabel);

                function showLoadBar(label) {
                    loadFill.style.width = "0%";
                    loadLabel.textContent = label || "";
                    loadBar.style.opacity = "1";
                }
                function setLoadProgress(pct, label) {
                    const p = Math.max(0, Math.min(100, pct || 0));
                    loadFill.style.width = p.toFixed(1) + "%";
                    if (label != null) loadLabel.textContent = label;
                }
                function hideLoadBar() {
                    loadBar.style.opacity = "0";
                }
                this._showLoadBar = showLoadBar;
                this._setLoadProgress = setLoadProgress;
                this._hideLoadBar = hideLoadBar;

                // Create iframe for gsplat.js viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1 1 0";
                iframe.style.minHeight = "0";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#1a1a1a";
                // Allow the in-iframe Fullscreen button to fill the whole screen.
                iframe.allowFullscreen = true;
                iframe.setAttribute("allow", "fullscreen");

                {
                    const cmW = (this.widgets || []).find(w => w.name === "camera_mode");
                    const wantSpectate = _isSpectate || (cmW && cmW.value === "spectate");
                    const _qs = wantSpectate ? "mode=spectate&" : "";
                    iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_gaussian.html?${_qs}v=` + Date.now();
                }

                // Create info panel
                const infoPanel = document.createElement("div");
                infoPanel.style.backgroundColor = "#1a1a1a";
                infoPanel.style.borderTop = "1px solid #444";
                infoPanel.style.padding = "6px 12px";
                infoPanel.style.fontSize = "10px";
                infoPanel.style.fontFamily = "monospace";
                infoPanel.style.color = "#ccc";
                infoPanel.style.lineHeight = "1.3";
                infoPanel.style.flexShrink = "0";
                infoPanel.style.overflow = "hidden";
                infoPanel.innerHTML = '<span style="color: #888;">Gaussian splat info will appear here after execution</span>';

                // Footer = info panel (left, grows) + key-press indicator (right).
                const keyPad = makeKeyPad();
                infoPanel.style.flex = "1 1 auto";
                const footer = document.createElement("div");
                Object.assign(footer.style, { display: "flex", alignItems: "stretch", flexShrink: "0" });
                footer.appendChild(infoPanel);
                footer.appendChild(keyPad.el);
                this.gaussianKeyPad = keyPad;

                // Add iframe + footer + loading-bar overlay to container.
                // loadBar last so its overlay sits above the iframe.
                container.appendChild(iframe);
                container.appendChild(footer);
                container.appendChild(loadBar);

                // --- Persistent camera-state plumbing ---------------------
                // Mirrors ComfyUI's built-in Load3D widget: state lives on
                // node.properties["Camera Config"] (named dict), NOT inside
                // widgets_values (positional). Putting the JSON string into
                // widgets_values shifts the standard fov/width/height widgets
                // by one slot on reload and breaks prompt validation.
                let pendingRestoreJSON = "";
                const sendRestore = () => {
                    if (!pendingRestoreJSON || !iframe.contentWindow) return;
                    try {
                        const state = JSON.parse(pendingRestoreJSON);
                        iframe.contentWindow.postMessage(
                            { type: "RESTORE_CAMERA_STATE", state }, "*",
                        );
                    } catch (e) {
                        console.warn("[GaussianPack] bad saved camera state:", e);
                    }
                };

                // Seed pendingRestoreJSON from node.properties (Load3D pattern)
                // if the workflow JSON had a saved pose.
                const savedCfg = this.properties && this.properties["Camera Config"];
                if (savedCfg && typeof savedCfg.state === "string") {
                    pendingRestoreJSON = savedCfg.state;
                }

                // Iframe-display widget. serialize: false -> never lands in
                // widgets_values; we own persistence via node.properties.
                const widget = this.addDOMWidget(
                    "preview_gaussian", "GAUSSIAN_PREVIEW", container,
                    { serialize: false },
                );

                // Store reference to node for dynamic resizing
                const node = this;
                let currentNodeSize = [512, 580];

                widget.computeSize = () => currentNodeSize;

                // Store references
                this.gaussianViewerIframe = iframe;
                this.gaussianInfoPanel = infoPanel;

                // Function to resize node dynamically
                this.resizeToAspectRatio = function(imageWidth, imageHeight) {
                    const aspectRatio = imageWidth / imageHeight;
                    const nodeWidth = 512;
                    const viewerHeight = Math.round(nodeWidth / aspectRatio);
                    const nodeHeight = viewerHeight + 60;  // Add space for info panel

                    currentNodeSize = [nodeWidth, nodeHeight];
                    node.setSize(currentNodeSize);
                    node.setDirtyCanvas(true, true);
                    app.graph.setDirtyCanvas(true, true);

                    console.log("[GaussianPack] Resized node to:", nodeWidth, "x", nodeHeight, "(aspect ratio:", aspectRatio.toFixed(2), ")");
                };

                // Track iframe load state
                let iframeLoaded = false;
                iframe.addEventListener('load', () => {
                    iframeLoaded = true;
                    // Push any pending saved camera state now that the iframe is alive
                    // (e.g. on workflow reload before the node has been re-queued).
                    sendRestore();
                });

                // camera_mode change → reload iframe with correct mode
                if (!_isSpectate) {
                    const _cmW = (this.widgets || []).find(w => w.name === "camera_mode");
                    if (_cmW) {
                        const _origCmCb = _cmW.callback;
                        _cmW.callback = function(v) {
                            propVals["camera_mode"] = v;
                            const mqs = v === "spectate" ? "mode=spectate&" : "";
                            iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_gaussian.html?${mqs}v=` + Date.now();
                            iframeLoaded = false;
                            return _origCmCb ? _origCmCb.apply(this, arguments) : undefined;
                        };
                    }
                }

                // Listen for messages from iframe
                window.addEventListener('message', async (event) => {
                    // Only react to messages from OUR iframe — otherwise a
                    // second PreviewGaussians node's iframe would stomp this
                    // node's state.
                    if (event.source !== iframe.contentWindow) return;

                    // Key-press indicator: light up the footer keypad.
                    if (event.data?.type === "KEY_STATE") {
                        this.gaussianKeyPad?.set(event.data.code, !!event.data.down);
                        return;
                    }

                    // Camera-state pushes from iframe -> persist on
                    // node.properties["Camera Config"] (Load3D pattern).
                    if (event.data?.type === "CAMERA_STATE" && event.data.state) {
                        const s = event.data.state;
                        node.properties = node.properties || {};
                        const cfg = node.properties["Camera Config"] || {};
                        cfg.cameraType = cfg.cameraType || "perspective";
                        cfg.fov = (typeof s.fov === "number") ? s.fov : (cfg.fov ?? 50);
                        cfg.state = JSON.stringify(s);
                        node.properties["Camera Config"] = cfg;
                        return;
                    }
                    // Iframe just finished a PLY load -> if we have a saved
                    // pose, replay it. The iframe's loadPLYFromData already
                    // frames on bounds first, so this restore wins.
                    if (event.data?.type === "MESH_LOADED") {
                        node._hideLoadBar?.();
                        sendRestore();
                    }
                    if (event.data?.type === "MESH_ERROR") {
                        // Surface a red bar so failure is visible
                        if (loadBar && loadFill) {
                            loadFill.style.background = "#c44";
                            loadFill.style.width = "100%";
                            loadLabel.textContent = "Error: " + (event.data.error || "load failed");
                        }
                    }
                    // Handle screenshot messages
                    if (event.data.type === 'SCREENSHOT' && event.data.image) {
                        try {
                            // Convert base64 data URL to blob
                            const base64Data = event.data.image.split(',')[1];
                            const byteString = atob(base64Data);
                            const arrayBuffer = new ArrayBuffer(byteString.length);
                            const uint8Array = new Uint8Array(arrayBuffer);

                            for (let i = 0; i < byteString.length; i++) {
                                uint8Array[i] = byteString.charCodeAt(i);
                            }

                            const blob = new Blob([uint8Array], { type: 'image/png' });

                            // Generate filename with timestamp
                            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                            const filename = `gaussian-screenshot-${timestamp}.png`;

                            // Create FormData for upload
                            const formData = new FormData();
                            formData.append('image', blob, filename);
                            formData.append('type', 'output');
                            formData.append('subfolder', '');

                            // Upload to ComfyUI backend
                            const response = await fetch('/upload/image', {
                                method: 'POST',
                                body: formData
                            });

                            if (response.ok) {
                                const result = await response.json();
                                console.log('[GaussianPack] Screenshot saved:', result.name);
                            } else {
                                throw new Error(`Upload failed: ${response.status}`);
                            }

                        } catch (error) {
                            console.error('[GaussianPack] Error saving screenshot:', error);
                        }
                    }
                    // Handle error messages from iframe
                    else if (event.data.type === 'MESH_ERROR' && event.data.error) {
                        console.error('[GaussianPack] Error from viewer:', event.data.error);
                        if (infoPanel) {
                            infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error: ${event.data.error}</div>`;
                        }
                    }
                });

                // Set initial node size
                this.setSize([512, 580]);

                // Download cache — skip re-fetch when same file is re-queued
                let _cachedSplat = null; // { key, buffer }

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("[GaussianPack] onExecuted called with:", message);
                    onExecuted?.apply(this, arguments);

                    // Check for errors
                    if (message?.error && message.error[0]) {
                        infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error: ${message.error[0]}</div>`;
                        return;
                    }

                    // The message IS the UI data (not message.ui)
                    if (message?.ply_file && message.ply_file[0]) {
                        const filename = message.ply_file[0];
                        const displayName = message.filename?.[0] || filename;
                        const fileSizeMb = message.file_size_mb?.[0] || 'N/A';
                        const numGaussians = message.num_gaussians?.[0] || 0;

                        // Extract camera parameters if provided
                        const extrinsics = message.extrinsics?.[0] || null;
                        const intrinsics = message.intrinsics?.[0] || null;
                        const renderer = message.renderer?.[0] || "spark";
                        const transportFormat = message.transport_format?.[0] || "ply";
                        // Which Comfy folder the PLY lives under — "input" or
                        // "output". Set by the Python preview node so /view can
                        // resolve files dropped via LoadPLY (under input/) as
                        // well as artifacts written by GaussianMerge / Export
                        // / LiToExportPLY (under output/).
                        const plyType = message.ply_type?.[0] || "output";
                        const plySubfolder = message.ply_subfolder?.[0] || "";

                        // Resize node to match image aspect ratio from intrinsics.
                        // When width/height are 0 (auto), keep the current node
                        // size and let the viewer fill whatever space is available.
                        if (intrinsics && intrinsics[0] && intrinsics[1]) {
                            const imageWidth = intrinsics[0][2] * 2;   // cx * 2
                            const imageHeight = intrinsics[1][2] * 2;  // cy * 2
                            if (imageWidth > 0 && imageHeight > 0) {
                                this.resizeToAspectRatio(imageWidth, imageHeight);
                            }
                        }

                        // Reload iframe if camera mode changed
                        const _cameraMode = message.camera_mode?.[0] || (_isSpectate ? "spectate" : "orbit");
                        const iframeHasSpectate = iframe.src && iframe.src.includes("mode=spectate");
                        const wantSpectate = _cameraMode === "spectate";
                        if (wantSpectate !== iframeHasSpectate) {
                            const mqs = wantSpectate ? "mode=spectate&" : "";
                            iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_gaussian.html?${mqs}v=` + Date.now();
                            iframeLoaded = false;
                        }

                        // Update info panel
                        const gaussianRow = numGaussians > 0
                            ? `<span style="color: #888;">Gaussians:</span>
                               <span>${numGaussians.toLocaleString()}</span>`
                            : '';
                        infoPanel.innerHTML = `
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 2px 8px;">
                                <span style="color: #888;">File:</span>
                                <span style="color: #6cc;">${displayName}</span>
                                <span style="color: #888;">Size:</span>
                                <span>${fileSizeMb} MB</span>
                                ${gaussianRow}
                            </div>
                        `;

                        // Wire URL + iframe filename based on transport_format.
                        //   "ply" -> ComfyUI's /view streams the raw PLY.
                        //   "spz" -> /gaussianpack/spz lazy-transcodes
                        //           via vendored spz-js (~9× smaller). On
                        //           first call the server pays the transcode
                        //           cost; subsequent calls hit the cached
                        //           sibling .spz file.
                        // The iframe filename gets the matching extension so
                        // the renderer's auto-detect (Spark sniffs bytes
                        // either way, but extension keeps the info panel
                        // honest).
                        const qsType = encodeURIComponent(plyType);
                        const qsSub  = encodeURIComponent(plySubfolder);
                        // Freshness token: changes whenever the upstream file is
                        // rewritten under the same name, so a re-merged PLY isn't
                        // served stale from the in-memory or browser HTTP cache.
                        const tok = message.mtime?.[0] ?? 0;
                        let filepath, iframeFilename, loadLabel;
                        if (transportFormat === "spz") {
                            filepath       = `/gaussianpack/spz?filename=${encodeURIComponent(filename)}&type=${qsType}&subfolder=${qsSub}&_=${tok}`;
                            iframeFilename = filename.replace(/\.ply$/i, ".spz");
                            loadLabel      = "Transcoding + downloading SPZ...";
                        } else {
                            filepath       = `/view?filename=${encodeURIComponent(filename)}&type=${qsType}&subfolder=${qsSub}&_=${tok}`;
                            iframeFilename = filename;
                            loadLabel      = "Downloading PLY...";
                        }

                        const cacheKey = `${filename}|${plyType}|${plySubfolder}|${transportFormat}|${tok}`;

                        const sendToIframe = (arrayBuffer) => {
                            if (!iframe.contentWindow) return;
                            node._setLoadProgress?.(100, "Parsing splats...");
                            iframe.contentWindow.postMessage({
                                type: "LOAD_MESH_DATA",
                                data: arrayBuffer,
                                filename: iframeFilename,
                                extrinsics, intrinsics, renderer,
                                timestamp: Date.now()
                            }, "*", [arrayBuffer]);
                        };

                        const fetchAndSend = async () => {
                            if (!iframe.contentWindow) {
                                console.error("[GaussianPack] Iframe contentWindow not available");
                                return;
                            }

                            // Use cached data if same file
                            if (_cachedSplat && _cachedSplat.key === cacheKey) {
                                console.log("[GaussianPack] using cached splat data for", filename);
                                sendToIframe(_cachedSplat.buffer.slice(0));
                                return;
                            }

                            try {
                                const t0 = performance.now();
                                console.log("[GaussianPack] fetch start:", filepath);
                                node._showLoadBar?.(loadLabel);
                                const response = await fetch(filepath);
                                if (!response.ok) {
                                    let body = "";
                                    try { body = (await response.text() || "").slice(0, 400); } catch (_) {}
                                    throw new Error(`HTTP ${response.status}: ${response.statusText}${body ? " - " + body : ""}`);
                                }
                                const totalHdr = response.headers.get("content-length");
                                const total = totalHdr ? parseInt(totalHdr, 10) : 0;

                                const chunks = [];
                                let received = 0;
                                let lastPctLogged = -1;
                                const reader = response.body?.getReader();
                                if (reader) {
                                    for (;;) {
                                        const { done, value } = await reader.read();
                                        if (done) break;
                                        chunks.push(value);
                                        received += value.byteLength;
                                        if (total > 0) {
                                            const pct = (received / total) * 100;
                                            const mb = (received / (1024*1024)).toFixed(1);
                                            const tmb = (total / (1024*1024)).toFixed(1);
                                            node._setLoadProgress?.(pct, `Downloading: ${mb} / ${tmb} MB`);
                                            const ipct = Math.floor(pct);
                                            if (ipct !== lastPctLogged && (ipct % 5 === 0 || ipct === 100)) {
                                                console.log(`[GaussianPack] download ${ipct}%  ${mb}/${tmb} MB`);
                                                lastPctLogged = ipct;
                                            }
                                        } else {
                                            node._setLoadProgress?.(50, `Downloading: ${(received/(1024*1024)).toFixed(1)} MB`);
                                        }
                                    }
                                }
                                const u8 = new Uint8Array(received);
                                let off = 0;
                                for (const c of chunks) { u8.set(c, off); off += c.byteLength; }
                                const arrayBuffer = u8.buffer;
                                const dt = performance.now() - t0;
                                const speed = received > 0 ? (received / (1024*1024)) / (dt / 1000) : 0;
                                console.log(`[GaussianPack] fetch done: ${arrayBuffer.byteLength} bytes in ${dt.toFixed(0)} ms (${speed.toFixed(1)} MB/s)`);

                                // Cache a copy, send the original (transferred)
                                _cachedSplat = { key: cacheKey, buffer: arrayBuffer.slice(0) };
                                sendToIframe(arrayBuffer);
                            } catch (error) {
                                console.error("[GaussianPack] Error fetching splat:", error);
                                infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error loading splat: ${error.message}</div>`;
                            }
                        };

                        if (iframeLoaded) {
                            fetchAndSend();
                        } else {
                            setTimeout(fetchAndSend, 500);
                        }
                    }
                };

                return r;
            };
        }

        // ==============================================================
        // Preview Gaussian Dual — side-by-side / slider comparison
        // ==============================================================
        if (nodeData.name === "PreviewGaussianDual") {
            console.log("[GaussianPack] Registering Preview Gaussian Dual node");

            const _origOnConfigureDual = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                const r = _origOnConfigureDual ? _origOnConfigureDual.apply(this, arguments) : undefined;
                this.properties = this.properties || {};
                const saved = this.properties["Widget Values"] || {};
                for (const w of (this.widgets || [])) {
                    if (w && w.name && w.name in saved) {
                        const v = saved[w.name];
                        if (v !== undefined) w.value = v;
                    }
                }
                for (const w of (this.widgets || [])) {
                    const optsValues = w.options && Array.isArray(w.options.values);
                    const isCombo = w.type === "combo" || optsValues;
                    if (isCombo && optsValues && !w.options.values.includes(w.value)) {
                        w.value = w.options.default ?? w.options.values[0];
                    }
                }
                return r;
            };

            const onNodeCreatedDual = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreatedDual ? onNodeCreatedDual.apply(this, arguments) : undefined;
                const node = this;

                // Widget persistence
                const TRACKED_DUAL = new Set([
                    "layout", "fov_degrees", "camera_mode",
                    "renderer", "transport_format",
                ]);
                this.properties = this.properties || {};
                if (!this.properties["Widget Values"]) this.properties["Widget Values"] = {};
                const propValsDual = this.properties["Widget Values"];
                for (const w of (this.widgets || [])) {
                    if (!w || !TRACKED_DUAL.has(w.name)) continue;
                    if (propValsDual[w.name] === undefined) propValsDual[w.name] = w.value;
                    const origCb = w.callback;
                    w.callback = function(v) {
                        propValsDual[w.name] = v;
                        return origCb ? origCb.apply(this, arguments) : undefined;
                    };
                }

                const container = document.createElement("div");
                container.style.cssText = "width:100%;height:100%;display:flex;flex-direction:column;background:#1a1a1a;overflow:hidden;position:relative;";

                // Two independent loading bars — green for splat 1, blue for splat 2
                function _makeLoadBar(topOffset, gradientColors, textColor) {
                    const bar = document.createElement("div");
                    bar.style.cssText = `position:absolute;top:${topOffset}px;left:0;right:0;height:4px;background:rgba(0,0,0,0.35);z-index:100;transition:opacity 0.3s;pointer-events:none;opacity:0;`;
                    const fill = document.createElement("div");
                    fill.style.cssText = `height:100%;width:0%;background:linear-gradient(90deg,${gradientColors});transition:width 0.15s linear;`;
                    bar.appendChild(fill);
                    const lbl = document.createElement("div");
                    lbl.style.cssText = `position:absolute;top:6px;left:8px;font:10px monospace;color:${textColor};text-shadow:0 0 4px rgba(0,0,0,0.8);pointer-events:none;`;
                    bar.appendChild(lbl);
                    return { bar, fill, lbl };
                }
                const lb1 = _makeLoadBar(0, "#3a8,#6c6", "#cfc");
                const lb2 = _makeLoadBar(6, "#38a,#6ac", "#cef");

                function showLoadBar(lb, label) { lb.fill.style.width = "0%"; lb.lbl.textContent = label || ""; lb.bar.style.opacity = "1"; }
                function setLoadProgress(lb, pct, label) { lb.fill.style.width = Math.max(0,Math.min(100,pct||0)).toFixed(1)+"%"; if (label!=null) lb.lbl.textContent = label; }
                function hideLoadBar(lb) { lb.bar.style.opacity = "0"; }

                const iframe = document.createElement("iframe");
                iframe.style.cssText = "width:100%;flex:1 1 0;min-height:0;border:none;background:#1a1a1a;";
                iframe.allowFullscreen = true;
                iframe.setAttribute("allow", "fullscreen");

                const infoPanel = document.createElement("div");
                infoPanel.style.cssText = "background:#1a1a1a;border-top:1px solid #444;padding:6px 12px;font:10px monospace;color:#ccc;line-height:1.3;flex-shrink:0;overflow:hidden;";
                infoPanel.innerHTML = '<span style="color:#888;">Dual view info will appear after execution</span>';

                // Footer = info panel (left) + key-press indicator (right).
                const keyPad = makeKeyPad();
                infoPanel.style.flex = "1 1 auto";
                const footer = document.createElement("div");
                Object.assign(footer.style, { display: "flex", alignItems: "stretch", flexShrink: "0" });
                footer.appendChild(infoPanel);
                footer.appendChild(keyPad.el);
                this.gaussianKeyPad = keyPad;

                container.appendChild(iframe);
                container.appendChild(footer);
                container.appendChild(lb1.bar);
                container.appendChild(lb2.bar);

                const widget = this.addDOMWidget("preview_gaussian_dual", "GAUSSIAN_DUAL_PREVIEW", container, { serialize: false });
                let currentNodeSize = [768, 500];
                widget.computeSize = () => currentNodeSize;

                this.gaussianDualIframe = iframe;
                this.gaussianDualInfoPanel = infoPanel;
                this.setSize([768, 500]);

                let iframeLoaded = false;
                iframe.addEventListener('load', () => { iframeLoaded = true; });

                // Download cache for both splats
                let _cachedDual = { key1: null, buf1: null, key2: null, buf2: null };

                // Camera state persistence
                this.properties = this.properties || {};
                window.addEventListener('message', (event) => {
                    if (event.source !== iframe.contentWindow) return;
                    if (event.data?.type === "KEY_STATE") {
                        this.gaussianKeyPad?.set(event.data.code, !!event.data.down);
                        return;
                    }
                    if (event.data?.type === "CAMERA_STATE" && event.data.state) {
                        node.properties["Camera Config"] = {
                            cameraType: "perspective",
                            state: JSON.stringify(event.data.state),
                        };
                    }
                    if (event.data?.type === "MESH_LOADED") {
                        hideLoadBar(lb1); hideLoadBar(lb2);
                        const saved = node.properties["Camera Config"];
                        if (saved?.state && iframe.contentWindow) {
                            try {
                                iframe.contentWindow.postMessage({
                                    type: "RESTORE_CAMERA_STATE",
                                    state: JSON.parse(saved.state),
                                }, "*");
                            } catch(_) {}
                        }
                    }
                    if (event.data?.type === "MESH_ERROR") {
                        lb1.fill.style.background = "#c44"; lb1.fill.style.width = "100%";
                        lb2.fill.style.background = "#c44"; lb2.fill.style.width = "100%";
                    }
                });

                const onExecutedDual = this.onExecuted;
                this.onExecuted = function(message) {
                    onExecutedDual?.apply(this, arguments);
                    if (message?.error?.[0]) {
                        infoPanel.innerHTML = `<div style="color:#ff6b6b;">Error: ${message.error[0]}</div>`;
                        return;
                    }
                    if (!message?.ply_file_1?.[0] || !message?.ply_file_2?.[0]) return;

                    const layout = message.layout?.[0] || "side_by_side";
                    const cameraMode = message.camera_mode?.[0] || "orbit";
                    const renderer = message.renderer?.[0] || "playcanvas";
                    const transport = message.transport_format?.[0] || "ply";
                    const fov = message.fov_degrees?.[0] || 50;

                    // Set iframe src with layout + mode params
                    const modeQs = cameraMode === "spectate" ? "&mode=spectate" : "";
                    iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_gaussian_dual.html?layout=${layout}${modeQs}&v=${Date.now()}`;
                    iframeLoaded = false;

                    // Resize node for dual view
                    currentNodeSize = layout === "side_by_side" ? [768, 500] : [512, 580];
                    node.setSize(currentNodeSize);
                    node.setDirtyCanvas(true, true);

                    // Freshness tokens: change whenever the upstream file is
                    // rewritten under the same name, so a re-merged PLY isn't
                    // served stale from the in-memory or browser HTTP cache.
                    const tok1 = message.mtime_1?.[0] ?? 0;
                    const tok2 = message.mtime_2?.[0] ?? 0;

                    // Build URLs for both PLYs
                    function buildUrl(filename, type, subfolder, tok) {
                        const qsType = encodeURIComponent(type);
                        const qsSub = encodeURIComponent(subfolder);
                        if (transport === "spz") {
                            return `/gaussianpack/spz?filename=${encodeURIComponent(filename)}&type=${qsType}&subfolder=${qsSub}&_=${tok}`;
                        }
                        return `/view?filename=${encodeURIComponent(filename)}&type=${qsType}&subfolder=${qsSub}&_=${tok}`;
                    }

                    const url1 = buildUrl(message.ply_file_1[0], message.ply_type_1?.[0] || "output", message.ply_subfolder_1?.[0] || "", tok1);
                    const url2 = buildUrl(message.ply_file_2[0], message.ply_type_2?.[0] || "output", message.ply_subfolder_2?.[0] || "", tok2);
                    const fn1 = transport === "spz" ? message.ply_file_1[0].replace(/\.ply$/i, ".spz") : message.ply_file_1[0];
                    const fn2 = transport === "spz" ? message.ply_file_2[0].replace(/\.ply$/i, ".spz") : message.ply_file_2[0];

                    // Info panel
                    const n1 = message.num_gaussians_1?.[0] || 0;
                    const n2 = message.num_gaussians_2?.[0] || 0;
                    const s1 = message.file_size_mb_1?.[0] || '?';
                    const s2 = message.file_size_mb_2?.[0] || '?';
                    infoPanel.innerHTML = `
                        <div style="display:flex;gap:16px;">
                            <div><span style="color:#888;">G1:</span> <span style="color:#6cc;">${message.ply_file_1[0]}</span> (${s1} MB, ${n1.toLocaleString()} splats)</div>
                            <div><span style="color:#888;">G2:</span> <span style="color:#6cc;">${message.ply_file_2[0]}</span> (${s2} MB, ${n2.toLocaleString()} splats)</div>
                        </div>`;

                    // Streaming fetch with per-side progress bars
                    async function streamFetch(url, lb, label) {
                        const response = await fetch(url);
                        if (!response.ok) throw new Error(`HTTP ${response.status} for ${label}`);
                        const totalHdr = response.headers.get("content-length");
                        const total = totalHdr ? parseInt(totalHdr, 10) : 0;
                        const chunks = [];
                        let received = 0;
                        const reader = response.body?.getReader();
                        if (reader) {
                            for (;;) {
                                const { done, value } = await reader.read();
                                if (done) break;
                                chunks.push(value);
                                received += value.byteLength;
                                if (total > 0) {
                                    const pct = (received / total) * 100;
                                    const mb = (received / (1024*1024)).toFixed(1);
                                    const tmb = (total / (1024*1024)).toFixed(1);
                                    setLoadProgress(lb, pct, `${label}: ${mb} / ${tmb} MB`);
                                } else {
                                    setLoadProgress(lb, 50, `${label}: ${(received/(1024*1024)).toFixed(1)} MB`);
                                }
                            }
                        }
                        setLoadProgress(lb, 100, `${label}: done`);
                        const u8 = new Uint8Array(received);
                        let off = 0;
                        for (const c of chunks) { u8.set(c, off); off += c.byteLength; }
                        return u8.buffer;
                    }

                    const key1 = `${message.ply_file_1[0]}|${message.ply_type_1?.[0]||"output"}|${message.ply_subfolder_1?.[0]||""}|${transport}|${tok1}`;
                    const key2 = `${message.ply_file_2[0]}|${message.ply_type_2?.[0]||"output"}|${message.ply_subfolder_2?.[0]||""}|${transport}|${tok2}`;

                    const fetchAndSendDual = async () => {
                        if (!iframe.contentWindow) { setTimeout(fetchAndSendDual, 300); return; }
                        try {
                            let ab1, ab2;

                            if (_cachedDual.key1 === key1 && _cachedDual.buf1) {
                                console.log("[GaussianPack] using cached G1");
                                ab1 = _cachedDual.buf1.slice(0);
                            } else {
                                showLoadBar(lb1, "Downloading G1...");
                                ab1 = await streamFetch(url1, lb1, "G1");
                                _cachedDual.key1 = key1;
                                _cachedDual.buf1 = ab1.slice(0);
                            }

                            if (_cachedDual.key2 === key2 && _cachedDual.buf2) {
                                console.log("[GaussianPack] using cached G2");
                                ab2 = _cachedDual.buf2.slice(0);
                            } else {
                                showLoadBar(lb2, "Downloading G2...");
                                ab2 = await streamFetch(url2, lb2, "G2");
                                _cachedDual.key2 = key2;
                                _cachedDual.buf2 = ab2.slice(0);
                            }

                            iframe.contentWindow.postMessage({
                                type: "LOAD_DUAL_GAUSSIAN",
                                data1: ab1, data2: ab2,
                                filename1: fn1, filename2: fn2,
                                renderer: renderer,
                                fov: fov,
                                timestamp: Date.now(),
                            }, "*", [ab1, ab2]);
                        } catch (err) {
                            infoPanel.innerHTML = `<div style="color:#ff6b6b;">Error: ${err.message}</div>`;
                            lb1.fill.style.background = "#c44"; lb1.fill.style.width = "100%";
                            lb2.fill.style.background = "#c44"; lb2.fill.style.width = "100%";
                        }
                    };

                    iframe.addEventListener('load', () => { iframeLoaded = true; fetchAndSendDual(); }, { once: true });
                };

                return r;
            };
        }
    }
});
