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
                    "fov_degrees", "image_width", "image_height",
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

                // Point to gsplat.js HTML viewer (with cache buster)
                iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_gaussian.html?v=` + Date.now();

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

                // Add iframe + info panel + loading-bar overlay to container.
                // loadBar last so its overlay sits above the iframe.
                container.appendChild(iframe);
                container.appendChild(infoPanel);
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

                // Listen for messages from iframe
                window.addEventListener('message', async (event) => {
                    // Only react to messages from OUR iframe — otherwise a
                    // second PreviewGaussians node's iframe would stomp this
                    // node's state.
                    if (event.source !== iframe.contentWindow) return;

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

                        // Resize node to match image aspect ratio from intrinsics
                        if (intrinsics && intrinsics[0] && intrinsics[1]) {
                            const imageWidth = intrinsics[0][2] * 2;   // cx * 2
                            const imageHeight = intrinsics[1][2] * 2;  // cy * 2
                            this.resizeToAspectRatio(imageWidth, imageHeight);
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
                        let filepath, iframeFilename, loadLabel;
                        if (transportFormat === "spz") {
                            filepath       = `/gaussianpack/spz?filename=${encodeURIComponent(filename)}&type=${qsType}&subfolder=${qsSub}`;
                            iframeFilename = filename.replace(/\.ply$/i, ".spz");
                            loadLabel      = "Transcoding + downloading SPZ...";
                        } else {
                            filepath       = `/view?filename=${encodeURIComponent(filename)}&type=${qsType}&subfolder=${qsSub}`;
                            iframeFilename = filename;
                            loadLabel      = "Downloading PLY...";
                        }

                        // Function to fetch and send data to iframe
                        const fetchAndSend = async () => {
                            if (!iframe.contentWindow) {
                                console.error("[GaussianPack] Iframe contentWindow not available");
                                return;
                            }

                            try {
                                // Fetch the file from parent context (authenticated).
                                // Stream so we can log download progress.
                                const t0 = performance.now();
                                console.log("[GaussianPack] fetch start:", filepath);
                                node._showLoadBar?.(loadLabel);
                                const response = await fetch(filepath);
                                if (!response.ok) {
                                    // Bubble the server's body up — for /gaussianpack/spz
                                    // we put an actionable message there (e.g. "spz not installed").
                                    let body = "";
                                    try { body = (await response.text() || "").slice(0, 400); } catch (_) {}
                                    throw new Error(`HTTP ${response.status}: ${response.statusText}${body ? " - " + body : ""}`);
                                }
                                const totalHdr = response.headers.get("content-length");
                                const total = totalHdr ? parseInt(totalHdr, 10) : 0;
                                console.log(`[GaussianPack] fetch response 200; content-length=${total || "unknown"}`);

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
                                            if (chunks.length % 64 === 0) {
                                                console.log(`[GaussianPack] downloaded ${(received/(1024*1024)).toFixed(1)} MB so far`);
                                            }
                                        }
                                    }
                                }
                                // Stitch chunks into one ArrayBuffer.
                                const u8 = new Uint8Array(received);
                                let off = 0;
                                for (const c of chunks) { u8.set(c, off); off += c.byteLength; }
                                const arrayBuffer = u8.buffer;
                                const dt = performance.now() - t0;
                                const speed = received > 0 ? (received / (1024*1024)) / (dt / 1000) : 0;
                                console.log(`[GaussianPack] fetch done: ${arrayBuffer.byteLength} bytes in ${dt.toFixed(0)} ms (${speed.toFixed(1)} MB/s)`);

                                // Send the data to iframe — filename carries
                                // the extension so the renderer can sniff
                                // format (Spark/PlayCanvas both auto-detect).
                                console.log("[GaussianPack] posting LOAD_MESH_DATA to iframe, renderer:", renderer, "filename:", iframeFilename);
                                node._setLoadProgress?.(100, "Parsing splats...");
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_MESH_DATA",
                                    data: arrayBuffer,
                                    filename: iframeFilename,
                                    extrinsics: extrinsics,
                                    intrinsics: intrinsics,
                                    renderer: renderer,
                                    timestamp: Date.now()
                                }, "*", [arrayBuffer]);
                            } catch (error) {
                                console.error("[GaussianPack] Error fetching splat:", error);
                                infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error loading splat: ${error.message}</div>`;
                            }
                        };

                        // Fetch and send when iframe is ready
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
    }
});
