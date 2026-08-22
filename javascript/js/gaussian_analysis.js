import { app } from "../../../scripts/app.js";

// Show the GaussianAnalysis report inside the node, mirroring the in-node
// text-panel pattern used by comfy_3d_viewers (addDOMWidget + onExecuted).
app.registerExtension({
    name: "gaussianpack.analysisreport",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GaussianAnalysis") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const container = document.createElement("div");
            Object.assign(container.style, {
                padding: "8px",
                backgroundColor: "#1e1e1e",
                color: "#cfcfcf",
                fontSize: "11px",
                fontFamily: "monospace",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                overflow: "auto",
                height: "100%",
                boxSizing: "border-box",
                borderRadius: "4px",
                marginTop: "4px",
            });
            container.textContent = "Run to analyze the splat…";

            const widget = this.addDOMWidget("report_display", "TEXT_DISPLAY", container, {
                getValue() { return container.textContent; },
                setValue(v) { container.textContent = v ?? ""; },
                // Don't serialize the report into the workflow JSON.
                serialize: false,
            });
            widget.computeSize = () => [this.size?.[0] ?? 320, 220];

            this._reportContainer = container;

            const onExecuted = this.onExecuted;
            this.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const txt = message?.text?.[0];
                if (txt != null) this._reportContainer.textContent = txt;
            };

            // Give the node a sensible starting size for the panel.
            if (this.size && this.size[1] < 260) this.setSize([Math.max(this.size[0], 320), 300]);

            return r;
        };
    },
});
