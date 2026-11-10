import { createC2pa, Reader } from "@contentauth/c2pa-web";
import wasmSrc from "@contentauth/c2pa-web/resources/c2pa.wasm?url";
import "./style.css";

type ValidationState = "Trusted" | "Valid" | "Invalid" | null | undefined;

const app = document.querySelector<HTMLDivElement>("#app");

if (!app) {
  throw new Error("Application root element was not found.");
}

app.innerHTML = `
  <main class="page-shell">
    <header class="topbar">
      <a class="brand" href="#" aria-label="Image Credentials home">
        <span class="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 32 32" fill="none">
            <path d="M16 3.5 26 7v7.4c0 6.6-4.1 11.6-10 14.1-5.9-2.5-10-7.5-10-14.1V7l10-3.5Z" stroke="currentColor" stroke-width="1.8"/>
            <path d="m11.4 15.8 3.1 3.1 6.5-6.7" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </span>
        <span>Image<span class="brand-light">Credentials</span></span>
      </a>
      <span class="prototype-tag"><span></span> RESEARCH PROTOTYPE</span>
    </header>

    <section class="intro">
      <p class="eyebrow">IMAGE PROVENANCE CHECK</p>
      <h1>Know what your image can tell you.</h1>
      <p class="intro-copy">Inspect Content Credentials attached to an image and review its available provenance and validation information.</p>
    </section>

    <section class="workspace" aria-label="Image credential inspection">
      <div class="upload-panel" id="drop-zone">
        <input id="file-input" type="file" accept="image/*" hidden />
        <div class="upload-icon" aria-hidden="true">
          <svg viewBox="0 0 32 32" fill="none">
            <path d="M16 21V6m0 0-5.5 5.5M16 6l5.5 5.5M6.5 20.5v4A2.5 2.5 0 0 0 9 27h14a2.5 2.5 0 0 0 2.5-2.5v-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <div class="upload-content">
          <h2>Choose an image to inspect</h2>
          <p>Drop an image here, or <button class="text-button" id="browse-button" type="button">browse files</button></p>
          <span class="file-hint">JPG, PNG, WebP or other supported image · up to 25 MB</span>
        </div>
        <button class="choose-button" id="choose-button" type="button">Select image</button>
      </div>

      <p class="privacy-note"><span class="lock-icon" aria-hidden="true">⌑</span> Your image is inspected in this browser and is not uploaded by this prototype.</p>

      <div class="notice" id="notice" role="status" aria-live="polite" hidden></div>

      <section class="result-card" id="result-card" aria-live="polite" hidden>
        <div class="result-heading">
          <div>
            <p class="eyebrow">INSPECTION RESULT</p>
            <h2 id="result-title"></h2>
          </div>
          <span class="state-badge" id="state-badge"></span>
        </div>
        <div class="asset-row">
          <div class="preview-frame">
            <img id="image-preview" alt="Preview of the selected image" />
          </div>
          <div class="asset-details">
            <strong id="file-name"></strong>
            <span id="file-facts"></span>
            <button type="button" class="text-button change-button" id="change-button">Inspect another image</button>
          </div>
        </div>
        <div class="result-body" id="result-body"></div>
        <details class="raw-details">
          <summary>View credential data</summary>
          <pre id="raw-data"></pre>
        </details>
      </section>
    </section>

    <aside class="caution">
      <span class="caution-icon" aria-hidden="true">i</span>
      <p><strong>Credentials are one signal, not a verdict.</strong> Missing credentials do not mean an image is AI-generated or fraudulent. A credential check alone cannot establish whether a claim is genuine.</p>
    </aside>

    <footer>
      <span>Powered by the open-source C2PA JavaScript SDK</span>
      <a href="https://contentcredentials.org/" target="_blank" rel="noreferrer">About Content Credentials <span aria-hidden="true">↗</span></a>
    </footer>
  </main>
`;

const input = requiredElement<HTMLInputElement>("#file-input");
const dropZone = requiredElement<HTMLDivElement>("#drop-zone");
const notice = requiredElement<HTMLDivElement>("#notice");
const resultCard = requiredElement<HTMLElement>("#result-card");
const resultTitle = requiredElement<HTMLHeadingElement>("#result-title");
const stateBadge = requiredElement<HTMLSpanElement>("#state-badge");
const imagePreview = requiredElement<HTMLImageElement>("#image-preview");
const fileName = requiredElement<HTMLElement>("#file-name");
const fileFacts = requiredElement<HTMLElement>("#file-facts");
const resultBody = requiredElement<HTMLDivElement>("#result-body");
const rawData = requiredElement<HTMLPreElement>("#raw-data");

let c2paPromise: ReturnType<typeof createC2pa> | undefined;
let previewUrl: string | undefined;

requiredElement<HTMLButtonElement>("#browse-button").addEventListener("click", () => input.click());
requiredElement<HTMLButtonElement>("#choose-button").addEventListener("click", () => input.click());
requiredElement<HTMLButtonElement>("#change-button").addEventListener("click", () => input.click());

input.addEventListener("change", () => {
  const file = input.files?.[0];
  if (file) void inspectFile(file);
  input.value = "";
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("is-dragging");
});

dropZone.addEventListener("dragleave", (event) => {
  if (!dropZone.contains(event.relatedTarget as Node | null)) {
    dropZone.classList.remove("is-dragging");
  }
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");
  const file = event.dataTransfer?.files[0];
  if (file) void inspectFile(file);
});

async function inspectFile(file: File): Promise<void> {
  clearNotice();
  resultCard.hidden = true;

  if (!file.type.startsWith("image/")) {
    showNotice("Choose an image file to inspect.", "error");
    return;
  }

  if (file.size > 25 * 1024 * 1024) {
    showNotice("This image is larger than 25 MB. Choose a smaller file.", "error");
    return;
  }

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  fileName.textContent = file.name;
  fileFacts.textContent = `${formatBytes(file.size)} · ${file.type || "Image"}`;

  showNotice("Reading image credentials…", "loading");

  try {
    c2paPromise ??= createC2pa({ wasmSrc });
    const c2pa = await c2paPromise;
    const reader = await Reader.fromBlob(c2pa, file.type, file);

    if (!reader) {
      showNoCredentials();
      return;
    }

    try {
      const store = await reader.manifestStore();
      const manifest = await reader.activeManifest();
      showCredentials(store, manifest, file.name);
    } finally {
      await reader.free();
    }
  } catch (error) {
    const detail = error instanceof Error ? error.message : "An unexpected error occurred.";
    showNotice(`Could not inspect this image. ${detail}`, "error");
  }
}

function showNoCredentials(): void {
  clearNotice();
  resultCard.hidden = false;
  resultTitle.textContent = "No Content Credentials found";
  setBadge("Not found", "neutral");
  resultBody.replaceChildren(
    makeParagraph("This image did not contain a C2PA manifest that the verifier could read."),
    makeParagraph("That does not tell us whether the image is real, AI-generated, edited, or fraudulent.", "muted"),
  );
  rawData.textContent = "No C2PA manifest was detected in this image.";
}

function showCredentials(
  store: Awaited<ReturnType<Reader["manifestStore"]>>,
  manifest: Awaited<ReturnType<Reader["activeManifest"]>>,
  sourceName: string,
): void {
  clearNotice();
  resultCard.hidden = false;
  resultTitle.textContent = "Content Credentials found";

  const state: ValidationState = store.validation_state;
  const stateInfo = describeValidationState(state);
  setBadge(stateInfo.label, stateInfo.tone);

  const fields: Array<[string, string]> = [
    ["Source file", manifest.title || sourceName],
    ["Format", manifest.format || "Not recorded"],
    ["Created with", manifest.claim_generator_info?.map((item) => item.name).filter(Boolean).join(", ") || manifest.claim_generator || "Not recorded"],
    ["Signer", manifest.signature_info?.common_name || manifest.signature_info?.issuer || "Not recorded"],
    ["Signed", manifest.signature_info?.time || "Not recorded"],
  ];

  const metadataSection = document.createElement("section");
  metadataSection.className = "metadata-section";
  const heading = document.createElement("h3");
  heading.textContent = "Provenance details";
  metadataSection.append(heading);
  const list = document.createElement("dl");
  list.className = "metadata-list";

  for (const [label, value] of fields) {
    const row = document.createElement("div");
    row.className = "metadata-item";
    const term = document.createElement("dt");
    term.textContent = label;
    const description = document.createElement("dd");
    description.textContent = value;
    row.append(term, description);
    list.append(row);
  }
  metadataSection.append(list);

  const validationSection = document.createElement("section");
  validationSection.className = "validation-section";
  const validationHeading = document.createElement("h3");
  validationHeading.textContent = "Credential validation";
  validationSection.append(validationHeading);
  validationSection.append(makeParagraph(stateInfo.description, "validation-summary"));

  const statuses = store.validation_status ?? [];
  if (statuses.length > 0) {
    const statusList = document.createElement("ul");
    statusList.className = "status-list";
    for (const status of statuses) {
      const item = document.createElement("li");
      const statusTitle = document.createElement("strong");
      statusTitle.textContent = status.code;
      item.append(statusTitle);
      if (status.explanation) {
        const explanation = document.createElement("span");
        explanation.textContent = status.explanation;
        item.append(explanation);
      }
      statusList.append(item);
    }
    validationSection.append(statusList);
  } else {
    validationSection.append(makeParagraph("No additional validation messages were returned.", "muted"));
  }

  resultBody.replaceChildren(metadataSection, validationSection);
  rawData.textContent = JSON.stringify(store, null, 2);
}

function describeValidationState(state: ValidationState): { label: string; tone: string; description: string } {
  switch (state) {
    case "Trusted":
      return {
        label: "Trusted",
        tone: "positive",
        description: "The SDK reports that the credential is trusted according to its configured trust settings.",
      };
    case "Valid":
      return {
        label: "Valid",
        tone: "positive",
        description: "The SDK reports a structurally and cryptographically valid credential. This does not necessarily mean its signer is trusted.",
      };
    case "Invalid":
      return {
        label: "Invalid",
        tone: "negative",
        description: "The SDK reports that this credential did not pass validation. Review the messages and investigate the image separately.",
      };
    default:
      return {
        label: "Status unavailable",
        tone: "neutral",
        description: "A credential was found, but the SDK did not return an overall validation state.",
      };
  }
}

function setBadge(label: string, tone: string): void {
  stateBadge.textContent = label;
  stateBadge.className = `state-badge ${tone}`;
}

function makeParagraph(text: string, className = ""): HTMLParagraphElement {
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  if (className) paragraph.className = className;
  return paragraph;
}

function showNotice(message: string, kind: "loading" | "error"): void {
  notice.hidden = false;
  notice.className = `notice ${kind}`;
  notice.textContent = message;
}

function clearNotice(): void {
  notice.hidden = true;
  notice.textContent = "";
}

function formatBytes(bytes: number): string {
  return bytes < 1024 * 1024
    ? `${Math.max(1, Math.round(bytes / 1024))} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function requiredElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Required element not found: ${selector}`);
  return element;
}
